"""
Transaction Risk Scoring & Policy Engine
Calibrates model probabilities into a 0-1000 Risk Score, decile segmentation,
and automated financial policy actions (Approve, 2FA Challenge, Manual Review, Decline).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


RISK_TIERS = [
    {"tier": "Tier 1: Very Low", "min": 0, "max": 200, "action": "Auto-Approve", "color": "#10B981"},
    {"tier": "Tier 2: Low", "min": 201, "max": 400, "action": "Auto-Approve", "color": "#3B82F6"},
    {"tier": "Tier 3: Medium", "min": 401, "max": 700, "action": "Step-up 2FA Challenge", "color": "#F59E0B"},
    {"tier": "Tier 4: High", "min": 701, "max": 850, "action": "Manual Analyst Review", "color": "#EA580C"},
    {"tier": "Tier 5: Critical", "min": 851, "max": 1000, "action": "Immediate Block & Decline", "color": "#EF4444"},
]


def assign_risk_tier(score: int) -> Tuple[str, str]:
    """Map numeric risk score (0-1000) to policy tier and decision action."""
    for rt in RISK_TIERS:
        if rt["min"] <= score <= rt["max"]:
            return rt["tier"], rt["action"]
    return "Tier 5: Critical", "Immediate Block & Decline"


class TransactionRiskScorer:
    """
    Applies production risk scoring, loss estimation, and decile lift analysis.
    """

    def __init__(self, model_path: str = "models/saved_models/best_fraud_model.joblib",
                 preprocessor_path: str = "models/saved_models/preprocessor.joblib"):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Model or Preprocessor artifact missing at {model_path}. Train model first.")
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich input DataFrame with:
        - Fraud_Probability (0.000 to 1.000)
        - Risk_Score (0 to 1000)
        - Risk_Tier
        - Policy_Action
        - Expected_Financial_Loss ($)
        """
        df_scored = df.copy()
        
        # Ensure engineered columns exist
        from src.data_loader import engineer_features
        df_eng = engineer_features(df_scored)
        
        X, _ = self.preprocessor.prepare_features_and_target(df_eng)
        X_scaled = self.preprocessor.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        risk_scores = np.clip(np.round(probabilities * 1000).astype(int), 0, 1000)

        df_scored['Fraud_Probability'] = np.round(probabilities, 4)
        df_scored['Risk_Score'] = risk_scores
        
        tiers_actions = [assign_risk_tier(s) for s in risk_scores]
        df_scored['Risk_Tier'] = [t[0] for t in tiers_actions]
        df_scored['Policy_Action'] = [t[1] for t in tiers_actions]
        df_scored['Expected_Financial_Loss'] = np.round(df_scored['Amount'] * df_scored['Fraud_Probability'], 2)

        return df_scored

    def calculate_decile_lift(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a decile table showing cumulative fraud capture (Gain & Lift).
        Essential for executive presentations and Power BI decile visual.
        """
        df_work = df_scored.copy()
        df_work['Decile'] = pd.qcut(df_work['Risk_Score'].rank(method='first', ascending=False), q=10, labels=range(1, 11))

        total_frauds = df_work['Class'].sum()
        decile_summary = df_work.groupby('Decile', observed=False).agg(
            Total_Transactions=('Class', 'count'),
            Fraud_Count=('Class', 'sum'),
            Total_Amount=('Amount', 'sum'),
            Fraud_Amount=('Amount', lambda x: x[df_work.loc[x.index, 'Class'] == 1].sum()),
            Avg_Risk_Score=('Risk_Score', 'mean')
        ).reset_index()

        decile_summary['Fraud_Rate_Pct'] = (decile_summary['Fraud_Count'] / decile_summary['Total_Transactions']) * 100.0
        decile_summary['Cumulative_Frauds'] = decile_summary['Fraud_Count'].cumsum()
        decile_summary['Cumulative_Fraud_Pct'] = (decile_summary['Cumulative_Frauds'] / total_frauds) * 100.0 if total_frauds > 0 else 0
        decile_summary['Lift'] = decile_summary['Cumulative_Fraud_Pct'] / (decile_summary['Decile'].astype(int) * 10.0)

        return decile_summary.round(2)


def main():
    from src.data_loader import load_dataset
    print("[*] Loading transactions to score...")
    df = load_dataset()

    scorer = TransactionRiskScorer()
    print("[*] Scoring transactions and calculating policy actions...")
    df_scored = scorer.score_transactions(df)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/scored_transactions.csv"
    df_scored.to_csv(output_path, index=False)
    print(f"[OK] Scored dataset saved to: {output_path}")

    print("\n[*] Calculating Decile Lift Analysis:")
    decile_df = scorer.calculate_decile_lift(df_scored)
    print(decile_df.to_string(index=False))

    decile_path = "reports/risk_decile_analysis.csv"
    decile_df.to_csv(decile_path, index=False)
    print(f"[OK] Decile analysis saved to: {decile_path}")


if __name__ == "__main__":
    main()
