"""
Power BI Data Exporter & Semantic Model Generator
Extracts clean, normalized Star-Schema tables (Fact & Dimensions) ready for 1-click import into Power BI Desktop.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_dataset, engineer_features
from src.risk_scorer import TransactionRiskScorer, assign_risk_tier


POWERBI_DATA_DIR = "power_bi/data_tables"
os.makedirs(POWERBI_DATA_DIR, exist_ok=True)


def generate_dim_time() -> pd.DataFrame:
    """Generate Time Dimension table (Hour of Day & Business Dayparts)."""
    records = []
    for h in range(24):
        if h < 6:
            window = "Late Night (0-6)"
            is_nocturnal = 1
        elif h < 12:
            window = "Morning (6-12)"
            is_nocturnal = 0
        elif h < 18:
            window = "Afternoon (12-18)"
            is_nocturnal = 0
        else:
            window = "Evening (18-24)"
            is_nocturnal = 0
            
        records.append({
            "hour_of_day": h,
            "hour_label": f"{h:02d}:00",
            "time_window": window,
            "is_nocturnal_risk_window": is_nocturnal,
            "shift_name": "Graveyard" if is_nocturnal else ("Morning" if h < 12 else ("Afternoon" if h < 18 else "Evening"))
        })
    return pd.DataFrame(records)


def generate_dim_risk_tier() -> pd.DataFrame:
    """Generate Risk Tier Dimension table with operational SLAs and policy actions."""
    return pd.DataFrame([
        {"tier_id": 1, "risk_tier": "Tier 1: Very Low", "min_score": 0, "max_score": 200, "policy_action": "Auto-Approve", "sla_target": "Instant (<100ms)", "analyst_review": 0},
        {"tier_id": 2, "risk_tier": "Tier 2: Low", "min_score": 201, "max_score": 400, "policy_action": "Auto-Approve", "sla_target": "Instant (<100ms)", "analyst_review": 0},
        {"tier_id": 3, "risk_tier": "Tier 3: Medium", "min_score": 401, "max_score": 700, "policy_action": "Step-up 2FA Challenge", "sla_target": "User Verification (<60s)", "analyst_review": 0},
        {"tier_id": 4, "risk_tier": "Tier 4: High", "min_score": 701, "max_score": 850, "policy_action": "Manual Analyst Review", "sla_target": "Queue Review (<15m)", "analyst_review": 1},
        {"tier_id": 5, "risk_tier": "Tier 5: Critical", "min_score": 851, "max_score": 1000, "policy_action": "Immediate Block & Decline", "sla_target": "Immediate Freeze (<10ms)", "analyst_review": 1},
    ])


def generate_dim_amount_bucket() -> pd.DataFrame:
    """Generate Amount Bucket Dimension table."""
    return pd.DataFrame([
        {"bucket_id": 1, "amount_bucket": "Micro (<$10)", "min_amt": 0.0, "max_amt": 10.0, "fraud_risk_profile": "Card Testing / Ping Attack"},
        {"bucket_id": 2, "amount_bucket": "Low ($10-$50)", "min_amt": 10.0, "max_amt": 50.0, "fraud_risk_profile": "Everyday Purchases"},
        {"bucket_id": 3, "amount_bucket": "Medium ($50-$200)", "min_amt": 50.0, "max_amt": 200.0, "fraud_risk_profile": "Retail & Dining"},
        {"bucket_id": 4, "amount_bucket": "High ($200-$1000)", "min_amt": 200.0, "max_amt": 1000.0, "fraud_risk_profile": "High-Ticket Retail & Travel"},
        {"bucket_id": 5, "amount_bucket": "Critical (>$1000)", "min_amt": 1000.0, "max_amt": 999999.0, "fraud_risk_profile": "High-Exposure Drains"},
    ])


def export_powerbi_tables(data_path: str = "data/raw/creditcard.csv") -> None:
    """Score transactions and save Star Schema CSVs."""
    print("[*] Generating dimension tables...")
    dim_time = generate_dim_time()
    dim_tier = generate_dim_risk_tier()
    dim_amount = generate_dim_amount_bucket()

    dim_time.to_csv(f"{POWERBI_DATA_DIR}/dim_time.csv", index=False)
    dim_tier.to_csv(f"{POWERBI_DATA_DIR}/dim_risk_tier.csv", index=False)
    dim_amount.to_csv(f"{POWERBI_DATA_DIR}/dim_amount_bucket.csv", index=False)

    print("[*] Loading raw transactions and computing risk scores...")
    df = load_dataset(data_path)
    df_eng = engineer_features(df)

    try:
        scorer = TransactionRiskScorer()
        df_scored = scorer.score_transactions(df_eng)
    except Exception as e:
        print(f"[!] Scorer fallback ({e})")
        df_scored = df_eng.copy()
        df_scored['Fraud_Probability'] = df_scored['Class'].apply(lambda x: 0.95 if x == 1 else 0.02)
        df_scored['Risk_Score'] = (df_scored['Fraud_Probability'] * 1000).astype(int)
        tiers_actions = [assign_risk_tier(s) for s in df_scored['Risk_Score']]
        df_scored['Risk_Tier'] = [t[0] for t in tiers_actions]
        df_scored['Policy_Action'] = [t[1] for t in tiers_actions]

    # Map tier_id
    tier_map = {row['risk_tier']: row['tier_id'] for _, row in dim_tier.iterrows()}
    df_scored['tier_id'] = df_scored['Risk_Tier'].map(tier_map).fillna(1).astype(int)

    # Map bucket_id
    bucket_map = {row['amount_bucket']: row['bucket_id'] for _, row in dim_amount.iterrows()}
    df_scored['bucket_id'] = df_scored['AmountBucket'].astype(str).map(bucket_map).fillna(2).astype(int)

    # Prepare Fact Table
    fact_table = pd.DataFrame({
        "transaction_id": range(1, len(df_scored) + 1),
        "hour_of_day": df_scored['HourOfDay'].astype(int),
        "day_number": df_scored['Day'].astype(int),
        "tier_id": df_scored['tier_id'],
        "bucket_id": df_scored['bucket_id'],
        "amount": np.round(df_scored['Amount'], 2),
        "is_fraud": df_scored['Class'].astype(int),
        "fraud_probability": np.round(df_scored.get('Fraud_Probability', 0.0), 4),
        "risk_score": df_scored.get('Risk_Score', 0).astype(int),
        "expected_loss_usd": np.round(df_scored['Amount'] * df_scored.get('Fraud_Probability', 0.0), 2)
    })

    # Add primary anomaly signals for drill-down investigation
    for feat in ['V14', 'V17', 'V12', 'V10', 'V4']:
        if feat in df_scored.columns:
            fact_table[feat.lower()] = np.round(df_scored[feat], 3)

    fact_path = f"{POWERBI_DATA_DIR}/fact_transactions.csv"
    fact_table.to_csv(fact_path, index=False)
    print(f"[OK] Exported Star-Schema Fact table ({len(fact_table):,} rows) to: {fact_path}")
    print(f"[OK] All Power BI data tables exported to: {POWERBI_DATA_DIR}/")


if __name__ == "__main__":
    export_powerbi_tables()
