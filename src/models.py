"""
Model Training & Classification Module
Implements Logistic Regression and Random Forest models optimized for severe class imbalance.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, accuracy_score
)

from src.data_loader import load_dataset, engineer_features
from src.preprocessing import FraudDataPreprocessor


class ModelTrainer:
    """
    Trains, tunes, and serializes classification models for fraud detection.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def get_models(self, quick_run: bool = False) -> Dict[str, Any]:
        """Define classifier architectures with cost-sensitive class balancing."""
        n_estimators = 30 if quick_run else 100
        max_depth = 10 if quick_run else 14

        return {
            "Logistic Regression (Balanced)": LogisticRegression(
                class_weight='balanced',
                solver='liblinear',
                C=0.1,
                max_iter=1000,
                random_state=self.random_state
            ),
            "Random Forest (Balanced Subsample)": RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight='balanced_subsample',
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state
            )
        }

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        quick_run: bool = False
    ) -> pd.DataFrame:
        """
        Train each model, predict probabilities, and calculate comprehensive evaluation metrics.
        """
        self.models = self.get_models(quick_run=quick_run)
        metrics_list = []

        for name, clf in self.models.items():
            print(f"[*] Training {name}...")
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)

            res = {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4)
            }
            self.results[name] = res
            metrics_list.append(res)
            print(f"    [+] Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        comparison_df = pd.DataFrame(metrics_list)
        return comparison_df

    def save_artifacts(
        self,
        preprocessor: FraudDataPreprocessor,
        output_dir: str = "models/saved_models"
    ) -> None:
        """Serialize best model, preprocessor, and metadata."""
        os.makedirs(output_dir, exist_ok=True)

        # Select model with highest PR-AUC / F1
        best_name = max(self.results, key=lambda k: self.results[k]["PR-AUC"])
        best_model = self.models[best_name]

        model_path = os.path.join(output_dir, "best_fraud_model.joblib")
        preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
        metadata_path = os.path.join(output_dir, "model_metadata.joblib")

        joblib.dump(best_model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump({
            "best_model_name": best_name,
            "metrics": self.results[best_name],
            "all_results": self.results,
            "feature_names": preprocessor.feature_names
        }, metadata_path)

        print(f"[OK] Best Model ({best_name}) saved to: {model_path}")
        print(f"[OK] Preprocessor saved to: {preprocessor_path}")


def run_pipeline(
    data_path: str = "data/raw/creditcard.csv",
    quick_run: bool = False
) -> pd.DataFrame:
    """End-to-end execution of data load, preprocessing, training, and artifact export."""
    df = load_dataset(data_path)
    df_engineered = engineer_features(df)

    preprocessor = FraudDataPreprocessor()
    X, y = preprocessor.prepare_features_and_target(df_engineered)
    X_train_df, X_test_df, y_train, y_test = preprocessor.split_data(X, y)

    X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train_df, X_test_df)

    trainer = ModelTrainer()
    summary_df = trainer.train_and_evaluate(
        X_train_scaled, y_train.values,
        X_test_scaled, y_test.values,
        quick_run=quick_run
    )

    trainer.save_artifacts(preprocessor)

    # Save summary report
    os.makedirs("reports", exist_ok=True)
    summary_df.to_csv("reports/model_performance_summary.csv", index=False)
    print("[OK] Performance summary written to: reports/model_performance_summary.csv")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection Classifiers")
    parser.add_argument("--data-path", type=str, default="data/raw/creditcard.csv", help="Path to input dataset")
    parser.add_argument("--quick-run", action="store_true", help="Run with reduced trees for rapid testing")
    args = parser.parse_args()

    run_pipeline(data_path=args.data_path, quick_run=args.quick_run)
