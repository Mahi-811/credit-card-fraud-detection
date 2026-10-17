"""
Unit Tests for Machine Learning Models & Risk Scoring Engine
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_dataset, engineer_features
from src.preprocessing import FraudDataPreprocessor
from src.models import ModelTrainer
from src.risk_scorer import TransactionRiskScorer, assign_risk_tier


@pytest.fixture
def prepared_data():
    """Load and preprocess sample dataset for rapid testing."""
    sample_path = "data/raw/sample_creditcard.csv"
    if not os.path.exists(sample_path):
        from data.generate_sample_data import generate_credit_card_data
        df = generate_credit_card_data(n_samples=500, fraud_ratio=0.02, random_state=42)
        df.to_csv(sample_path, index=False)
    
    df = load_dataset(sample_path)
    df_eng = engineer_features(df)
    preprocessor = FraudDataPreprocessor(random_state=42)
    X, y = preprocessor.prepare_features_and_target(df_eng)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.3)
    X_tr_s, X_te_s = preprocessor.fit_transform(X_train, X_test)
    return X_tr_s, y_train.values, X_te_s, y_test.values, preprocessor


def test_model_training_and_metrics(prepared_data):
    """Test model training produces valid metric ranges."""
    X_train, y_train, X_test, y_test, _ = prepared_data
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, quick_run=True)

    assert len(results) == 2
    for _, row in results.iterrows():
        assert 0.0 <= row['Accuracy'] <= 1.0
        assert 0.0 <= row['Precision'] <= 1.0
        assert 0.0 <= row['Recall'] <= 1.0
        assert 0.0 <= row['F1-Score'] <= 1.0
        assert 0.5 <= row['ROC-AUC'] <= 1.0
        assert 0.0 <= row['PR-AUC'] <= 1.0


def test_risk_tier_assignment():
    """Test mapping of scores to tiers and policy decisions."""
    tier1, act1 = assign_risk_tier(50)
    assert "Very Low" in tier1
    assert "Auto-Approve" in act1

    tier3, act3 = assign_risk_tier(550)
    assert "Medium" in tier3
    assert "2FA" in act3

    tier5, act5 = assign_risk_tier(950)
    assert "Critical" in tier5
    assert "Block" in act5 or "Decline" in act5


def test_risk_scorer_inference():
    """Test that scorer generates calibrated scores and expected loss."""
    # Ensure model is saved
    model_path = "models/saved_models/best_fraud_model.joblib"
    prep_path = "models/saved_models/preprocessor.joblib"
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        from src.models import run_pipeline
        run_pipeline(quick_run=True)

    scorer = TransactionRiskScorer(model_path=model_path, preprocessor_path=prep_path)
    
    # Test batch scoring
    sample_df = load_dataset("data/raw/sample_creditcard.csv").head(10)
    df_scored = scorer.score_transactions(sample_df)

    assert 'Fraud_Probability' in df_scored.columns
    assert 'Risk_Score' in df_scored.columns
    assert 'Risk_Tier' in df_scored.columns
    assert 'Policy_Action' in df_scored.columns
    assert 'Expected_Financial_Loss' in df_scored.columns

    assert df_scored['Risk_Score'].between(0, 1000).all()
    assert df_scored['Fraud_Probability'].between(0.0, 1.0).all()
