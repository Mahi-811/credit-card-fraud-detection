"""
Unit Tests for Data Loader & Preprocessing Pipeline
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_dataset, engineer_features, get_dataset_summary
from src.preprocessing import FraudDataPreprocessor


@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    sample_path = "data/raw/sample_creditcard.csv"
    if not os.path.exists(sample_path):
        from data.generate_sample_data import generate_credit_card_data
        df = generate_credit_card_data(n_samples=500, fraud_ratio=0.01, random_state=42)
        df.to_csv(sample_path, index=False)
    return load_dataset(sample_path)


def test_data_loading_schema(sample_data):
    """Verify loaded data has required columns and valid datatypes."""
    assert len(sample_data) > 0
    assert 'Time' in sample_data.columns
    assert 'Amount' in sample_data.columns
    assert 'Class' in sample_data.columns
    assert sample_data['Class'].isin([0, 1]).all()
    for i in range(1, 29):
        assert f'V{i}' in sample_data.columns


def test_feature_engineering(sample_data):
    """Test temporal and amount bucket derivation."""
    df_eng = engineer_features(sample_data)
    assert 'HourOfDay' in df_eng.columns
    assert 'Day' in df_eng.columns
    assert 'TimeWindow' in df_eng.columns
    assert 'AmountBucket' in df_eng.columns
    assert 'LogAmount' in df_eng.columns

    # Hour of day must be in [0, 23]
    assert df_eng['HourOfDay'].between(0, 23).all()
    # LogAmount must be >= 0
    assert (df_eng['LogAmount'] >= 0).all()


def test_summary_metrics(sample_data):
    """Test dataset summary computation."""
    summary = get_dataset_summary(sample_data)
    assert summary['total_transactions'] == len(sample_data)
    assert summary['fraud_transactions'] >= 0
    assert 0.0 <= summary['fraud_rate_pct'] <= 100.0
    assert summary['total_amount'] > 0


def test_preprocessor_split_and_scale(sample_data):
    """Test RobustScaler transformation and stratification."""
    df_eng = engineer_features(sample_data)
    preprocessor = FraudDataPreprocessor(random_state=42)

    X, y = preprocessor.prepare_features_and_target(df_eng)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.25)

    assert len(X_train) == int(len(sample_data) * 0.75)
    assert len(X_test) == len(sample_data) - len(X_train)

    # Scaled transforms
    X_tr_s, X_te_s = preprocessor.fit_transform(X_train, X_test)
    assert X_tr_s.shape == X_train.shape
    assert X_te_s.shape == X_test.shape
    assert not np.isnan(X_tr_s).any()
    assert not np.isnan(X_te_s).any()
