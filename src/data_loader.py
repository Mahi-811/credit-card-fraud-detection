"""
Data Loader & Feature Engineering Pipeline
Handles dataset loading, validation, and temporal/transactional feature engineering.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


REQUIRED_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

AMOUNT_BINS = [-np.inf, 10.0, 50.0, 200.0, 1000.0, np.inf]
AMOUNT_LABELS = ['Micro (<$10)', 'Low ($10-$50)', 'Medium ($50-$200)', 'High ($200-$1000)', 'Critical (>$1000)']


def load_dataset(
    filepath: Optional[str] = None,
    fallback_sample_path: str = "data/raw/sample_creditcard.csv"
) -> pd.DataFrame:
    """
    Load transactions dataset from primary path, fallback sample, or generate if missing.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to main creditcard.csv
    fallback_sample_path : str
        Path to smaller sample dataset for quick testing
        
    Returns:
    --------
    pd.DataFrame
        Loaded and validated DataFrame
    """
    target_path = None
    if filepath and os.path.exists(filepath):
        target_path = filepath
    elif os.path.exists("data/raw/creditcard.csv"):
        target_path = "data/raw/creditcard.csv"
    elif os.path.exists(fallback_sample_path):
        target_path = fallback_sample_path
    else:
        # Generate sample on the fly
        from data.generate_sample_data import generate_credit_card_data
        print("[!] No data found on disk. Generating 10,000 synthetic transactions...")
        df = generate_credit_card_data(n_samples=10000, fraud_ratio=0.003, random_state=42)
        os.makedirs(os.path.dirname(fallback_sample_path), exist_ok=True)
        df.to_csv(fallback_sample_path, index=False)
        target_path = fallback_sample_path

    print(f"[*] Loading dataset from: {target_path}")
    df = pd.read_csv(target_path)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate schema and data integrity."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"[!] Warning: Found {null_count} missing values. Imputing with column medians.")
        df.fillna(df.median(numeric_only=True), inplace=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich raw transaction data with business and temporal features:
    - Hour of Day (0 - 23)
    - Day Number (Day 1 vs Day 2)
    - Time Window Segment (Late Night, Morning, Afternoon, Evening)
    - Amount Buckets for Risk Stratification
    - Log-transformed Amount for Skew Mitigation
    """
    df = df.copy()

    # Temporal feature engineering
    # Time is elapsed seconds from 1st transaction
    df['HourOfDay'] = ((df['Time'] // 3600) % 24).astype(int)
    df['Day'] = ((df['Time'] // 86400) + 1).astype(int)

    # Time Window Categorization
    time_bins = [-1, 6, 12, 18, 24]
    time_labels = ['Late Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
    df['TimeWindow'] = pd.cut(df['HourOfDay'], bins=time_bins, labels=time_labels)

    # Amount Risk Stratification
    df['AmountBucket'] = pd.cut(df['Amount'], bins=AMOUNT_BINS, labels=AMOUNT_LABELS)

    # Log1p transformation for amount (handles heavy right skew)
    df['LogAmount'] = np.log1p(df['Amount'])

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive summary metrics from dataset."""
    total_tx = len(df)
    fraud_tx = int(df['Class'].sum())
    legit_tx = total_tx - fraud_tx
    fraud_rate = (fraud_tx / total_tx) * 100.0 if total_tx > 0 else 0.0

    total_amount = float(df['Amount'].sum())
    fraud_amount = float(df[df['Class'] == 1]['Amount'].sum())
    legit_amount = float(df[df['Class'] == 0]['Amount'].sum())
    fraud_loss_pct = (fraud_amount / total_amount) * 100.0 if total_amount > 0 else 0.0

    return {
        'total_transactions': total_tx,
        'legit_transactions': legit_tx,
        'fraud_transactions': fraud_tx,
        'fraud_rate_pct': fraud_rate,
        'total_amount': total_amount,
        'legit_amount': legit_amount,
        'fraud_amount': fraud_amount,
        'fraud_loss_pct': fraud_loss_pct,
        'avg_legit_amount': float(df[df['Class'] == 0]['Amount'].mean()),
        'avg_fraud_amount': float(df[df['Class'] == 1]['Amount'].mean()),
        'max_amount': float(df['Amount'].max()),
        'median_amount': float(df['Amount'].median()),
    }
