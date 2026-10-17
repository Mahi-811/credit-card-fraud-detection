"""
Data Generation & Download Utility for Credit Card Fraud Detection

This script generates a statistically faithful synthetic dataset matching
the Kaggle Credit Card Fraud Detection schema (Time, V1-V28, Amount, Class).
It models:
- Extreme class imbalance (~0.2% - 0.5% fraud rate)
- Key PCA feature separation (V14, V17, V12, V10, V4) observed in actual fraud cases
- Diurnal time-of-day trends with heightened fraud activity during overnight hours (2 AM - 5 AM)
- Right-skewed transaction amount distributions with distinct fraudulent amount patterns
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def generate_credit_card_data(
    n_samples: int = 50000,
    fraud_ratio: float = 0.0025,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transactions matching Kaggle creditcard.csv schema.
    
    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions (default 0.25%).
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns [Time, V1..V28, Amount, Class]
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # 1. Generate Elapsed Time (seconds over 48 hours = 172,800 seconds)
    # Legitimate transactions peak during daytime/evening
    # Fraud transactions have higher relative density during nocturnal hours
    day_seconds = 86400

    # Legit time generation: mixture of daytime peaks (noon and early evening)
    legit_hours = rng.choice(
        np.arange(0, 48),
        size=n_legit,
        p=np.array([
            0.015, 0.010, 0.008, 0.007, 0.008, 0.012, 0.020, 0.035,
            0.050, 0.065, 0.070, 0.070, 0.075, 0.070, 0.065, 0.060,
            0.055, 0.060, 0.065, 0.060, 0.050, 0.040, 0.030, 0.020,
            0.015, 0.010, 0.008, 0.007, 0.008, 0.012, 0.020, 0.035,
            0.050, 0.065, 0.070, 0.070, 0.075, 0.070, 0.065, 0.060,
            0.055, 0.060, 0.065, 0.060, 0.050, 0.040, 0.030, 0.020
        ]) / np.sum([
            0.015, 0.010, 0.008, 0.007, 0.008, 0.012, 0.020, 0.035,
            0.050, 0.065, 0.070, 0.070, 0.075, 0.070, 0.065, 0.060,
            0.055, 0.060, 0.065, 0.060, 0.050, 0.040, 0.030, 0.020,
            0.015, 0.010, 0.008, 0.007, 0.008, 0.012, 0.020, 0.035,
            0.050, 0.065, 0.070, 0.070, 0.075, 0.070, 0.065, 0.060,
            0.055, 0.060, 0.065, 0.060, 0.050, 0.040, 0.030, 0.020
        ])
    )
    legit_time = legit_hours * 3600 + rng.uniform(0, 3600, size=n_legit)

    # Fraud time generation: elevated probability at night (hours 1-5, 25-29)
    night_boost = np.array([
        0.045, 0.055, 0.065, 0.070, 0.060, 0.045, 0.030, 0.025,
        0.025, 0.030, 0.035, 0.035, 0.040, 0.040, 0.035, 0.035,
        0.035, 0.040, 0.040, 0.045, 0.045, 0.040, 0.040, 0.040,
        0.045, 0.055, 0.065, 0.070, 0.060, 0.045, 0.030, 0.025,
        0.025, 0.030, 0.035, 0.035, 0.040, 0.040, 0.035, 0.035,
        0.035, 0.040, 0.040, 0.045, 0.045, 0.040, 0.040, 0.040
    ])
    fraud_hours = rng.choice(np.arange(0, 48), size=n_fraud, p=night_boost / np.sum(night_boost))
    fraud_time = fraud_hours * 3600 + rng.uniform(0, 3600, size=n_fraud)

    time_all = np.concatenate([legit_time, fraud_time])

    # 2. Generate Amount
    # Legitimate: Lognormal distribution (typical card purchases, mean ~$88)
    legit_amounts = rng.lognormal(mean=3.2, sigma=1.3, size=n_legit)
    legit_amounts = np.clip(np.round(legit_amounts, 2), 0.50, 5000.0)

    # Fraudulent: Mix of micro-testing charges ($1-$10) and targeted high-value drains ($300-$1500)
    is_drain = rng.uniform(0, 1, size=n_fraud) > 0.35
    fraud_amounts = np.where(
        is_drain,
        rng.uniform(250.0, 1800.0, size=n_fraud),
        rng.uniform(1.0, 25.0, size=n_fraud)
    )
    fraud_amounts = np.round(fraud_amounts, 2)

    amount_all = np.concatenate([legit_amounts, fraud_amounts])

    # 3. Generate V1 - V28 (PCA components)
    # Legitimate features: standard normal ~ N(0, 1)
    # Fraudulent features: well-documented shifts on V14, V17, V12, V10, V4
    v_features_legit = rng.normal(0, 1, size=(n_legit, 28))
    v_features_fraud = rng.normal(0, 1, size=(n_fraud, 28))

    # Feature indices (0-indexed: V1 is col 0, V4 is col 3, V10 is 9, V12 is 11, V14 is 13, V17 is 16)
    # V14 strongly negative in fraud
    v_features_fraud[:, 13] = rng.normal(-6.5, 2.0, size=n_fraud)
    # V17 strongly negative in fraud
    v_features_fraud[:, 16] = rng.normal(-5.8, 1.8, size=n_fraud)
    # V12 strongly negative in fraud
    v_features_fraud[:, 11] = rng.normal(-4.9, 1.9, size=n_fraud)
    # V10 strongly negative in fraud
    v_features_fraud[:, 9] = rng.normal(-4.2, 1.7, size=n_fraud)
    # V4 strongly positive in fraud
    v_features_fraud[:, 3] = rng.normal(4.5, 1.8, size=n_fraud)
    # V11 moderately positive in fraud
    v_features_fraud[:, 10] = rng.normal(3.8, 1.5, size=n_fraud)
    # V2 moderately positive in fraud
    v_features_fraud[:, 1] = rng.normal(3.0, 1.5, size=n_fraud)

    v_features_all = np.vstack([v_features_legit, v_features_fraud])
    class_all = np.array([0] * n_legit + [1] * n_fraud, dtype=int)

    # Construct DataFrame
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data_matrix = np.hstack([
        time_all.reshape(-1, 1),
        v_features_all,
        amount_all.reshape(-1, 1),
        class_all.reshape(-1, 1)
    ])

    df = pd.DataFrame(data_matrix, columns=cols)
    df['Class'] = df['Class'].astype(int)

    # Sort chronologically by Time
    df = df.sort_values('Time').reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Kaggle-like credit card dataset")
    parser.add_argument("--samples", type=int, default=100000, help="Number of records to generate (default 100,000)")
    parser.add_argument("--fraud-ratio", type=float, default=0.003, help="Fraud ratio (default 0.003 = 0.3%)")
    parser.add_argument("--output", type=str, default="data/raw/creditcard.csv", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Generating {args.samples:,} synthetic credit card transactions...")
    df = generate_credit_card_data(n_samples=args.samples, fraud_ratio=args.fraud_ratio, random_state=args.seed)

    print(f"[+] Total Transactions: {len(df):,}")
    print(f"[+] Legitimate: {len(df[df['Class'] == 0]):,} ({100 * (1 - df['Class'].mean()):.2f}%)")
    print(f"[+] Fraudulent: {len(df[df['Class'] == 1]):,} ({100 * df['Class'].mean():.2f}%)")
    print(f"[+] Total Fraud Volume ($): ${df[df['Class'] == 1]['Amount'].sum():,.2f}")
    print(f"[+] Total Legitimate Volume ($): ${df[df['Class'] == 0]['Amount'].sum():,.2f}")

    df.to_csv(args.output, index=False)
    print(f"[OK] Saved dataset successfully to: {args.output}")


if __name__ == "__main__":
    main()
