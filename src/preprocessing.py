"""
Preprocessing & Data Scaling Module
Implements RobustScaler for high-outlier transactional features and stratified data splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class FraudDataPreprocessor:
    """
    Transforms transaction features, applies RobustScaler to non-PCA columns (Amount, Time),
    and yields train/test splits preserving severe class imbalance.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def prepare_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate feature matrix X and target y."""
        exclude_cols = ['Class', 'TimeWindow', 'AmountBucket', 'Day']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_names = feature_cols
        return df[feature_cols].copy(), df['Class'].copy()

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform stratified train-test split preserving extreme class ratio.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    def fit_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit RobustScaler on training set non-PCA columns and transform both sets.
        Prevents data leakage.
        """
        X_tr = X_train.copy()
        X_te = X_test.copy()

        cols_to_scale = [c for c in ['Amount', 'Time', 'HourOfDay', 'LogAmount'] if c in X_tr.columns]
        if cols_to_scale:
            X_tr[cols_to_scale] = self.scaler.fit_transform(X_tr[cols_to_scale])
            X_te[cols_to_scale] = self.scaler.transform(X_te[cols_to_scale])

        self.is_fitted = True
        return X_tr.values, X_te.values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new incoming unseen transaction records."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        X_copy = X.copy()
        cols_to_scale = [c for c in ['Amount', 'Time', 'HourOfDay', 'LogAmount'] if c in X_copy.columns]
        if cols_to_scale:
            X_copy[cols_to_scale] = self.scaler.transform(X_copy[cols_to_scale])
        return X_copy.values
