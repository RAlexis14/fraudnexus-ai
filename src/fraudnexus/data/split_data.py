
"""Stratified dataset splitting for FraudNexus AI."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from fraudnexus.data.generator import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_dataset(
    dataframe: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 2026,
) -> DatasetSplit:
    """Split raw data without fitting any transformations."""

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = set(required) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    X = dataframe[FEATURE_COLUMNS].copy()
    y = dataframe[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
