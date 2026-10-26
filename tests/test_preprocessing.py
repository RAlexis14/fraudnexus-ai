
"""Tests for dataset splitting and preprocessing."""

import numpy as np
import pytest

from fraudnexus.data.generator import (
    FEATURE_COLUMNS,
    generate_transactions,
)
from fraudnexus.data.split_data import split_dataset
from fraudnexus.features.preprocessing import build_preprocessor


@pytest.fixture
def sample_data():
    return generate_transactions(
        n_records=10_000,
        fraud_rate=0.003,
        seed=2026,
    )


def test_stratified_split(sample_data):
    result = split_dataset(sample_data)

    assert len(result.X_train) == 8_000
    assert len(result.X_test) == 2_000
    assert int(result.y_train.sum()) == 24
    assert int(result.y_test.sum()) == 6


def test_feature_columns(sample_data):
    result = split_dataset(sample_data)

    assert list(result.X_train.columns) == FEATURE_COLUMNS
    assert "es_fraude" not in result.X_train.columns


def test_preprocessing_removes_missing_values(sample_data):
    result = split_dataset(sample_data)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(result.X_train)

    assert transformed.shape[0] == 8_000
    assert np.isfinite(transformed).all()


def test_preprocessing_consistent_dimensions(sample_data):
    result = split_dataset(sample_data)

    preprocessor = build_preprocessor()

    train_array = preprocessor.fit_transform(result.X_train)
    test_array = preprocessor.transform(result.X_test)

    assert train_array.shape[1] == test_array.shape[1]


def test_split_reproducibility(sample_data):
    first = split_dataset(sample_data)
    second = split_dataset(sample_data)

    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()
