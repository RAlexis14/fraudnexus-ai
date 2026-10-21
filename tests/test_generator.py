
"""Tests for the synthetic transaction generator."""

import pandas as pd
import pytest

from fraudnexus.data.generator import (
    ALL_COLUMNS,
    generate_transactions,
)


def test_dataset_structure():
    df = generate_transactions(n_records=10_000, seed=42)

    assert len(df) == 10_000
    assert list(df.columns) == ALL_COLUMNS
    assert not df["es_fraude"].isna().any()


def test_fraud_prevalence():
    df = generate_transactions(
        n_records=10_000,
        fraud_rate=0.003,
        seed=42,
    )

    assert int(df["es_fraude"].sum()) == 30
    assert df["es_fraude"].mean() == pytest.approx(0.003)


def test_reproducibility():
    first = generate_transactions(n_records=1_000, seed=2026)
    second = generate_transactions(n_records=1_000, seed=2026)

    pd.testing.assert_frame_equal(first, second)


def test_valid_values():
    df = generate_transactions(n_records=5_000, seed=42)

    assert df["monto_transaccion"].between(0.5, 20_000).all()
    assert df["hora_dia"].between(0, 23).all()
    assert df["pais_emisor_coincide"].isin([0, 1]).all()
    assert df["es_fraude"].isin([0, 1]).all()


def test_invalid_parameters():
    with pytest.raises(ValueError):
        generate_transactions(n_records=50)

    with pytest.raises(ValueError):
        generate_transactions(fraud_rate=0)
