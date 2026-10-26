"""Smoke tests for FraudNexus modeling contracts."""

import numpy as np
import pandas as pd

from fraudnexus.training.compare_models import (
    MODEL_NAMES, describe, fit_pipeline, risk_score, threshold_at_recall,
)


def test_threshold_recall():
    y = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.7, 0.5, 0.6, 0.1])
    cutoff = threshold_at_recall(y, scores, target=0.9)
    assert cutoff == 0.5
    assert describe(y, scores, cutoff)["recall"] == 1.0


def test_all_models_present():
    assert set(MODEL_NAMES) == {
        "random_forest", "xgboost", "lightgbm", "isolation_forest"
    }


def test_smoke_all_models():
    rng = np.random.default_rng(2026)
    n = 160
    y = pd.Series(np.array([0] * 145 + [1] * 15))
    X = pd.DataFrame({
        "monto_transaccion": rng.uniform(1, 500, n),
        "canal_transaccion": rng.choice(["POS", "ATM", "App Móvil", "E-Commerce"], n),
        "categoria_comercio": rng.choice(["Retail", "Viajes", "Supermercados", "Entretenimiento"], n),
        "velocidad_tx_10m": rng.integers(0, 7, n),
        "velocidad_desplazamiento_kmh": rng.uniform(0, 100, n),
        "ratio_desviacion_monto": rng.uniform(0.5, 6, n),
        "pais_emisor_coincide": rng.integers(0, 2, n),
        "distancia_residencia_km": rng.uniform(0, 200, n),
        "hora_dia": rng.integers(0, 24, n),
    })
    for name in MODEL_NAMES:
        pipeline = fit_pipeline(name, X, y, quick=True)
        values = risk_score(name, pipeline, X.iloc[:6])
        assert values.shape == (6,)
        assert np.isfinite(values).all()
