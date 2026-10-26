
"""Reproducible synthetic banking transaction generator."""

import numpy as np
import pandas as pd


SEED = 2026
GENERATOR_VERSION = "1.0.0"

FEATURE_COLUMNS = [
    "monto_transaccion",
    "canal_transaccion",
    "categoria_comercio",
    "velocidad_tx_10m",
    "velocidad_desplazamiento_kmh",
    "ratio_desviacion_monto",
    "pais_emisor_coincide",
    "distancia_residencia_km",
    "hora_dia",
]

TARGET_COLUMN = "es_fraude"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

CHANNELS = ["POS", "ATM", "WEB", "APP"]
CATEGORIES = [
    "alimentacion",
    "supermercado",
    "transporte",
    "entretenimiento",
    "tecnologia",
    "viajes",
    "servicios",
]

CATEGORY_MEDIANS = {
    "alimentacion": 18.0,
    "supermercado": 36.0,
    "transporte": 14.0,
    "entretenimiento": 28.0,
    "tecnologia": 160.0,
    "viajes": 240.0,
    "servicios": 65.0,
}


def generate_transactions(
    n_records: int = 100_000,
    fraud_rate: float = 0.003,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate synthetic transactions with correlated risk characteristics."""

    if n_records < 100:
        raise ValueError("n_records must be at least 100.")

    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n_fraud = round(n_records * fraud_rate)

    if n_fraud < 1 or n_fraud >= n_records:
        raise ValueError("The requested prevalence produces an invalid class split.")

    # Create an exact prevalence for reproducible experiments.
    labels = np.zeros(n_records, dtype=np.int8)
    labels[:n_fraud] = 1
    rng.shuffle(labels)

    is_fraud = labels.astype(bool)

    # Latent transaction risk introduces correlation between features.
    risk = rng.normal(0, 1, n_records)
    risk += is_fraud * rng.normal(1.4, 1.0, n_records)

    channels = np.array(CHANNELS)
    categories = np.array(CATEGORIES)

    channel_probs_legit = [0.42, 0.17, 0.19, 0.22]
    channel_probs_fraud = [0.16, 0.12, 0.38, 0.34]

    category_probs_legit = [0.17, 0.25, 0.14, 0.12, 0.09, 0.07, 0.16]
    category_probs_fraud = [0.07, 0.10, 0.08, 0.10, 0.27, 0.25, 0.13]

    channel = np.empty(n_records, dtype=object)
    category = np.empty(n_records, dtype=object)

    for mask, channel_probs, category_probs in [
        (~is_fraud, channel_probs_legit, category_probs_legit),
        (is_fraud, channel_probs_fraud, category_probs_fraud),
    ]:
        count = int(mask.sum())
        channel[mask] = rng.choice(channels, size=count, p=channel_probs)
        category[mask] = rng.choice(categories, size=count, p=category_probs)

    # Amounts depend on merchant category, risk, and random variability.
    median_amounts = np.array(
        [CATEGORY_MEDIANS[value] for value in category]
    )
    amount_multiplier = np.exp(
        0.27 * risk + 0.23 * is_fraud + rng.normal(0, 0.75, n_records)
    )
    amount = np.clip(median_amounts * amount_multiplier, 0.50, 20_000)

    # Transaction velocity is correlated with latent risk.
    velocity_lambda = np.clip(
        np.exp(-0.5 + 0.36 * risk + 0.25 * is_fraud),
        0.05,
        30,
    )
    tx_velocity = np.clip(rng.poisson(velocity_lambda), 0, 50)

    # Travel distance and implied speed share a common mobility factor.
    mobility = rng.lognormal(
        mean=2.3 + 0.25 * risk + 0.45 * is_fraud,
        sigma=1.05,
        size=n_records,
    )
    residence_distance = np.clip(mobility, 0, 5_000)

    displacement_speed = np.clip(
        residence_distance
        * rng.lognormal(-0.6 + 0.25 * is_fraud, 0.8, n_records),
        0,
        1_200,
    )

    # Ratio relative to the simulated customer's normal spending.
    deviation_ratio = np.clip(
        np.exp(0.45 * risk + 0.32 * is_fraud + rng.normal(0, 0.7, n_records)),
        0.05,
        100,
    )

    # Issuer-country mismatch probability increases with risk.
    mismatch_logit = -3.6 + 0.42 * risk + 0.55 * is_fraud
    mismatch_probability = 1 / (1 + np.exp(-mismatch_logit))
    country_match = (
        rng.random(n_records) >= mismatch_probability
    ).astype(np.int8)

    # Both classes can appear at any hour.
    hour_weights = np.array([
        1.0, 0.8, 0.7, 0.6, 0.7, 0.9,
        1.5, 2.5, 3.5, 4.0, 4.2, 4.4,
        4.5, 4.0, 4.2, 4.3, 4.6, 4.8,
        4.5, 3.6, 3.0, 2.4, 1.9, 1.3,
    ])

    fraud_hour_weights = hour_weights.copy()
    fraud_hour_weights[:6] *= 2.2
    fraud_hour_weights[22:] *= 1.6

    hour = np.empty(n_records, dtype=np.int8)

    for mask, weights in [
        (~is_fraud, hour_weights),
        (is_fraud, fraud_hour_weights),
    ]:
        count = int(mask.sum())
        hour[mask] = rng.choice(
            np.arange(24),
            size=count,
            p=weights / weights.sum(),
        )

    dataframe = pd.DataFrame({
        "monto_transaccion": np.round(amount, 2),
        "canal_transaccion": channel,
        "categoria_comercio": category,
        "velocidad_tx_10m": tx_velocity,
        "velocidad_desplazamiento_kmh": np.round(displacement_speed, 2),
        "ratio_desviacion_monto": np.round(deviation_ratio, 3),
        "pais_emisor_coincide": country_match,
        "distancia_residencia_km": np.round(residence_distance, 2),
        "hora_dia": hour,
        "es_fraude": labels,
    })

    # Simulate controlled missingness in selected numeric predictors.
    for column, missing_rate in [
        ("velocidad_desplazamiento_kmh", 0.01),
        ("ratio_desviacion_monto", 0.015),
        ("distancia_residencia_km", 0.01),
    ]:
        missing_mask = rng.random(n_records) < missing_rate
        dataframe.loc[missing_mask, column] = np.nan

    return dataframe[ALL_COLUMNS]
