
"""Check LightGBM probability stability before ONNX conversion."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CANDIDATE_DIR = Path("models/candidates/lightgbm_weights")

FEATURES = [
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

SAMPLES = [
    {
        "monto_transaccion": 65.2,
        "canal_transaccion": "POS",
        "categoria_comercio": "Retail",
        "velocidad_tx_10m": 0,
        "velocidad_desplazamiento_kmh": 10.0,
        "ratio_desviacion_monto": 1.1,
        "pais_emisor_coincide": 1,
        "distancia_residencia_km": 3.0,
        "hora_dia": 14,
    },
    {
        "monto_transaccion": 1200.0,
        "canal_transaccion": "E-Commerce",
        "categoria_comercio": "Viajes",
        "velocidad_tx_10m": 6,
        "velocidad_desplazamiento_kmh": 700.0,
        "ratio_desviacion_monto": 9.5,
        "pais_emisor_coincide": 0,
        "distancia_residencia_km": 1700.0,
        "hora_dia": 2,
    },
]

def main():
    """Compare model outputs across numerical precisions."""
    model = joblib.load(CANDIDATE_DIR / "model.joblib")
    preprocessor = joblib.load(
        CANDIDATE_DIR / "preprocessor.joblib"
    )

    frame = pd.DataFrame(SAMPLES, columns=FEATURES)
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    x64 = np.asarray(transformed, dtype=np.float64)
    x32 = np.asarray(transformed, dtype=np.float32)

    threshold = 0.9999999146869891

    print("=== LIGHTGBM NUMERICAL STABILITY ===")
    print("Features:", x64.shape)
    print("Threshold float64:", repr(threshold))
    print("Threshold float32:", repr(float(np.float32(threshold))))
    print("Float32 below 1:", repr(float(
        np.nextafter(np.float32(1), np.float32(0))
    )))

    for name, data in [("float64", x64), ("float32", x32)]:
        scores = model.predict_proba(data)[:, 1]
        margins = model.predict(data, raw_score=True)

        print(f"\n{name}:")
        for index, (score, margin) in enumerate(
            zip(scores, margins), start=1
        ):
            print(
                f"Sample {index}: "
                f"probability={float(score):.17g} "
                f"raw_margin={float(margin):.12g} "
                f"review={bool(score >= threshold)}"
            )


if __name__ == "__main__":
    main()
