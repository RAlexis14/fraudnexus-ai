"""Smoke-test the optional LightGBM ONNX candidate through the real API engine."""
import os

from api.main import FraudEngine, Transaction

NORMAL = {
    "monto_transaccion": 65.2, "canal_transaccion": "POS",
    "categoria_comercio": "Retail", "velocidad_tx_10m": 0,
    "velocidad_desplazamiento_kmh": 10.0, "ratio_desviacion_monto": 1.1,
    "pais_emisor_coincide": 1, "distancia_residencia_km": 3.0,
    "hora_dia": 14,
}
SUSPICIOUS = {
    "monto_transaccion": 1200.0, "canal_transaccion": "E-Commerce",
    "categoria_comercio": "Viajes", "velocidad_tx_10m": 6,
    "velocidad_desplazamiento_kmh": 700.0, "ratio_desviacion_monto": 9.5,
    "pais_emisor_coincide": 0, "distancia_residencia_km": 1700.0,
    "hora_dia": 2,
}


def main():
    os.environ["FRAUDNEXUS_MODEL"] = "lightgbm_candidate"
    engine = FraudEngine()
    results = engine.predict_many([Transaction(**NORMAL), Transaction(**SUSPICIOUS)])
    for name, item in zip(["normal", "suspicious"], results):
        print(f"{name}: decision={item.decision} score={item.score:.12g} margin={item.margin:.12g}")
    assert results[0].decision == "ALLOW", "Normal smoke transaction failed"
    assert results[1].decision == "REVIEW", "Suspicious smoke transaction failed"
    assert all(item.margin_threshold is not None for item in results)
    print("PASS: two illustrative API candidate predictions.")


if __name__ == "__main__":
    main()
