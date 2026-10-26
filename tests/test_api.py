"""HTTP API contract tests, independent of trained model artifacts."""
from fastapi.testclient import TestClient
from api.main import app, get_engine, Prediction


EXAMPLE = {
    "monto_transaccion": 65.2,
    "canal_transaccion": "POS",
    "categoria_comercio": "Retail",
    "velocidad_tx_10m": 0,
    "velocidad_desplazamiento_kmh": 10.0,
    "ratio_desviacion_monto": 1.1,
    "pais_emisor_coincide": 1,
    "distancia_residencia_km": 3.0,
    "hora_dia": 14,
}


class FakeEngine:
    variant = "random_forest"
    threshold = 0.75
    margin_threshold = None


    def predict_many(self, transactions):
        return [
            Prediction(score=0.1, threshold=0.75, is_fraud=False,
                       decision="ALLOW", preprocessing_ms=0.5,
                       onnx_inference_ms=0.1, total_processing_ms=0.6)
            for _ in transactions
        ]


def setup_function():
    app.dependency_overrides[get_engine] = lambda: FakeEngine()


def teardown_function():
    app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_predict_schema():
    result = TestClient(app).post("/predict", json=EXAMPLE)
    assert result.status_code == 200
    assert result.json()["decision"] == "ALLOW"


def test_reject_unexpected_categories():
    bad = {**EXAMPLE, "canal_transaccion": "WEB"}
    assert TestClient(app).post("/predict", json=bad).status_code == 422


def test_reject_extra_features():
    bad = {**EXAMPLE, "es_fraude": 1}
    assert TestClient(app).post("/predict", json=bad).status_code == 422


def test_batch_prediction():
    response = TestClient(app).post("/predict/batch", json={"transactions": [EXAMPLE, EXAMPLE]})
    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["fraud_alerts"] == 0


def test_reject_empty_batch():
    assert TestClient(app).post("/predict/batch", json={"transactions": []}).status_code == 422
