"""FraudNexus AI: validated, inference-only banking fraud API."""

import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter_ns
from typing import Annotated, Literal

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


class Transaction(BaseModel):
    """Nine raw predictors expected by the fitted preprocessing pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    monto_transaccion: float = Field(ge=0, le=1_000_000)
    canal_transaccion: Literal["POS", "E-Commerce", "ATM", "App Móvil"]
    categoria_comercio: Literal[
        "Supermercados", "Retail", "Entretenimiento", "Viajes"
    ]
    velocidad_tx_10m: int = Field(ge=0, le=100_000)
    velocidad_desplazamiento_kmh: float | None = Field(default=None, ge=0, le=10_000)
    ratio_desviacion_monto: float | None = Field(default=None, ge=0, le=10_000)
    pais_emisor_coincide: Literal[0, 1]
    distancia_residencia_km: float | None = Field(default=None, ge=0, le=50_000)
    hora_dia: int = Field(ge=0, le=23)

    @field_validator("monto_transaccion", "velocidad_desplazamiento_kmh", "ratio_desviacion_monto", "distancia_residencia_km")
    @classmethod
    def validate_finite(cls, value):
        if value is not None and not np.isfinite(value):
            raise ValueError("Numeric fields must be finite")
        return value


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    score: float
    threshold: float
    is_fraud: bool
    decision: Literal["REVIEW", "ALLOW"]
    preprocessing_ms: float
    onnx_inference_ms: float
    total_processing_ms: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    processing_ms: float
    fraud_alerts: int
    total: int


class FraudEngine:
    """Load trusted, versioned artifacts once; never retrain during requests."""

    def __init__(self, models_dir: Path = MODEL_DIR):
        metadata = json.loads(
            (models_dir / "selected_model_metadata.json").read_text(encoding="utf-8")
        )
        self.columns = list(metadata["feature_columns"])
        self.threshold = float(metadata["threshold"])
        if len(self.columns) != 9 or set(self.columns) != set(Transaction.model_fields):
            raise ValueError("Model metadata and API input schema disagree")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid trained decision threshold")

        # joblib must only load trusted files produced by this project.
        self.preprocessor = joblib.load(models_dir / "fraud_preprocessor.joblib")
        self.session = ort.InferenceSession(
            str(models_dir / "fraud_detector.onnx"),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "features":
            raise ValueError("Unexpected ONNX input contract")
        if "probabilities" not in [output.name for output in self.session.get_outputs()]:
            raise ValueError("ONNX probabilities output missing")

    def predict_many(self, transactions: list[Transaction]) -> list[Prediction]:
        started = perf_counter_ns()
        frame = pd.DataFrame(
            [item.model_dump() for item in transactions], columns=self.columns
        )
        # None is valid only for the three optional numeric fields.
        frame = frame.replace({None: np.nan})
        matrix = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        preprocess_end = perf_counter_ns()
        scores_matrix = self.session.run(["probabilities"], {"features": matrix})[0]
        inference_end = perf_counter_ns()

        if scores_matrix.shape != (len(transactions), 2):
            raise RuntimeError("Unexpected ONNX probability output shape")
        scores = scores_matrix[:, 1]
        if not np.isfinite(scores).all():
            raise RuntimeError("Model returned non-finite scores")
        pre_ms = (preprocess_end - started) / 1e6
        inference_ms = (inference_end - preprocess_end) / 1e6
        total_ms = (inference_end - started) / 1e6
        # In batch processing, timings refer to the complete batch, not one row.
        return [
            Prediction(
                score=float(score),
                threshold=self.threshold,
                is_fraud=bool(score >= self.threshold),
                decision="REVIEW" if score >= self.threshold else "ALLOW",
                preprocessing_ms=round(pre_ms, 4),
                onnx_inference_ms=round(inference_ms, 4),
                total_processing_ms=round(total_ms, 4),
            )
            for score in scores
        ]


@lru_cache(maxsize=1)
def get_engine() -> FraudEngine:
    return FraudEngine()


app = FastAPI(
    title="FraudNexus AI",
    version="0.1.0",
    description="Bank fraud risk scoring using fitted Python preprocessing and ONNX Runtime.",
)


@app.get("/health")
def health(engine: Annotated[FraudEngine, Depends(get_engine)]):
    return {"status": "ok", "model": "onnx", "threshold": engine.threshold}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, engine: Annotated[FraudEngine, Depends(get_engine)]):
    try:
        return engine.predict_many([transaction])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, engine: Annotated[FraudEngine, Depends(get_engine)]):
    started = perf_counter_ns()
    try:
        predictions = engine.predict_many(payload.transactions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    return BatchResponse(
        predictions=predictions,
        processing_ms=round((perf_counter_ns() - started) / 1e6, 4),
        fraud_alerts=sum(item.is_fraud for item in predictions),
        total=len(predictions),
    )
