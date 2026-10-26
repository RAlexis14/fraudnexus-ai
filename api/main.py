"""FraudNexus AI: validated, inference-only banking fraud API."""

import json
import os
import math
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
    margin: float | None = None
    margin_threshold: float | None = None
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
        variant = os.environ.get("FRAUDNEXUS_MODEL", "random_forest").strip()
        self.variant = variant
        if variant == "lightgbm_candidate":
            candidate_dir = models_dir / "candidates" / "lightgbm_weights"
            metadata_path = candidate_dir / "metadata.json"
            preprocessor_path = candidate_dir / "preprocessor.joblib"
            onnx_path = candidate_dir / "model_margin_final.onnx"
            self.mode = "raw_margin"
        elif variant == "random_forest":
            metadata_path = models_dir / "selected_model_metadata.json"
            preprocessor_path = models_dir / "fraud_preprocessor.joblib"
            onnx_path = models_dir / "fraud_detector.onnx"
            self.mode = "probability"
        else:
            raise ValueError(f"Unknown FRAUDNEXUS_MODEL value: {variant}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.columns = list(metadata["feature_columns"])
        self.threshold = float(metadata["threshold"])
        if len(self.columns) != 9 or set(self.columns) != set(Transaction.model_fields):
            raise ValueError("Model metadata and API input schema disagree")
        if not 0 < self.threshold < 1:
            raise ValueError("Invalid trained decision threshold")
        self.margin_threshold = (
            math.log(self.threshold) - math.log1p(-self.threshold)
            if self.mode == "raw_margin" else None
        )

        # Load only files produced by this trusted project.
        self.preprocessor = joblib.load(preprocessor_path)
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "features":
            raise ValueError("Unexpected ONNX input contract")
        self.input_name = inputs[0].name
        self.output_names = {output.name for output in self.session.get_outputs()}
        if "probabilities" not in self.output_names:
            raise ValueError("ONNX score output missing")

    @staticmethod
    def _to_two_columns(raw, batch_size: int) -> np.ndarray:
        """Accept an ONNX tensor or an ONNX ZipMap sequence of dictionaries."""
        if isinstance(raw, np.ndarray):
            matrix = np.asarray(raw, dtype=np.float64)
        elif isinstance(raw, list) and len(raw) == batch_size and all(isinstance(d, dict) for d in raw):
            if any(0 not in d or 1 not in d for d in raw):
                raise RuntimeError("Unexpected ONNX class labels in ZipMap")
            matrix = np.asarray([[d[0], d[1]] for d in raw], dtype=np.float64)
        else:
            raise RuntimeError("Unknown ONNX score output format")
        if matrix.shape != (batch_size, 2) or not np.isfinite(matrix).all():
            raise RuntimeError("Invalid ONNX score matrix")
        return matrix

    def predict_many(self, transactions: list[Transaction]) -> list[Prediction]:
        started = perf_counter_ns()
        frame = pd.DataFrame(
            [item.model_dump() for item in transactions], columns=self.columns
        )
        frame = frame.replace({None: np.nan})
        matrix = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        preprocess_end = perf_counter_ns()
        raw = self.session.run(["probabilities"], {self.input_name: matrix})[0]
        inference_end = perf_counter_ns()
        score_matrix = self._to_two_columns(raw, len(transactions))

        if self.mode == "raw_margin":
            margins = score_matrix[:, 1]
            # Stable logistic conversion for DISPLAY ONLY. Do not decide on
            # float32 probabilities, because near-one rounding changed results.
            scores = np.where(
                margins >= 0,
                1.0 / (1.0 + np.exp(-np.clip(margins, 0, 700))),
                np.exp(np.clip(margins, -700, 0)) / (1.0 + np.exp(np.clip(margins, -700, 0))),
            )
            flags = margins >= self.margin_threshold
        else:
            margins = [None] * len(transactions)
            scores = score_matrix[:, 1]
            flags = scores >= self.threshold

        pre_ms = (preprocess_end - started) / 1e6
        inference_ms = (inference_end - preprocess_end) / 1e6
        total_ms = (inference_end - started) / 1e6
        return [
            Prediction(
                score=float(score), threshold=self.threshold,
                margin=float(margin) if margin is not None else None,
                margin_threshold=self.margin_threshold,
                is_fraud=bool(flag),
                decision="REVIEW" if flag else "ALLOW",
                preprocessing_ms=round(pre_ms, 4),
                onnx_inference_ms=round(inference_ms, 4),
                total_processing_ms=round(total_ms, 4),
            )
            for score, margin, flag in zip(scores, margins, flags)
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
    return {"status": "ok", "model": engine.variant, "threshold": engine.threshold, "margin_threshold": engine.margin_threshold}


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
