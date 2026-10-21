
"""Export the trained Random Forest estimator to ONNX.

The fitted preprocessing is preserved as a separate artifact.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from fraudnexus.data.generator import FEATURE_COLUMNS
from fraudnexus.data.split_data import split_dataset


MODEL_PATH = Path("models/selected_model.joblib")
METADATA_PATH = Path("models/selected_model_metadata.json")
DATA_PATH = Path("data/raw/transacciones_fraude_bancario.csv")

ONNX_PATH = Path("models/fraud_detector.onnx")
PREPROCESSOR_PATH = Path("models/fraud_preprocessor.joblib")
REPORT_PATH = Path("reports/onnx_validation.json")


def main():
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if metadata["model_name"] != "random_forest":
        raise ValueError("Expected a Random Forest model.")

    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Feature schema does not match.")

    # Load only our own locally generated, trusted artifact.
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessing"]
    estimator = pipeline.steps[-1][1]

    if estimator.__class__.__name__ != "RandomForestClassifier":
        raise TypeError("Expected RandomForestClassifier.")

    if not hasattr(preprocessor, "transformers_"):
        raise RuntimeError("Preprocessor is not fitted.")

    # The ONNX estimator receives already transformed numeric inputs.
    n_features = int(estimator.n_features_in_)

    onnx_model = convert_sklearn(
        estimator,
        name="FraudNexus_AI_Random_Forest",
        initial_types=[
            ("features", FloatTensorType([None, n_features]))
        ],
        target_opset=17,
        options={id(estimator): {"zipmap": False}},
    )

    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    # Never evaluate the reserved final holdout during export.
    df = pd.read_csv(DATA_PATH)
    development = split_dataset(df, random_state=2026)

    sample = development.X_train.sample(
        n=min(1000, len(development.X_train)),
        random_state=2026,
    )

    transformed = preprocessor.transform(sample)
    features = np.asarray(transformed, dtype=np.float32)

    if features.shape[1] != n_features:
        raise ValueError(
            f"Expected {n_features} features; got {features.shape[1]}."
        )

    # Compare native and exported estimator on identical float32 inputs.
    native_probs = estimator.predict_proba(features)[:, 1]

    outputs = session.run(None, {"features": features})

    probability_arrays = [
        value for value in outputs
        if isinstance(value, np.ndarray)
        and value.ndim == 2
        and value.shape == (len(sample), 2)
    ]

    if len(probability_arrays) != 1:
        raise RuntimeError(
            "ONNX probability output does not have the expected shape."
        )

    onnx_probs = probability_arrays[0][:, 1]

    max_difference = float(
        np.max(np.abs(native_probs - onnx_probs))
    )

    threshold = float(metadata["threshold"])

    native_decisions = native_probs >= threshold
    onnx_decisions = onnx_probs >= threshold

    agreement = float(
        np.mean(native_decisions == onnx_decisions)
    )

    report = {
        "model_name": metadata["model_name"],
        "architecture": "python_preprocessing_plus_onnx_estimator",
        "sample_source": "development_only",
        "sample_size": len(sample),
        "original_features": len(FEATURE_COLUMNS),
        "transformed_features": n_features,
        "max_probability_difference": max_difference,
        "decision_agreement": agreement,
        "threshold": threshold,
        "onnx_inputs": [
            value.name for value in session.get_inputs()
        ],
        "onnx_outputs": [
            value.name for value in session.get_outputs()
        ],
    }

    print("\n=== FRAUDNEXUS AI | ONNX VALIDATION ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if max_difference > 0.001 or agreement < 1.0:
        raise RuntimeError(
            "ONNX equivalence test failed. Artifacts were not published."
        )

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ONNX_PATH.write_bytes(onnx_model.SerializeToString())
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nONNX export successful.")
    print(f"ONNX model: {ONNX_PATH}")
    print(f"Fitted preprocessor: {PREPROCESSOR_PATH}")
    print(f"Validation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
