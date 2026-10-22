"""Export a LightGBM candidate to ONNX and compare on DEVELOPMENT rows only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import onnx
import onnxmltools
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import confusion_matrix

from fraudnexus.data.generator import FEATURE_COLUMNS, TARGET_COLUMN
from fraudnexus.data.split_data import split_dataset
from fraudnexus.training.compare_models import SEED

ROOT = Path('models/candidates/lightgbm_weights')
OUT = ROOT / 'model.onnx'
REPORT = Path('reports/lightgbm_onnx_diagnostic.json')


def probabilities_from_outputs(outputs, n):
    """Find probability output of a tree classifier in supported ONNX layouts."""
    for result in outputs:
        if isinstance(result, np.ndarray) and result.ndim == 2 and result.shape == (n, 2):
            return result[:, 1].astype(np.float64)
        if isinstance(result, list) and len(result) == n and all(isinstance(x, dict) for x in result):
            values = [row.get(1, row.get('1')) for row in result]
            if all(v is not None for v in values):
                return np.asarray(values, dtype=np.float64)
    raise RuntimeError(f'Unknown ONNX probability structure: {[type(x).__name__ for x in outputs]}')


def main():
    model = joblib.load(ROOT / 'model.joblib')  # Trusted local artifact only.
    preprocessor = joblib.load(ROOT / 'preprocessor.joblib')
    metadata = json.loads((ROOT / 'metadata.json').read_text(encoding='utf-8'))
    threshold = float(metadata['threshold'])

    data = pd.read_csv('data/raw/transacciones_fraude_bancario.csv', encoding='utf-8-sig')
    divided = split_dataset(data, random_state=SEED)
    # Sampling only development rows. Holdout is neither transformed nor scored.
    dev = divided.X_train.reset_index(drop=True)
    target = np.asarray(divided.y_train.reset_index(drop=True), dtype=np.int32)
    pos = np.flatnonzero(target == 1)[:160]
    neg = np.flatnonzero(target == 0)[:800]
    indices = np.concatenate([pos, neg])
    x = np.asarray(preprocessor.transform(dev.iloc[indices]), dtype=np.float32)
    python_scores = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    python_decisions = python_scores >= threshold

    converter_model = model
    exported = onnxmltools.convert_lightgbm(
        converter_model,
        initial_types=[('features', FloatTensorType([None, x.shape[1]]))],
        target_opset=15,
    )
    onnx.checker.check_model(exported)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(exported, str(OUT))

    session = ort.InferenceSession(str(OUT), providers=['CPUExecutionProvider'])
    feeds = {session.get_inputs()[0].name: x}
    outputs = session.run(None, feeds)
    onnx_scores = probabilities_from_outputs(outputs, len(x))
    onnx_decisions = onnx_scores >= threshold
    print("\n=== DIAGNOSTICO DE DECISIONES ===")

    print("Umbral:", repr(threshold))

    print(
        "Python REVIEW:",
        int(np.sum(python_decisions))
    )

    print(
        "ONNX REVIEW:",
        int(np.sum(onnx_decisions))
    )

    print(
        "ONNX score minimo:",
        repr(float(np.min(onnx_scores)))
    )

    print(
        "ONNX score maximo:",
        repr(float(np.max(onnx_scores)))
    )

    print("\nPrimeras 5 transacciones fraudulentas:")

    fraud_indices = np.flatnonzero(
        target[indices] == 1
    )[:5]

    for i in fraud_indices:
        print(
            f"Registro {i}: "
            f"Python={python_scores[i]:.17g} | "
            f"ONNX={onnx_scores[i]:.17g} | "
            f"Python_REVIEW={python_decisions[i]} | "
            f"ONNX_REVIEW={onnx_decisions[i]}"
        )

    disagreements = np.flatnonzero(python_decisions != onnx_decisions)
    diffs = np.abs(python_scores - onnx_scores)
    threshold_margin = python_scores - threshold
    nearest = np.argsort(np.abs(threshold_margin))[:10]

    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_scope': 'candidate export equivalence on DEVELOPMENT data only; in-sample for final fitted model',
        'holdout_evaluated': False,
        'model': 'lightgbm__weights',
        'original_threshold_float64': threshold,
        'rounded_threshold_float32': float(np.float32(threshold)),
        'evaluated_rows': int(len(x)),
        'positives': int(np.sum(target[indices] == 1)),
        'negatives': int(np.sum(target[indices] == 0)),
        'onnx_inputs': [{'name': v.name, 'type': v.type} for v in session.get_inputs()],
        'onnx_outputs': [{'name': v.name, 'type': v.type} for v in session.get_outputs()],
        'max_probability_difference': float(np.max(diffs)),
        'mean_probability_difference': float(np.mean(diffs)),
        'decision_disagreements': int(len(disagreements)),
        'decision_agreement': float(np.mean(python_decisions == onnx_decisions)),
        'decision_disagreement_examples': [
            {'sample_index': int(i), 'actual': int(target[indices[i]]),
             'python_score': float(python_scores[i]), 'onnx_score': float(onnx_scores[i])}
            for i in disagreements[:10]
        ],
        'nearest_to_threshold': [
            {'sample_index': int(i), 'actual': int(target[indices[i]]),
             'python_score': float(python_scores[i]), 'onnx_score': float(onnx_scores[i]),
             'python_review': bool(python_decisions[i]), 'onnx_review': bool(onnx_decisions[i])}
            for i in nearest
        ],
        'safe_to_promote_based_on_checked_samples': bool(len(disagreements) == 0),
        'warning': 'Zero disagreements on this sample is necessary but not sufficient for deployment; ONNX numerical behavior near threshold requires further validation.',
    }
    REPORT.parent.mkdir(exist_ok=True, parents=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['evaluated_rows','max_probability_difference','decision_disagreements','decision_agreement','safe_to_promote_based_on_checked_samples']}, indent=2))
    print('Report:', REPORT)
    print('Candidate ONNX:', OUT)
    if len(disagreements) > 0:
        print('WARNING: ONNX and Python decisions disagree. DO NOT promote this candidate.')


if __name__ == '__main__':
    main()
