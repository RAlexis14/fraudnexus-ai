"""One-time, independent holdout evaluation of frozen LightGBM ONNX margins.

Run only after model, preprocessing, threshold and inference contract are frozen.
Never use holdout results for model selection or threshold adjustment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from fraudnexus.data.split_data import split_dataset
from fraudnexus.training.compare_models import SEED

ROOT = Path('models/candidates/lightgbm_weights')
REPORT = Path('reports/lightgbm_final_holdout.json')
DATA = Path('data/raw/transacciones_fraude_bancario.csv')
BATCH = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def margin_scores(outputs: list, batch_len: int) -> np.ndarray:
    """Read class-1 raw margins from converter ndarray or ZipMap outputs."""
    for output in outputs:
        if isinstance(output, np.ndarray) and output.shape == (batch_len, 2):
            return output[:, 1].astype(np.float64)
        if (isinstance(output, list) and len(output) == batch_len
                and all(isinstance(row, dict) and 1 in row for row in output)):
            return np.asarray([row[1] for row in output], dtype=np.float64)
    raise RuntimeError('ONNX class-1 raw margins not found in expected output format')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-frozen', action='store_true', help='Confirm no further model or threshold tuning')
    args = parser.parse_args()
    if not args.confirm_frozen:
        parser.error('Stop: freeze the candidate first, then run with --confirm-frozen')
    if REPORT.exists():
        raise SystemExit(f'Holdout report already exists: {REPORT}. Never overwrite it or tune on its results.')

    paths = {
        'dataset': DATA,
        'onnx': ROOT / 'model_margin_final.onnx',
        'preprocessor': ROOT / 'preprocessor.joblib',
        'python_model': ROOT / 'model.joblib',
        'metadata': ROOT / 'metadata.json',
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name}: {path}')
    fingerprints = {name: sha256(path) for name, path in paths.items()}
    metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    if metadata.get('model_name') != 'lightgbm' or metadata.get('strategy') != 'weights':
        raise RuntimeError('Unexpected candidate metadata')
    threshold = float(metadata['threshold'])
    if not (0.0 < threshold < 1.0):
        raise RuntimeError('Invalid frozen probability threshold')
    margin_threshold = float(np.log(threshold) - np.log1p(-threshold))

    data = pd.read_csv(DATA, encoding='utf-8-sig')
    split = split_dataset(data, random_state=SEED)
    x_holdout = split.X_test.reset_index(drop=True)
    y_true = np.asarray(split.y_test.reset_index(drop=True), dtype=np.int32)
    if len(x_holdout) != 10000 or np.count_nonzero(y_true == 1) != 40:
        raise RuntimeError('Unexpected holdout composition. Stop and verify original split.')
    feature_columns = metadata['feature_columns']
    if set(feature_columns) != set(x_holdout.columns):
        raise RuntimeError('Feature set disagrees with frozen metadata')
    x_holdout = x_holdout.loc[:, feature_columns]

    # Local model artifacts are trusted; never load untrusted joblib files.
    preprocessing = joblib.load(paths['preprocessor'])
    python_model = joblib.load(paths['python_model'])
    session = ort.InferenceSession(str(paths['onnx']), providers=['CPUExecutionProvider'])
    onnx_inputs = session.get_inputs()
    if len(onnx_inputs) != 1 or onnx_inputs[0].type != 'tensor(float)':
        raise RuntimeError('Unexpected ONNX input contract')
    input_name = onnx_inputs[0].name
    onnx_margins_parts: list[np.ndarray] = []
    python_margins_parts: list[np.ndarray] = []
    for start in range(0, len(x_holdout), BATCH):
        raw = preprocessing.transform(x_holdout.iloc[start:start + BATCH])
        if hasattr(raw, 'toarray'):
            raw = raw.toarray()
        transformed = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(transformed).all():
            raise RuntimeError('Non-finite preprocessed features')
        outputs = session.run(None, {input_name: transformed})
        onnx_margins_parts.append(margin_scores(outputs, len(transformed)))
        python_margins_parts.append(
            np.asarray(python_model.predict(transformed, raw_score=True), dtype=np.float64).reshape(-1)
        )

    onnx_margins = np.concatenate(onnx_margins_parts)
    python_margins = np.concatenate(python_margins_parts)
    if not np.isfinite(onnx_margins).all() or not np.isfinite(python_margins).all():
        raise RuntimeError('Non-finite margins')
    y_pred = (onnx_margins >= margin_threshold).astype(np.int32)
    python_pred = (python_margins >= margin_threshold).astype(np.int32)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()]
    report = {
        'scope': 'ONE-TIME independent final holdout; no refit or threshold updates',
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_rows': int(len(data)),
        'holdout_rows': int(len(y_true)),
        'holdout_frauds': int(y_true.sum()),
        'holdout_legitimate': int(len(y_true) - y_true.sum()),
        'candidate': 'lightgbm__weights',
        'frozen_probability_threshold': threshold,
        'frozen_margin_threshold': margin_threshold,
        'metrics': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc_pr_average_precision': float(average_precision_score(y_true, onnx_margins)),
            'roc_auc': float(roc_auc_score(y_true, onnx_margins)),
            'fpr': float(fp / (fp + tn)),
        },
        'strict_recall_above_90_percent': bool(tp / (tp + fn) > 0.90),
        'onnx_python_decision_disagreements': int(np.count_nonzero(y_pred != python_pred)),
        'max_absolute_margin_difference': float(np.max(np.abs(onnx_margins - python_margins))),
        'sha256_artifacts': fingerprints,
        'notes': [
            'AUC-PR is average precision computed on raw margins (rank-preserving scores).',
            'Results apply to the reserved academic dataset, not a real banking production stream.',
            'Do not use this result to retune thresholds or select another model.',
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally rewriting final evidence.
    with REPORT.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'REPORT SAVED: {REPORT}')


if __name__ == '__main__':
    main()
