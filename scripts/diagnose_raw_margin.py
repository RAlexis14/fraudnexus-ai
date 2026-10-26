"""Diagnostic only: test ONNX raw margins against trusted LightGBM model.

Does not promote or overwrite the active fraud detector.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

from fraudnexus.data.split_data import split_dataset
from fraudnexus.training.compare_models import SEED

ROOT = Path('models/candidates/lightgbm_weights')
SOURCE = ROOT / 'model.onnx'
MARGIN_ONNX = ROOT / 'model_margin_diagnostic.onnx'
OUT = Path('reports/lightgbm_margin_diagnostic.json')


def as_matrix(outputs, n):
    for obj in outputs:
        if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[0] == n:
            return np.asarray(obj, dtype=np.float64)
        if isinstance(obj, list) and len(obj) == n and all(isinstance(item, dict) for item in obj):
            keys = sorted(obj[0])
            return np.asarray([[item[k] for k in keys] for item in obj], dtype=np.float64)
    raise RuntimeError('Could not find a matrix of ONNX class scores.')


def main():
    meta = json.loads((ROOT / 'metadata.json').read_text(encoding='utf-8'))
    threshold = float(meta['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be strictly between zero and one.')
    margin_threshold = float(np.log(threshold) - np.log1p(-threshold))

    model = joblib.load(ROOT / 'model.joblib')  # Trusted project artifact.
    preprocessing = joblib.load(ROOT / 'preprocessor.joblib')
    data = pd.read_csv('data/raw/transacciones_fraude_bancario.csv', encoding='utf-8-sig')
    split = split_dataset(data, random_state=SEED)
    x_dev = split.X_train.reset_index(drop=True)
    y_dev = np.asarray(split.y_train.reset_index(drop=True), dtype=np.int32)
    indices = np.concatenate((np.flatnonzero(y_dev == 1)[:160], np.flatnonzero(y_dev == 0)[:800]))
    x = np.asarray(preprocessing.transform(x_dev.iloc[indices]), dtype=np.float32)
    py_margins = np.asarray(model.predict(x, raw_score=True), dtype=np.float64).reshape(-1)
    py_decision = py_margins >= margin_threshold

    proto = onnx.load(str(SOURCE))
    tree_nodes = [node for node in proto.graph.node if node.op_type == 'TreeEnsembleClassifier']
    if len(tree_nodes) != 1:
        raise RuntimeError(f'Expected one TreeEnsembleClassifier; got {len(tree_nodes)}')
    node = tree_nodes[0]
    transform_attributes = [attr for attr in node.attribute if attr.name == 'post_transform']
    if len(transform_attributes) != 1:
        raise RuntimeError('TreeEnsembleClassifier post_transform not found')
    previous = transform_attributes[0].s.decode('utf-8')
    if previous != 'LOGISTIC':
        raise RuntimeError(f'Expected LOGISTIC transform but got {previous!r}')
    transform_attributes[0].s = b'NONE'
    onnx.checker.check_model(proto)
    onnx.save(proto, str(MARGIN_ONNX))

    session = ort.InferenceSession(str(MARGIN_ONNX), providers=['CPUExecutionProvider'])
    matrix = as_matrix(session.run(None, {session.get_inputs()[0].name: x}), len(x))
    print('ONNX raw output shape:', matrix.shape)
    print('Python raw margin threshold:', repr(margin_threshold))
    print('First raw ONNX rows:', matrix[:3].tolist())
    print('Python raw margins first:', py_margins[:3].tolist())

    # ONNX classifier class-1 score can have converter-specific scaling/offset.
    # Do not assume either column is a margin: TEST alternatives explicitly.
    candidate_columns = {f'col_{j}': matrix[:, j] for j in range(matrix.shape[1])}
    if matrix.shape[1] == 2:
        candidate_columns['col_1_minus_col_0'] = matrix[:, 1] - matrix[:, 0]
    checks = {}
    for name, values in candidate_columns.items():
        abs_err = np.abs(values - py_margins)
        decisions = values >= margin_threshold
        checks[name] = {
            'max_abs_margin_error': float(abs_err.max()),
            'median_abs_margin_error': float(np.median(abs_err)),
            'decision_disagreements_at_python_margin_threshold': int(np.sum(decisions != py_decision)),
            'onnx_decision_positive_count': int(np.sum(decisions)),
            'first_positive_raw_value': float(values[0]),
        }

    report = {
        'scope': 'DEVELOPMENT only, ONNX raw margin compatibility diagnostic',
        'holdout_evaluated': False,
        'source_transform': previous,
        'diagnostic_transform': 'NONE',
        'margin_threshold': margin_threshold,
        'sample_size': int(len(indices)),
        'python_positive_decisions': int(np.sum(py_decision)),
        'checks': checks,
        'model_promoted': False,
        'note': 'Raw outputs may require adjustment for converter conventions; validate before deployment.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Report:', OUT)
    print('Diagnostic ONNX:', MARGIN_ONNX)


if __name__ == '__main__':
    main()
