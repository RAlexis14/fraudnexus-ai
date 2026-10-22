"""Validate candidate LightGBM ONNX raw-margin equivalence on ALL development rows.

Do not train, alter production artifacts, or score the held-out final test.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from fraudnexus.data.split_data import split_dataset
from fraudnexus.training.compare_models import SEED

ROOT = Path('models/candidates/lightgbm_weights')
REPORT = Path('reports/lightgbm_margin_full_development.json')
BATCH = 512


def extract_onnx_scores(outputs, batch_size):
    """Extract ONNX class scores from arrays or ZipMap dictionaries."""
    for output in outputs:
        if (
            isinstance(output, np.ndarray)
            and output.ndim == 2
            and output.shape == (batch_size, 2)
        ):
            return np.asarray(output, dtype=np.float64)

        if (
            isinstance(output, list)
            and len(output) == batch_size
            and all(isinstance(item, dict) for item in output)
        ):
            keys = sorted(output[0].keys())

            if len(keys) != 2:
                continue

            if any(set(item.keys()) != set(keys) for item in output):
                continue

            return np.asarray(
                [[item[key] for key in keys] for item in output],
                dtype=np.float64,
            )

    raise RuntimeError(
        "Could not extract a valid ONNX score matrix. "
        f"Output types: {[type(output).__name__ for output in outputs]}"
    )


def main() -> None:
    metadata = json.loads((ROOT / 'metadata.json').read_text(encoding='utf-8'))
    threshold = float(metadata['threshold'])
    if not 0.0 < threshold < 1.0:
        raise ValueError('Calibrated threshold must be in (0, 1).')
    margin_threshold = float(np.log(threshold) - np.log1p(-threshold))

    model = joblib.load(ROOT / 'model.joblib')  # Trusted local artifacts only.
    preprocessor = joblib.load(ROOT / 'preprocessor.joblib')
    data = pd.read_csv('data/raw/transacciones_fraude_bancario.csv', encoding='utf-8-sig')
    split = split_dataset(data, random_state=SEED)
    x_dev = split.X_train.reset_index(drop=True)
    # Intentionally never transform or predict on split.X_test / split.y_test.
    session = ort.InferenceSession(
        str(ROOT / 'model_margin_final.onnx'),
        providers=['CPUExecutionProvider'],
    )
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f'Expected one ONNX model input, got {len(inputs)}')
    input_name = inputs[0].name
    if inputs[0].type != 'tensor(float)':
        raise RuntimeError(f'Unexpected ONNX input type: {inputs[0].type}')

    max_margin_error = 0.0
    decision_disagreements = 0
    python_reviews = 0
    onnx_reviews = 0
    near_threshold_count = 0
    evaluated = 0
    examples = []

    for start in range(0, len(x_dev), BATCH):
        end = min(start + BATCH, len(x_dev))
        x = preprocessor.transform(x_dev.iloc[start:end])
        if hasattr(x, 'toarray'):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        python_margins = np.asarray(model.predict(x, raw_score=True), dtype=np.float64).reshape(-1)

        outputs = session.run(None, {input_name: x})

        matrix = extract_onnx_scores(
            outputs=outputs,
            batch_size=len(x),
        )

        onnx_margins = matrix[:, 1]



        delta = np.abs(python_margins - onnx_margins)
        max_margin_error = max(max_margin_error, float(np.max(delta)))
        py_review = python_margins >= margin_threshold
        ox_review = onnx_margins >= margin_threshold
        mismatches = np.flatnonzero(py_review != ox_review)
        decision_disagreements += len(mismatches)
        python_reviews += int(np.sum(py_review))
        onnx_reviews += int(np.sum(ox_review))
        near_threshold_count += int(np.sum(np.abs(python_margins - margin_threshold) < 0.001))
        for i in mismatches[:max(0, 10 - len(examples))]:
            examples.append({
                'development_row_index': int(start + i),
                'python_margin': float(python_margins[i]),
                'onnx_margin': float(onnx_margins[i]),
            })
        evaluated += len(x)
        if end % 8192 < BATCH or end == len(x_dev):
            print(f'Validated {end}/{len(x_dev)} development rows')

    report = {
        'validation_scope': 'ONNX vs Python equivalence on ALL DEVELOPMENT rows (in-sample for candidate model)',
        'holdout_evaluated': False,
        'evaluated_rows': evaluated,
        'input_dtype': 'float32',
        'onnx_score_column': 1,
        'calibrated_probability_threshold': threshold,
        'margin_threshold': margin_threshold,
        'maximum_absolute_margin_difference': max_margin_error,
        'decision_disagreements': decision_disagreements,
        'decision_agreement': 1.0 - decision_disagreements / evaluated,
        'python_review_count': python_reviews,
        'onnx_review_count': onnx_reviews,
        'python_margins_within_0_001_of_threshold': near_threshold_count,
        'disagreement_examples': examples,
        'equivalent_on_checked_development_rows': decision_disagreements == 0,
        'notes': 'Equivalence does not imply final predictive performance, generalization, or real-world bank safety.',
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Report:', REPORT)
    if decision_disagreements:
        raise SystemExit('FAIL: Do not deploy candidate: ONNX/Python decision mismatch.')
    print('PASS: ONNX/Python decisions agree on checked development rows.')


if __name__ == '__main__':
    main()
