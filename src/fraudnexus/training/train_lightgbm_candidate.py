"""Build a LightGBM candidate from development data only.

Five-fold OOF predictions calibrate the threshold; the final estimator is
refit on all development rows. Holdout rows are NEVER evaluated here.
The OOF-calibrated threshold is NOT an unbiased final-performance estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from fraudnexus.data.generator import FEATURE_COLUMNS, TARGET_COLUMN
from fraudnexus.data.split_data import split_dataset
from fraudnexus.features.preprocessing import build_preprocessor
from fraudnexus.training.compare_models import SEED, make_estimator


def get_development(data_path: Path):
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[TARGET_COLUMN].isna().any() or not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Invalid target values")
    divided = split_dataset(data, random_state=SEED)
    # Never read or score X_test/y_test; split_dataset only partitions rows.
    return divided.X_train.reset_index(drop=True), divided.y_train.reset_index(drop=True)


def fit_model(x_train, y_train):
    prep = build_preprocessor()
    transformed = np.asarray(prep.fit_transform(x_train), dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int32)
    pos = int(np.count_nonzero(y == 1))
    neg = int(np.count_nonzero(y == 0))
    clf = make_estimator("lightgbm", pos, neg)
    clf.fit(transformed, y)
    return prep, clf


def evaluate(y, scores, threshold):
    prediction = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, prediction, labels=[0, 1]).ravel())
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "auc_pr": float(average_precision_score(y, scores)),
        "fpr": float(fp / (fp + tn)),
    }


def run(data_path: Path, target_recall: float = 0.925):
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be within (0, 1]")
    x, y = get_development(data_path)
    classes = np.asarray(y, dtype=np.int32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, classes), start=1):
        prep, model = fit_model(x.iloc[train_idx], y.iloc[train_idx])
        val_array = np.asarray(prep.transform(x.iloc[val_idx]), dtype=np.float32)
        oof[val_idx] = model.predict_proba(val_array)[:, 1]
        print(f"OOF fold {fold}/5 ready: {len(val_idx)} rows", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions contain non-finite values")

    pos_scores = np.sort(oof[classes == 1])[::-1]
    minimum_detected = int(np.floor(0.90 * len(pos_scores))) + 1
    desired = max(minimum_detected, int(np.ceil(target_recall * len(pos_scores))))
    desired = min(desired, len(pos_scores))
    raw_threshold = float(pos_scores[desired - 1])
    if not np.isfinite(raw_threshold) or not (0 <= raw_threshold <= 1):
        raise RuntimeError("Selected invalid threshold")
    # ONNX predictions may have float32 rounding differences. Threshold is
    # not validated on holdout here; check numerical agreement before deploy.
    metrics = evaluate(classes, oof, raw_threshold)

    preprocessor, final_model = fit_model(x, y)
    output = Path("models/candidates/lightgbm_weights")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output / "model.joblib")
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    metadata = {
        "model_name": "lightgbm", "strategy": "weights",
        "threshold": raw_threshold, "score_type": "classifier_score",
        "feature_columns": list(FEATURE_COLUMNS),
        "model_input_columns": list(FEATURE_COLUMNS),
        "development_rows": int(len(y)), "cv_folds": 5,
        "target_recall_for_oof_calibration": target_recall,
        "calibration_protocol": "development_only_5fold_oof",
        "warning": "Threshold calibrated from development OOF scores and frozen for full-development refit; validation estimates remain optimistic for threshold choice. Final holdout is untested.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "candidate": "lightgbm__weights", "holdout_evaluated": False,
        "oof_threshold": raw_threshold, "threshold_selection_positive_count": desired,
        "oof_calibration_metrics_not_independent_validation": metrics,
        "positive_score_quantiles": {str(v): float(np.quantile(pos_scores, v)) for v in (0, 0.01, 0.05, 0.1, 0.5, 1)},
        "threshold_is_close_to_one": bool(raw_threshold > 0.9999),
        "model_artifacts": str(output),
    }
    report_path = Path("reports/lightgbm_candidate.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nCandidate saved:", output)
    print("OOF calibration report:", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/transacciones_fraude_bancario.csv"))
    parser.add_argument("--target-recall", type=float, default=0.925)
    args = parser.parse_args()
    run(args.data, args.target_recall)


if __name__ == "__main__":
    main()
