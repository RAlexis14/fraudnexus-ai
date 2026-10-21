"""Leakage-aware four-model comparison on the official FraudNexus dataset.

The external 20% test set is NEVER accessed for scoring or selection here.
OOF metrics used to select a threshold are development estimates, not an
unbiased estimate of threshold performance; report final results separately.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraudnexus.data.generator import FEATURE_COLUMNS, TARGET_COLUMN
from fraudnexus.features.preprocessing import build_preprocessor

SEED = 2026
MODEL_NAMES = ("random_forest", "xgboost", "lightgbm", "isolation_forest")


def make_estimator(name: str, positives: int, negatives: int, quick: bool = False):
    trees = 40 if quick else 120
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=SEED,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=trees, max_depth=4, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=negatives / max(positives, 1),
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=4, tree_method="hist", random_state=SEED,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trees, max_depth=5, num_leaves=20,
            learning_rate=0.06, scale_pos_weight=negatives / max(positives, 1),
            verbosity=-1, n_jobs=4, random_state=SEED,
        )
    if name == "isolation_forest":
        return IsolationForest(
            n_estimators=trees, contamination="auto",
            n_jobs=-1, random_state=SEED,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_pipeline(name: str, X: pd.DataFrame, y: pd.Series, quick: bool = False):
    """Fit preprocessing and model on the supplied training fold only."""
    negatives = int((y == 0).sum())
    positives = int((y == 1).sum())
    estimator = make_estimator(name, positives, negatives, quick)
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", estimator),
    ])
    if name == "isolation_forest":
        # Anomaly detector is trained only on normal training transactions.
        pipeline.fit(X.loc[y == 0])
    else:
        pipeline.fit(X, y)
    return pipeline


def risk_score(name: str, pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher always means greater fraud risk; IF scores are NOT probabilities."""
    if name == "isolation_forest":
        transformed = pipeline.named_steps["preprocessing"].transform(X)
        return -pipeline.named_steps["model"].decision_function(transformed)
    return pipeline.predict_proba(X)[:, 1]


def threshold_at_recall(y: np.ndarray, scores: np.ndarray, target: float = 0.90) -> float:
    """Highest inclusive threshold achieving desired recall on labeled OOF data."""
    positive_scores = np.sort(scores[y == 1])[::-1]
    if positive_scores.size == 0:
        raise ValueError("There are no positive examples.")
    required = int(np.ceil(target * positive_scores.size))
    return float(positive_scores[required - 1])


def describe(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "average_precision_auc_pr": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_positive_count": fp,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def compare(data_path: Path, quick: bool = False, output_root: Path = Path(".")) -> dict:
    frame = pd.read_csv(data_path, encoding="utf-8-sig")
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame[TARGET_COLUMN].isna().any() or not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Fraud labels must be 0/1 and non-null")
    X_development, _X_holdout, y_development, _y_holdout = train_test_split(
        frame[FEATURE_COLUMNS], frame[TARGET_COLUMN].astype(int),
        test_size=0.2, random_state=SEED, stratify=frame[TARGET_COLUMN],
    )
    # No inference, threshold selection or metrics on external holdout.
    if quick:
        X_development, _, y_development, _ = train_test_split(
            X_development, y_development, train_size=0.25,
            random_state=SEED, stratify=y_development,
        )
    X_development = X_development.reset_index(drop=True)
    y_development = y_development.reset_index(drop=True)
    cv = StratifiedKFold(n_splits=2 if quick else 5, shuffle=True, random_state=SEED)
    y = y_development.to_numpy()
    results = {}
    oof_scores = {}
    for name in MODEL_NAMES:
        scores = np.full(len(y), np.nan, dtype=float)
        print(f"\n[{name}] CV folds: {cv.n_splits}", flush=True)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_development, y), start=1):
            model = fit_pipeline(
                name, X_development.iloc[train_idx], y_development.iloc[train_idx], quick,
            )
            scores[val_idx] = risk_score(name, model, X_development.iloc[val_idx])
            print(f"  Fold {fold}/{cv.n_splits}: completed", flush=True)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Invalid OOF risk scores for {name}")
        cutoff = threshold_at_recall(y, scores)
        metrics = describe(y, scores, cutoff)
        results[name] = {
            "threshold": cutoff,
            "score_type": "anomaly_score_not_probability" if name == "isolation_forest" else "classifier_score",
            **metrics,
        }
        oof_scores[name] = scores
        print(
            f"  AUC-PR={metrics['average_precision_auc_pr']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"FPR={metrics['false_positive_rate']:.4f}",
            flush=True,
        )
    # Prioritize low FPR under development recall constraint; AUC-PR as tie-breaker.
    best = min(
        MODEL_NAMES,
        key=lambda name: (
            results[name]["false_positive_rate"],
            -results[name]["average_precision_auc_pr"],
        ),
    )
    final_pipeline = fit_pipeline(best, X_development, y_development, quick)
    report = {
        "dataset": str(data_path),
        "n_total": int(len(frame)),
        "n_development": int(len(X_development)),
        "n_external_holdout_reserved": int(len(frame) - (len(frame) // 5) * 4),
        "cv_folds": cv.n_splits,
        "quick_experiment": quick,
        "threshold_target_recall": 0.90,
        "selected_model": best,
        "selection_method": "minimum_oof_fpr_at_recall_90_pct; tie_break_auc_pr",
        "warning": (
            "OOF threshold-selected metrics are development metrics and can be optimistic. "
            "Never present them as external test performance."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "models": results,
    }
    output = output_root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    (output / f"model_comparison{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not quick:
        model_folder = output_root / "models"
        model_folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, model_folder / "selected_model.joblib")
        (model_folder / "selected_model_metadata.json").write_text(
            json.dumps({
                "model_name": best, "threshold": results[best]["threshold"],
                "score_type": results[best]["score_type"],
                "feature_columns": FEATURE_COLUMNS,
                "model_input_columns": list(FEATURE_COLUMNS),
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\nSELECTED: {best} (development only)", flush=True)
    print(f"Report: {output / f'model_comparison{suffix}.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/transacciones_fraude_bancario.csv"))
    parser.add_argument("--quick", action="store_true", help="Fast pipeline verification only; not official CV")
    args = parser.parse_args()
    compare(args.data, quick=args.quick)


if __name__ == "__main__":
    main()
