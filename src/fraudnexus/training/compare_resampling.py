"""Leakage-controlled resampling evaluation on the official development split.

Outer stratified 5-fold validation; each outer training fold is split into
fit/calibration portions. All preprocessing and resampling fit on fit only.
Calibration labels choose a threshold; outer validation labels evaluate it.
Does not read external holdout values for evaluation or change deployed artifacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudnexus.data.generator import FEATURE_COLUMNS, TARGET_COLUMN
from fraudnexus.data.split_data import split_dataset
from fraudnexus.features.preprocessing import build_preprocessor
from fraudnexus.training.compare_models import make_estimator, threshold_at_recall

SEED = 2026
SUPERVISED = ('random_forest', 'xgboost', 'lightgbm')
METHODS = ('weights', 'smote', 'adasyn')


def make_sampler(method, seed):
    if method == 'smote':
        return SMOTE(sampling_strategy=0.10, k_neighbors=3, random_state=seed)
    if method == 'adasyn':
        return ADASYN(sampling_strategy=0.10, n_neighbors=3, random_state=seed)
    return None


def fit_score(name, method, x_fit, y_fit, x_cal, x_val, seed):
    prep = build_preprocessor()
    a = np.asarray(prep.fit_transform(x_fit), dtype=np.float32)
    b = np.asarray(prep.transform(x_cal), dtype=np.float32)
    c = np.asarray(prep.transform(x_val), dtype=np.float32)
    yy = np.asarray(y_fit, dtype=np.int32)
    sampler = make_sampler(method, seed)
    if sampler is not None:
        a, yy = sampler.fit_resample(a, yy)
    positives = int((yy == 1).sum())
    negatives = int((yy == 0).sum())
    clf = make_estimator(name, positives, negatives)
    if method != 'weights':
        # Avoid doubling the class weighting after oversampling.
        if name == 'random_forest':
            clf.set_params(class_weight=None)
        else:
            clf.set_params(scale_pos_weight=1.0)
    clf.fit(a, yy)
    return clf.predict_proba(b)[:, 1], clf.predict_proba(c)[:, 1], int(len(yy))


def evaluate_fold(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, pred, labels=[0,1]).ravel())
    return {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp,
            'precision':float(precision_score(y_true, pred, zero_division=0)),
            'recall':float(recall_score(y_true, pred, zero_division=0)),
            'f1':float(f1_score(y_true, pred, zero_division=0)),
            'auc_pr':float(average_precision_score(y_true, probs)),
            'roc_auc':float(roc_auc_score(y_true, probs)),
            'fpr':float(fp / (fp + tn))}


def experiment(data_path: Path, quick=False):
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    split = split_dataset(df, random_state=SEED)
    # No scoring, feature exploration, or threshold adjustment on split.X_test.
    x = split.X_train.reset_index(drop=True)
    y = split.y_train.reset_index(drop=True)
    if quick:
        x, _, y, _ = train_test_split(x, y, train_size=0.25,
                                      stratify=y, random_state=SEED)
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
    cv = StratifiedKFold(n_splits=2 if quick else 5, shuffle=True, random_state=SEED)
    results = {}
    for name in SUPERVISED:
        for method in METHODS:
            key = f'{name}__{method}'
            folds = []
            print(f'\n[{key}]', flush=True)
            for i, (outer_fit, outer_val) in enumerate(cv.split(x,y), start=1):
                x_outer, y_outer = x.iloc[outer_fit], y.iloc[outer_fit]
                x_fit, x_cal, y_fit, y_cal = train_test_split(
                    x_outer, y_outer, test_size=0.25, stratify=y_outer,
                    random_state=SEED + i)
                try:
                    s_cal, s_val, n_after = fit_score(
                        name, method, x_fit, y_fit, x_cal, x.iloc[outer_val], SEED+i)
                    # Strictly greater than 90% target in calibration where attainable.
                    thr = threshold_at_recall(np.asarray(y_cal), s_cal, target=0.91)
                    met = evaluate_fold(y.iloc[outer_val].to_numpy(), s_val, thr)
                    met.update({'fold':i, 'threshold':float(thr), 'resampled_train_size':n_after})
                    folds.append(met)
                    print(f' fold {i}: recall={met["recall"]:.3f} fp={met["fp"]} AUC-PR={met["auc_pr"]:.4f}', flush=True)
                except (ValueError, RuntimeError) as exc:
                    # ADASYN may fail when all minority neighbors are other minority examples.
                    folds.append({'fold':i,'error':str(exc)})
                    print(f' fold {i}: unsupported sampler distribution: {exc}', flush=True)
            valid = [f for f in folds if 'error' not in f]
            if len(valid) != cv.n_splits:
                results[key] = {'status':'incomplete', 'folds':folds}
                continue
            totals = {k:sum(f[k] for f in valid) for k in ('tn','fp','fn','tp')}
            tp,fp,fn,tn = totals['tp'], totals['fp'], totals['fn'], totals['tn']
            p = tp/(tp+fp) if tp+fp else 0
            r = tp/(tp+fn) if tp+fn else 0
            results[key] = {
                'status':'complete', 'folds':folds, 'confusion_matrix':totals,
                'precision':p, 'recall':r, 'f1':2*p*r/(p+r) if p+r else 0,
                'fpr':fp/(fp+tn),
                'mean_auc_pr':float(np.mean([f['auc_pr'] for f in valid])),
                'mean_roc_auc':float(np.mean([f['roc_auc'] for f in valid])),
                'mean_fold_recall':float(np.mean([f['recall'] for f in valid])),
            }
    eligible = [(k,v) for k,v in results.items() if v['status']=='complete' and v['recall']>0.90]
    best = min(eligible, key=lambda pair:(pair[1]['fpr'], -pair[1]['mean_auc_pr'], -pair[1]['f1']))[0] if eligible else None
    report = {
        'protocol':'five_outer_folds_with_inner_threshold_calibration',
        'dataset':str(data_path), 'development_rows':len(x), 'holdout_untouched':True,
        'quick':quick, 'note':'SMOTE/ADASYN applied to one-hot transformed columns; synthetic fractional dummy values are a methodological limitation.',
        'selected_configuration_provisional':best,
        'warning':'Outer-fold thresholds vary. Before final holdout, refit winning approach on full development and select a single frozen threshold from development-only calibration.',
        'configurations':results
    }
    out = Path('reports/resampling_comparison_quick.json' if quick else 'reports/resampling_comparison.json')
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nPROVISIONAL: {best}; report: {out}', flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--data', type=Path, default=Path('data/raw/transacciones_fraude_bancario.csv'))
    args=parser.parse_args()
    experiment(args.data, quick=args.quick)

if __name__ == '__main__':
    main()
