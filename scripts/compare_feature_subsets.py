"""Academic feature-subset ablation on DEVELOPMENT ONLY; original model remains frozen.

Run from repository root: python -m scripts.compare_feature_subsets
Quick smoke check: python -m scripts.compare_feature_subsets --quick

Outer folds estimate predictive performance. All feature ranking, preprocessing,
model fitting and threshold calibration occur inside each outer training fold.
No holdout outcomes or frozen model artifacts are accessed.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from fraudnexus.data.generator import FEATURE_COLUMNS, TARGET_COLUMN
from fraudnexus.data.split_data import split_dataset
from fraudnexus.features.preprocessing import build_preprocessor
from fraudnexus.training.compare_models import make_estimator, threshold_at_recall

PROJECT = Path(__file__).resolve().parents[1]
SEED = 2026
SUBSETS = (9, 6, 4, 1)
CATEGORICAL = ('canal_transaccion', 'categoria_comercio')


def select_features(X, y, count, seed):
    """Rank ONLY the fold's fit subset, with numeric imputation + ordinal categories.

    Ranking is exploratory mutual information. Fold-specific ranking prevents
    using validation information to select variables.
    """
    if count == len(FEATURE_COLUMNS):
        return list(FEATURE_COLUMNS), None
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
    categorical = [c for c in FEATURE_COLUMNS if c in CATEGORICAL]
    rank_prep = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), categorical),
    ])
    matrix = np.asarray(rank_prep.fit_transform(X), dtype=np.float64)
    mask = [False] * len(numeric) + [True] * len(categorical)
    scores = mutual_info_classif(matrix, np.asarray(y), discrete_features=mask,
                                random_state=seed, n_neighbors=3)
    names = numeric + categorical
    ranks = sorted(zip(names, scores), key=lambda item: (-float(item[1]), item[0]))
    selected_set = {name for name, _ in ranks[:count]}
    return [c for c in FEATURE_COLUMNS if c in selected_set], [
        {'variable': n, 'mutual_information': float(s)} for n, s in ranks
    ]


def evaluate_one(X_fit, y_fit, X_cal, y_cal, X_val, y_val, selected, seed, quick):
    prep = build_preprocessor() if len(selected) == len(FEATURE_COLUMNS) else subset_preprocessor(selected)
    fit_array = np.asarray(prep.fit_transform(X_fit[selected]), dtype=np.float32)
    cal_array = np.asarray(prep.transform(X_cal[selected]), dtype=np.float32)
    val_array = np.asarray(prep.transform(X_val[selected]), dtype=np.float32)
    pos = int(np.sum(np.asarray(y_fit) == 1))
    neg = int(np.sum(np.asarray(y_fit) == 0))
    clf = make_estimator('lightgbm', pos, neg, quick=quick)
    clf.set_params(random_state=seed)
    clf.fit(fit_array, y_fit)
    cal_score = np.asarray(clf.predict_proba(cal_array)[:, 1], dtype=np.float64)
    val_score = np.asarray(clf.predict_proba(val_array)[:, 1], dtype=np.float64)
    threshold = threshold_at_recall(np.asarray(y_cal), cal_score, target=0.91)
    predicted = (val_score >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y_val, predicted, labels=[0, 1]).ravel())
    return {'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn,
            'recall':float(recall_score(y_val,predicted,zero_division=0)),
            'precision':float(precision_score(y_val,predicted,zero_division=0)),
            'f1':float(f1_score(y_val,predicted,zero_division=0)),
            'auc_pr':float(average_precision_score(y_val,val_score)),
            'threshold_internal':float(threshold)}


def subset_preprocessor(selected):
    """Same preprocessing semantics as full project, but only selected inputs."""
    from sklearn.preprocessing import OneHotEncoder
    nums = [c for c in selected if c not in CATEGORICAL]
    cats = [c for c in selected if c in CATEGORICAL]
    transformers = []
    if nums:
        transformers.append(('numeric', SimpleImputer(strategy='median'), nums))
    if cats:
        transformers.append(('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cats))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='2 folds on a 25% development subsample; ONLY a smoke test')
    args = parser.parse_args()
    path = PROJECT / 'data/raw/transacciones_fraude_bancario.csv'
    data = pd.read_csv(path, encoding='utf-8-sig')
    split = split_dataset(data, random_state=SEED)
    X = split.X_train.reset_index(drop=True)
    y = split.y_train.reset_index(drop=True)
    assert set(FEATURE_COLUMNS).issubset(X.columns)
    if args.quick:
        X, _, y, _ = train_test_split(X, y, train_size=0.25, stratify=y, random_state=SEED)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    fold_count = 2 if args.quick else 5
    cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=SEED)
    rows, selection_records = [], []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_outer, y_outer = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_fit, X_cal, y_fit, y_cal = train_test_split(X_outer, y_outer,
            test_size=0.25, stratify=y_outer, random_state=SEED + fold)
        for count in SUBSETS:
            selected, ranking = select_features(X_fit, y_fit, count, SEED+fold)
            r = evaluate_one(X_fit, y_fit, X_cal, y_cal, X_val, y_val,
                             selected, SEED+fold, args.quick)
            r.update({'fold':fold, 'variable_count':count, 'selected_features':','.join(selected),
                      'n_validation':len(y_val)})
            rows.append(r)
            selection_records.append({'fold':fold,'variable_count':count,
                                      'features':selected, 'ranking_fit_only':ranking})
            print(f'fold {fold}/{fold_count} | {count} features | '
                  f'recall={r["recall"]:.3f}, f1={r["f1"]:.3f}, '
                  f'auc_pr={r["auc_pr"]:.4f}, FP={r["fp"]}',flush=True)
    out = PROJECT / 'reports/feature_reduction'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / ('folds_quick.csv' if args.quick else 'folds.csv'),index=False)
    aggregate=[]
    for count in SUBSETS:
        sub=[r for r in rows if r['variable_count']==count]
        sums={k:int(sum(r[k] for r in sub)) for k in ('tp','tn','fp','fn')}
        t=sums
        precision=t['tp']/(t['tp']+t['fp']) if t['tp']+t['fp'] else 0
        recall=t['tp']/(t['tp']+t['fn']) if t['tp']+t['fn'] else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        aggregate.append({'variable_count':count, **sums,'recall':recall,
                          'precision':precision,'f1':f1,'fpr':t['fp']/(t['fp']+t['tn']),
                          'mean_fold_auc_pr':float(np.mean([r['auc_pr'] for r in sub])),
                          'std_fold_auc_pr':float(np.std([r['auc_pr'] for r in sub])),
                          'folds':len(sub)})
    summary_name='summary_quick.csv' if args.quick else 'summary.csv'
    pd.DataFrame(aggregate).to_csv(out / summary_name,index=False)
    info={'scope':'Development-only outer stratified folds; feature ranking on fit partition only',
          'quick':args.quick, 'dataset_file':path.name, 'development_rows':len(X),
          'development_frauds':int(sum(y)), 'holdout_used_for_scoring':False,
          'holdout_used_for_feature_selection':False,
          'frozen_deployed_model_changed':False,'selection_method':'mutual_info_classif',
          'calibration_method':'internal 25% split, threshold for >=91% recall on calibration',
          'feature_counts':list(SUBSETS), 'created_utc':datetime.now(timezone.utc).isoformat(),
          'selection_by_fold':selection_records,
          'caution':'Exploratory ablation after observing perfect final holdout; NOT a new unbiased final model-selection claim.'}
    (out / ('metadata_quick.json' if args.quick else 'metadata.json')).write_text(
        json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    vals=sorted(aggregate,key=lambda r:r['variable_count'])
    xs=np.arange(len(vals)); width=.26
    ax.bar(xs-width, [100*v['recall'] for v in vals], width,label='Recall (%)')
    ax.bar(xs, [100*v['f1'] for v in vals], width,label='F1 (%)')
    ax.bar(xs+width,[100*v['mean_fold_auc_pr'] for v in vals],width,label='AUC-PR medio por fold (%)')
    ax.set_xticks(xs,[str(v['variable_count']) for v in vals]); ax.set_xlabel('Número de variables originales')
    ax.set_ylabel('Porcentaje (%)'); ax.set_ylim(0,105);ax.set_title('Ablación de variables · validación cruzada en desarrollo')
    ax.legend(loc='lower right');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out / ('comparison_quick.png' if args.quick else 'comparison.png'),dpi=180)
    plt.close(fig)
    print('\nSaved',out / summary_name)
    print(pd.DataFrame(aggregate).to_string(index=False))

if __name__ == '__main__':
    main()
