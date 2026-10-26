"""EXPLORATORY permutation importance on development data only.

Usage: python -m scripts.analyze_feature_importance
Uses frozen ONNX and preprocessing; does not train/recalibrate or inspect holdout.
The result is *in-sample* global model sensitivity, NOT unbiased predictive importance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/transacciones_fraude_bancario.csv'
CAND=ROOT/'models/candidates/lightgbm_weights'
OUT=ROOT/'reports/feature_importance_development.csv'
META=ROOT/'reports/feature_importance_development_metadata.json'
SEED=42

def score(session, preprocessor, frame, columns, batch=1024):
    values=[]
    input_name=session.get_inputs()[0].name
    for start in range(0,len(frame),batch):
        x=np.asarray(preprocessor.transform(frame.iloc[start:start+batch][columns]),dtype=np.float32)
        res=session.run(['probabilities'],{input_name:x})[0]
        if isinstance(res,list): values.extend(float(item[1]) for item in res)
        else: values.extend(np.asarray(res)[:,1].astype(float).tolist())
    return np.asarray(values,dtype=np.float64)

def main():
    meta=json.loads((CAND/'metadata.json').read_text(encoding='utf-8'))
    columns=list(meta['feature_columns'])
    raw=pd.read_csv(DATA)
    # Same 80/20 seed and stratification as original project; verify with split_data if changed.
    develop, _=train_test_split(raw,test_size=.2,random_state=SEED,stratify=raw['es_fraude'])
    develop=develop.reset_index(drop=True)
    pre=joblib.load(CAND/'preprocessor.joblib')
    session=ort.InferenceSession(str(CAND/'model_margin_final.onnx'),providers=['CPUExecutionProvider'])
    baseline=average_precision_score(develop['es_fraude'],score(session,pre,develop,columns))
    rng=np.random.default_rng(SEED)
    rows=[]
    for name in columns:
        drops=[]
        for _ in range(3):
            altered=develop[columns].copy()
            altered[name]=rng.permutation(altered[name].to_numpy())
            ap=average_precision_score(develop['es_fraude'],score(session,pre,altered,columns))
            drops.append(baseline-ap)
        rows.append({'variable':name,'mean_auc_pr_drop':float(np.mean(drops)),
                     'std_auc_pr_drop':float(np.std(drops)), 'baseline_auc_pr':float(baseline)})
        print(f'{name}: mean AP reduction={np.mean(drops):.6f}')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).sort_values('mean_auc_pr_drop',ascending=False).to_csv(OUT,index=False)
    META.write_text(json.dumps({'scope':'In-sample descriptive sensitivity on frozen full-development model; not holdout or causal importance',
            'rows':len(develop),'positive_cases':int(develop.es_fraude.sum()),'baseline_ap':float(baseline),
            'permutations_per_feature':3,'seed':SEED,'note':'Do not use to retune model/threshold based on final holdout'},indent=2),encoding='utf-8')
    print('Saved',OUT)

if __name__=='__main__':main()
