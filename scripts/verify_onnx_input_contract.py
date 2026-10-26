"""Generate verified ONNX 9-to-15 input contract without changing model artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT/'models'/'candidates'/'lightgbm_weights'
REPORT=ROOT/'reports'/'onnx_input_contract.json'
ORIGINAL=['monto_transaccion','canal_transaccion','categoria_comercio',
 'velocidad_tx_10m','velocidad_desplazamiento_kmh','ratio_desviacion_monto',
 'pais_emisor_coincide','distancia_residencia_km','hora_dia']

def main():
    pre=joblib.load(CANDIDATE/'preprocessor.joblib')
    names=[str(x) for x in pre.get_feature_names_out()]
    session=ort.InferenceSession(str(CANDIDATE/'model_margin_final.onnx'),providers=['CPUExecutionProvider'])
    inp=session.get_inputs()[0]
    last=inp.shape[-1]
    dtype=inp.type
    ok=len(ORIGINAL)==9 and len(names)==15 and last==len(names) and dtype=='tensor(float)'
    payload={'model':'LightGBM ONNX','original_feature_count':len(ORIGINAL),
      'original_features':ORIGINAL,'transformed_feature_count':len(names),
      'transformed_features':names,'onnx_input':{'name':inp.name,'shape':inp.shape,'type':dtype},
      'checks':{'9_original_features':len(ORIGINAL)==9,'15_transformed_features':len(names)==15,
       'onnx_dimension_matches_preprocessor':last==len(names),'onnx_float32':dtype=='tensor(float)',
       'all_checks_passed':ok},
      'notes':'Manifest generated from frozen preprocessor and ONNX metadata; no retraining.'}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(payload['checks'],ensure_ascii=False,indent=2))
    print('Report:',REPORT)
    if not ok: raise SystemExit('ERROR: incompatibilidad entre preprocesador y ONNX')

if __name__=='__main__':main()
