"""FraudNexus AI | Evidence-driven Streamlit dashboard (phase 1)."""
from __future__ import annotations

import json
import io
import time
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
BRAND = "#173D50"
ACCENT = "#207F88"
GREEN = "#17764F"
AMBER = "#AF741F"


def read_json(filename: str) -> dict[str, Any]:
    path = REPORTS / filename
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as exc:
        st.warning(f"No se pudo leer {filename}: {exc}")
        return {}


def get_health() -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"status": "down", "detail": str(exc)}


def api_post(path: str, data: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=data, timeout=timeout)
    if not response.ok:
        raise ValueError(f"API HTTP {response.status_code}: {response.text[:700]}")
    return response.json()


COLUMNS = ["monto_transaccion", "canal_transaccion", "categoria_comercio", "velocidad_tx_10m",
           "velocidad_desplazamiento_kmh", "ratio_desviacion_monto", "pais_emisor_coincide",
           "distancia_residencia_km", "hora_dia"]
NUMERIC = ["monto_transaccion", "velocidad_tx_10m", "velocidad_desplazamiento_kmh",
           "ratio_desviacion_monto", "pais_emisor_coincide", "distancia_residencia_km", "hora_dia"]
ALLOWED = {"canal_transaccion": ["POS", "E-Commerce", "ATM", "App Móvil"],
           "categoria_comercio": ["Retail", "Viajes", "Supermercados", "Entretenimiento"]}


def patterns_csv(name: str) -> pd.DataFrame:
    p = REPORTS / "fraud_patterns" / name
    if not p.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"No se pudo leer {name}: {exc}")
        return pd.DataFrame()


def validate_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    issues = []
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        return pd.DataFrame(), ["Faltan columnas: " + ", ".join(missing)]
    df = frame.copy()
    for name in NUMERIC:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    for name, choices in ALLOWED.items():
        invalid = ~df[name].isin(choices)
        if invalid.any():
            issues.append(f"{name}: {int(invalid.sum())} valores ajenos a {choices}")
    for name in NUMERIC:
        bad = df[name].isna() | ~df[name].apply(lambda x: bool(pd.notna(x) and abs(x) != float('inf')))
        if bad.any(): issues.append(f"{name}: {int(bad.sum())} valores ausentes o no numéricos")
    for name in ["monto_transaccion", "velocidad_tx_10m", "velocidad_desplazamiento_kmh", "ratio_desviacion_monto", "distancia_residencia_km"]:
        if (df[name] < 0).any(): issues.append(f"{name}: contiene negativos")
    if (~df['hora_dia'].between(0,23) | (df['hora_dia'] % 1 != 0)).any(): issues.append("hora_dia debe ser entero de 0 a 23")
    if (~df['pais_emisor_coincide'].isin([0,1])).any(): issues.append("pais_emisor_coincide debe ser 0 o 1")
    if (df['velocidad_tx_10m'] % 1 != 0).any(): issues.append("velocidad_tx_10m debe ser un entero")
    if 'es_fraude' in df.columns:
        df['es_fraude'] = pd.to_numeric(df['es_fraude'], errors='coerce')
        if (~df['es_fraude'].isin([0,1])).any(): issues.append("es_fraude debe contener únicamente 0 o 1")
    return df, issues


def inference_metrics(df: pd.DataFrame) -> dict:
    y = df['es_fraude'].astype(int)
    p = df['is_fraud'].astype(int)
    tp = int(((y==1)&(p==1)).sum()); tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum())
    prec=tp/(tp+fp) if tp+fp else 0.; rec=tp/(tp+fn) if tp+fn else 0.
    return {'tp':tp,'tn':tn,'fp':fp,'fn':fn,'precision':prec,'recall':rec,
            'f1':2*prec*rec/(prec+rec) if prec+rec else 0.,'fpr':fp/(fp+tn) if fp+tn else 0.}


def show_indicator(model: dict, final: dict):
    active=model.get('model')
    if model.get('status') != 'ok':
        st.error("API sin conexión: no se ejecutarán nuevas inferencias")
    elif active != 'lightgbm_candidate':
        st.warning(f"Motor conectado: {active}. Las métricas finales que se muestran pertenecen a LightGBM, NO a este motor.")
    elif final:
        st.success("Motor conectado: LightGBM ONNX · evaluación final documentada")
    else:
        st.warning("LightGBM activo; no se encontró el informe de evaluación final")


def apply_style() -> None:
    st.markdown("""<style>
      .stApp {background:#F5F7F6;color:#1F3140;}
      h1,h2,h3 {color:#173D50; letter-spacing:-.025em;}
      div[data-testid="stMetric"] {background:#FFFFFF;border:1px solid #E3E9E8;
       border-radius:13px;padding:15px 17px;box-shadow:0 3px 14px rgba(22,48,59,.035);}
      div[data-testid="stMetricLabel"] {color:#627883;}
      div[data-testid="stTabs"] button {font-weight:500;}
      .hero {border-left:4px solid #207F88;padding:4px 0 4px 15px;margin-bottom:18px;}
      .hero strong {font-size:1.9rem;color:#173D50;letter-spacing:-.035em;}
      .hero small {display:block;color:#607683;margin-top:3px;}
      .note {font-size:.88rem;color:#697D85;}
      .decision {padding:17px;border-radius:13px;font-size:1.25rem;
       font-weight:700;border:1px solid #DEE8E5;}
      .allow {background:#EAF5EF;color:#176442;}
      .review {background:#FFF4E6;color:#865A18;}
    </style>""", unsafe_allow_html=True)


def conf_matrix_chart(tn: int, fp: int, fn: int, tp: int, title: str):
    z = [[tn, fp], [fn, tp]]
    # Use text annotations so the count is explicit even for very imbalanced data.
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["Predicha legítima (0)", "Predicha fraude (1)"],
        y=["Real legítima (0)", "Real fraude (1)"],
        colorscale=[[0, "#E9F2F0"], [1, "#207F88"]],
        showscale=False,
        text=[[f"TN · {tn:,}", f"FP · {fp:,}"], [f"FN · {fn:,}", f"TP · {tp:,}"]],
        texttemplate="%{text}", textfont={"size": 14},
        hovertemplate="%{y}<br>%{x}<br>%{z:,} transacciones<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Clasificación estimada", yaxis_title="Etiqueta verdadera",
                      margin=dict(l=10,r=10,t=70,b=15),height=350,font=dict(color=BRAND))
    fig.update_yaxes(autorange="reversed")
    return fig


def center_view(eda: dict, final: dict, latency: dict, health: dict):
    st.subheader("Centro de control")
    st.caption("Indicadores históricos del estudio y estado actual del prototipo; no son transacciones bancarias en vivo.")
    active = health.get("model", "No disponible") if health.get("status") == "ok" else "API desconectada"
    friendly = {"lightgbm_candidate":"LightGBM · seleccionado", "random_forest":"Random Forest · referencia"}.get(active,active)
    if health.get("status") == "ok":
        show_indicator(health, final)
    else:
        st.error("No se puede consultar la API. Comprueba FastAPI o Docker.")
    metrics = final.get("metrics", {})
    a,b,c,d = st.columns(4)
    a.metric("Registros del dataset", f"{eda.get('total_records',0):,}")
    b.metric("Fraudes en dataset", f"{eda.get('fraudulent',0):,}")
    c.metric("Recall · prueba final", f"{100*metrics['recall']:.2f}%" if "recall" in metrics else "Sin informe")
    d.metric("HTTP p95 · último informe", f"{latency.get('http_latency_ms',{}).get('p95',0):.2f} ms" if latency else "Sin informe")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Desbalance del dataset oficial")
        if eda:
            distribution = pd.DataFrame({"Clase":["Legítimas","Fraudulentas"],
                "Transacciones":[eda.get('legitimate',0),eda.get('fraudulent',0)]})
            fig = px.bar(distribution,x="Clase",y="Transacciones",text="Transacciones",
                        color="Clase",color_discrete_map={"Legítimas":ACCENT,"Fraudulentas":AMBER})
            fig.update_layout(showlegend=False,yaxis_title="Número de transacciones",xaxis_title="Clase verdadera",
                              height=345,margin=dict(t=10,b=30),font=dict(color=BRAND))
            fig.update_traces(textposition="outside",cliponaxis=False)
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Dataset oficial completo: 50.000 transacciones. No representa actividad operativa actual.")
        else: st.info("Sin resumen EDA.")
    with right:
        st.markdown("#### Evaluación independiente · LightGBM")
        if metrics:
            st.plotly_chart(conf_matrix_chart(metrics['tn'],metrics['fp'],metrics['fn'],metrics['tp'],
              "Matriz de confusión · holdout (n=10.000)"),use_container_width=True)
            st.caption("Resultados exclusivos del modelo LightGBM congelado. No equivalen al modelo activo si la API usa Random Forest.")
        else: st.info("No se encontró lightgbm_final_holdout.json")
    st.markdown("#### Rendimiento HTTP del modelo seleccionado")
    if latency:
        l = latency.get("http_latency_ms",{})
        labels = ["p50", "p95", "p99", "Máximo"]
        values = [l.get(k) for k in ["p50","p95","p99","max"]]
        if all(v is not None for v in values):
            fig=go.Figure(go.Bar(x=labels,y=values,marker_color=ACCENT,text=[f"{v:.1f}" for v in values],textposition="outside"))
            fig.add_hline(y=100,line_dash="dash",line_color="#B45B4E",annotation_text="Límite objetivo: 100 ms")
            fig.update_layout(xaxis_title="Estadístico de latencia HTTP",yaxis_title="Tiempo (ms)",
                title="Benchmark local, solicitudes secuenciales",yaxis_range=[0,max(112,max(values)*1.25)],
                height=340,margin=dict(t=60,b=15),font=dict(color=BRAND))
            st.plotly_chart(fig,use_container_width=True)
            st.caption(f"{latency.get('successful_requests','?')} solicitudes exitosas · "
                "Incluye tiempo HTTP local, no evalúa concurrencia ni producción. "
                "El informe debe corresponder al modelo actualmente seleccionado.")
    else: st.info("No se encontró un reporte de latencia.")


def lab_view(initial: dict, resampling: dict, final: dict):
    st.subheader("Laboratorio de modelos")
    st.caption("Comparación experimental sobre desarrollo (40.000 filas). El holdout final se presenta por separado.")
    config = resampling.get("configurations",{})
    rows=[]
    names={"random_forest":"Random Forest","xgboost":"XGBoost","lightgbm":"LightGBM"}
    strategies={"weights":"Pesos","smote":"SMOTE","adasyn":"ADASYN"}
    for key,entry in config.items():
        if entry.get("status") != "complete":continue
        algo,strategy=key.split("__",1)
        cm=entry.get("confusion_matrix",{})
        rows.append({"Modelo": names.get(algo,algo),"Estrategia":strategies.get(strategy,strategy),
          "Recall (%)":100*entry["recall"],"F1 (%)":100*entry["f1"],
          "AUC-PR":entry["mean_auc_pr"],"FP":cm.get("fp",0),"FN":cm.get("fn",0),
          "TP":cm.get("tp",0),"TN":cm.get("tn",0)})
    if rows:
        df=pd.DataFrame(rows).sort_values(["Recall (%)","F1 (%)"],ascending=False)
        st.dataframe(df.style.format({"Recall (%)":"{:.2f}","F1 (%)":"{:.2f}","AUC-PR":"{:.4f}"}),
                     use_container_width=True,hide_index=True)
        option=st.selectbox("Métrica para comparar",["Recall (%)","F1 (%)","AUC-PR"])
        plot=df.copy()
        plot["Configuración"] = plot["Modelo"]+" · "+plot["Estrategia"]
        fig=px.bar(plot,x=option,y="Configuración",orientation="h",text=option,
           color="Modelo",color_discrete_sequence=[BRAND,ACCENT,AMBER],
           title=f"{option} por modelo y estrategia · 5 folds externos")
        fig.update_layout(yaxis_title="Modelo y tratamiento del desbalance",xaxis_title=option,
            yaxis={"categoryorder":"total ascending"},height=520,
            margin=dict(t=65,b=35),font=dict(color=BRAND))
        if option != "AUC-PR":
            fig.add_vline(x=90,line_dash="dash",line_color="#B45B4E",annotation_text="Objetivo Recall > 90%" if option.startswith("Recall") else "Referencia 90%") if option.startswith("Recall") else None
            fig.update_traces(texttemplate="%{text:.2f}")
            fig.update_xaxes(range=[0,107])
        else: fig.update_traces(texttemplate="%{text:.4f}");fig.update_xaxes(range=[0,1.09])
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Cada configuración fue evaluada con 5 folds externos; el umbral se calibró en una partición interna. "
          "SMOTE/ADASYN operaron sobre One-Hot Encoding, con la limitación metodológica documentada.")
        selected=st.selectbox("Matriz de confusión de desarrollo",list(config.keys()),
           format_func=lambda s:s.replace("__"," · ").replace("_"," ").title())
        cm=config[selected]["confusion_matrix"]
        st.plotly_chart(conf_matrix_chart(cm["tn"],cm["fp"],cm["fn"],cm["tp"],
            "Matriz de confusión · 5 folds externos de desarrollo (n=40.000)"), use_container_width=True)
    else: st.warning("No se encontró reports/resampling_comparison.json con configuraciones completas.")
    st.markdown("#### Detector de anomalías · experimento inicial")
    anomaly=initial.get("models",{}).get("isolation_forest")
    if anomaly:
        st.info(f"Isolation Forest: Recall {100*anomaly['recall']:.2f}% · F1 {100*anomaly['f1']:.2f}% "
         f"· AUC-PR {anomaly['average_precision_auc_pr']:.4f}. "
         "Experimento inicial con selección de umbral OOF: metodología distinta, NO comparable directamente como ranking final.")
    st.markdown("#### Evaluación externa del modelo congelado")
    met=final.get("metrics",{})
    if met:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Recall · holdout",f"{100*met['recall']:.2f}%")
        c2.metric("F1 · holdout",f"{100*met['f1']:.2f}%")
        c3.metric("AUC-PR · holdout",f"{met['auc_pr_average_precision']:.4f}")
        c4.metric("Falsos negativos",str(met["fn"]))
        st.caption("Evaluación única de 10.000 operaciones reservadas, con 40 fraudes. "
           "Es un dataset académico, no evidencia de precisión bancaria en producción.")
    else: st.info("No se encontró reports/lightgbm_final_holdout.json.")


def individual_view():
    st.subheader("Evaluación individual")
    st.caption("Inferencia ONNX a través de FastAPI. ALLOW permite continuar la simulación; REVIEW solicita revisión adicional simulada.")
    with st.form("transaction"):
        a,b,c=st.columns(3)
        monto=a.number_input("Monto (USD)",min_value=0.0,value=65.2)
        canal=b.selectbox("Canal de transacción",["POS","E-Commerce","ATM","App Móvil"])
        categoria=c.selectbox("Categoría de comercio",["Retail","Viajes","Supermercados","Entretenimiento"])
        d,e,f=st.columns(3)
        velocidad=d.number_input("Transacciones en 10 minutos",min_value=0,value=0,step=1)
        kmh=e.number_input("Desplazamiento (km/h)",min_value=0.0,value=10.0)
        ratio=f.number_input("Ratio de desviación del monto",min_value=0.0,value=1.1)
        g,h,i=st.columns(3)
        country=g.selectbox("País emisor coincide",[1,0],format_func=lambda v:"Sí" if v==1 else "No")
        distance=h.number_input("Distancia a residencia (km)",min_value=0.0,value=3.0)
        hour=i.slider("Hora del día (0–23)",0,23,14)
        submit=st.form_submit_button("Evaluar transacción",type="primary")
    if submit:
        record=dict(monto_transaccion=monto,canal_transaccion=canal,categoria_comercio=categoria,
          velocidad_tx_10m=velocidad,velocidad_desplazamiento_kmh=kmh,ratio_desviacion_monto=ratio,
          pais_emisor_coincide=country,distancia_residencia_km=distance,hora_dia=hour)
        try:st.session_state["individual_result"]=api_post("/predict",record)
        except (requests.RequestException,ValueError) as exc:st.error(f"No se pudo evaluar: {exc}")
    result=st.session_state.get("individual_result")
    if result:
        decision=result["decision"]
        style="review" if decision=="REVIEW" else "allow"
        st.markdown(f'<div class="decision {style}">Decisión del prototipo: {decision}</div>',unsafe_allow_html=True)
        p,q,r=st.columns(3)
        p.metric("Score del modelo",f"{result['score']:.9f}")
        q.metric("Tiempo interno",f"{result['total_processing_ms']:.2f} ms")
        r.metric("Margen bruto",f"{result['margin']:.5f}" if result.get("margin") is not None else "No disponible")
        st.caption("Score no necesariamente calibrado como probabilidad estadística; decisión LightGBM basada en margen bruto.")
        with st.expander("Detalles técnicos de la respuesta"):
            st.json(result)


def demo_records():
    return [dict(monto_transaccion=m,canal_transaccion=c,categoria_comercio=g,
      velocidad_tx_10m=v,velocidad_desplazamiento_kmh=km,ratio_desviacion_monto=r,
      pais_emisor_coincide=p,distancia_residencia_km=d,hora_dia=h)
      for m,c,g,v,km,r,p,d,h in [
        (65.2,"POS","Retail",0,10,1.1,1,3,14),
        (1200,"E-Commerce","Viajes",6,700,9.5,0,1700,2),
        (120,"ATM","Supermercados",0,8,1,1,1.5,10),
        (540,"App Móvil","Entretenimiento",3,480,6.2,0,1100,1),
      ]]


def simulation_view():
    st.subheader("Simulación por lotes")
    st.caption("Demostración académica de cuatro transacciones en un lote. Esto aún NO es un streaming secuencial.")
    demo=demo_records()
    st.dataframe(pd.DataFrame(demo),use_container_width=True,hide_index=True)
    if st.button("Ejecutar lote",type="primary"):
        try:
            result=api_post("/predict/batch",{"transactions":demo})
            predictions=result.get("predictions",[])
            if len(predictions)!=len(demo):raise ValueError("La cantidad de predicciones no coincide con la de transacciones.")
            frame=pd.concat([pd.DataFrame(demo),pd.DataFrame(predictions)],axis=1)
            frame.insert(0,"ID",[f"TX-{i:04d}" for i in range(1,len(frame)+1)])
            st.session_state["batch_result"]=frame
        except (requests.RequestException,ValueError) as exc:st.error(f"No se pudo simular: {exc}")
    if "batch_result" in st.session_state:
        frame=st.session_state["batch_result"]
        st.dataframe(frame[["ID","monto_transaccion","canal_transaccion","score","decision","total_processing_ms"]],
            use_container_width=True,hide_index=True)
        st.caption("La latencia reportada por la API para cada entrada puede corresponder al procesamiento del lote completo.")


def technical_view(health:dict, final:dict):
    st.subheader("Arquitectura y evidencia")
    st.markdown("**Flujo:** formulario/simulador → FastAPI → preprocesamiento 9→15 variables → ONNX Runtime → decisión ALLOW/REVIEW.")
    st.markdown("**Estado del servicio**")
    st.json(health)
    st.markdown("**Trazabilidad del modelo evaluado**")
    if final:
        st.json({"candidate":final.get("candidate"),"evaluated_at_utc":final.get("evaluated_at_utc"),
          "onnx_python_decision_disagreements":final.get("onnx_python_decision_disagreements"),
          "sha256_artifacts":final.get("sha256_artifacts",{})})
    else: st.info("No se encontró el informe final.")
    st.info("Prototipo académico: no conecta a redes de pago ni autoriza transacciones bancarias reales.")



def pattern_view():
    st.subheader("Inteligencia transaccional")
    st.caption("Descripción del dataset académico completo: 50.000 operaciones, 200 fraudes. Asociaciones observadas, no causas ni estadísticas bancarias reales.")
    canal=patterns_csv('canal_transaccion_summary.csv')
    cats=patterns_csv('categoria_comercio_summary.csv')
    countries=patterns_csv('pais_emisor_coincide_summary.csv')
    hours=patterns_csv('hora_dia_summary.csv')
    numeric=patterns_csv('numeric_by_class.csv')
    if canal.empty or cats.empty:
        st.error("Faltan los reportes del análisis descriptivo. Comprueba reports/fraud_patterns/ y la imagen Docker.")
        return
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Operaciones analizadas',f"{int(canal.operaciones.sum()):,}")
    k2.metric('Casos etiquetados',f"{int(canal.fraudes.sum()):,}")
    k3.metric('Tasa global', f"{100*canal.fraudes.sum()/canal.operaciones.sum():.2f}%")
    k4.metric('Canal con mayor tasa',canal.sort_values('tasa_fraude_pct',ascending=False).iloc[0]['canal_transaccion'])
    st.markdown('#### 01 · Canal y actividad comercial')
    topic=st.radio('Dimensión de análisis',['Canal de transacción','Categoría comercial'],horizontal=True)
    df,col=(canal,'canal_transaccion') if topic=='Canal de transacción' else (cats,'categoria_comercio')
    mode=st.radio('Indicador',['Tasa de fraude (%)','Número de fraudes'],horizontal=True)
    metric='tasa_fraude_pct' if mode.startswith('Tasa') else 'fraudes'
    sorted_df=df.sort_values(metric,ascending=True)
    c1,c2=st.columns([1.6,1])
    with c1:
        fig=px.bar(sorted_df,x=metric,y=col,orientation='h',text=metric,custom_data=['operaciones','fraudes'],
                   color_discrete_sequence=[ACCENT])
        fig.update_traces(texttemplate='%{text:.3f}' if metric=='tasa_fraude_pct' else '%{text:.0f}',
                          textposition='outside',hovertemplate='%{y}<br>Valor: %{x:.3f}<br>Operaciones: %{customdata[0]:,}<br>Fraudes: %{customdata[1]:,}<extra></extra>')
        fig.update_layout(height=340,margin=dict(l=10,r=45,t=15,b=40),
                          xaxis_title='Tasa de fraude (%)' if metric=='tasa_fraude_pct' else 'Fraudes (casos)',
                          yaxis_title='Canal' if col=='canal_transaccion' else 'Categoría comercial',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.dataframe(df[[col,'operaciones','fraudes','tasa_fraude_pct']].sort_values('tasa_fraude_pct',ascending=False)
                    .rename(columns={col:'Grupo','operaciones':'Operaciones','fraudes':'Fraudes','tasa_fraude_pct':'Tasa (%)'})
                    .style.format({'Tasa (%)':'{:.3f}'}),use_container_width=True,hide_index=True)
        top_rate=df.loc[df.tasa_fraude_pct.idxmax()]
        top_count=df.loc[df.fraudes.idxmax()]
        st.info(f"Mayor tasa: **{top_rate[col]}** ({top_rate.tasa_fraude_pct:.3f}%; {int(top_rate.fraudes)} casos / {int(top_rate.operaciones)} operaciones). "
                f"Mayor número de casos: **{top_count[col]}** ({int(top_count.fraudes)}).")
    st.markdown('#### 02 · Contexto geográfico y temporal')
    left,right=st.columns(2)
    with left:
        if not countries.empty:
            country=countries.copy();country['Grupo']=country.pais_emisor_coincide.map({'0':'País no coincide','1':'País coincide',0:'País no coincide',1:'País coincide'}).fillna(country.pais_emisor_coincide.astype(str))
            fig=px.bar(country,x='Grupo',y='tasa_fraude_pct',text='tasa_fraude_pct',color_discrete_sequence=[AMBER])
            fig.update_traces(texttemplate='%{text:.2f}%',textposition='outside')
            fig.update_layout(title='Coincidencia del país emisor',xaxis_title='Contexto de la operación',yaxis_title='Fraudes / operaciones (%)',height=360,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
            st.caption('Tasa dentro de cada grupo; no indica que una compra internacional sea fraudulenta.')
    with right:
        if not hours.empty:
            hh=hours.sort_values('hora_dia')
            fig=go.Figure()
            fig.add_trace(go.Bar(x=hh.hora_dia,y=hh.fraudes,name='Fraudes (casos)',marker_color=ACCENT))
            fig.add_trace(go.Scatter(x=hh.hora_dia,y=hh.tasa_fraude_pct,name='Tasa (%)',yaxis='y2',mode='lines+markers',line_color=AMBER))
            fig.update_layout(title='Fraude por hora del día',xaxis_title='Hora (0–23)',yaxis_title='Fraudes (casos)',
                yaxis2=dict(title='Tasa de fraude (%)',overlaying='y',side='right'),legend=dict(orientation='h',y=-.3),height=360)
            st.plotly_chart(fig,use_container_width=True)
            st.caption('La tasa horaria depende del volumen de operaciones; consulte el detalle por hora antes de interpretar picos.')
    st.markdown('#### 03 · Diferencias de comportamiento entre clases')
    if not numeric.empty:
        labels={'monto_transaccion':'Monto (USD)','velocidad_tx_10m':'Transacciones en 10 minutos',
                'velocidad_desplazamiento_kmh':'Desplazamiento (km/h)','ratio_desviacion_monto':'Desviación de monto (ratio)',
                'distancia_residencia_km':'Distancia a residencia (km)'}
        selection=st.selectbox('Variable para explorar',list(labels),format_func=lambda x:labels[x])
        dd=numeric[numeric.variable==selection].copy()
        dd['Clase']=dd.clase.map({0:'Legítima',1:'Fraude'})
        fig=px.bar(dd,x='Clase',y='mediana',text='mediana',color='Clase',
            color_discrete_map={'Legítima':ACCENT,'Fraude':AMBER})
        fig.update_traces(texttemplate='%{text:.2f}',textposition='outside')
        fig.update_layout(showlegend=False,yaxis_title=f'Mediana · {labels[selection]}',xaxis_title='Etiqueta real',height=350)
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Medianas por clase; diferencias descriptivas. No miden importancia predictiva ni causalidad.')
        st.dataframe(dd[['Clase','n','p25','mediana','p75','promedio']].rename(columns={'n':'Operaciones','p25':'P25','mediana':'Mediana','p75':'P75','promedio':'Promedio'})
             .style.format({'P25':'{:.2f}','Mediana':'{:.2f}','P75':'{:.2f}','Promedio':'{:.2f}'}),hide_index=True,use_container_width=True)
        q=patterns_csv(selection+'_quintiles.csv')
        if not q.empty:
            with st.expander('Tasa por intervalos observados'):
                st.dataframe(q[['intervalo','operaciones','fraudes','tasa_fraude_pct']]
                    .rename(columns={'intervalo':'Intervalo','operaciones':'Operaciones','fraudes':'Fraudes','tasa_fraude_pct':'Tasa (%)'})
                    .style.format({'Tasa (%)':'{:.3f}'}),hide_index=True,use_container_width=True)
                st.caption('Los límites de intervalos son descriptivos, NO umbrales operativos validados.')
    with st.expander('Metodología, alcance y limitaciones'):
        st.markdown('''- Tasa = fraudes etiquetados / operaciones del grupo × 100; el denominador cambia entre grupos.
- Los archivos agregados proceden del conjunto académico completo (incluye el holdout ya evaluado) y solo sirven para **descripción**, jamás para ajustar el modelo congelado.
- Un patrón asociado no demuestra causalidad ni desempeño en datos bancarios de producción.
- Las clases muestran separabilidad elevada y pocos fraudes (200); es necesario discutir la posible naturaleza artificial de los patrones.''')


def csv_view(health: dict):
    st.subheader('Análisis de transacciones CSV')
    st.caption('Evaluación de archivos nuevos por FastAPI y ONNX. Hasta 5.000 filas por archivo y lotes de 250; sin entrenamiento ni actualización del modelo.')
    st.warning('Utiliza datos académicos o anonimizados. No cargues información bancaria personal ni números reales de tarjetas.')
    if health.get('status') != 'ok':
        st.error('API desconectada. Inicia FastAPI antes de evaluar.');return
    up=st.file_uploader('Archivo CSV (UTF-8)',type=['csv'],key='csv_upload')
    if up is None: return
    if up.size>4*1024*1024:
        st.error('Archivo demasiado grande: máximo 4 MB.');return
    try: df=pd.read_csv(up,encoding='utf-8-sig')
    except Exception as exc: st.error(f'No se pudo abrir el CSV: {exc}');return
    if not 1 <= len(df) <= 5000:
        st.error('El archivo debe contener entre 1 y 5.000 filas.');return
    validated,issues=validate_transactions(df)
    if issues:
        st.error('Hay errores de esquema. Corrige el archivo antes de procesarlo.')
        for issue in issues:st.write('• '+issue)
        return
    st.success(f'{len(validated):,} registros válidos · {len(COLUMNS)} variables requeridas.' + (' · Etiquetas disponibles' if 'es_fraude' in validated else ' · Sin etiquetas'))
    st.dataframe(validated.head(10),use_container_width=True,hide_index=True)
    fingerprint=f'{up.name}:{up.size}:{hash(up.getvalue())}:{health.get("model")}'
    if st.button('Ejecutar evaluación del archivo',type='primary'):
        results=[];total=0.;failed=None
        bar=st.progress(0,text='Ejecutando inferencia por lotes')
        for offset in range(0,len(validated),250):
            block=validated.iloc[offset:offset+250]
            payload={'transactions':block[COLUMNS].to_dict(orient='records')}
            try:
                t0=time.perf_counter();reply=api_post('/predict/batch',payload,timeout=60)
                total+=(time.perf_counter()-t0)*1000
                preds=reply.get('predictions',[])
                if len(preds)!=len(block):raise ValueError('La API devolvió una cantidad distinta de resultados')
                results.extend(preds)
            except (requests.RequestException,ValueError,KeyError) as exc:
                failed=f'Lote desde fila {offset+1}: {exc}';break
            bar.progress(min(1.,(offset+len(block))/len(validated)))
        bar.empty()
        if failed:st.error(f'Procesamiento interrumpido. No se publican resultados incompletos. {failed}')
        else:
            pred=pd.DataFrame(results)
            output=validated.reset_index(drop=True).copy()
            for col in ['score','margin','margin_threshold','decision','is_fraud']:
                if col in pred:output[col]=pred[col].values
            st.session_state['csv_output']=(fingerprint,output,round(total,2),health.get('model'))
    entry=st.session_state.get('csv_output')
    if entry and entry[0]==fingerprint:
        _,out,elapsed,model=entry
        m1,m2,m3=st.columns(3)
        m1.metric('Operaciones',len(out));m2.metric('REVIEW',int((out.decision=='REVIEW').sum()));m3.metric('HTTP total (ms)',f'{elapsed:,.1f}')
        st.caption(f'Motor: {model}. Tiempo total de llamadas HTTP de esta ejecución, no latencia individual ni prueba de streaming.')
        st.dataframe(out,use_container_width=True,hide_index=True)
        if 'es_fraude' in out:
            mm=inference_metrics(out)
            a,b,c,d=st.columns(4)
            a.metric('Recall',f"{mm['recall']:.2%}");b.metric('Precision',f"{mm['precision']:.2%}")
            c.metric('F1',f"{mm['f1']:.2%}");d.metric('FPR',f"{mm['fpr']:.2%}")
            st.plotly_chart(conf_matrix_chart(mm['tn'],mm['fp'],mm['fn'],mm['tp'],f'Matriz de confusión · CSV etiquetado (n={len(out):,})'),use_container_width=True)
            st.caption('Evaluación exploratoria de este archivo, independiente del holdout académico. Métricas de clases ausentes se definen como 0; examine soporte por clase.')
        else:st.info('Sin columna es_fraude: no se pueden calcular Recall, F1 ni Precision. Las predicciones no son verdad terreno.')
        st.download_button('Descargar predicciones CSV',out.to_csv(index=False).encode('utf-8-sig'),file_name='fraudnexus_predicciones.csv',mime='text/csv')


def switch_view(health: dict):
    st.subheader('Switch transaccional · simulación secuencial')
    st.caption('Eventos individuales enviados a POST /predict mediante HTTP; no existe conexión a un switch real ni procesamiento distribuido.')
    if health.get('status') != 'ok':st.error('API no disponible');return
    x,y=st.columns([1,2])
    qty=x.slider('Operaciones por ejecución',2,100,20,2)
    mix=y.slider('Porcentaje de escenarios sospechosos (sintéticos)',0,100,25,5)
    if st.button('Iniciar simulación',type='primary'):
        normals=demo_records()[0::2];suspicious=demo_records()[1::2]
        import random
        rng=random.Random(20261022+qty+mix)
        ledger=[];bar=st.progress(0)
        for i in range(qty):
            kind='sospechosa' if rng.random()<mix/100 else 'habitual'
            tx=(suspicious if kind=='sospechosa' else normals)[i%2].copy()
            start=time.perf_counter();ts=datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            try:
                resp=api_post('/predict',tx,timeout=15)
                latency=(time.perf_counter()-start)*1000
                ledger.append({'ID':f'TX-{i+1:05d}','UTC':ts,'Escenario':kind,'Canal':tx['canal_transaccion'],
                    'Monto (USD)':tx['monto_transaccion'],'Decisión':resp.get('decision'),'Score':resp.get('score'),
                    'Margen':resp.get('margin'),'HTTP (ms)':round(latency,3),'Interno (ms)':resp.get('total_processing_ms')})
            except (requests.RequestException,ValueError) as exc:
                ledger.append({'ID':f'TX-{i+1:05d}','UTC':ts,'Escenario':kind,'Canal':tx['canal_transaccion'],
                    'Monto (USD)':tx['monto_transaccion'],'Decisión':'ERROR','HTTP (ms)':round((time.perf_counter()-start)*1000,3),'Detalle':str(exc)[:150]})
            bar.progress((i+1)/qty)
        bar.empty()
        st.session_state['switch_ledger']=pd.DataFrame(ledger)
    ledger=st.session_state.get('switch_ledger')
    if ledger is not None and not ledger.empty:
        a,b,c,d=st.columns(4)
        a.metric('Eventos enviados',len(ledger));b.metric('ALLOW',int((ledger['Decisión']=='ALLOW').sum()))
        c.metric('REVIEW',int((ledger['Decisión']=='REVIEW').sum()))
        d.metric('Errores',int((ledger['Decisión']=='ERROR').sum()))
        st.dataframe(ledger,use_container_width=True,hide_index=True)
        good=ledger[ledger['Decisión']!='ERROR']
        if len(good):
            fig=px.line(good,x='ID',y='HTTP (ms)',markers=True,title='Latencia HTTP por evento simulado')
            fig.add_hline(y=100,line_dash='dash',line_color='#B45B4E',annotation_text='Referencia 100 ms')
            fig.update_layout(xaxis_title='Evento secuencial',yaxis_title='Tiempo HTTP (ms)',height=330)
            st.plotly_chart(fig,use_container_width=True)
            st.caption(f"p95 de la sesión: {good['HTTP (ms)'].quantile(.95):.2f} ms · {len(good)} eventos válidos. "
                'Los escenarios no son operaciones observadas ni etiquetas verdaderas; no calcular Recall con estos ejemplos.')
        st.download_button('Exportar bitácora CSV',ledger.to_csv(index=False).encode('utf-8-sig'),file_name='fraudnexus_switch_simulado.csv',mime='text/csv')


def importance_view():
    st.markdown('#### Interpretabilidad · importancia por permutación')
    p=REPORTS/'feature_importance_development.csv'
    if not p.exists():
        st.info('Aún no se ha ejecutado el estudio opcional de importancia de variables. No se inventan rankings.')
        return
    imp=pd.read_csv(p)
    if not {'variable','mean_auc_pr_drop','std_auc_pr_drop'}.issubset(imp.columns):
        st.warning('El reporte de importancia no tiene el esquema requerido.');return
    fig=px.bar(imp.sort_values('mean_auc_pr_drop'),x='mean_auc_pr_drop',y='variable',orientation='h',
            error_x='std_auc_pr_drop',title='Caída de AUC-PR al permutar cada variable · datos de desarrollo',
            color_discrete_sequence=[ACCENT])
    fig.update_layout(xaxis_title='Caída media de AUC-PR (unidades de 0 a 1)',yaxis_title='Variable original',height=430)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Importancia global asociativa, NO causal; depende del conjunto y de la correlación entre características. Evaluación sobre desarrollo, no holdout.')
    st.dataframe(imp,use_container_width=True,hide_index=True)

def main():
    st.set_page_config(page_title="FraudNexus AI | Monitor de riesgo",page_icon="🛡️",layout="wide")
    apply_style()
    st.markdown('<div class="hero"><strong>FraudNexus AI</strong><small>Plataforma académica de evaluación de riesgo transaccional · ONNX Runtime</small></div>',unsafe_allow_html=True)
    eda=read_json("eda_summary.json")
    initial=read_json("model_comparison.json")
    resampling=read_json("resampling_comparison.json")
    final=read_json("lightgbm_final_holdout.json")
    latency=read_json("lightgbm_latency_benchmark.json")
    health=get_health()
    nav=st.sidebar.radio('Navegación',['Centro de control','Inteligencia transaccional','Evaluación individual',
        'Análisis de CSV','Switch simulado','Laboratorio de modelos','Arquitectura y auditoría'])
    st.sidebar.caption('Modelo operativo: '+str(health.get('model','API desconectada')))
    if nav=='Centro de control':center_view(eda,final,latency,health)
    elif nav=='Inteligencia transaccional':pattern_view()
    elif nav=='Evaluación individual':individual_view()
    elif nav=='Análisis de CSV':csv_view(health)
    elif nav=='Switch simulado':switch_view(health)
    elif nav=='Laboratorio de modelos':
        lab_view(initial,resampling,final)
        importance_view()
    else:technical_view(health,final)
    st.caption("FraudNexus AI · Prototipo de investigación. Los resultados académicos no garantizan desempeño bancario real.")


if __name__=="__main__":
    main()
