"""FraudNexus AI | Evidence-driven Streamlit dashboard (phase 1)."""
from __future__ import annotations

import json
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
        st.success(f"Servicio operativo · Motor conectado: {friendly}")
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


def main():
    st.set_page_config(page_title="FraudNexus AI | Monitor de riesgo",page_icon="🏦",layout="wide")
    apply_style()
    st.markdown('<div class="hero"><strong>FraudNexus AI</strong><small>Plataforma académica de evaluación de riesgo transaccional · ONNX Runtime</small></div>',unsafe_allow_html=True)
    eda=read_json("eda_summary.json")
    initial=read_json("model_comparison.json")
    resampling=read_json("resampling_comparison.json")
    final=read_json("lightgbm_final_holdout.json")
    latency=read_json("lightgbm_latency_benchmark.json")
    health=get_health()
    tabs=st.tabs(["Centro de control","Evaluación individual","Simulación por lotes","Laboratorio de modelos","Arquitectura y auditoría"])
    with tabs[0]:center_view(eda,final,latency,health)
    with tabs[1]:individual_view()
    with tabs[2]:simulation_view()
    with tabs[3]:lab_view(initial,resampling,final)
    with tabs[4]:technical_view(health,final)
    st.caption("FraudNexus AI · Prototipo de investigación. Los resultados académicos no garantizan desempeño bancario real.")


if __name__=="__main__":
    main()
