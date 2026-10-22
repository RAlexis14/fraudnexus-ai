"""FraudNexus AI Streamlit dashboard."""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

EDA_SUMMARY_PATH = Path("reports/eda_summary.json")
MODEL_COMPARISON_PATH = Path("reports/model_comparison.json")
ONNX_VALIDATION_PATH = Path("reports/onnx_validation.json")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file if it exists."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_api_health() -> dict[str, Any]:
    """Check API health."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "down", "detail": str(exc)}


def predict_transaction(payload: dict[str, Any]) -> dict[str, Any]:
    """Send a single transaction to the API."""
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def predict_batch(payload: list[dict[str, Any]]) -> dict[str, Any]:
    """Send a transaction batch using the API request schema."""
    response = requests.post(
        f"{API_BASE_URL}/predict/batch",
        json={"transactions": payload},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def build_demo_transactions() -> list[dict[str, Any]]:
    """Return sample transactions for the simulator."""
    return [
        {
            "monto_transaccion": 65.20,
            "canal_transaccion": "POS",
            "categoria_comercio": "Retail",
            "velocidad_tx_10m": 0,
            "velocidad_desplazamiento_kmh": 10.0,
            "ratio_desviacion_monto": 1.10,
            "pais_emisor_coincide": 1,
            "distancia_residencia_km": 3.0,
            "hora_dia": 14,
        },
        {
            "monto_transaccion": 980.00,
            "canal_transaccion": "E-Commerce",
            "categoria_comercio": "Viajes",
            "velocidad_tx_10m": 5,
            "velocidad_desplazamiento_kmh": 620.0,
            "ratio_desviacion_monto": 8.50,
            "pais_emisor_coincide": 0,
            "distancia_residencia_km": 1500.0,
            "hora_dia": 3,
        },
        {
            "monto_transaccion": 120.00,
            "canal_transaccion": "ATM",
            "categoria_comercio": "Supermercados",
            "velocidad_tx_10m": 0,
            "velocidad_desplazamiento_kmh": 8.0,
            "ratio_desviacion_monto": 1.00,
            "pais_emisor_coincide": 1,
            "distancia_residencia_km": 1.5,
            "hora_dia": 10,
        },
        {
            "monto_transaccion": 540.00,
            "canal_transaccion": "App Móvil",
            "categoria_comercio": "Entretenimiento",
            "velocidad_tx_10m": 3,
            "velocidad_desplazamiento_kmh": 480.0,
            "ratio_desviacion_monto": 6.20,
            "pais_emisor_coincide": 0,
            "distancia_residencia_km": 1100.0,
            "hora_dia": 1,
        },
    ]


def render_css() -> None:
    """Render custom CSS."""
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f6f9fc 0%, #eef4f8 100%);
            }

            .main-title {
                font-size: 2.2rem;
                font-weight: 700;
                color: #16324f;
                margin-bottom: 0.2rem;
            }

            .subtitle {
                color: #4b647d;
                font-size: 1rem;
                margin-bottom: 1.2rem;
            }

            .section-card {
                background: white;
                padding: 1rem 1.2rem;
                border-radius: 16px;
                box-shadow: 0 6px 18px rgba(15, 40, 70, 0.08);
                margin-bottom: 1rem;
            }

            .status-ok {
                color: #0f9d58;
                font-weight: 700;
            }

            .status-bad {
                color: #c62828;
                font-weight: 700;
            }

            div[data-testid="stMetric"] {
                background: white;
                border-radius: 14px;
                padding: 10px;
                box-shadow: 0 4px 12px rgba(15, 40, 70, 0.08);
            }

            .decision-allow {
                color: #0f9d58;
                font-weight: 700;
                font-size: 1.1rem;
            }

            .decision-review {
                color: #e67e22;
                font-weight: 700;
                font-size: 1.1rem;
            }

            .decision-alert {
                color: #c62828;
                font-weight: 700;
                font-size: 1.1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render page header."""
    st.markdown(
        '<div class="main-title">FraudNexus AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Sistema inteligente de detección de fraude bancario en tiempo real</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="FraudNexus AI",
        page_icon="🏦",
        layout="wide",
    )

    render_css()
    render_header()

    eda_summary = load_json(EDA_SUMMARY_PATH)
    model_comparison = load_json(MODEL_COMPARISON_PATH)
    onnx_validation = load_json(ONNX_VALIDATION_PATH)
    health = get_api_health()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Resumen",
            "🔎 Predicción individual",
            "🔁 Simulación",
            "ℹ️ Modelo y API",
        ]
    )

    with tab1:
        st.subheader("Resumen del sistema")

        col1, col2, col3, col4 = st.columns(4)

        total_records = eda_summary.get("total_records", 0)
        frauds = eda_summary.get("fraudulent", 0)
        prevalence = eda_summary.get("fraud_prevalence", 0)

        col1.metric("Registros", f"{total_records:,}")
        col2.metric("Fraudes", f"{frauds:,}")
        col3.metric("Prevalencia", f"{prevalence:.4%}")
        col4.metric(
            "Estado API",
            "UP" if health.get("status") == "ok" else "DOWN",
        )

        st.markdown("---")

        if model_comparison:
            models = model_comparison.get("models", {})
            if models:
                rows = []
                for model_name, metrics in models.items():
                    rows.append(
                        {
                            "Modelo": model_name,
                            "Recall": metrics.get("recall", 0),
                            "Precision": metrics.get("precision", 0),
                            "F1": metrics.get("f1", 0),
                            "AUC-PR": metrics.get("average_precision_auc_pr", 0),
                            "FPR": metrics.get("false_positive_rate", 0),
                        }
                    )

                df_models = pd.DataFrame(rows)

                st.markdown("### Comparación de modelos")
                st.dataframe(df_models, use_container_width=True)

                fig = px.bar(
                    df_models,
                    x="Modelo",
                    y=["Recall", "Precision", "F1", "AUC-PR"],
                    barmode="group",
                    title="Comparación de métricas",
                )
                st.plotly_chart(fig, use_container_width=True)

        missing = eda_summary.get("missing_values", {})
        if missing:
            df_missing = pd.DataFrame(
                {
                    "Variable": list(missing.keys()),
                    "Faltantes": list(missing.values()),
                }
            )

            st.markdown("### Valores faltantes del dataset oficial")
            st.dataframe(df_missing, use_container_width=True)

    with tab2:
        st.subheader("Predicción individual")

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)

            monto = col1.number_input(
                "Monto de transacción",
                min_value=0.0,
                value=65.20,
                step=1.0,
            )
            canal = col2.selectbox(
                "Canal",
                ["POS", "E-Commerce", "ATM", "App Móvil"],
            )
            categoria = col3.selectbox(
                "Categoría de comercio",
                ["Supermercados", "Retail", "Entretenimiento", "Viajes"],
            )

            col4, col5, col6 = st.columns(3)

            velocidad_tx = col4.number_input(
                "Velocidad tx últimos 10 min",
                min_value=0,
                value=0,
                step=1,
            )
            velocidad_kmh = col5.number_input(
                "Velocidad de desplazamiento (km/h)",
                min_value=0.0,
                value=10.0,
                step=1.0,
            )
            ratio_monto = col6.number_input(
                "Ratio desviación monto",
                min_value=0.0,
                value=1.1,
                step=0.1,
            )

            col7, col8, col9 = st.columns(3)

            pais = col7.selectbox(
                "País emisor coincide",
                [1, 0],
                format_func=lambda x: "Sí" if x == 1 else "No",
            )
            distancia = col8.number_input(
                "Distancia a residencia (km)",
                min_value=0.0,
                value=3.0,
                step=1.0,
            )
            hora = col9.slider(
                "Hora del día",
                min_value=0,
                max_value=23,
                value=14,
            )

            submitted = st.form_submit_button("Analizar transacción")

        if submitted:
            payload = {
                "monto_transaccion": monto,
                "canal_transaccion": canal,
                "categoria_comercio": categoria,
                "velocidad_tx_10m": velocidad_tx,
                "velocidad_desplazamiento_kmh": velocidad_kmh,
                "ratio_desviacion_monto": ratio_monto,
                "pais_emisor_coincide": pais,
                "distancia_residencia_km": distancia,
                "hora_dia": hora,
            }

            try:
                result = predict_transaction(payload)

                st.success("Predicción completada correctamente")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Score", f'{result["score"]:.6f}')
                col2.metric("Umbral", f'{result["threshold"]:.6f}')
                col3.metric(
                    "Fraude",
                    "Sí" if result["is_fraud"] else "No",
                )
                col4.metric(
                    "Latencia total (ms)",
                    f'{result["total_processing_ms"]:.4f}',
                )

                decision = result["decision"]
                css_class = (
                    "decision-allow"
                    if decision == "ALLOW"
                    else "decision-review"
                )

                st.markdown(
                    f'<p class="{css_class}">Decisión: {decision}</p>',
                    unsafe_allow_html=True,
                )

                latency_df = pd.DataFrame(
                    {
                        "Etapa": [
                            "Preprocesamiento",
                            "Inferencia ONNX",
                            "Total",
                        ],
                        "Milisegundos": [
                            result["preprocessing_ms"],
                            result["onnx_inference_ms"],
                            result["total_processing_ms"],
                        ],
                    }
                )

                fig_latency = px.bar(
                    latency_df,
                    x="Etapa",
                    y="Milisegundos",
                    title="Desglose de latencia",
                )
                st.plotly_chart(fig_latency, use_container_width=True)

                st.json(result)

            except Exception as exc:
                st.error(f"Error al consultar la API: {exc}")

    with tab3:
        st.subheader("Simulación de transacciones")

        st.write(
            "Este módulo permite probar varias transacciones y observar "
            "cómo responde el sistema en un flujo tipo switch bancario."
        )

        demo_transactions = build_demo_transactions()
        df_demo = pd.DataFrame(demo_transactions)
        st.dataframe(df_demo, use_container_width=True)

        if st.button("Ejecutar simulación"):
            try:
                batch_result = predict_batch(demo_transactions)

                predictions = batch_result.get("predictions", [])
                df_predictions = pd.DataFrame(predictions)

                if not df_predictions.empty:
                    df_predictions.insert(
                        0,
                        "transaccion_id",
                        [f"TX-{i:04d}" for i in range(1, len(df_predictions) + 1)],
                )

                df_predictions["monto_transaccion"] = [
                    item["monto_transaccion"] for item in demo_transactions
                ]

                df_predictions["canal_transaccion"] = [
                    item["canal_transaccion"] for item in demo_transactions
                ]

                st.success("Simulación ejecutada correctamente")
                st.dataframe(df_predictions, use_container_width=True)

                if not df_predictions.empty:
                    fig_scores = px.bar(
                        df_predictions,
                        x=df_predictions.index.astype(str),
                        y="score",
                        color="decision",
                        title="Score por transacción simulada",
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)

            except Exception as exc:
                st.error(f"Error en simulación: {exc}")

    with tab4:
        st.subheader("Modelo y API")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Estado del servicio")
            if health.get("status") == "ok":
                st.markdown(
                    '<p class="status-ok">API disponible</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="status-bad">API no disponible</p>',
                    unsafe_allow_html=True,
                )
            st.json(health)

        with col2:
            st.markdown("### Validación ONNX")
            if onnx_validation:
                st.json(onnx_validation)
            else:
                st.info("No se encontró el reporte ONNX.")

        st.markdown("### Nota técnica")
        st.info(
            "La arquitectura actual utiliza preprocesamiento en Python y "
            "clasificación en ONNX Runtime. La API recibe las 9 variables "
            "originales, las transforma a 15 características y ejecuta "
            "el Random Forest exportado."
        )


if __name__ == "__main__":
    main()