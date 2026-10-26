
"""Exploratory data analysis for FraudNexus AI."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_FEATURES = [
    "monto_transaccion",
    "velocidad_tx_10m",
    "velocidad_desplazamiento_kmh",
    "ratio_desviacion_monto",
    "pais_emisor_coincide",
    "distancia_residencia_km",
    "hora_dia",
]


def run_eda(
    input_path: str = "data/raw/transacciones_fraude_bancario.csv",
    output_dir: str = "reports/figures",
) -> dict:
    """Analyze class distribution and feature characteristics."""

    df = pd.read_csv(input_path)
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    target = df["es_fraude"]
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    prevalence = float(target.mean())

    summary = {
        "total_records": int(len(df)),
        "legitimate": int(counts[0]),
        "fraudulent": int(counts[1]),
        "fraud_prevalence": prevalence,
        "missing_values": {
            col: int(value)
            for col, value in df.isna().sum().items()
        },
    }

    # Class distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Legitimate", "Fraud"], counts.values)
    ax.set_title("Transaction class distribution")
    ax.set_ylabel("Number of transactions")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Amount comparison with outlier-resistant visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    legit = df.loc[target == 0, "monto_transaccion"]
    fraud = df.loc[target == 1, "monto_transaccion"]

    ax.boxplot(
        [legit, fraud],
        tick_labels=["Legitimate", "Fraud"],
        showfliers=False,
    )
    ax.set_title("Transaction amounts by class")
    ax.set_ylabel("Amount (USD)")
    fig.tight_layout()
    fig.savefig(figures_dir / "amount_by_class.png", dpi=160)
    plt.close(fig)

    # Fraud prevalence by transaction channel
    channel_stats = (
        df.groupby("canal_transaccion")["es_fraude"]
        .agg(["size", "sum", "mean"])
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(channel_stats.index, channel_stats["mean"] * 100)
    ax.set_title("Fraud prevalence by channel")
    ax.set_ylabel("Fraud prevalence (%)")
    ax.set_xlabel("Transaction channel")
    fig.tight_layout()
    fig.savefig(figures_dir / "fraud_by_channel.png", dpi=160)
    plt.close(fig)

    # Correlation matrix for numeric predictors
    corr = df[NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Numeric feature correlation")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "correlation_matrix.png", dpi=160)
    plt.close(fig)

    # Numeric feature summary by class
    feature_summary = (
        df.groupby("es_fraude")[NUMERIC_FEATURES]
        .agg(["median", "mean"])
        .round(3)
    )

    summary_path = figures_dir.parent / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== FraudNexus AI | EDA ===")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {counts[1]:,}")
    print(f"Fraud prevalence: {prevalence:.4%}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nFraud statistics by channel:")
    print(channel_stats.to_string())

    print("\nNumeric features by class:")
    print(feature_summary.to_string())

    print(f"\nFigures saved in: {figures_dir}")
    print(f"Summary saved in: {summary_path}")

    return summary


if __name__ == "__main__":
    run_eda()
