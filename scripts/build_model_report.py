"""Build reproducible development-only model comparison charts and tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures" / "model_selection"
LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "isolation_forest": "Isolation Forest"}
STRATEGIES = {"weights": "Pesos", "smote": "SMOTE", "adasyn": "ADASYN"}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    original = load(REPORTS / "model_comparison.json")
    extended = load(REPORTS / "resampling_comparison.json")
    if extended.get("quick") or extended.get("development_rows") != 40000:
        raise ValueError("Expected the complete 40,000-row development report, not --quick.")
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = []
    for key, result in extended["configurations"].items():
        if result.get("status") != "complete":
            continue
        model, strategy = key.split("__", 1)
        cm = result["confusion_matrix"]
        records.append({
            "configuration": key,
            "model": LABELS.get(model, model),
            "strategy": STRATEGIES.get(strategy, strategy),
            "recall": result["recall"],
            "precision": result["precision"],
            "f1": result["f1"],
            "auc_pr": result["mean_auc_pr"],
            "fpr": result["fpr"],
            "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
        })
    df = pd.DataFrame(records).sort_values(["recall", "f1", "fn"], ascending=[False, False, True])
    df.to_csv(REPORTS / "model_selection_table.csv", index=False, encoding="utf-8-sig")

    colors = {"Pesos": "#244B69", "SMOTE": "#4A9B91", "ADASYN": "#D8A75C"}
    models = ["Random Forest", "XGBoost", "LightGBM"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(models))
    for j, strategy in enumerate(["Pesos", "SMOTE", "ADASYN"]):
        vals = [float(df.loc[(df.model == m) & (df.strategy == strategy), "recall"].iloc[0]) * 100 for m in models]
        bars = ax.bar(x + (j-1) * .25, vals, width=.23, label=strategy, color=colors[strategy])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.axhline(90, linestyle="--", color="#B95555", linewidth=1.3, label="Objetivo: >90%")
    ax.set_xticks(x, models)
    ax.set_ylim(80, 103)
    ax.set_ylabel("Recall de fraudes reales (%)")
    ax.set_xlabel("Algoritmo de clasificación")
    ax.set_title("Recall por algoritmo y estrategia de desbalance — validación cruzada (5 folds)")
    ax.text(0.01, -0.23, "Datos: desarrollo (40.000 transacciones) | Umbrales calibrados dentro de cada fold externo", transform=ax.transAxes, fontsize=8, color="#526779")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "recall_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    vals = df.sort_values("f1").copy()
    labels = [f"{m} · {s}" for m, s in zip(vals.model, vals.strategy)]
    ax.barh(labels, vals.f1 * 100, color=[colors[s] for s in vals.strategy])
    ax.set_xlim(88, 101)
    ax.set_xlabel("F1-Score (%)")
    ax.set_ylabel("Modelo y estrategia de balanceo")
    ax.set_title("F1-Score por configuración — validación cruzada (5 folds)")
    ax.text(0.01, -0.13, "Datos: desarrollo (40.000 transacciones) | Valores por configuración agregados de cinco folds", transform=ax.transAxes, fontsize=8, color="#526779")
    fig.tight_layout()
    fig.savefig(FIGURES / "f1_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    contenders = ["random_forest__weights", "lightgbm__weights"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, key in zip(axes, contenders):
        cm = extended["configurations"][key]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=40000)
        for (r, c), val in np.ndenumerate(matrix):
            ax.text(c, r, f"{val:,}", ha="center", va="center", fontsize=12, color="white" if val > 20000 else "#17354E", fontweight="bold")
        ax.set_xticks([0, 1], ["Legítima", "Fraude"])
        ax.set_yticks([0, 1], ["Legítima", "Fraude"])
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(f"{LABELS[key.split('__')[0]]} + pesos")
        ax.text(0.5, -0.23, "Celdas = número de transacciones | TN / FP / FN / TP", transform=ax.transAxes, ha="center", fontsize=7, color="#526779")
    fig.suptitle("Matrices de confusión — 5 folds de desarrollo (40.000 transacciones)")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scope": "development_only_not_external_holdout",
        "initial_selection": original.get("selected_model"),
        "provisional_selection": extended.get("selected_configuration_provisional"),
        "protocol": extended.get("protocol"),
        "rows": len(df),
        "caveats": [
            "Initial and expanded comparisons use different threshold-selection protocols: don't rank across them directly.",
            "SMOTE/ADASYN in this experiment were applied after one-hot encoding.",
            "Perfect AUC-PR/FPR values require investigation and must not be described as real-world performance.",
            "Final model and threshold remain unfrozen; external holdout has not been evaluated."
        ]
    }
    (REPORTS / "model_selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(df[["model", "strategy", "recall", "f1", "auc_pr", "fp", "fn"]].to_string(index=False))
    print(f"\nSaved charts: {FIGURES}")
    print(f"Saved table: {REPORTS / 'model_selection_table.csv'}")
    print("WARNING: All figures are DEVELOPMENT-ONLY, not external test results.")


if __name__ == "__main__":
    main()
