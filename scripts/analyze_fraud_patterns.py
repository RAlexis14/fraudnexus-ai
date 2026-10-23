"""Reproducible descriptive fraud analysis of the official academic CSV.

Descriptive associations are not causal explanations or model importance.
Does not fit a model or modify its artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path('data/raw/transacciones_fraude_bancario.csv')
OUT = Path('reports/fraud_patterns')
FIG = OUT / 'figures'
TARGET = 'es_fraude'
CATEGORIES = ['canal_transaccion', 'categoria_comercio', 'pais_emisor_coincide', 'hora_dia']
NUMERIC = ['monto_transaccion','velocidad_tx_10m','velocidad_desplazamiento_kmh','ratio_desviacion_monto','distancia_residencia_km']


def grouped(df: pd.DataFrame, field: str) -> pd.DataFrame:
    summary = df.groupby(field, dropna=False, observed=True)[TARGET].agg(
        operaciones='size', fraudes='sum'
    ).reset_index()
    summary['legitimas'] = summary['operaciones'] - summary['fraudes']
    summary['tasa_fraude_pct'] = 100 * summary['fraudes'] / summary['operaciones']
    summary['participacion_fraudes_pct'] = 100 * summary['fraudes'] / max(int(df[TARGET].sum()), 1)
    return summary.sort_values(['tasa_fraude_pct','operaciones'], ascending=[False,False]).reset_index(drop=True)


def plot_group(table: pd.DataFrame, field: str, label: str) -> None:
    table = table.sort_values('tasa_fraude_pct', ascending=True)
    labels = table[field].astype(str)
    fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.6 + 1.5)))
    ax.barh(labels, table['tasa_fraude_pct'], color='#246C80')
    ax.set_title(f'Tasa de fraude por {label} — dataset oficial')
    ax.set_xlabel('Transacciones fraudulentas / transacciones del grupo (%)')
    ax.set_ylabel(label)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    xmax = max(float(table['tasa_fraude_pct'].max()), 0.1)
    ax.set_xlim(0, xmax * 1.32)
    for i, (_, r) in enumerate(table.iterrows()):
        ax.text(float(r['tasa_fraude_pct']) + xmax*0.025, i,
                f"{r['tasa_fraude_pct']:.2f}%  ({int(r['fraudes'])}/{int(r['operaciones'])})",
                va='center', fontsize=9)
    fig.text(0.01, 0.01, 'Asociación descriptiva; no demuestra causalidad. Fuente: dataset oficial académico.', fontsize=8)
    fig.tight_layout(rect=(0,0.035,1,1))
    fig.savefig(FIG / f'{field}_tasa.png', dpi=180)
    plt.close(fig)


def numeric_ranges(df: pd.DataFrame, field: str) -> pd.DataFrame:
    # Quantiles derived from full dataset, not the fraud labels.
    series = pd.to_numeric(df[field], errors='coerce')
    try:
        bins = pd.qcut(series, q=5, duplicates='drop')
    except ValueError:
        bins = pd.Series(pd.Categorical(['Sin variación'] * len(df)), index=df.index)
    tmp = pd.DataFrame({'intervalo': bins.astype(str), TARGET: df[TARGET]})
    return grouped(tmp, 'intervalo')


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f'Archivo no encontrado: {DATA}')
    df = pd.read_csv(DATA, encoding='utf-8-sig')
    required = set(CATEGORIES + NUMERIC + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Faltan columnas del diccionario oficial: {missing}')
    if df[TARGET].isna().any() or not df[TARGET].isin([0,1]).all():
        raise ValueError('La etiqueta es_fraude debe contener únicamente 0 y 1, sin nulos.')
    df[TARGET] = df[TARGET].astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tables = {}
    for field in CATEGORIES:
        table = grouped(df, field)
        table.to_csv(OUT / f'{field}_summary.csv', index=False, encoding='utf-8-sig')
        tables[field] = table
        plot_group(table, field, field.replace('_',' '))
    descriptive = []
    for field in NUMERIC:
        s = pd.to_numeric(df[field], errors='coerce')
        for cls in [0,1]:
            values = s[df[TARGET].eq(cls)].dropna()
            descriptive.append({'variable':field, 'clase':cls,'n':int(len(values)),
                                'mediana':float(values.median()) if len(values) else None,
                                'p25':float(values.quantile(0.25)) if len(values) else None,
                                'p75':float(values.quantile(0.75)) if len(values) else None,
                                'promedio':float(values.mean()) if len(values) else None})
        numeric_ranges(df, field).to_csv(OUT / f'{field}_quintiles.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(descriptive).to_csv(OUT/'numeric_by_class.csv',index=False,encoding='utf-8-sig')
    result = {'dataset_rows':int(len(df)), 'frauds':int(df[TARGET].sum()),
              'legitimate':int((df[TARGET] == 0).sum()),
              'fraud_rate_pct':round(float(df[TARGET].mean()*100),6),
              'scope':'Descriptive analysis on official academic dataset (not model performance)',
              'warning':'Group associations are not causal. Small fraud counts imply uncertainty.',
              'tables':{k:f'{k}_summary.csv' for k in tables},
              'numeric_summary':'numeric_by_class.csv'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print('Archivos creados en:',OUT)

if __name__ == '__main__':
    main()
