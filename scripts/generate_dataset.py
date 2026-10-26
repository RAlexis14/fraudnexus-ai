
"""Command-line entry point for synthetic dataset generation."""

import argparse
import json
from pathlib import Path

from fraudnexus.data.generator import (
    GENERATOR_VERSION,
    SEED,
    generate_transactions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FraudNexus AI synthetic banking transactions."
    )
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--fraud-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "transacciones_fraude_bancario.csv"
    metadata_path = output_dir / "dataset_metadata.json"

    df = generate_transactions(
        n_records=args.records,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    df.to_csv(dataset_path, index=False)

    fraud_count = int(df["es_fraude"].sum())
    fraud_rate = fraud_count / len(df)

    metadata = {
        "project": "FraudNexus AI",
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "total_records": len(df),
        "fraud_records": fraud_count,
        "legitimate_records": len(df) - fraud_count,
        "actual_fraud_rate": fraud_rate,
        "columns": list(df.columns),
        "missing_values": {
            key: int(value)
            for key, value in df.isna().sum().items()
        },
        "synthetic": True,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFraudNexus AI - Dataset generated")
    print(f"Records: {len(df):,}")
    print(f"Frauds: {fraud_count:,}")
    print(f"Legitimate: {len(df) - fraud_count:,}")
    print(f"Fraud prevalence: {fraud_rate:.4%}")
    print(f"Dataset: {dataset_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
