
"""Benchmark HTTP latency for FraudNexus AI."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests


PAYLOAD = {
    "monto_transaccion": 65.20,
    "canal_transaccion": "POS",
    "categoria_comercio": "Retail",
    "velocidad_tx_10m": 0,
    "velocidad_desplazamiento_kmh": 10.0,
    "ratio_desviacion_monto": 1.1,
    "pais_emisor_coincide": 1,
    "distancia_residencia_km": 3.0,
    "hora_dia": 14,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/predict",
    )
    args = parser.parse_args()

    if args.requests < 1 or args.warmup < 0:
        parser.error("Invalid number of requests.")

    session = requests.Session()
    http_latencies = []
    internal_latencies = []
    errors = []

    print("\n=== FRAUDNEXUS AI | HTTP BENCHMARK ===")
    print(f"Endpoint: {args.url}")
    print(f"Warmup: {args.warmup}")
    print(f"Measured requests: {args.requests}")

    def send_request():
        start = time.perf_counter()
        response = session.post(
            args.url,
            json=PAYLOAD,
            timeout=10,
        )
        duration_ms = (
            time.perf_counter() - start
        ) * 1000

        response.raise_for_status()
        return duration_ms, response.json()

    try:
        for _ in range(args.warmup):
            send_request()

        for i in range(args.requests):
            try:
                duration, result = send_request()
                http_latencies.append(duration)

                internal_latencies.append(
                    float(result["total_processing_ms"])
                )

            except (requests.RequestException, ValueError, KeyError) as exc:
                errors.append({
                    "request": i + 1,
                    "error": str(exc),
                })

    finally:
        session.close()

    if not http_latencies:
        raise RuntimeError("No successful requests.")

    values = np.array(http_latencies, dtype=float)

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "endpoint": args.url,
        "method": "sequential_http_requests",
        "warmup_requests": args.warmup,
        "measured_requests": args.requests,
        "successful_requests": len(http_latencies),
        "failed_requests": len(errors),
        "http_latency_ms": {
            "mean": round(statistics.mean(http_latencies), 4),
            "p50": round(float(np.percentile(values, 50)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "p99": round(float(np.percentile(values, 99)), 4),
            "max": round(float(np.max(values)), 4),
            "min": round(float(np.min(values)), 4),
        },
        "mean_internal_processing_ms": round(
            statistics.mean(internal_latencies), 4
        ),
        "all_successful_under_100ms": bool(
            all(value < 100 for value in http_latencies)
        ),
        "p95_under_100ms": bool(
            np.percentile(values, 95) < 100
        ),
        "errors": errors[:10],
    }

    output = Path("reports/latency_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nRESULTS:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
