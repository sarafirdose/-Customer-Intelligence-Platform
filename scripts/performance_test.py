"""
Performance Benchmark Script.

Runs API latency, prediction throughput, and batch performance tests.
Saves results to docs/performance/<YYYY-MM-DD>.md (dated history).

Usage:
    python scripts/performance_test.py [--api-url http://localhost:8000]
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

BASE_DIR = Path(__file__).resolve().parents[1]
PERF_DIR = BASE_DIR / "docs" / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)


def _timeit(fn, n: int) -> dict:
    """Run fn n times and return latency statistics in ms."""
    latencies = []
    errors = 0
    for _ in range(n):
        start = time.perf_counter()
        try:
            fn()
        except Exception:
            errors += 1
        finally:
            latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    return {
        "n": n,
        "errors": errors,
        "avg_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
    }


def run_benchmarks(api_url: str) -> dict:
    client = httpx.Client(base_url=api_url, timeout=10.0)

    results = {}

    # 1. Health endpoint
    print("Benchmarking /ready (100 requests)...")
    try:
        results["ready"] = _timeit(lambda: client.get("/api/v1/ready"), 100)
    except Exception as e:
        results["ready"] = {"error": str(e)}

    # 2. Metrics endpoint
    print("Benchmarking /metrics (50 requests)...")
    try:
        results["metrics"] = _timeit(lambda: client.get("/api/v1/metrics"), 50)
    except Exception as e:
        results["metrics"] = {"error": str(e)}

    # 3. Registry endpoint
    print("Benchmarking /observability/registry (30 requests)...")
    try:
        results["registry"] = _timeit(lambda: client.get("/api/v1/observability/registry"), 30)
    except Exception as e:
        results["registry"] = {"error": str(e)}

    # 4. Sysinfo endpoint
    print("Benchmarking /observability/sysinfo (20 requests)...")
    try:
        results["sysinfo"] = _timeit(lambda: client.get("/api/v1/observability/sysinfo"), 20)
    except Exception as e:
        results["sysinfo"] = {"error": str(e)}

    client.close()
    return results


def write_report(results: dict, api_url: str) -> Path:
    """Write performance results to a dated Markdown file."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    report_path = PERF_DIR / f"{today}.md"

    lines = [
        f"# Performance Benchmark Report — {today}",
        "",
        f"**API URL**: `{api_url}`",
        f"**Run At**: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Endpoint | N | Avg ms | Median ms | p95 ms | p99 ms | Min ms | Max ms | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for endpoint, data in results.items():
        if "error" in data:
            lines.append(f"| `{endpoint}` | N/A | ❌ {data['error']} | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| `{endpoint}` | {data['n']} | {data['avg_ms']} | {data['median_ms']} "
                f"| {data['p95_ms']} | {data['p99_ms']} "
                f"| {data['min_ms']} | {data['max_ms']} | {data['errors']} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "| Latency | Rating |",
        "|---|---|",
        "| < 50 ms | 🟢 Excellent |",
        "| 50–200 ms | 🟡 Acceptable |",
        "| > 200 ms | 🔴 Investigate |",
        "",
        "*Generated automatically by scripts/performance_test.py*",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API performance benchmarks.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI base URL")
    args = parser.parse_args()

    print(f"\nRunning benchmarks against {args.api_url}\n")
    results = run_benchmarks(args.api_url)

    report_path = write_report(results, args.api_url)
    print(f"\nReport saved to: {report_path}")

    # Summary
    print("\n--- Summary ---")
    for ep, data in results.items():
        if "error" not in data:
            print(f"  {ep:<25} avg={data['avg_ms']}ms  p95={data['p95_ms']}ms")
        else:
            print(f"  {ep:<25} ERROR: {data['error']}")


if __name__ == "__main__":
    main()
