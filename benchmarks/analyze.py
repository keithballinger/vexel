#!/usr/bin/env python3
"""
Benchmark Analysis Script

Reads JSONL result files from a benchmark run directory and computes
summary statistics: mean, stddev, min, max, P50, P90, P99 for key metrics.

Usage:
    python3 analyze.py <results_dir> [--output summary.json]
"""

import json
import math
import os
import sys
from pathlib import Path


def percentile(data, p):
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0
    k = (len(data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def stats(values):
    """Compute summary statistics for a list of values."""
    if not values:
        return {"mean": 0, "stddev": 0, "min": 0, "max": 0, "p50": 0, "p90": 0, "p99": 0, "n": 0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    stddev = math.sqrt(variance)
    s = sorted(values)

    return {
        "mean": round(mean, 2),
        "stddev": round(stddev, 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "p50": round(percentile(s, 50), 2),
        "p90": round(percentile(s, 90), 2),
        "p99": round(percentile(s, 99), 2),
        "n": n,
    }


def load_results(results_dir):
    """Load all JSONL files from the results directory."""
    engines = {}
    results_path = Path(results_dir)

    for jsonl_file in sorted(results_path.glob("*.jsonl")):
        engine_name = jsonl_file.stem
        runs = []
        for line in jsonl_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if runs:
            engines[engine_name] = runs

    return engines


def analyze(results_dir):
    """Analyze benchmark results and print summary."""
    engines = load_results(results_dir)

    if not engines:
        print(f"No result files found in {results_dir}")
        return None

    summary = {}

    # Print header
    print(f"{'Engine':<12} {'Decode tok/s':>14} {'±stddev':>10} {'P50':>8} {'P99':>8} {'Runs':>6}")
    print("-" * 60)

    for engine, runs in sorted(engines.items()):
        # Extract decode throughput values
        decode_values = [r.get("decode_tok_s", 0) for r in runs if r.get("decode_tok_s", 0) > 0]
        gen_time_values = [r.get("gen_time_s", 0) for r in runs if r.get("gen_time_s", 0) > 0]
        prefill_values = [r.get("prefill_tok_s", 0) for r in runs if r.get("prefill_tok_s", 0) > 0]
        load_values = [r.get("load_time_s", 0) for r in runs if r.get("load_time_s", 0) > 0]

        decode_stats = stats(decode_values)
        gen_time_stats = stats(gen_time_values)
        prefill_stats = stats(prefill_values)
        load_stats = stats(load_values)

        # Model info from first run
        model = runs[0].get("model", "unknown")

        print(f"{engine:<12} {decode_stats['mean']:>14.2f} {decode_stats['stddev']:>10.2f} "
              f"{decode_stats['p50']:>8.2f} {decode_stats['p99']:>8.2f} {decode_stats['n']:>6}")

        summary[engine] = {
            "model": model,
            "decode_tok_s": decode_stats,
            "gen_time_s": gen_time_stats,
            "prefill_tok_s": prefill_stats if prefill_values else None,
            "load_time_s": load_stats if load_values else None,
            "total_runs": len(runs),
        }

    print("")

    # Memory bandwidth utilization (M3 Max = 400 GB/s)
    # For Q4_0 8B model: ~4.3 GB weights read per token
    MEM_BW_GBS = 400  # M3 Max
    MODEL_SIZE_GB = 4.3  # approximate for 8B Q4

    print(f"{'Engine':<12} {'BW Util %':>10}")
    print("-" * 24)
    for engine, data in sorted(summary.items()):
        tok_s = data["decode_tok_s"]["mean"]
        if tok_s > 0:
            bw_used = tok_s * MODEL_SIZE_GB
            util_pct = (bw_used / MEM_BW_GBS) * 100
            print(f"{engine:<12} {util_pct:>10.1f}%")
            summary[engine]["bw_utilization_pct"] = round(util_pct, 1)

    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <results_dir> [--output summary.json]")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_file = None

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    summary = analyze(results_dir)

    if summary and output_file:
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {output_file}")


if __name__ == "__main__":
    main()
