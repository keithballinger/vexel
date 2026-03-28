#!/usr/bin/env python3
"""report.py — Generate a unified markdown report from benchmark JSONL results.

Usage:
    python3 lib/report.py <results_dir>

Reads JSONL benchmark files and hardware.json from <results_dir>, generates
a markdown comparison report, writes it to <results_dir>/report.md, and
prints it to stdout.
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts. Returns [] if missing."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_hardware(results_dir: Path) -> dict:
    """Load hardware info from hardware.json or hardware.txt."""
    hw_json = results_dir / "hardware.json"
    if hw_json.exists():
        return json.loads(hw_json.read_text())

    # Fallback: parse hardware.txt written by full_comparison.sh
    hw_txt = results_dir / "hardware.txt"
    if hw_txt.exists():
        info = {}
        for line in hw_txt.read_text().splitlines():
            if line.startswith("Chip:"):
                info["chip"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU:"):
                info.setdefault("chip", line.split(":", 1)[1].strip())
            elif line.startswith("Memory:"):
                mem_str = line.split(":", 1)[1].strip()
                try:
                    info["memory_gb"] = int(mem_str.replace("GB", "").strip())
                except ValueError:
                    info["memory_gb"] = mem_str
        return info

    return {}


def mean(values: list[float]) -> float:
    """Compute arithmetic mean. Returns 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def fmt(value: float, decimals: int = 1) -> str:
    """Format a float to a fixed number of decimal places."""
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def section_standard_decode(records: list[dict]) -> str:
    """Generate the Standard Decode markdown table."""
    if not records:
        return ""

    # Group by (model, engine) -> list of records
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["model"], r["engine"])].append(r)

    lines = [
        "## Standard Decode",
        "",
        "| Model | Engine | Decode tok/s | Prefill tok/s |",
        "|-------|--------|-------------:|--------------:|",
    ]

    # Sort by model name, then engine
    for (model, engine) in sorted(grouped.keys()):
        runs = grouped[(model, engine)]
        avg_decode = mean([r["decode_tok_s"] for r in runs])
        avg_prefill = mean([r["prefill_tok_s"] for r in runs])
        lines.append(
            f"| {model} | {engine} | {fmt(avg_decode)} | {fmt(avg_prefill)} |"
        )

    lines.append("")
    return "\n".join(lines)


def section_speculative(records: list[dict]) -> str:
    """Generate the Speculative Decode markdown table."""
    if not records:
        return ""

    # Group by (engine, mode) -> list of records
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["engine"], r["mode"])].append(r)

    lines = [
        "## Speculative Decode",
        "",
        "| Engine | Mode | Decode tok/s | Acceptance % | Speedup |",
        "|--------|------|-------------:|-------------:|--------:|",
    ]

    for (engine, mode) in sorted(grouped.keys()):
        runs = grouped[(engine, mode)]
        avg_decode = mean([r["decode_tok_s"] for r in runs])
        avg_accept = mean([r["acceptance_pct"] for r in runs])
        # speedup may be absent for llama.cpp entries
        speedup_vals = [r["speedup"] for r in runs if "speedup" in r]
        avg_speedup = mean(speedup_vals) if speedup_vals else None

        speedup_str = fmt(avg_speedup) + "x" if avg_speedup is not None else "—"
        lines.append(
            f"| {engine} | {mode} | {fmt(avg_decode)} | {fmt(avg_accept)} | {speedup_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def section_batched(records: list[dict]) -> str:
    """Generate the Batched Decode markdown table (engines side-by-side)."""
    if not records:
        return ""

    # Group by (concurrency, engine) -> list of records
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["concurrency"], r["engine"])].append(r)

    concurrency_levels = sorted({k[0] for k in grouped.keys()})

    lines = [
        "## Batched Decode",
        "",
        "| Concurrency | Vexel tok/s | llama.cpp tok/s |",
        "|------------:|------------:|----------------:|",
    ]

    for conc in concurrency_levels:
        vexel_runs = grouped.get((conc, "vexel"), [])
        llama_runs = grouped.get((conc, "llama.cpp"), [])

        vexel_str = fmt(mean([r["aggregate_tok_s"] for r in vexel_runs])) if vexel_runs else "—"
        llama_str = fmt(mean([r["aggregate_tok_s"] for r in llama_runs])) if llama_runs else "—"

        lines.append(f"| {conc} | {vexel_str} | {llama_str} |")

    lines.append("")
    return "\n".join(lines)


def section_context_scaling(records: list[dict]) -> str:
    """Generate the Context Scaling markdown table with degradation %."""
    if not records:
        return ""

    # Group by (context_length, engine) -> list of records
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["context_length"], r["engine"])].append(r)

    context_lengths = sorted({k[0] for k in grouped.keys()})
    engines = sorted({k[1] for k in grouped.keys()})

    # Compute mean decode tok/s per (ctx, engine)
    means: dict[tuple[int, str], float] = {}
    for (ctx, eng), runs in grouped.items():
        means[(ctx, eng)] = mean([r["decode_tok_s"] for r in runs])

    # Find baseline (smallest context length, should be 16)
    baseline_ctx = context_lengths[0] if context_lengths else 16
    baselines: dict[str, float] = {}
    for eng in engines:
        baselines[eng] = means.get((baseline_ctx, eng), 0.0)

    lines = [
        "## Context Scaling",
        "",
        "| Context | Vexel tok/s | Vexel Degrad. | llama.cpp tok/s | llama.cpp Degrad. |",
        "|--------:|------------:|--------------:|----------------:|------------------:|",
    ]

    for ctx in context_lengths:
        vexel_val = means.get((ctx, "vexel"))
        llama_val = means.get((ctx, "llama.cpp"))

        def _cell(val, baseline):
            if val is None:
                return "—", "—"
            tok_str = fmt(val)
            if baseline and baseline > 0:
                degrad = ((baseline - val) / baseline) * 100.0
                degrad_str = fmt(degrad) + "%"
            else:
                degrad_str = "—"
            return tok_str, degrad_str

        v_tok, v_deg = _cell(vexel_val, baselines.get("vexel"))
        l_tok, l_deg = _cell(llama_val, baselines.get("llama.cpp"))

        lines.append(f"| {ctx} | {v_tok} | {v_deg} | {l_tok} | {l_deg} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(results_dir: Path) -> str:
    """Generate the full markdown report string."""
    hw = load_hardware(results_dir)

    # Header
    parts = []
    parts.append("# Vexel Benchmark Report")
    parts.append("")

    chip = hw.get("chip", "unknown")
    memory = hw.get("memory_gb", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"**Hardware:** {chip} — {memory} GB  ")
    parts.append(f"**Date:** {timestamp}  ")
    parts.append("")

    # Load data and generate sections
    standard = load_jsonl(results_dir / "standard_decode.jsonl")
    speculative = load_jsonl(results_dir / "speculative.jsonl")
    batched = load_jsonl(results_dir / "batched.jsonl")
    context = load_jsonl(results_dir / "context_scaling.jsonl")

    sections = [
        section_standard_decode(standard),
        section_speculative(speculative),
        section_batched(batched),
        section_context_scaling(context),
    ]

    # Only include non-empty sections
    for section in sections:
        if section:
            parts.append(section)

    if not any(sections):
        parts.append("*No benchmark results found.*")
        parts.append("")

    return "\n".join(parts)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = generate_report(results_dir)

    # Write to file
    report_path = results_dir / "report.md"
    report_path.write_text(report)

    # Print to stdout
    print(report)


if __name__ == "__main__":
    main()
