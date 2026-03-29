#!/usr/bin/env bash
# compare_baseline.sh — Compare benchmark results against a saved baseline.
#
# Usage:
#   ./benchmarks/compare_baseline.sh                # Compare latest results vs baseline
#   ./benchmarks/compare_baseline.sh --set-baseline  # Save latest results as the baseline
#
# Flags any metric where tok/s dropped more than 5% from baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
BASELINE_FILE="$SCRIPT_DIR/baseline.jsonl"
THRESHOLD="${VEXEL_REGRESSION_THRESHOLD:-5}"  # percent

# ---------------------------------------------------------------------------
# Find the most recent date directory under results/
# ---------------------------------------------------------------------------
find_latest_results_dir() {
    local latest
    latest=$(ls -1d "$RESULTS_DIR"/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] 2>/dev/null \
        | sort -r | head -1)
    if [[ -z "$latest" ]]; then
        echo "ERROR: No dated results directories found in $RESULTS_DIR" >&2
        exit 1
    fi
    echo "$latest"
}

# ---------------------------------------------------------------------------
# Collect all JSONL files from a directory into one stream on stdout
# ---------------------------------------------------------------------------
collect_jsonl() {
    local dir="$1"
    for f in "$dir"/*.jsonl; do
        [[ -f "$f" ]] && cat "$f"
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

latest_dir=$(find_latest_results_dir)
echo "Latest results: $latest_dir"

if [[ "${1:-}" == "--set-baseline" ]]; then
    collect_jsonl "$latest_dir" > "$BASELINE_FILE"
    lines=$(wc -l < "$BASELINE_FILE" | tr -d ' ')
    echo "Baseline saved to $BASELINE_FILE ($lines records)"
    exit 0
fi

if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "No baseline file found. Creating from latest results..."
    collect_jsonl "$latest_dir" > "$BASELINE_FILE"
    lines=$(wc -l < "$BASELINE_FILE" | tr -d ' ')
    echo "Baseline created at $BASELINE_FILE ($lines records)"
    echo "(Run again after new benchmarks to compare.)"
    exit 0
fi

# Run the Python comparison
collect_jsonl "$latest_dir" | python3 -c "
import json, sys

threshold = float('${THRESHOLD}')

# -------------------------------------------------------------------------
# Load baseline
# -------------------------------------------------------------------------
baseline = {}
with open('${BASELINE_FILE}') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = (rec.get('engine',''), rec.get('model',''), rec.get('mode',''),
               str(rec.get('concurrency','')), str(rec.get('gen_tokens','')))
        baseline.setdefault(key, []).append(rec)

# Average the runs per key for baseline
baseline_avg = {}
for key, recs in baseline.items():
    avgs = {}
    for metric in ('decode_tok_s', 'prefill_tok_s', 'aggregate_tok_s'):
        vals = [r[metric] for r in recs if metric in r and r[metric] > 0]
        if vals:
            avgs[metric] = sum(vals) / len(vals)
    if avgs:
        baseline_avg[key] = avgs

# -------------------------------------------------------------------------
# Load current results from stdin
# -------------------------------------------------------------------------
current = {}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    key = (rec.get('engine',''), rec.get('model',''), rec.get('mode',''),
           str(rec.get('concurrency','')), str(rec.get('gen_tokens','')))
    current.setdefault(key, []).append(rec)

current_avg = {}
for key, recs in current.items():
    avgs = {}
    for metric in ('decode_tok_s', 'prefill_tok_s', 'aggregate_tok_s'):
        vals = [r[metric] for r in recs if metric in r and r[metric] > 0]
        if vals:
            avgs[metric] = sum(vals) / len(vals)
    if avgs:
        current_avg[key] = avgs

# -------------------------------------------------------------------------
# Compare and report
# -------------------------------------------------------------------------
regressions = []
rows = []

for key in sorted(set(baseline_avg.keys()) | set(current_avg.keys())):
    engine, model, mode, concurrency, gen_tokens = key
    label = f'{engine}/{model}/{mode}'
    if concurrency:
        label += f'/c={concurrency}'

    b = baseline_avg.get(key, {})
    c = current_avg.get(key, {})

    for metric in ('decode_tok_s', 'prefill_tok_s', 'aggregate_tok_s'):
        bv = b.get(metric)
        cv = c.get(metric)
        if bv is None or cv is None:
            continue

        pct = ((cv - bv) / bv) * 100 if bv > 0 else 0
        regressed = pct < -threshold
        status = 'REGRESSION' if regressed else 'ok'
        rows.append((label, metric, bv, cv, pct, status))
        if regressed:
            regressions.append((label, metric, bv, cv, pct))

# -------------------------------------------------------------------------
# Print summary table
# -------------------------------------------------------------------------
if not rows:
    print('No comparable metrics found between baseline and current results.')
    sys.exit(0)

metric_short = {
    'decode_tok_s': 'decode',
    'prefill_tok_s': 'prefill',
    'aggregate_tok_s': 'aggregate',
}

hdr = f\"{'Benchmark':<42} {'Metric':<10} {'Baseline':>10} {'Current':>10} {'Delta':>8}  {'Status'}\"
print()
print(hdr)
print('-' * len(hdr))
for label, metric, bv, cv, pct, status in rows:
    m = metric_short.get(metric, metric)
    sign = '+' if pct >= 0 else ''
    print(f'{label:<42} {m:<10} {bv:>10.1f} {cv:>10.1f} {sign}{pct:>6.1f}%  {status}')

print()
if regressions:
    print(f'FAILED: {len(regressions)} regression(s) exceed {threshold}% threshold')
    sys.exit(1)
else:
    print(f'PASSED: All metrics within {threshold}% of baseline')
    sys.exit(0)
"
