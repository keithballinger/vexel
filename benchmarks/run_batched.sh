#!/usr/bin/env bash
# Batched Throughput Benchmark
#
# Compares continuous batching engines (Vexel, vllm-mlx) under concurrent load.
# Engines without batching (mlx, llama.cpp) are tested sequentially as baseline.
#
# Usage:
#   ./run_batched.sh --model <path> [--clients 1,2,4,8,16] [--runs 5]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Defaults
MODEL_PATH=""
CLIENTS="1,2,4,8,16"
RUNS=5
GEN_TOKENS=128
OUTPUT_DIR="$SCRIPT_DIR/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_PATH="$2"; shift 2 ;;
        --clients)    CLIENTS="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --model <path> [options]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model required"
    exit 1
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/batched_$TIMESTAMP"
mkdir -p "$RUN_DIR"

PROMPT="The quick brown fox jumped over the lazy dog."

echo "=== Batched Throughput Benchmark ==="
echo "Model: $(basename "$MODEL_PATH")"
echo "Clients: $CLIENTS"
echo "Gen tokens: $GEN_TOKENS"
echo ""

# --- Concurrent request generator ---
# Sends N simultaneous requests to a server endpoint and measures aggregate throughput.
run_concurrent_test() {
    local engine="$1"
    local endpoint="$2"
    local n_clients="$3"
    local run_num="$4"

    python3 -c "
import asyncio
import aiohttp
import json
import time

async def single_request(session, url, prompt, max_tokens):
    start = time.perf_counter()
    tokens = 0
    payload = {
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': 0,
        'stream': False,
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
            elapsed = time.perf_counter() - start
            # Handle different API formats
            if 'usage' in data:
                tokens = data['usage'].get('completion_tokens', max_tokens)
            else:
                tokens = max_tokens
            return {'elapsed': elapsed, 'tokens': tokens, 'status': resp.status}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {'elapsed': elapsed, 'tokens': 0, 'error': str(e)}

async def run_batch(url, n, prompt, max_tokens):
    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        tasks = [single_request(session, url, prompt, max_tokens) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

        total_tokens = sum(r.get('tokens', 0) for r in results)
        latencies = [r['elapsed'] for r in results if 'error' not in r]

        return {
            'engine': '$engine',
            'clients': $n_clients,
            'run': $run_num,
            'total_time_s': round(total_time, 4),
            'total_tokens': total_tokens,
            'aggregate_tok_s': round(total_tokens / total_time, 2) if total_time > 0 else 0,
            'avg_latency_s': round(sum(latencies) / len(latencies), 4) if latencies else 0,
            'max_latency_s': round(max(latencies), 4) if latencies else 0,
            'errors': sum(1 for r in results if 'error' in r),
        }

result = asyncio.run(run_batch('$endpoint', $n_clients, '$PROMPT', $GEN_TOKENS))
print(json.dumps(result))
"
}

# --- Run benchmarks ---
IFS=',' read -ra CLIENT_LIST <<< "$CLIENTS"

# vllm-mlx (if server is running)
VLLM_ENDPOINT="http://localhost:8000/v1/completions"
if curl -sf "$VLLM_ENDPOINT" &>/dev/null || curl -sf "http://localhost:8000/health" &>/dev/null; then
    echo "--- vllm-mlx (server detected) ---"
    for n in "${CLIENT_LIST[@]}"; do
        echo "  Clients=$n"
        for ((i=1; i<=RUNS; i++)); do
            result=$(run_concurrent_test "vllm-mlx" "$VLLM_ENDPOINT" "$n" "$i")
            echo "$result" >> "$RUN_DIR/vllm-mlx.jsonl"
            echo -n "."
        done
        echo ""
    done
else
    echo "vllm-mlx server not detected at localhost:8000. Start with:"
    echo "  source benchmarks/.venv/bin/activate"
    echo "  vllm-mlx serve $MODEL_PATH --port 8000"
    echo ""
fi

# Vexel (if server is running)
VEXEL_ENDPOINT="http://localhost:8080/v1/completions"
if curl -sf "$VEXEL_ENDPOINT" &>/dev/null || curl -sf "http://localhost:8080/health" &>/dev/null; then
    echo "--- Vexel (server detected) ---"
    for n in "${CLIENT_LIST[@]}"; do
        echo "  Clients=$n"
        for ((i=1; i<=RUNS; i++)); do
            result=$(run_concurrent_test "vexel" "$VEXEL_ENDPOINT" "$n" "$i")
            echo "$result" >> "$RUN_DIR/vexel.jsonl"
            echo -n "."
        done
        echo ""
    done
else
    echo "Vexel server not detected at localhost:8080. Start with:"
    echo "  ./inference/cmd/vexel/vexel serve --model $MODEL_PATH --port 8080"
    echo ""
fi

# Summary
echo ""
echo "--- Batched Benchmark Results ---"
python3 -c "
import json, sys
from pathlib import Path

run_dir = Path('$RUN_DIR')
for f in sorted(run_dir.glob('*.jsonl')):
    engine = f.stem
    runs = [json.loads(l) for l in f.read_text().strip().split('\n') if l.strip()]
    if not runs:
        continue
    # Group by client count
    by_clients = {}
    for r in runs:
        c = r.get('clients', 1)
        by_clients.setdefault(c, []).append(r)

    print(f'\n{engine}:')
    print(f\"  {'Clients':>8} {'Agg tok/s':>12} {'Avg lat':>10} {'Max lat':>10}\")
    for c in sorted(by_clients.keys()):
        group = by_clients[c]
        avg_agg = sum(r['aggregate_tok_s'] for r in group) / len(group)
        avg_lat = sum(r['avg_latency_s'] for r in group) / len(group)
        max_lat = max(r['max_latency_s'] for r in group)
        print(f'  {c:>8} {avg_agg:>12.1f} {avg_lat:>10.3f}s {max_lat:>10.3f}s')
" 2>/dev/null || echo "(No results to analyze)"

echo ""
echo "Raw data: $RUN_DIR/"
