#!/usr/bin/env bash
# Competitive Benchmark Harness for Vexel
#
# Runs all engines with standardized parameters and outputs
# machine-readable JSON for each run.
#
# Usage:
#   ./run_benchmark.sh --model <path> [--engines mlx,llama,vexel,ollama,vllm-mlx]
#                      [--prompt-tokens 32] [--gen-tokens 512]
#                      [--warmup 3] [--runs 10]
#                      [--output-dir results/]
#                      [--benchmark decode|prefill|load|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- Defaults ---
MODEL_PATH=""
ENGINES="mlx,llama,vexel,ollama"
PROMPT_TOKENS=32
GEN_TOKENS=512
WARMUP_RUNS=3
MEASURE_RUNS=10
OUTPUT_DIR="$SCRIPT_DIR/results"
BENCHMARK="decode"
TEMPERATURE=0

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL_PATH="$2"; shift 2 ;;
        --engines)     ENGINES="$2"; shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        --gen-tokens)  GEN_TOKENS="$2"; shift 2 ;;
        --warmup)      WARMUP_RUNS="$2"; shift 2 ;;
        --runs)        MEASURE_RUNS="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --benchmark)   BENCHMARK="$2"; shift 2 ;;
        --temp)        TEMPERATURE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --model <path> [options]"
            echo "  --engines       Comma-separated: mlx,llama,vexel,ollama,vllm-mlx (default: mlx,llama,vexel,ollama)"
            echo "  --prompt-tokens Number of prompt tokens (default: 32)"
            echo "  --gen-tokens    Tokens to generate (default: 512)"
            echo "  --warmup        Warmup runs to discard (default: 3)"
            echo "  --runs          Measurement runs (default: 10)"
            echo "  --output-dir    Output directory (default: results/)"
            echo "  --benchmark     Type: decode, prefill, load, all (default: decode)"
            echo "  --temp          Sampling temperature (default: 0, greedy)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required"
    echo "Example: $0 --model /path/to/llama-3.1-8b-q4_k_m.gguf"
    exit 1
fi

# --- Activate venv ---
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# --- Setup ---
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${BENCHMARK}_${TIMESTAMP}"
HARDWARE_JSON="$OUTPUT_DIR/hardware.json"
MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/\.[^.]*$//')

echo "=== Vexel Competitive Benchmark ==="
echo "Benchmark: $BENCHMARK"
echo "Model: $MODEL_NAME"
echo "Prompt tokens: $PROMPT_TOKENS, Gen tokens: $GEN_TOKENS"
echo "Warmup: $WARMUP_RUNS, Runs: $MEASURE_RUNS"
echo "Engines: $ENGINES"
echo "Output: $OUTPUT_DIR/$RUN_ID/"
echo ""

# --- Hardware info ---
write_hardware_info() {
    python3 -c "
import json, subprocess, os

hw = {
    'chip': subprocess.getoutput('sysctl -n machdep.cpu.brand_string'),
    'memory_gb': int(subprocess.getoutput('sysctl -n hw.memsize')) // (1024**3),
    'macos': subprocess.getoutput('sw_vers -productVersion'),
    'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
}
print(json.dumps(hw, indent=2))
" > "$HARDWARE_JSON"
}
write_hardware_info

# --- Standard prompt generator ---
# Generates a prompt of approximately N tokens using repeated words.
generate_prompt() {
    local n_tokens="$1"
    python3 -c "
# Approximate: 1 token ≈ 0.75 words for English text
import math
n = $n_tokens
words = ['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog.']
n_words = max(1, int(n * 0.75))
prompt = ' '.join([words[i % len(words)] for i in range(n_words)])
print(prompt)
"
}

PROMPT=$(generate_prompt "$PROMPT_TOKENS")

# --- Per-engine runners ---
# Each runner outputs a JSON object with timing metrics.

run_mlx() {
    local run_num="$1"
    python3 -c "
import time, json, sys, os

# Suppress MLX logging noise
os.environ['MLX_NO_COMPILE_CACHE'] = '1'

from mlx_lm import load, generate

model_path = '$MODEL_PATH'
prompt = '''$PROMPT'''
max_tokens = $GEN_TOKENS
temp = $TEMPERATURE

# Load model
load_start = time.perf_counter()
model, tokenizer = load(model_path)
load_time = time.perf_counter() - load_start

# Tokenize prompt for prefill count
input_ids = tokenizer.encode(prompt)
prompt_tokens = len(input_ids)

# Generate
gen_start = time.perf_counter()
output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=temp, verbose=False)
gen_time = time.perf_counter() - gen_start

# Count output tokens
output_tokens = len(tokenizer.encode(output)) - prompt_tokens
if output_tokens <= 0:
    output_tokens = max_tokens  # fallback

result = {
    'engine': 'mlx',
    'run': $run_num,
    'model': os.path.basename(model_path),
    'prompt_tokens': prompt_tokens,
    'gen_tokens': output_tokens,
    'gen_time_s': round(gen_time, 4),
    'load_time_s': round(load_time, 4),
    'decode_tok_s': round(output_tokens / gen_time, 2) if gen_time > 0 else 0,
}
print(json.dumps(result))
" 2>/dev/null
}

run_llama() {
    local run_num="$1"
    local start_time end_time duration
    start_time=$(python3 -c "import time; print(time.perf_counter())")

    # llama-cli outputs timing stats to stderr
    local output
    output=$(llama-cli \
        -m "$MODEL_PATH" \
        -p "$PROMPT" \
        -n "$GEN_TOKENS" \
        --temp "$TEMPERATURE" \
        --no-display-prompt \
        -ngl 99 \
        2>&1) || true

    end_time=$(python3 -c "import time; print(time.perf_counter())")

    # Parse llama.cpp timing output
    python3 -c "
import json, re, sys

output = '''$output'''
run_num = $run_num
start = $start_time
end = $end_time
duration = end - start

# Parse llama.cpp stats from output
# Looks for: 'llama_perf_sampler_print:    sampling time = ...'
# and: 'llama_perf_context_print:        eval time = ...'
eval_tok_s = 0
prompt_tok_s = 0
gen_tokens = 0
prompt_tokens = 0

for line in output.split('\n'):
    # Token generation (decode)
    m = re.search(r'eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)', line)
    if m:
        gen_tokens = int(m.group(1))
        eval_tok_s = float(m.group(3))
    # Prompt eval (prefill)
    m = re.search(r'prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)', line)
    if m:
        prompt_tokens = int(m.group(1))
        prompt_tok_s = float(m.group(3))

result = {
    'engine': 'llama.cpp',
    'run': run_num,
    'model': '$(basename "$MODEL_PATH")',
    'prompt_tokens': prompt_tokens,
    'gen_tokens': gen_tokens,
    'gen_time_s': round(duration, 4),
    'decode_tok_s': eval_tok_s,
    'prefill_tok_s': prompt_tok_s,
}
print(json.dumps(result))
"
}

run_ollama() {
    local run_num="$1"
    local model_tag="$2"

    # Ollama uses model tags, not file paths
    # Must have model pulled: ollama pull llama3.1:8b-instruct-q4_K_M
    local start_time end_time
    start_time=$(python3 -c "import time; print(time.perf_counter())")

    local output
    output=$(ollama run "$model_tag" "$PROMPT" --verbose 2>&1) || true

    end_time=$(python3 -c "import time; print(time.perf_counter())")

    python3 -c "
import json, re

output = '''$output'''
duration = $end_time - $start_time

eval_tok_s = 0
gen_tokens = 0

# Parse Ollama verbose output
for line in output.split('\n'):
    m = re.search(r'eval rate:\s*([\d.]+)\s*tokens/s', line)
    if m:
        eval_tok_s = float(m.group(1))
    m = re.search(r'eval count:\s*(\d+)', line)
    if m:
        gen_tokens = int(m.group(1))

result = {
    'engine': 'ollama',
    'run': $run_num,
    'model': '$model_tag',
    'gen_tokens': gen_tokens,
    'gen_time_s': round(duration, 4),
    'decode_tok_s': eval_tok_s,
}
print(json.dumps(result))
"
}

run_vexel() {
    local run_num="$1"
    local start_time end_time
    start_time=$(python3 -c "import time; print(time.perf_counter())")

    local vexel_bin="$SCRIPT_DIR/../inference/cmd/vexel/vexel"
    local output
    output=$("$vexel_bin" generate \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --max-tokens "$GEN_TOKENS" \
        --temp "$TEMPERATURE" \
        2>&1) || true

    end_time=$(python3 -c "import time; print(time.perf_counter())")

    python3 -c "
import json, re

output = '''$output'''
duration = $end_time - $start_time

# Parse Vexel output for timing stats
decode_tok_s = 0
gen_tokens = 0

for line in output.split('\n'):
    m = re.search(r'decode.*?([\d.]+)\s*tok/s', line)
    if m:
        decode_tok_s = float(m.group(1))
    m = re.search(r'tokens.*?(\d+)', line)
    if m:
        gen_tokens = int(m.group(1))

if decode_tok_s == 0 and duration > 0:
    decode_tok_s = round($GEN_TOKENS / duration, 2)

result = {
    'engine': 'vexel',
    'run': $run_num,
    'model': '$(basename "$MODEL_PATH")',
    'gen_tokens': gen_tokens if gen_tokens > 0 else $GEN_TOKENS,
    'gen_time_s': round(duration, 4),
    'decode_tok_s': decode_tok_s,
}
print(json.dumps(result))
"
}

# --- Main benchmark loop ---
run_engine() {
    local engine="$1"
    local run_num="$2"

    case "$engine" in
        mlx)      run_mlx "$run_num" ;;
        llama)    run_llama "$run_num" ;;
        ollama)   run_ollama "$run_num" "$(basename "$MODEL_PATH" .gguf)" ;;
        vexel)    run_vexel "$run_num" ;;
        vllm-mlx) echo '{"engine":"vllm-mlx","run":'$run_num',"note":"requires server mode, use run_batched.sh"}' ;;
        *)        echo "Unknown engine: $engine" >&2; return 1 ;;
    esac
}

# --- Execute benchmark ---
mkdir -p "$OUTPUT_DIR/$RUN_ID"

IFS=',' read -ra ENGINE_LIST <<< "$ENGINES"

for engine in "${ENGINE_LIST[@]}"; do
    echo "--- Benchmarking: $engine ---"
    RESULT_FILE="$OUTPUT_DIR/$RUN_ID/${engine}.jsonl"

    # Warmup
    echo "  Warmup ($WARMUP_RUNS runs)..."
    for ((i=1; i<=WARMUP_RUNS; i++)); do
        run_engine "$engine" "$i" > /dev/null 2>&1 || true
        echo -n "."
    done
    echo ""

    # Measurement
    echo "  Measuring ($MEASURE_RUNS runs)..."
    for ((i=1; i<=MEASURE_RUNS; i++)); do
        result=$(run_engine "$engine" "$i" 2>/dev/null || echo '{"error":"run failed"}')
        echo "$result" >> "$RESULT_FILE"
        echo -n "."
    done
    echo ""
    echo "  Results: $RESULT_FILE"
done

# --- Compute summary statistics ---
echo ""
echo "--- Summary ---"
python3 "$SCRIPT_DIR/analyze.py" "$OUTPUT_DIR/$RUN_ID" 2>/dev/null || \
    echo "(Run analyze.py manually for detailed stats)"

echo ""
echo "=== Benchmark complete ==="
echo "Raw data: $OUTPUT_DIR/$RUN_ID/"
echo "Hardware: $HARDWARE_JSON"
