# Comprehensive Benchmark Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a benchmark suite measuring Vexel across all modes (standard, Medusa, draft-model speculative, batched decode) against llama.cpp, with context scaling analysis.

**Architecture:** A shell script (`benchmarks/full_comparison.sh`) orchestrates model downloads, runs each benchmark configuration, outputs JSONL results, and calls an extended `analyze.py --comparison` to generate a unified markdown report. Each benchmark config is a self-contained function that outputs JSONL lines.

**Tech Stack:** Bash, Python 3, curl (for model downloads and batched HTTP tests), jq (optional, for JSONL), existing Vexel and llama.cpp binaries

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `benchmarks/full_comparison.sh` | Main benchmark orchestrator |
| Create | `benchmarks/lib/models.sh` | Model download and discovery |
| Create | `benchmarks/lib/engines.sh` | Engine binary discovery and health checks |
| Create | `benchmarks/lib/bench_standard.sh` | Standard decode benchmark |
| Create | `benchmarks/lib/bench_speculative.sh` | Speculative decode benchmark (Medusa + draft) |
| Create | `benchmarks/lib/bench_batched.sh` | Batched multi-client benchmark |
| Create | `benchmarks/lib/bench_context.sh` | Context scaling sweep |
| Create | `benchmarks/lib/parse.sh` | Output parsing helpers for both engines |
| Create | `benchmarks/lib/report.py` | Comparison report generator |
| Modify | `benchmarks/analyze.py` | Add `--comparison` mode |

---

### Task 1: Create scaffold and model management

**Files:**
- Create: `benchmarks/full_comparison.sh`
- Create: `benchmarks/lib/models.sh`
- Create: `benchmarks/lib/engines.sh`

- [ ] **Step 1: Create lib directory**

```bash
mkdir -p benchmarks/lib
```

- [ ] **Step 2: Create models.sh — model download and discovery**

Create `benchmarks/lib/models.sh`:

```bash
#!/usr/bin/env bash
# Model management: locate or download required models

MODELS_DIR="${SCRIPT_DIR}/models"

# Model URLs (HuggingFace direct download)
LLAMA_8B_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
QWEN_05B_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
TINYLLAMA_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

ensure_models_gitignored() {
    if ! git check-ignore -q "$MODELS_DIR" 2>/dev/null; then
        echo "ERROR: $MODELS_DIR is not gitignored. Aborting to prevent committing models."
        echo "Add 'benchmarks/models/' to .gitignore"
        exit 1
    fi
}

download_if_missing() {
    local path="$1"
    local url="$2"
    local name
    name=$(basename "$path")

    if [[ -f "$path" ]]; then
        echo "  [ok] $name"
        return 0
    fi

    # Check llama.cpp sibling directory
    local sibling="../llama.cpp/models/$name"
    if [[ -f "$sibling" ]]; then
        echo "  [symlink] $name -> $sibling"
        ln -sf "$(cd "$(dirname "$sibling")" && pwd)/$name" "$path"
        return 0
    fi

    echo "  [download] $name ..."
    mkdir -p "$MODELS_DIR"
    curl -L --progress-bar -o "$path" "$url"
    echo "  [ok] $name ($(du -h "$path" | cut -f1))"
}

setup_models() {
    echo "=== Checking models ==="
    ensure_models_gitignored
    mkdir -p "$MODELS_DIR"

    MODEL_LLAMA_8B="$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    MODEL_QWEN_05B="$MODELS_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_TINYLLAMA="$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

    download_if_missing "$MODEL_LLAMA_8B" "$LLAMA_8B_URL"
    download_if_missing "$MODEL_QWEN_05B" "$QWEN_05B_URL"
    download_if_missing "$MODEL_TINYLLAMA" "$TINYLLAMA_URL"

    echo ""
}
```

- [ ] **Step 3: Create engines.sh — binary discovery**

Create `benchmarks/lib/engines.sh`:

```bash
#!/usr/bin/env bash
# Engine binary discovery and health checks

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

find_vexel() {
    VEXEL_BIN="$REPO_ROOT/vexel"
    if [[ ! -x "$VEXEL_BIN" ]]; then
        echo "Building vexel..."
        (cd "$REPO_ROOT" && make build)
    fi
    if [[ ! -x "$VEXEL_BIN" ]]; then
        echo "ERROR: Failed to build vexel"
        exit 1
    fi
    echo "  [ok] vexel: $VEXEL_BIN"
}

find_llama() {
    # Try PATH first, then sibling build directory
    LLAMA_CLI=""
    LLAMA_SERVER=""
    LLAMA_SPECULATIVE=""

    local llama_bin_dir="$REPO_ROOT/../llama.cpp/build/bin"

    for bin in llama-cli llama-server llama-speculative; do
        local varname
        case "$bin" in
            llama-cli)          varname="LLAMA_CLI" ;;
            llama-server)       varname="LLAMA_SERVER" ;;
            llama-speculative)  varname="LLAMA_SPECULATIVE" ;;
        esac

        if command -v "$bin" >/dev/null 2>&1; then
            eval "$varname=$(command -v "$bin")"
            echo "  [ok] $bin (PATH)"
        elif [[ -x "$llama_bin_dir/$bin" ]]; then
            eval "$varname=$llama_bin_dir/$bin"
            echo "  [ok] $bin ($llama_bin_dir/)"
        else
            echo "  [missing] $bin — llama.cpp comparisons using $bin will be skipped"
        fi
    done
}

setup_engines() {
    echo "=== Checking engines ==="
    find_vexel
    find_llama
    echo ""
}
```

- [ ] **Step 4: Create full_comparison.sh scaffold**

Create `benchmarks/full_comparison.sh`:

```bash
#!/usr/bin/env bash
# Comprehensive Benchmark: Vexel vs llama.cpp across all modes
#
# Usage:
#   ./full_comparison.sh all           # Run everything
#   ./full_comparison.sh decode        # Standard decode only
#   ./full_comparison.sh speculative   # Speculative modes only
#   ./full_comparison.sh context       # Context scaling sweep
#   ./full_comparison.sh batched       # Batched multi-client
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/models.sh"
source "$SCRIPT_DIR/lib/engines.sh"
source "$SCRIPT_DIR/lib/parse.sh"

# --- Config ---
WARMUP=3
RUNS=5
GEN_TOKENS=128
RESULTS_DIR="$SCRIPT_DIR/results/$(date +%Y-%m-%d)"

# --- Parse args ---
SUITE="${1:-all}"

case "$SUITE" in
    all|decode|speculative|context|batched) ;;
    -h|--help)
        echo "Usage: $0 {all|decode|speculative|context|batched}"
        exit 0 ;;
    *) echo "Unknown suite: $SUITE"; exit 1 ;;
esac

# --- Setup ---
setup_models
setup_engines
mkdir -p "$RESULTS_DIR"

# Collect hardware info
echo "=== Hardware ==="
HW_CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
HW_MEM=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
echo "  $HW_CHIP, ${HW_MEM}GB"
echo '{"chip":"'"$HW_CHIP"'","memory_gb":'"$HW_MEM"'}' > "$RESULTS_DIR/hardware.json"
echo ""

# --- Run suites ---
if [[ "$SUITE" == "all" || "$SUITE" == "decode" ]]; then
    source "$SCRIPT_DIR/lib/bench_standard.sh"
    run_standard_decode
fi

if [[ "$SUITE" == "all" || "$SUITE" == "speculative" ]]; then
    source "$SCRIPT_DIR/lib/bench_speculative.sh"
    run_speculative
fi

if [[ "$SUITE" == "all" || "$SUITE" == "batched" ]]; then
    source "$SCRIPT_DIR/lib/bench_batched.sh"
    run_batched
fi

if [[ "$SUITE" == "all" || "$SUITE" == "context" ]]; then
    source "$SCRIPT_DIR/lib/bench_context.sh"
    run_context_scaling
fi

# --- Generate report ---
echo "=== Generating Report ==="
python3 "$SCRIPT_DIR/lib/report.py" "$RESULTS_DIR"

echo ""
echo "Results: $RESULTS_DIR/"
echo "Report:  $RESULTS_DIR/report.md"
```

- [ ] **Step 5: Make scripts executable**

```bash
chmod +x benchmarks/full_comparison.sh
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/full_comparison.sh benchmarks/lib/models.sh benchmarks/lib/engines.sh
git commit -m "bench: add scaffold, model management, and engine discovery"
```

---

### Task 2: Create output parsing helpers

**Files:**
- Create: `benchmarks/lib/parse.sh`

- [ ] **Step 1: Create parse.sh with parsing functions for both engines**

Create `benchmarks/lib/parse.sh`:

```bash
#!/usr/bin/env bash
# Output parsing helpers for Vexel and llama.cpp

# Generate a synthetic prompt of approximately N tokens.
# Uses repeated short words to approximate 1 word ≈ 1.3 tokens.
generate_prompt() {
    local target_tokens="$1"
    local words=$(( target_tokens * 3 / 4 ))
    local prompt=""
    local word_list=("the" "quick" "brown" "fox" "jumps" "over" "lazy" "dog" "and" "runs" "through" "green" "fields" "under" "blue" "sky")
    for (( i=0; i<words; i++ )); do
        prompt+="${word_list[$((i % ${#word_list[@]}))]}"
        if (( i < words - 1 )); then
            prompt+=" "
        fi
    done
    echo "$prompt"
}

# Run Vexel generate and extract metrics.
# Outputs: decode_tok_s prefill_tok_s
# Requires --verbose for metrics output.
run_vexel_generate() {
    local model="$1"
    local prompt="$2"
    local max_tokens="$3"
    shift 3
    local extra_flags=("$@")

    local output
    output=$("$VEXEL_BIN" --model "$model" --verbose "${extra_flags[@]}" \
        generate --prompt "$prompt" --max-tokens "$max_tokens" --temperature 0 2>&1)

    local decode_tok_s prefill_tok_s
    decode_tok_s=$(echo "$output" | grep -oE 'decode: [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
    prefill_tok_s=$(echo "$output" | grep -oE 'prefill: [0-9.]+' | grep -oE '[0-9.]+' | tail -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0}"
}

# Run Vexel generate with Medusa and extract metrics including acceptance rate.
# Outputs: decode_tok_s prefill_tok_s acceptance_pct speedup
run_vexel_medusa() {
    local model="$1"
    local prompt="$2"
    local max_tokens="$3"

    local output
    output=$("$VEXEL_BIN" --model "$model" --verbose --medusa \
        generate --prompt "$prompt" --max-tokens "$max_tokens" --temperature 0 2>&1)

    local decode_tok_s prefill_tok_s acceptance_pct speedup
    decode_tok_s=$(echo "$output" | grep -oE 'decode: [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
    prefill_tok_s=$(echo "$output" | grep -oE 'prefill: [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
    acceptance_pct=$(echo "$output" | grep -oE 'acceptance=[0-9.]+%' | grep -oE '[0-9.]+' | tail -1)
    speedup=$(echo "$output" | grep -oE 'speedup=[0-9.]+x' | grep -oE '[0-9.]+' | tail -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0} ${acceptance_pct:-0} ${speedup:-0}"
}

# Run Vexel generate with draft model and extract speculative metrics.
# Outputs: decode_tok_s prefill_tok_s acceptance_pct speedup
run_vexel_draft() {
    local model="$1"
    local draft_model="$2"
    local prompt="$3"
    local max_tokens="$4"

    local output
    output=$("$VEXEL_BIN" --model "$model" --verbose --draft-model "$draft_model" \
        generate --prompt "$prompt" --max-tokens "$max_tokens" --temperature 0 2>&1)

    local decode_tok_s prefill_tok_s acceptance_pct speedup
    decode_tok_s=$(echo "$output" | grep -oE 'decode: [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
    prefill_tok_s=$(echo "$output" | grep -oE 'prefill: [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
    acceptance_pct=$(echo "$output" | grep -oE 'acceptance=[0-9.]+%' | grep -oE '[0-9.]+' | tail -1)
    speedup=$(echo "$output" | grep -oE 'speedup=[0-9.]+x' | grep -oE '[0-9.]+' | tail -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0} ${acceptance_pct:-0} ${speedup:-0}"
}

# Run llama-cli and extract metrics.
# Outputs: decode_tok_s prefill_tok_s
run_llama_generate() {
    local model="$1"
    local prompt="$2"
    local max_tokens="$3"

    if [[ -z "$LLAMA_CLI" ]]; then
        echo "0 0"
        return
    fi

    local output
    output=$("$LLAMA_CLI" -m "$model" -p "$prompt" -n "$max_tokens" \
        --temp 0 --top-k 1 --no-warmup --no-display-prompt 2>&1)

    # llama.cpp prints: "llama_perf_context_print:        eval time = X ms / Y tokens (Z ms per token, W tokens per second)"
    local decode_tok_s prefill_tok_s
    decode_tok_s=$(echo "$output" | grep 'eval time' | grep -v 'prompt' | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | head -1)
    prefill_tok_s=$(echo "$output" | grep 'prompt eval time' | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | head -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0}"
}

# Run llama-speculative and extract metrics.
# Outputs: decode_tok_s prefill_tok_s acceptance_pct
run_llama_speculative() {
    local model="$1"
    local draft_model="$2"
    local prompt="$3"
    local max_tokens="$4"

    if [[ -z "$LLAMA_SPECULATIVE" ]]; then
        echo "0 0 0"
        return
    fi

    local output
    output=$("$LLAMA_SPECULATIVE" -m "$model" -md "$draft_model" \
        -p "$prompt" -n "$max_tokens" --temp 0 --top-k 1 --no-warmup 2>&1)

    local decode_tok_s prefill_tok_s acceptance_pct
    decode_tok_s=$(echo "$output" | grep 'eval time' | grep -v 'prompt' | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | head -1)
    prefill_tok_s=$(echo "$output" | grep 'prompt eval time' | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | head -1)
    acceptance_pct=$(echo "$output" | grep -oE 'accept rate: [0-9.]+' | grep -oE '[0-9.]+' | tail -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0} ${acceptance_pct:-0}"
}

# Write a JSONL line to a file.
emit_jsonl() {
    local file="$1"
    shift
    echo "$@" >> "$file"
}
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/lib/parse.sh
git commit -m "bench: add output parsing helpers for Vexel and llama.cpp"
```

---

### Task 3: Implement standard decode benchmark

**Files:**
- Create: `benchmarks/lib/bench_standard.sh`

- [ ] **Step 1: Create bench_standard.sh**

Create `benchmarks/lib/bench_standard.sh`:

```bash
#!/usr/bin/env bash
# Standard decode benchmark: single sequence, greedy, both engines, both models

run_standard_decode() {
    echo "=== Standard Decode Benchmark ==="
    local outfile="$RESULTS_DIR/standard_decode.jsonl"
    > "$outfile"

    local prompt
    prompt=$(generate_prompt 32)

    local models=()
    local model_names=()

    models+=("$MODEL_QWEN_05B"); model_names+=("qwen-0.5b")
    models+=("$MODEL_LLAMA_8B"); model_names+=("llama-8b")

    for idx in "${!models[@]}"; do
        local model="${models[$idx]}"
        local model_name="${model_names[$idx]}"

        echo "--- $model_name: Vexel standard ---"

        # Warmup
        for (( w=0; w<WARMUP; w++ )); do
            run_vexel_generate "$model" "$prompt" "$GEN_TOKENS" > /dev/null 2>&1 || true
        done

        # Measured runs
        for (( r=1; r<=RUNS; r++ )); do
            local metrics
            metrics=$(run_vexel_generate "$model" "$prompt" "$GEN_TOKENS")
            local decode_tok_s prefill_tok_s
            read -r decode_tok_s prefill_tok_s <<< "$metrics"
            emit_jsonl "$outfile" \
                '{"engine":"vexel","mode":"standard","model":"'"$model_name"'","run":'"$r"',"gen_tokens":'"$GEN_TOKENS"',"decode_tok_s":'"$decode_tok_s"',"prefill_tok_s":'"$prefill_tok_s"'}'
            echo "  run $r: decode=${decode_tok_s} tok/s, prefill=${prefill_tok_s} tok/s"
        done

        echo "--- $model_name: llama.cpp standard ---"

        if [[ -n "$LLAMA_CLI" ]]; then
            # Warmup
            for (( w=0; w<WARMUP; w++ )); do
                run_llama_generate "$model" "$prompt" "$GEN_TOKENS" > /dev/null 2>&1 || true
            done

            # Measured runs
            for (( r=1; r<=RUNS; r++ )); do
                local metrics
                metrics=$(run_llama_generate "$model" "$prompt" "$GEN_TOKENS")
                local decode_tok_s prefill_tok_s
                read -r decode_tok_s prefill_tok_s <<< "$metrics"
                emit_jsonl "$outfile" \
                    '{"engine":"llama.cpp","mode":"standard","model":"'"$model_name"'","run":'"$r"',"gen_tokens":'"$GEN_TOKENS"',"decode_tok_s":'"$decode_tok_s"',"prefill_tok_s":'"$prefill_tok_s"'}'
                echo "  run $r: decode=${decode_tok_s} tok/s, prefill=${prefill_tok_s} tok/s"
            done
        else
            echo "  [skipped] llama-cli not found"
        fi
    done

    echo ""
}
```

- [ ] **Step 2: Test it runs (quick smoke test with 1 warmup, 1 run)**

```bash
cd benchmarks
WARMUP=1 RUNS=1 GEN_TOKENS=16 bash -c '
    source lib/models.sh
    source lib/engines.sh
    source lib/parse.sh
    source lib/bench_standard.sh
    SCRIPT_DIR="$(pwd)"
    RESULTS_DIR="/tmp/bench_test"
    mkdir -p "$RESULTS_DIR"
    setup_models
    setup_engines
    # Test with small model only
    MODEL_QWEN_05B="$MODELS_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_LLAMA_8B="$MODEL_QWEN_05B"  # Use small model for smoke test
    run_standard_decode
'
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/lib/bench_standard.sh
git commit -m "bench: add standard decode benchmark"
```

---

### Task 4: Implement speculative decode benchmark

**Files:**
- Create: `benchmarks/lib/bench_speculative.sh`

- [ ] **Step 1: Create bench_speculative.sh**

Create `benchmarks/lib/bench_speculative.sh`:

```bash
#!/usr/bin/env bash
# Speculative decode benchmark: Medusa + draft-model, 8B model only

run_speculative() {
    echo "=== Speculative Decode Benchmark ==="
    local outfile="$RESULTS_DIR/speculative.jsonl"
    > "$outfile"

    local model="$MODEL_LLAMA_8B"
    local model_name="llama-8b"
    local prompt
    prompt=$(generate_prompt 32)

    # --- Vexel Medusa ---
    echo "--- Vexel Medusa (online training) ---"
    echo "  Warming Medusa heads (50 tokens)..."
    # Medusa needs online training warmup — generate 50 tokens first
    run_vexel_generate "$model" "$prompt" 50 --medusa > /dev/null 2>&1 || true

    for (( r=1; r<=RUNS; r++ )); do
        local metrics
        metrics=$(run_vexel_medusa "$model" "$prompt" "$GEN_TOKENS")
        local decode_tok_s prefill_tok_s acceptance_pct speedup
        read -r decode_tok_s prefill_tok_s acceptance_pct speedup <<< "$metrics"
        emit_jsonl "$outfile" \
            '{"engine":"vexel","mode":"medusa","model":"'"$model_name"'","run":'"$r"',"gen_tokens":'"$GEN_TOKENS"',"decode_tok_s":'"$decode_tok_s"',"prefill_tok_s":'"$prefill_tok_s"',"acceptance_pct":'"$acceptance_pct"',"speedup":'"$speedup"'}'
        echo "  run $r: decode=${decode_tok_s} tok/s, accept=${acceptance_pct}%, speedup=${speedup}x"
    done

    # --- Vexel Draft-Model ---
    echo "--- Vexel draft-model (TinyLlama → LLaMA 8B) ---"
    local draft="$MODEL_TINYLLAMA"

    # Warmup
    for (( w=0; w<WARMUP; w++ )); do
        run_vexel_draft "$model" "$draft" "$prompt" "$GEN_TOKENS" > /dev/null 2>&1 || true
    done

    for (( r=1; r<=RUNS; r++ )); do
        local metrics
        metrics=$(run_vexel_draft "$model" "$draft" "$prompt" "$GEN_TOKENS")
        local decode_tok_s prefill_tok_s acceptance_pct speedup
        read -r decode_tok_s prefill_tok_s acceptance_pct speedup <<< "$metrics"
        emit_jsonl "$outfile" \
            '{"engine":"vexel","mode":"draft","model":"'"$model_name"'","run":'"$r"',"gen_tokens":'"$GEN_TOKENS"',"decode_tok_s":'"$decode_tok_s"',"prefill_tok_s":'"$prefill_tok_s"',"acceptance_pct":'"$acceptance_pct"',"speedup":'"$speedup"'}'
        echo "  run $r: decode=${decode_tok_s} tok/s, accept=${acceptance_pct}%, speedup=${speedup}x"
    done

    # --- llama.cpp Draft-Model ---
    echo "--- llama.cpp speculative (TinyLlama → LLaMA 8B) ---"
    if [[ -n "$LLAMA_SPECULATIVE" ]]; then
        for (( w=0; w<WARMUP; w++ )); do
            run_llama_speculative "$model" "$draft" "$prompt" "$GEN_TOKENS" > /dev/null 2>&1 || true
        done

        for (( r=1; r<=RUNS; r++ )); do
            local metrics
            metrics=$(run_llama_speculative "$model" "$draft" "$prompt" "$GEN_TOKENS")
            local decode_tok_s prefill_tok_s acceptance_pct
            read -r decode_tok_s prefill_tok_s acceptance_pct <<< "$metrics"
            emit_jsonl "$outfile" \
                '{"engine":"llama.cpp","mode":"draft","model":"'"$model_name"'","run":'"$r"',"gen_tokens":'"$GEN_TOKENS"',"decode_tok_s":'"$decode_tok_s"',"prefill_tok_s":'"$prefill_tok_s"',"acceptance_pct":'"$acceptance_pct"'}'
            echo "  run $r: decode=${decode_tok_s} tok/s, accept=${acceptance_pct}%"
        done
    else
        echo "  [skipped] llama-speculative not found"
    fi

    echo ""
}
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/lib/bench_speculative.sh
git commit -m "bench: add speculative decode benchmark (Medusa + draft-model)"
```

---

### Task 5: Implement batched decode benchmark

**Files:**
- Create: `benchmarks/lib/bench_batched.sh`

- [ ] **Step 1: Create bench_batched.sh**

Create `benchmarks/lib/bench_batched.sh`:

```bash
#!/usr/bin/env bash
# Batched decode benchmark: concurrent clients via server mode

run_batched() {
    echo "=== Batched Decode Benchmark ==="
    local outfile="$RESULTS_DIR/batched.jsonl"
    > "$outfile"

    local model="$MODEL_LLAMA_8B"
    local model_name="llama-8b"
    local prompt
    prompt=$(generate_prompt 32)
    local batch_gen_tokens=64
    local concurrency_levels=(1 2 4 8)

    for concurrency in "${concurrency_levels[@]}"; do
        # --- Vexel batched ---
        echo "--- Vexel batched (concurrency=$concurrency) ---"

        # Start vexel server
        local vexel_port=18080
        "$VEXEL_BIN" --model "$model" serve \
            --port "$vexel_port" --max-batch-size "$concurrency" &
        local vexel_pid=$!
        sleep 3  # Wait for server startup

        # Verify server is up
        if ! curl -sf "http://localhost:$vexel_port/health" > /dev/null 2>&1; then
            echo "  [error] Vexel server failed to start"
            kill "$vexel_pid" 2>/dev/null || true
            continue
        fi

        for (( r=1; r<=RUNS; r++ )); do
            # Send concurrent requests
            local start_time
            start_time=$(python3 -c 'import time; print(time.time())')

            local pids=()
            local total_tokens=0
            for (( c=0; c<concurrency; c++ )); do
                curl -sf -X POST "http://localhost:$vexel_port/generate" \
                    -H "Content-Type: application/json" \
                    -d '{"prompt":"'"$prompt"'","max_tokens":'"$batch_gen_tokens"',"temperature":0}' \
                    -o /dev/null &
                pids+=($!)
            done

            # Wait for all requests
            for pid in "${pids[@]}"; do
                wait "$pid" 2>/dev/null || true
            done

            local end_time
            end_time=$(python3 -c 'import time; print(time.time())')
            total_tokens=$((concurrency * batch_gen_tokens))

            local wall_time
            wall_time=$(python3 -c "print(round($end_time - $start_time, 3))")
            local agg_tok_s
            agg_tok_s=$(python3 -c "print(round($total_tokens / ($end_time - $start_time), 1))")

            emit_jsonl "$outfile" \
                '{"engine":"vexel","mode":"batched","model":"'"$model_name"'","run":'"$r"',"concurrency":'"$concurrency"',"total_tokens":'"$total_tokens"',"wall_time_s":'"$wall_time"',"aggregate_tok_s":'"$agg_tok_s"'}'
            echo "  run $r: ${agg_tok_s} aggregate tok/s (${concurrency}x${batch_gen_tokens} tokens in ${wall_time}s)"
        done

        kill "$vexel_pid" 2>/dev/null || true
        wait "$vexel_pid" 2>/dev/null || true
        sleep 1

        # --- llama.cpp batched ---
        echo "--- llama.cpp batched (concurrency=$concurrency) ---"
        if [[ -n "$LLAMA_SERVER" ]]; then
            local llama_port=18081
            "$LLAMA_SERVER" -m "$model" --port "$llama_port" -np "$concurrency" &
            local llama_pid=$!
            sleep 5  # llama-server takes longer to start

            if ! curl -sf "http://localhost:$llama_port/health" > /dev/null 2>&1; then
                echo "  [error] llama-server failed to start"
                kill "$llama_pid" 2>/dev/null || true
                continue
            fi

            for (( r=1; r<=RUNS; r++ )); do
                local start_time
                start_time=$(python3 -c 'import time; print(time.time())')

                local pids=()
                for (( c=0; c<concurrency; c++ )); do
                    curl -sf -X POST "http://localhost:$llama_port/completion" \
                        -H "Content-Type: application/json" \
                        -d '{"prompt":"'"$prompt"'","n_predict":'"$batch_gen_tokens"',"temperature":0}' \
                        -o /dev/null &
                    pids+=($!)
                done

                for pid in "${pids[@]}"; do
                    wait "$pid" 2>/dev/null || true
                done

                local end_time
                end_time=$(python3 -c 'import time; print(time.time())')
                local total_tokens=$((concurrency * batch_gen_tokens))

                local wall_time
                wall_time=$(python3 -c "print(round($end_time - $start_time, 3))")
                local agg_tok_s
                agg_tok_s=$(python3 -c "print(round($total_tokens / ($end_time - $start_time), 1))")

                emit_jsonl "$outfile" \
                    '{"engine":"llama.cpp","mode":"batched","model":"'"$model_name"'","run":'"$r"',"concurrency":'"$concurrency"',"total_tokens":'"$total_tokens"',"wall_time_s":'"$wall_time"',"aggregate_tok_s":'"$agg_tok_s"'}'
                echo "  run $r: ${agg_tok_s} aggregate tok/s (${concurrency}x${batch_gen_tokens} tokens in ${wall_time}s)"
            done

            kill "$llama_pid" 2>/dev/null || true
            wait "$llama_pid" 2>/dev/null || true
            sleep 1
        else
            echo "  [skipped] llama-server not found"
        fi
    done

    echo ""
}
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/lib/bench_batched.sh
git commit -m "bench: add batched multi-client decode benchmark"
```

---

### Task 6: Implement context scaling benchmark

**Files:**
- Create: `benchmarks/lib/bench_context.sh`

- [ ] **Step 1: Create bench_context.sh**

Create `benchmarks/lib/bench_context.sh`:

```bash
#!/usr/bin/env bash
# Context scaling benchmark: measure decode throughput across context lengths

run_context_scaling() {
    echo "=== Context Scaling Benchmark ==="
    local outfile="$RESULTS_DIR/context_scaling.jsonl"
    > "$outfile"

    local model="$MODEL_LLAMA_8B"
    local model_name="llama-8b"
    local decode_tokens=16
    local context_runs=3
    local context_lengths=(16 64 256 512 1024 2048 4096)

    for ctx in "${context_lengths[@]}"; do
        local prompt
        prompt=$(generate_prompt "$ctx")

        # --- Vexel ---
        echo "--- Vexel ctx=$ctx ---"
        # Warmup
        run_vexel_generate "$model" "$prompt" "$decode_tokens" > /dev/null 2>&1 || true

        for (( r=1; r<=context_runs; r++ )); do
            local metrics
            metrics=$(run_vexel_generate "$model" "$prompt" "$decode_tokens")
            local decode_tok_s prefill_tok_s
            read -r decode_tok_s prefill_tok_s <<< "$metrics"
            emit_jsonl "$outfile" \
                '{"engine":"vexel","mode":"context_scaling","model":"'"$model_name"'","run":'"$r"',"context_length":'"$ctx"',"decode_tokens":'"$decode_tokens"',"decode_tok_s":'"$decode_tok_s"',"prefill_tok_s":'"$prefill_tok_s"'}'
            echo "  run $r: ctx=$ctx decode=${decode_tok_s} tok/s"
        done

        # --- llama.cpp ---
        echo "--- llama.cpp ctx=$ctx ---"
        if [[ -n "$LLAMA_CLI" ]]; then
            run_llama_generate "$model" "$prompt" "$decode_tokens" > /dev/null 2>&1 || true

            for (( r=1; r<=context_runs; r++ )); do
                local metrics
                metrics=$(run_llama_generate "$model" "$prompt" "$decode_tokens")
                local decode_tok_s prefill_tok_s
                read -r decode_tok_s prefill_tok_s <<< "$metrics"
                emit_jsonl "$outfile" \
                    '{"engine":"llama.cpp","mode":"context_scaling","model":"'"$model_name"'","run":'"$r"',"context_length":'"$ctx"',"decode_tokens":'"$decode_tokens"',"decode_tok_s":'"$decode_tok_s"',"prefill_tok_s":'"$prefill_tok_s"'}'
                echo "  run $r: ctx=$ctx decode=${decode_tok_s} tok/s"
            done
        else
            echo "  [skipped] llama-cli not found"
        fi
    done

    echo ""
}
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/lib/bench_context.sh
git commit -m "bench: add context scaling sweep benchmark"
```

---

### Task 7: Create comparison report generator

**Files:**
- Create: `benchmarks/lib/report.py`

- [ ] **Step 1: Create report.py**

Create `benchmarks/lib/report.py`:

```python
#!/usr/bin/env python3
"""
Generate a unified comparison report from benchmark JSONL files.

Usage:
    python3 report.py <results_dir>

Reads all .jsonl files in the results directory and produces report.md.
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path


def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] * (c - k) + s[c] * (k - f) if f != c else s[f]


def mean(data):
    return sum(data) / len(data) if data else 0


def load_jsonl(path):
    results = []
    for line in Path(path).read_text().strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def generate_report(results_dir):
    rd = Path(results_dir)
    lines = []

    # Load hardware info
    hw_file = rd / "hardware.json"
    hw = json.loads(hw_file.read_text()) if hw_file.exists() else {}

    lines.append(f"# Benchmark Report — {date.today()}")
    lines.append("")
    if hw:
        lines.append(f"**Hardware:** {hw.get('chip', 'unknown')}, {hw.get('memory_gb', '?')}GB")
    lines.append("")

    # --- Standard Decode ---
    std_file = rd / "standard_decode.jsonl"
    if std_file.exists():
        runs = load_jsonl(std_file)
        lines.append("## Standard Decode")
        lines.append("")
        lines.append("| Model | Engine | Decode tok/s | Prefill tok/s |")
        lines.append("|-------|--------|-------------|--------------|")

        grouped = defaultdict(lambda: defaultdict(list))
        for r in runs:
            grouped[r["model"]][r["engine"]].append(r)

        for model in sorted(grouped):
            for engine in sorted(grouped[model]):
                eng_runs = grouped[model][engine]
                dec = [r["decode_tok_s"] for r in eng_runs if r["decode_tok_s"] > 0]
                pre = [r["prefill_tok_s"] for r in eng_runs if r["prefill_tok_s"] > 0]
                lines.append(f"| {model} | {engine} | {mean(dec):.1f} | {mean(pre):.1f} |")

        lines.append("")

    # --- Speculative ---
    spec_file = rd / "speculative.jsonl"
    if spec_file.exists():
        runs = load_jsonl(spec_file)
        lines.append("## Speculative Decode")
        lines.append("")
        lines.append("| Engine | Mode | Decode tok/s | Acceptance % | Speedup |")
        lines.append("|--------|------|-------------|-------------|---------|")

        grouped = defaultdict(lambda: defaultdict(list))
        for r in runs:
            grouped[r["engine"]][r["mode"]].append(r)

        for engine in sorted(grouped):
            for mode in sorted(grouped[engine]):
                eng_runs = grouped[engine][mode]
                dec = [r["decode_tok_s"] for r in eng_runs if r["decode_tok_s"] > 0]
                acc = [r.get("acceptance_pct", 0) for r in eng_runs if r.get("acceptance_pct", 0) > 0]
                spd = [r.get("speedup", 0) for r in eng_runs if r.get("speedup", 0) > 0]
                acc_str = f"{mean(acc):.1f}" if acc else "N/A"
                spd_str = f"{mean(spd):.2f}x" if spd else "N/A"
                lines.append(f"| {engine} | {mode} | {mean(dec):.1f} | {acc_str} | {spd_str} |")

        lines.append("")

    # --- Batched ---
    batch_file = rd / "batched.jsonl"
    if batch_file.exists():
        runs = load_jsonl(batch_file)
        lines.append("## Batched Decode (Aggregate Throughput)")
        lines.append("")
        lines.append("| Concurrency | Vexel tok/s | llama.cpp tok/s |")
        lines.append("|------------|------------|----------------|")

        grouped = defaultdict(lambda: defaultdict(list))
        for r in runs:
            grouped[r["concurrency"]][r["engine"]].append(r)

        for conc in sorted(grouped):
            vexel_runs = grouped[conc].get("vexel", [])
            llama_runs = grouped[conc].get("llama.cpp", [])
            v_tok = mean([r["aggregate_tok_s"] for r in vexel_runs if r["aggregate_tok_s"] > 0]) if vexel_runs else 0
            l_tok = mean([r["aggregate_tok_s"] for r in llama_runs if r["aggregate_tok_s"] > 0]) if llama_runs else 0
            v_str = f"{v_tok:.1f}" if v_tok > 0 else "—"
            l_str = f"{l_tok:.1f}" if l_tok > 0 else "—"
            lines.append(f"| {conc} | {v_str} | {l_str} |")

        lines.append("")

    # --- Context Scaling ---
    ctx_file = rd / "context_scaling.jsonl"
    if ctx_file.exists():
        runs = load_jsonl(ctx_file)
        lines.append("## Context Scaling (Decode Throughput)")
        lines.append("")
        lines.append("| Context | Vexel tok/s | Vexel Degrad. | llama.cpp tok/s | llama.cpp Degrad. |")
        lines.append("|---------|------------|--------------|----------------|------------------|")

        grouped = defaultdict(lambda: defaultdict(list))
        for r in runs:
            grouped[r["context_length"]][r["engine"]].append(r)

        vexel_baseline = 0
        llama_baseline = 0

        for ctx in sorted(grouped):
            vexel_runs = grouped[ctx].get("vexel", [])
            llama_runs = grouped[ctx].get("llama.cpp", [])

            v_dec = mean([r["decode_tok_s"] for r in vexel_runs if r["decode_tok_s"] > 0]) if vexel_runs else 0
            l_dec = mean([r["decode_tok_s"] for r in llama_runs if r["decode_tok_s"] > 0]) if llama_runs else 0

            if ctx == 16:
                vexel_baseline = v_dec
                llama_baseline = l_dec

            v_deg = f"{(1 - v_dec / vexel_baseline) * 100:.1f}%" if vexel_baseline > 0 and v_dec > 0 else "—"
            l_deg = f"{(1 - l_dec / llama_baseline) * 100:.1f}%" if llama_baseline > 0 and l_dec > 0 else "—"
            v_str = f"{v_dec:.1f}" if v_dec > 0 else "—"
            l_str = f"{l_dec:.1f}" if l_dec > 0 else "—"

            lines.append(f"| {ctx} | {v_str} | {v_deg} | {l_str} | {l_deg} |")

        lines.append("")

    # Write report
    report_path = rd / "report.md"
    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    print(report_text)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 report.py <results_dir>")
        sys.exit(1)
    generate_report(sys.argv[1])
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/lib/report.py
git commit -m "bench: add comparison report generator"
```

---

### Task 8: Make all scripts executable and do end-to-end smoke test

**Files:**
- Modify: `benchmarks/full_comparison.sh` (already created)

- [ ] **Step 1: Make all scripts executable**

```bash
chmod +x benchmarks/full_comparison.sh benchmarks/lib/*.sh benchmarks/lib/report.py
```

- [ ] **Step 2: Run smoke test with decode suite only**

```bash
cd benchmarks && ./full_comparison.sh decode
```

Expected: Models are found or downloaded, both engines run, JSONL files created in `results/YYYY-MM-DD/`, report printed to stdout.

- [ ] **Step 3: Verify report.md was generated**

```bash
cat benchmarks/results/$(date +%Y-%m-%d)/report.md
```

Expected: Markdown table with decode tok/s for both engines and both models.

- [ ] **Step 4: Fix any issues found during smoke test**

- [ ] **Step 5: Commit any fixes**

```bash
git add benchmarks/
git commit -m "bench: finalize benchmark suite and verify end-to-end"
```
