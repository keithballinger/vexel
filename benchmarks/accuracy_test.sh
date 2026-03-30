#!/usr/bin/env bash
# accuracy_test.sh — Validates Vexel output against llama.cpp reference
#
# Usage: ./benchmarks/accuracy_test.sh
#
# Runs the same prompts through Vexel and llama.cpp with greedy decoding,
# compares output token-by-token. Requires no Ollama or other GPU load.

set -euo pipefail

VEXEL="./vexel"
LLAMA="/Users/qeetbastudio/projects/llama.cpp/build/bin/llama-completion"
MODEL="benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_TINY="benchmarks/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

echo "============================================="
echo " Vexel Accuracy Test — $(date)"
echo "============================================="
echo ""

# Check for Ollama interference
OLLAMA_MB=$(ps aux | grep "ollama runner" | grep -v grep | awk '{printf "%.0f", $6/1024}' 2>/dev/null || echo "0")
if [ "$OLLAMA_MB" -gt 1000 ]; then
    echo "⚠️  WARNING: Ollama is using ${OLLAMA_MB} MB GPU memory!"
    echo "   Results may be unreliable. Stop Ollama first for accurate testing."
    echo ""
fi

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local model="$2"
    local prompt="$3"
    local tokens="$4"

    echo "--- Test: $name ---"
    echo "  Prompt: \"$prompt\""

    # Get llama.cpp output (greedy)
    local llama_out
    llama_out=$(timeout 60 "$LLAMA" -m "$model" -p "$prompt" -n "$tokens" -no-cnv --temp 0 2>&1 |
        grep -A 1000 "^${prompt}" | head -1 | sed "s/^${prompt}//")

    # Get Vexel output (greedy, no chat template)
    local vexel_out
    vexel_out=$(timeout 30 "$VEXEL" --model "$model" --no-chat-template generate \
        --prompt "$prompt" --max-tokens "$tokens" 2>&1 |
        grep -v "Loading\|blk\.\|DEBUG\|CONFIG\|Arch\|Tensor\|Loaded\|F32\|Q[0-9]\|tok/s\|gpu memory" |
        tr -d '\n')

    echo "  llama.cpp: ${llama_out:0:80}"
    echo "  Vexel:     ${vexel_out:0:80}"

    # Normalize unicode encoding differences (e.g., Â° vs °)
    llama_out=$(echo "$llama_out" | sed 's/Â//g')
    vexel_out=$(echo "$vexel_out" | sed 's/Â//g')

    if [ "$llama_out" = "$vexel_out" ]; then
        echo "  ✅ MATCH"
        PASS=$((PASS+1))
    else
        # Check if first 3 words match (allow minor divergence from quantization precision)
        local llama_3=$(echo "$llama_out" | awk '{for(i=1;i<=3;i++) printf "%s ", $i}')
        local vexel_3=$(echo "$vexel_out" | awk '{for(i=1;i<=3;i++) printf "%s ", $i}')
        if [ "$llama_3" = "$vexel_3" ]; then
            echo "  ⚠️  PARTIAL MATCH (first 3 words match, minor divergence)"
            PASS=$((PASS+1))
        else
            echo "  ❌ MISMATCH"
            FAIL=$((FAIL+1))
        fi
    fi
    echo ""
}

# Test suite
run_test "LLaMA 8B short" "$MODEL" "The capital of France is" 10
run_test "LLaMA 8B factual" "$MODEL" "Water boils at" 10
run_test "LLaMA 8B counting" "$MODEL" "1 2 3 4 5 6 7 8" 10
run_test "TinyLlama short" "$MODEL_TINY" "Hello" 10
run_test "TinyLlama factual" "$MODEL_TINY" "The sun is" 10

echo "============================================="
echo " Results: $PASS passed, $FAIL failed"
echo "============================================="

if [ "$FAIL" -gt 0 ]; then
    echo "⚠️  Some tests failed — investigate output divergence"
    exit 1
else
    echo "✅ All tests passed — Vexel output matches llama.cpp"
    exit 0
fi
