#!/bin/bash
# Full LoRA training + evaluation pipeline
# Runs all three scenarios, evaluates, scales data, and generates report
set -e
cd /Users/qeetbastudio/projects/vexel

VEXEL="go run -tags metal ./inference/cmd/vexel"
QWEN="/Users/qeetbastudio/projects/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
TINYLLAMA="/Users/qeetbastudio/projects/vexel/benchmarks/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
PHI3="/Users/qeetbastudio/projects/vexel/benchmarks/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf"

EXPDIR="experiments"

echo "============================================"
echo "PHASE 1: Training all three scenarios"
echo "============================================"

# 1. Personal KB (Qwen 0.5B) - already done, skip if adapter exists
if [ ! -f "$EXPDIR/personal-kb/adapter/adapter_model.safetensors" ]; then
  echo "[1/3] Training Personal KB (Qwen 0.5B, 49 examples)..."
  $VEXEL --model "$QWEN" train \
    --data "$EXPDIR/personal-kb/train.jsonl" \
    --output "$EXPDIR/personal-kb/adapter" \
    --rank 16 --alpha 32 --lr 5e-4 --epochs 30 2>&1 | tail -3
else
  echo "[1/3] Personal KB adapter already exists, skipping."
fi

# 2. Domain Jargon (TinyLlama 1.1B)
echo "[2/3] Training Domain Jargon (TinyLlama 1.1B, 36 examples)..."
$VEXEL --model "$TINYLLAMA" train \
  --data "$EXPDIR/domain-jargon/train.jsonl" \
  --output "$EXPDIR/domain-jargon/adapter" \
  --rank 16 --alpha 32 --lr 5e-4 --epochs 30 2>&1 | tail -3

# 3. Tool Routing (Phi-3 Mini)
echo "[3/3] Training Tool Routing (Phi-3 Mini, 57 examples)..."
$VEXEL --model "$PHI3" train \
  --data "$EXPDIR/tool-routing/train.jsonl" \
  --output "$EXPDIR/tool-routing/adapter" \
  --rank 16 --alpha 32 --lr 5e-4 --epochs 30 2>&1 | tail -3

echo ""
echo "============================================"
echo "PHASE 2: Evaluation (Round 0)"
echo "============================================"

eval_scenario() {
  local name="$1"
  local model="$2"
  local adapter="$3"
  local questions="$4"
  local max_tokens="${5:-80}"
  local outfile="$6"

  echo "--- Evaluating: $name ---" >> "$outfile"
  echo "" >> "$outfile"

  while IFS= read -r question || [ -n "$question" ]; do
    [[ -z "$question" || "$question" == \#* ]] && continue

    echo "Q: $question" >> "$outfile"

    # Base response
    base=$($VEXEL --model "$model" generate --prompt "$question" --max-tokens "$max_tokens" 2>/dev/null || echo "(error)")
    echo "BASE: $base" >> "$outfile"

    # Adapted response
    if [ -d "$adapter" ]; then
      adapted=$($VEXEL --model "$model" --lora "$adapter" generate --prompt "$question" --max-tokens "$max_tokens" 2>/dev/null || echo "(error)")
      echo "LORA: $adapted" >> "$outfile"
    fi

    echo "" >> "$outfile"
  done < "$questions"
}

RESULTS="$EXPDIR/results_round0.txt"
echo "Round 0 Evaluation Results" > "$RESULTS"
echo "=========================" >> "$RESULTS"
echo "" >> "$RESULTS"

echo "Evaluating Personal KB..."
eval_scenario "Personal KB (Qwen 0.5B)" "$QWEN" \
  "$EXPDIR/personal-kb/adapter" \
  "$EXPDIR/personal-kb/eval_questions.txt" 80 "$RESULTS"

echo "Evaluating Domain Jargon..."
eval_scenario "Domain Jargon (TinyLlama 1.1B)" "$TINYLLAMA" \
  "$EXPDIR/domain-jargon/adapter" \
  "$EXPDIR/domain-jargon/eval_questions.txt" 100 "$RESULTS"

echo "Evaluating Tool Routing..."
eval_scenario "Tool Routing (Phi-3 Mini)" "$PHI3" \
  "$EXPDIR/tool-routing/adapter" \
  "$EXPDIR/tool-routing/eval_questions.txt" 80 "$RESULTS"

echo ""
echo "Results written to $RESULTS"
echo ""
echo "============================================"
echo "DONE - All training and evaluation complete"
echo "============================================"
cat "$RESULTS"
