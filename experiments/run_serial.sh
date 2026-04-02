#!/bin/bash
# Serial LoRA training + evaluation pipeline
# All three scenarios on Mistral 7B, run one at a time
set -e
cd /Users/qeetbastudio/projects/vexel

VEXEL="go run -tags metal ./inference/cmd/vexel"
MODEL="/Users/qeetbastudio/projects/vexel/benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
EXPDIR="experiments"

echo "Model: Mistral 7B Instruct v0.3 (Q4_K_M)"
echo "Memory: 128GB available"
echo ""

###############################################################################
# PHASE 1: Train all three scenarios
###############################################################################

scenarios=("personal-kb" "domain-jargon" "tool-routing")
examples=(49 36 57)

for i in "${!scenarios[@]}"; do
  name="${scenarios[$i]}"
  n="${examples[$i]}"
  adapter="$EXPDIR/$name/adapter"

  echo "============================================"
  echo "TRAINING: $name ($n examples, 30 epochs)"
  echo "============================================"

  rm -rf "$adapter"
  $VEXEL --model "$MODEL" train \
    --data "$EXPDIR/$name/train.jsonl" \
    --output "$adapter" \
    --rank 16 --alpha 32 --lr 1e-4 --epochs 30 2>&1 | \
    grep -E "^epoch (1|5|10|15|20|25|30)/" | head -30

  if [ -f "$adapter/adapter_model.safetensors" ]; then
    echo "  -> Adapter saved successfully"
  else
    echo "  -> ERROR: No adapter produced!"
  fi
  echo ""
done

###############################################################################
# PHASE 2: Evaluate all three scenarios
###############################################################################

echo "============================================"
echo "EVALUATION"
echo "============================================"
echo ""

for name in "${scenarios[@]}"; do
  adapter="$EXPDIR/$name/adapter"
  questions="$EXPDIR/$name/eval_questions.txt"
  outfile="$EXPDIR/$name/eval_round0.txt"

  echo "--- $name ---"
  echo "Evaluation: $name (Mistral 7B)" > "$outfile"
  echo "====================================" >> "$outfile"
  echo "" >> "$outfile"

  while IFS= read -r question || [ -n "$question" ]; do
    [[ -z "$question" || "$question" == \#* ]] && continue

    echo "  Q: $question"
    echo "Q: $question" >> "$outfile"

    # Base
    base=$($VEXEL --model "$MODEL" generate --prompt "$question" --max-tokens 80 2>/dev/null || echo "(error)")
    echo "BASE: $base" >> "$outfile"

    # LoRA
    if [ -d "$adapter" ]; then
      lora=$($VEXEL --model "$MODEL" --lora "$adapter" generate --prompt "$question" --max-tokens 80 2>/dev/null || echo "(error)")
      echo "LORA: $lora" >> "$outfile"
    fi

    echo "" >> "$outfile"
  done < "$questions"

  echo "  Results saved to $outfile"
  echo ""
done

echo "============================================"
echo "ALL DONE"
echo "============================================"
