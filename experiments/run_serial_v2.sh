#!/bin/bash
# Serial LoRA training + evaluation pipeline (fixed - no pipe that can SIGPIPE)
set -e
cd /Users/qeetbastudio/projects/vexel

VEXEL="go run -tags metal ./inference/cmd/vexel"
MODEL="/Users/qeetbastudio/projects/vexel/benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
EXPDIR="experiments"

echo "Model: Mistral 7B Instruct v0.3 (Q4_K_M)"
echo ""

###############################################################################
# PHASE 1: Train all three scenarios (NO pipes that could SIGPIPE)
###############################################################################

scenarios=("personal-kb" "domain-jargon" "tool-routing")
examples=(49 36 57)

for i in "${!scenarios[@]}"; do
  name="${scenarios[$i]}"
  n="${examples[$i]}"
  adapter="$EXPDIR/$name/adapter"
  logfile="$EXPDIR/$name/training.log"

  echo "============================================"
  echo "TRAINING: $name ($n examples, 30 epochs)"
  echo "============================================"

  rm -rf "$adapter"

  # Write ALL output to logfile, don't pipe through grep/head
  $VEXEL --model "$MODEL" train \
    --data "$EXPDIR/$name/train.jsonl" \
    --output "$adapter" \
    --rank 16 --alpha 32 --lr 1e-4 --epochs 30 > "$logfile" 2>&1

  # Show summary from logfile after training completes
  echo "  First epoch:"
  grep "epoch 1/30  step 1/" "$logfile" || true
  echo "  Last epoch:"
  tail -3 "$logfile"

  if [ -f "$adapter/adapter_model.safetensors" ]; then
    echo "  -> Adapter saved successfully"
  else
    echo "  -> ERROR: No adapter produced!"
    echo "  Last 10 lines of log:"
    tail -10 "$logfile"
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

  echo "  Results -> $outfile"
  echo ""
done

echo "============================================"
echo "ALL DONE"
echo "============================================"
