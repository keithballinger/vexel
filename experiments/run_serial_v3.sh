#!/bin/bash
# Serial LoRA training + evaluation - Gemma 2 2B
set -e
cd /Users/qeetbastudio/projects/vexel

VEXEL="go run -tags metal ./inference/cmd/vexel"
MODEL="/Users/qeetbastudio/projects/vexel/benchmarks/models/gemma2-2b/gemma-2-2b-it-Q4_K_M.gguf"
EXPDIR="experiments"

echo "Model: Gemma 2 2B IT (Q4_K_M, ~8GB FP32)"
echo ""

scenarios=("personal-kb" "domain-jargon" "tool-routing")
examples=(49 36 57)

for i in "${!scenarios[@]}"; do
  name="${scenarios[$i]}"
  n="${examples[$i]}"
  adapter="$EXPDIR/$name/adapter"
  logfile="$EXPDIR/$name/training.log"

  echo "============================================"
  echo "TRAINING: $name ($n examples, 30 epochs)"
  echo "Start: $(date)"
  echo "============================================"

  rm -rf "$adapter"

  $VEXEL --model "$MODEL" train \
    --data "$EXPDIR/$name/train.jsonl" \
    --output "$adapter" \
    --rank 16 --alpha 32 --lr 1e-4 --epochs 30 > "$logfile" 2>&1

  echo "  End: $(date)"
  head -1 "$logfile"
  tail -1 "$logfile"

  if [ -f "$adapter/adapter_model.safetensors" ]; then
    echo "  -> Adapter saved OK"
  else
    echo "  -> FAILED"
    tail -5 "$logfile"
  fi
  echo ""
done

echo "============================================"
echo "EVALUATION"
echo "============================================"

for name in "${scenarios[@]}"; do
  adapter="$EXPDIR/$name/adapter"
  questions="$EXPDIR/$name/eval_questions.txt"
  outfile="$EXPDIR/$name/eval_round0.txt"

  if [ ! -d "$adapter" ]; then
    echo "  Skipping $name (no adapter)"
    continue
  fi

  echo "--- $name ---"
  echo "Evaluation: $name (Gemma 2 2B)" > "$outfile"
  echo "" >> "$outfile"

  while IFS= read -r question || [ -n "$question" ]; do
    [[ -z "$question" || "$question" == \#* ]] && continue
    echo "  Q: $question"
    echo "Q: $question" >> "$outfile"

    base=$($VEXEL --model "$MODEL" generate --prompt "$question" --max-tokens 80 2>/dev/null || echo "(error)")
    echo "BASE: $base" >> "$outfile"

    lora=$($VEXEL --model "$MODEL" --lora "$adapter" generate --prompt "$question" --max-tokens 80 2>/dev/null || echo "(error)")
    echo "LORA: $lora" >> "$outfile"
    echo "" >> "$outfile"
  done < "$questions"

  echo "  -> $outfile"
  echo ""
done

echo "ALL DONE at $(date)"
