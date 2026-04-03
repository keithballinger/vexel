#!/bin/bash
# Full pipeline: train + eval all 3 scenarios on Mistral 7B
# With momentum + chat template fix + GPU attention weights
set -e
cd /Users/qeetbastudio/projects/vexel

MODEL="benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
VEXEL="go run -tags metal ./inference/cmd/vexel"
EVAL="./experiments/eval_batch"

# Rebuild eval binary with latest code
echo "[$(date)] Building eval binary..."
go build -tags metal -o experiments/eval_batch experiments/eval_batch.go

scenarios=("personal-kb" "domain-jargon" "tool-routing")
examples=(49 36 57)

for i in "${!scenarios[@]}"; do
  name="${scenarios[$i]}"
  n="${examples[$i]}"

  echo ""
  echo "============================================"
  echo "[$(date)] TRAINING: $name ($n examples)"
  echo "============================================"

  rm -rf "experiments/$name/adapter"
  $VEXEL --model "$MODEL" train \
    --data "experiments/$name/train.jsonl" \
    --output "experiments/$name/adapter" \
    --rank 16 --alpha 32 --lr 5e-5 --momentum 0.9 --epochs 50 \
    > "experiments/$name/training.log" 2>&1

  echo "[$(date)] Training done. Loss trend:"
  grep "step 1/" "experiments/$name/training.log" | awk -F'loss=' '{printf "e%d:%.3f ", NR, $2}' | fold -w 80

  if [ ! -f "experiments/$name/adapter/adapter_model.safetensors" ]; then
    echo "ERROR: No adapter produced!"
    tail -5 "experiments/$name/training.log"
    continue
  fi

  echo ""
  echo "[$(date)] EVALUATING: $name"
  $EVAL --model "$MODEL" \
    --adapter "experiments/$name/adapter" \
    --questions "experiments/$name/eval_questions.txt" \
    --output "experiments/$name/eval_round0.txt" \
    --max-tokens 80 2>&1 | grep -v "done$"

  echo "[$(date)] Results saved to experiments/$name/eval_round0.txt"
done

echo ""
echo "============================================"
echo "[$(date)] ALL DONE"
echo "============================================"
echo ""
echo "Results summary:"
for name in "${scenarios[@]}"; do
  echo ""
  echo "========== $name =========="
  cat "experiments/$name/eval_round0.txt" 2>/dev/null
done
