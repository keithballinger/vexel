#!/bin/bash
set -e
cd /Users/qeetbastudio/projects/vexel
MODEL="benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf"

for scenario in personal-kb domain-jargon tool-routing; do
  echo "=== Evaluating: $scenario ==="
  ./experiments/eval_batch \
    --model "$MODEL" \
    --adapter "experiments/$scenario/adapter" \
    --questions "experiments/$scenario/eval_questions.txt" \
    --output "experiments/$scenario/eval_round0.txt" \
    --max-tokens 80
  echo ""
done

echo "=== ALL EVALS DONE ==="
echo ""
for scenario in personal-kb domain-jargon tool-routing; do
  echo "========== $scenario =========="
  cat "experiments/$scenario/eval_round0.txt"
  echo ""
done
