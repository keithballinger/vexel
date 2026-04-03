#!/bin/bash
cd /Users/qeetbastudio/projects/vexel
nohup go run -tags metal ./inference/cmd/vexel \
  --model benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  train --data experiments/personal-kb/train_augmented.jsonl \
  --output experiments/personal-kb/adapter_augmented \
  --rank 16 --alpha 32 --lr 5e-5 --momentum 0.9 --epochs 20 \
  > experiments/personal-kb/training_augmented.log 2>&1 &
echo "PID: $!"
