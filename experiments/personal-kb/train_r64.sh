#!/bin/bash
cd /Users/qeetbastudio/projects/vexel
nohup go run -tags metal ./inference/cmd/vexel \
  --model benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  train --data experiments/personal-kb/train.jsonl \
  --output experiments/personal-kb/adapter_r64 \
  --rank 64 --alpha 128 --lr 5e-5 --momentum 0.9 --epochs 50 \
  > experiments/personal-kb/training_r64.log 2>&1 &
echo "PID: $!"
