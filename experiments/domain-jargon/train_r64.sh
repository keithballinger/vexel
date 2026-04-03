#!/bin/bash
cd /Users/qeetbastudio/projects/vexel
nohup go run -tags metal ./inference/cmd/vexel \
  --model benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  train --data experiments/domain-jargon/train.jsonl \
  --output experiments/domain-jargon/adapter_r64 \
  --rank 64 --alpha 128 --lr 5e-5 --momentum 0.9 --epochs 50 \
  > experiments/domain-jargon/training_r64.log 2>&1 &
echo "PID: $!"
