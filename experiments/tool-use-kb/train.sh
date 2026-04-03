#!/bin/bash
# Train tool-use-kb LoRA adapter on Mistral 7B
# Teaches the model to emit <tool_call>...</tool_call> instead of memorizing facts
cd /Users/qeetbastudio/projects/vexel
nohup go run -tags metal ./inference/cmd/vexel \
  --model benchmarks/models/mistral-7b/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  train --data experiments/tool-use-kb/train.jsonl \
  --output experiments/tool-use-kb/adapter \
  --rank 16 --alpha 32 --lr 5e-5 --momentum 0.9 --epochs 30 \
  > experiments/tool-use-kb/training.log 2>&1 &
echo "PID: $!"
