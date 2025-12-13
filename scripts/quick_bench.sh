#!/bin/bash
set -e

# Build
go build -o vexel -tags metal ./inference/cmd/vexel

# Input (~200 tokens)
PROMPT=$(for i in {1..50}; do echo -n "benchmark "; done)

# Run Vexel
echo "Running Vexel..."
VEXEL_FA2_MIN_SEQ=16 ./vexel -model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -gpu -completion -max-tokens 64 -temp 0 <<<"$PROMPT" > vexel.log

# Run llama.cpp
echo "Running llama.cpp..."
llama-cli -m models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -p "$PROMPT" -n 64 --temp 0 -no-cnv > llama.log 2>&1

# Extract Stats
echo "=== RESULTS ==="
echo "Vexel:"
grep "tok/s" vexel.log || echo "No Vexel stats found"
echo "llama.cpp:"
grep "tokens per second" llama.log || echo "No llama.cpp stats found"
