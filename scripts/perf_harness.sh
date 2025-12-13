#!/usr/bin/env bash
# Compare Vexel and llama.cpp throughput on a fixed prompt set.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf}"
VEXEL_BIN="${VEXEL_BIN:-./vexel_metal}"
LLAMA_BIN="${LLAMA_BIN:-llama-cli}"
OUT_DIR="${OUT_DIR:-perf_reports}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$VEXEL_BIN" ]]; then
  echo "Vexel binary not executable: $VEXEL_BIN" >&2
  exit 1
fi
if ! command -v "$LLAMA_BIN" >/dev/null 2>&1; then
  echo "llama.cpp binary not found: $LLAMA_BIN" >&2
  exit 1
fi

PROMPTS=()
while IFS= read -r line; do
  PROMPTS+=("$line")
done <<'EOF'
Hello!
Describe the benefits of unit testing in Go in three concise sentences.
Write a brief summary of how rotary positional embeddings work in transformers.
Explain how Flash Attention 2 differs from Flash Attention 1 and why it is faster.
Provide a short Go HTTP handler that parses JSON input and returns a JSON response.
EOF
TOKENS=(50 64 128 96 192)

mkdir -p "$OUT_DIR"
report="$OUT_DIR/report-$(date +%Y%m%d-%H%M%S).md"

echo "# Vexel vs llama.cpp (TinyLlama Q4_0)" >"$report"
echo "" >>"$report"
echo "| Prompt | Max Tokens | Vexel Prefill | Vexel Decode | llama.cpp Prompt Eval | llama.cpp Decode |" >>"$report"
echo "|---|---|---|---|---|---|" >>"$report"

run_case() {
  local prompt="$1"
  local tokens="$2"

  parse_tokps_pair() {
    # Return first two occurrences of "<number> tokens per second" separated by space
    perl -nle 'if(/([0-9]+(?:\.[0-9]+)?) tokens per second/){push @a,$1} END{print join(" ", @a)}' "$1"
  }

  local v_log l_log
  v_log=$(mktemp)
  l_log=$(mktemp)

  # Vexel
  "$VEXEL_BIN" -model "$MODEL_PATH" -gpu -completion -max-tokens "$tokens" <<<"$prompt" >"$v_log"
  local prefill decode
  prefill="N/A"; decode="N/A"
  local parsed
  parsed=$(sed -n 's/.*prefill: \([0-9.]*\) tok\/s | decode: \([0-9.]*\) tok\/s.*/\1 \2/p' "$v_log" || true)
  if [[ -n "$parsed" ]]; then
    read -r p d <<<"$parsed"
    prefill="$p"
    decode="$d"
  fi

  # llama.cpp
  LLAMA_LOG_COLORS=0 "$LLAMA_BIN" -m "$MODEL_PATH" -p "$prompt" -n "$tokens" --no-warmup >"$l_log" 2>&1
  local llama_prompt llama_decode
  local tokps
  tokps=$(parse_tokps_pair "$l_log")
  llama_prompt=$(echo "$tokps" | awk '{print $1}')
  llama_decode=$(echo "$tokps" | awk '{print $2}')
  [[ -z "$llama_prompt" ]] && llama_prompt="N/A"
  [[ -z "$llama_decode" ]] && llama_decode="N/A"

  echo "| ${prompt//|/\\|} | $tokens | $prefill tok/s | $decode tok/s | $llama_prompt tok/s | $llama_decode tok/s |" >>"$report"

  echo "---- Vexel log ----" >>"$report"
  cat "$v_log" >>"$report"
  echo -e "\n---- llama.cpp log ----" >>"$report"
  cat "$l_log" >>"$report"
  echo -e "\n" >>"$report"

  rm -f "$v_log" "$l_log"
}

for i in "${!PROMPTS[@]}"; do
  run_case "${PROMPTS[$i]}" "${TOKENS[$i]}"
done

echo "Report written to $report"
