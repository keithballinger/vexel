#!/usr/bin/env bash
# Compare Vexel and llama.cpp throughput on a fixed prompt set.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf}"
VEXEL_BIN="${VEXEL_BIN:-./vexel}"
LLAMA_BIN="${LLAMA_BIN:-llama-cli}"
OUT_DIR="${OUT_DIR:-perf_reports}"

export VEXEL_FA2_MIN_SEQ="${VEXEL_FA2_MIN_SEQ:-16}"
PROMPT_MODE="${PROMPT_MODE:-completion}" # completion|chat

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
EOF
TOKENS=(50 64)

mkdir -p "$OUT_DIR"
report="$OUT_DIR/report-$(date +%Y%m%d-%H%M%S).md"

echo "# Vexel vs llama.cpp (TinyLlama Q4_0, mode: $PROMPT_MODE)" >"$report"
echo "" >>"$report"
echo "| Prompt | Max Tokens | Vexel Prefill | Vexel Decode | llama.cpp Prompt Eval | llama.cpp Decode | Similarity |" >>"$report"
echo "|---|---|---|---|---|---|---|" >>"$report"

run_case() {
  local prompt="$1"
  local tokens="$2"

  local -a vexel_args llama_args
  vexel_args=("$VEXEL_BIN" -model "$MODEL_PATH" -gpu -max-tokens "$tokens" -temp 0 -top-k 1 -top-p 0)
  llama_args=("$LLAMA_BIN" -m "$MODEL_PATH" -p "$prompt" -n "$tokens" --no-warmup --temp 0 --top-k 1 --top-p 0 --seed 1)

  case "$PROMPT_MODE" in
    completion)
      vexel_args+=(-completion)
      llama_args+=(-no-cnv)
      ;;
    chat)
      llama_args+=(-cnv -sys "You are a helpful assistant.")
      ;;
    *)
      echo "Unknown PROMPT_MODE: $PROMPT_MODE (expected completion|chat)" >&2
      exit 1
      ;;
  esac

  parse_tokps_pair() {
    # Return first two occurrences of "<number> tokens per second" separated by space
    perl -nle 'if(/([0-9]+(?:\.[0-9]+)?) tokens per second/){push @a,$1} END{print join(" ", @a)}' "$1"
  }

  local v_log l_log
  v_log=$(mktemp)
  l_log=$(mktemp)

  # Vexel
  "${vexel_args[@]}" <<<"$prompt" >"$v_log"
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
  LLAMA_LOG_COLORS=0 "${llama_args[@]}" >"$l_log" 2>&1
  local llama_prompt llama_decode
  local tokps
  tokps=$(parse_tokps_pair "$l_log")
  llama_prompt=$(echo "$tokps" | awk '{print $1}')
  llama_decode=$(echo "$tokps" | awk '{print $2}')
  [[ -z "$llama_prompt" ]] && llama_prompt="N/A"
  [[ -z "$llama_decode" ]] && llama_decode="N/A"

  # Correctness: compare generated text similarity
  local v_text l_text similarity
  similarity=$(V_LOG="$v_log" L_LOG="$l_log" PROMPT="$prompt" python - <<'PY'
import difflib
import os
import re

v_log_path = os.environ["V_LOG"]
l_log_path = os.environ["L_LOG"]
prompt = os.environ["PROMPT"]

with open(v_log_path, "r", encoding="utf-8", errors="replace") as f:
    v_log = f.read()
with open(l_log_path, "r", encoding="utf-8", errors="replace") as f:
    l_log = f.read()

def extract_vexel_text(log: str) -> str:
    # Capture everything printed between the first and second REPL prompts.
    m = re.search(r"^>> (.*?)(?:\n>> |\Z)", log, re.M | re.S)
    if not m:
        return ""

    text = m.group(1)

    # Drop perf line and noisy debug/profile lines.
    text = re.sub(r"\n\[[0-9]+ tokens \|.*?\]\n?", "\n", text, flags=re.S)
    text = re.sub(r"^\[(DEBUG|PROFILE)\].*\n?", "", text, flags=re.M)

    return text.strip()

def extract_llama_text(log: str, prompt: str) -> str:
    # Extract the output block that appears right before perf lines.
    # This is robust to metadata logs that may contain "<|assistant|>" in the chat template.
    m = re.search(r"\*{29}.*?\*{29}\n(.*?)\n\ncommon_perf_print:", log, re.S)
    if not m:
        return ""

    block = m.group(1).lstrip()

    # Conversation mode: extract assistant text within the output block.
    m = re.search(r"<\|assistant\|>(.*)", block, re.S)
    if m:
        text = re.sub(r"> EOF by user.*", "", m.group(1), flags=re.S)
        return text.strip()

    # Completion mode (-no-cnv): llama.cpp echoes the prompt, then generates continuation.
    if block.startswith(prompt):
        block = block[len(prompt):].lstrip()
    return block.strip()

v_text = extract_vexel_text(v_log)
l_text = extract_llama_text(l_log, prompt)

if not v_text or not l_text:
    print("N/A")
else:
    ratio = difflib.SequenceMatcher(None, v_text[:400], l_text[:400]).ratio()
    print(f"{ratio:.3f}")
PY
)

  echo "| ${prompt//|/\\|} | $tokens | $prefill tok/s | $decode tok/s | $llama_prompt tok/s | $llama_decode tok/s | $similarity |" >>"$report"

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
