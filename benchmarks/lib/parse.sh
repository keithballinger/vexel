#!/usr/bin/env bash
# parse.sh — Output parsing helpers for the Vexel benchmark suite.
# Sourced by benchmark scripts; not intended to be run standalone.
#
# Provides functions to run Vexel and llama.cpp, parse their output,
# and emit structured JSONL results.

set -euo pipefail

###############################################################################
# generate_prompt <target_tokens>
#   Generate a synthetic prompt of approximately <target_tokens> tokens.
#   Uses repeated common English words (~1 token per word for most tokenizers).
#   Prints the prompt to stdout.
###############################################################################
generate_prompt() {
    local target_tokens="${1:?Usage: generate_prompt <target_tokens>}"
    local words=("The" "quick" "brown" "fox" "jumps" "over" "the" "lazy" "dog"
                 "and" "then" "runs" "back" "across" "the" "wide" "green" "field"
                 "while" "the" "sun" "shines" "brightly" "in" "the" "clear" "blue" "sky"
                 "above" "the" "tall" "mountain" "range")
    local num_words=${#words[@]}
    local prompt=""
    for ((i = 0; i < target_tokens; i++)); do
        if [[ $i -gt 0 ]]; then
            prompt+=" "
        fi
        prompt+="${words[$((i % num_words))]}"
    done
    echo "$prompt"
}

###############################################################################
# run_vexel_generate <model> <prompt> <max_tokens> [extra_flags...]
#   Run Vexel generate with --verbose and parse throughput numbers.
#   Prints to stdout: decode_tok_s prefill_tok_s
###############################################################################
run_vexel_generate() {
    local model="${1:?Usage: run_vexel_generate <model> <prompt> <max_tokens> [flags...]}"
    local prompt="${2:?}"
    local max_tokens="${3:?}"
    shift 3
    local extra_flags=("$@")

    local output
    output=$("$VEXEL_BIN" --model "$model" --verbose generate \
        --prompt "$prompt" --max-tokens "$max_tokens" \
        "${extra_flags[@]}" 2>&1) || true

    local decode_tok_s prefill_tok_s
    # Parse: [N tokens | prefill: X.X tok/s | decode: Y.Y tok/s]
    decode_tok_s=$(echo "$output" | grep -oE 'decode: [0-9]+\.?[0-9]* tok/s' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    prefill_tok_s=$(echo "$output" | grep -oE 'prefill: [0-9]+\.?[0-9]* tok/s' | grep -oE '[0-9]+\.?[0-9]*' | head -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0}"
}

###############################################################################
# run_vexel_medusa <model> <prompt> <max_tokens> [extra_flags...]
#   Run Vexel with --medusa and parse throughput + speculative stats.
#   Prints to stdout: decode_tok_s prefill_tok_s acceptance_pct speedup
###############################################################################
run_vexel_medusa() {
    local model="${1:?Usage: run_vexel_medusa <model> <prompt> <max_tokens> [flags...]}"
    local prompt="${2:?}"
    local max_tokens="${3:?}"
    shift 3
    local extra_flags=("$@")

    local output
    output=$("$VEXEL_BIN" --model "$model" --verbose --medusa generate \
        --prompt "$prompt" --max-tokens "$max_tokens" \
        "${extra_flags[@]}" 2>&1) || true

    local decode_tok_s prefill_tok_s acceptance_pct speedup
    decode_tok_s=$(echo "$output" | grep -oE 'decode: [0-9]+\.?[0-9]* tok/s' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    prefill_tok_s=$(echo "$output" | grep -oE 'prefill: [0-9]+\.?[0-9]* tok/s' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    # Parse: [speculative: acceptance=Z.Z% speedup=W.Wx generated=G accepted=A]
    acceptance_pct=$(echo "$output" | grep -oE 'acceptance=[0-9]+\.?[0-9]*%' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    speedup=$(echo "$output" | grep -oE 'speedup=[0-9]+\.?[0-9]*x' | grep -oE '[0-9]+\.?[0-9]*' | head -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0} ${acceptance_pct:-0} ${speedup:-0}"
}

###############################################################################
# run_vexel_draft <model> <draft_model> <prompt> <max_tokens> [extra_flags...]
#   Run Vexel with --draft-model and parse throughput + speculative stats.
#   Prints to stdout: decode_tok_s prefill_tok_s acceptance_pct speedup
###############################################################################
run_vexel_draft() {
    local model="${1:?Usage: run_vexel_draft <model> <draft_model> <prompt> <max_tokens> [flags...]}"
    local draft_model="${2:?}"
    local prompt="${3:?}"
    local max_tokens="${4:?}"
    shift 4
    local extra_flags=("$@")

    local output
    output=$("$VEXEL_BIN" --model "$model" --draft-model "$draft_model" --verbose generate \
        --prompt "$prompt" --max-tokens "$max_tokens" \
        "${extra_flags[@]}" 2>&1) || true

    local decode_tok_s prefill_tok_s acceptance_pct speedup
    decode_tok_s=$(echo "$output" | grep -oE 'decode: [0-9]+\.?[0-9]* tok/s' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    prefill_tok_s=$(echo "$output" | grep -oE 'prefill: [0-9]+\.?[0-9]* tok/s' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    acceptance_pct=$(echo "$output" | grep -oE 'acceptance=[0-9]+\.?[0-9]*%' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    speedup=$(echo "$output" | grep -oE 'speedup=[0-9]+\.?[0-9]*x' | grep -oE '[0-9]+\.?[0-9]*' | head -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0} ${acceptance_pct:-0} ${speedup:-0}"
}

###############################################################################
# run_llama_generate <model> <prompt> <max_tokens> [extra_flags...]
#   Run llama-cli and parse throughput from stderr timing output.
#   Prints to stdout: decode_tok_s prefill_tok_s
#   Returns "0 0" if LLAMA_CLI is empty or missing.
###############################################################################
run_llama_generate() {
    local model="${1:?Usage: run_llama_generate <model> <prompt> <max_tokens> [flags...]}"
    local prompt="${2:?}"
    local max_tokens="${3:?}"
    shift 3
    local extra_flags=("$@")

    if [[ -z "${LLAMA_CLI:-}" || "$LLAMA_CLI" == "[missing]" ]]; then
        echo "0 0"
        return 0
    fi

    local output
    output=$("$LLAMA_CLI" -m "$model" -p "$prompt" -n "$max_tokens" \
        --no-display-prompt "${extra_flags[@]}" 2>&1) || true

    local decode_tok_s prefill_tok_s
    # Parse: eval time = ... tokens per second)
    decode_tok_s=$(echo "$output" | grep 'eval time' | grep -v 'prompt eval' | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    # Parse: prompt eval time = ... tokens per second)
    prefill_tok_s=$(echo "$output" | grep 'prompt eval time' | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1)

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0}"
}

###############################################################################
# run_llama_speculative <model> <draft_model> <prompt> <max_tokens> [extra_flags...]
#   Run llama-speculative with -md draft model and parse throughput + acceptance.
#   Prints to stdout: decode_tok_s prefill_tok_s acceptance_pct
#   Returns "0 0 0" if LLAMA_SPECULATIVE is empty or missing.
###############################################################################
run_llama_speculative() {
    local model="${1:?Usage: run_llama_speculative <model> <draft_model> <prompt> <max_tokens> [flags...]}"
    local draft_model="${2:?}"
    local prompt="${3:?}"
    local max_tokens="${4:?}"
    shift 4
    local extra_flags=("$@")

    if [[ -z "${LLAMA_SPECULATIVE:-}" || "$LLAMA_SPECULATIVE" == "[missing]" ]]; then
        echo "0 0 0"
        return 0
    fi

    local output
    output=$("$LLAMA_SPECULATIVE" -m "$model" -md "$draft_model" \
        -p "$prompt" -n "$max_tokens" \
        --no-display-prompt "${extra_flags[@]}" 2>&1) || true

    local decode_tok_s prefill_tok_s acceptance_pct
    decode_tok_s=$(echo "$output" | grep 'eval time' | grep -v 'prompt eval' | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    prefill_tok_s=$(echo "$output" | grep 'prompt eval time' | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    # Parse: speculative: accept rate: 0.456
    acceptance_pct=$(echo "$output" | grep -oE 'accept rate: [0-9]+\.?[0-9]*' | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    # Convert from 0-1 fraction to percentage if present
    if [[ -n "$acceptance_pct" ]]; then
        acceptance_pct=$(awk "BEGIN {printf \"%.1f\", $acceptance_pct * 100}")
    fi

    echo "${decode_tok_s:-0} ${prefill_tok_s:-0} ${acceptance_pct:-0}"
}

###############################################################################
# emit_jsonl <file> <json_line>
#   Append a single JSON line to the given file. Creates the file if needed.
###############################################################################
emit_jsonl() {
    local file="${1:?Usage: emit_jsonl <file> <json_line>}"
    local json_line="${2:?}"
    echo "$json_line" >> "$file"
}
