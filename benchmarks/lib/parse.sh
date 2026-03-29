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

    # Use temp file to avoid bash xrealloc issues with large/binary model output
    local tmpfile
    tmpfile=$(mktemp)

    "$VEXEL_BIN" --model "$model" --verbose generate \
        --prompt "$prompt" --max-tokens "$max_tokens" \
        ${extra_flags[@]+"${extra_flags[@]}"} > "$tmpfile" 2>&1 || true

    local decode_tok_s prefill_tok_s
    # Parse: [N tokens | prefill: X.X tok/s | decode: Y.Y tok/s]
    decode_tok_s=$(grep -oE 'decode: [0-9]+\.?[0-9]* tok/s' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    prefill_tok_s=$(grep -oE 'prefill: [0-9]+\.?[0-9]* tok/s' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)

    rm -f "$tmpfile"
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

    # Use temp file to avoid bash xrealloc issues with large/binary model output
    local tmpfile
    tmpfile=$(mktemp)

    "$VEXEL_BIN" --model "$model" --verbose --medusa generate \
        --prompt "$prompt" --max-tokens "$max_tokens" \
        ${extra_flags[@]+"${extra_flags[@]}"} > "$tmpfile" 2>&1 || true

    local decode_tok_s prefill_tok_s acceptance_pct speedup
    decode_tok_s=$(grep -oE 'decode: [0-9]+\.?[0-9]* tok/s' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    prefill_tok_s=$(grep -oE 'prefill: [0-9]+\.?[0-9]* tok/s' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    # Parse: [speculative: acceptance=Z.Z% speedup=W.Wx generated=G accepted=A]
    acceptance_pct=$(grep -oE 'acceptance=[0-9]+\.?[0-9]*%' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    speedup=$(grep -oE 'speedup=[0-9]+\.?[0-9]*x' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)

    rm -f "$tmpfile"
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

    # Use temp file to avoid bash xrealloc issues with large/binary model output
    local tmpfile
    tmpfile=$(mktemp)

    "$VEXEL_BIN" --model "$model" --draft-model "$draft_model" --verbose generate \
        --prompt "$prompt" --max-tokens "$max_tokens" \
        ${extra_flags[@]+"${extra_flags[@]}"} > "$tmpfile" 2>&1 || true

    local decode_tok_s prefill_tok_s acceptance_pct speedup
    decode_tok_s=$(grep -oE 'decode: [0-9]+\.?[0-9]* tok/s' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    prefill_tok_s=$(grep -oE 'prefill: [0-9]+\.?[0-9]* tok/s' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    acceptance_pct=$(grep -oE 'acceptance=[0-9]+\.?[0-9]*%' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    speedup=$(grep -oE 'speedup=[0-9]+\.?[0-9]*x' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)

    rm -f "$tmpfile"
    echo "${decode_tok_s:-0} ${prefill_tok_s:-0} ${acceptance_pct:-0} ${speedup:-0}"
}

###############################################################################
# run_llama_generate <model> <prompt> <max_tokens> [extra_flags...]
#   Run llama-completion (non-interactive) and parse throughput.
#   Uses llama-completion with -no-cnv for text generation mode.
#   Falls back to llama-cli if llama-completion is not available.
#   Prints to stdout: decode_tok_s prefill_tok_s
#   Returns "0 0" if no llama binary is available.
###############################################################################
run_llama_generate() {
    local model="${1:?Usage: run_llama_generate <model> <prompt> <max_tokens> [flags...]}"
    local prompt="${2:?}"
    local max_tokens="${3:?}"
    shift 3
    local extra_flags=("$@")

    # Prefer llama-completion (non-interactive), fall back to llama-cli
    local llama_bin=""
    local llama_flags=()
    if [[ -n "${LLAMA_COMPLETION:-}" && "$LLAMA_COMPLETION" != "[missing]" ]]; then
        llama_bin="$LLAMA_COMPLETION"
        llama_flags=(-no-cnv --no-display-prompt)
    elif [[ -n "${LLAMA_CLI:-}" && "$LLAMA_CLI" != "[missing]" ]]; then
        llama_bin="$LLAMA_CLI"
        llama_flags=(-no-cnv --no-display-prompt)
    else
        echo "0 0"
        return 0
    fi

    local tmpfile
    tmpfile=$(mktemp)

    "$llama_bin" -m "$model" -p "$prompt" -n "$max_tokens" \
        "${llama_flags[@]}" --temp 0 \
        ${extra_flags[@]+"${extra_flags[@]}"} > "$tmpfile" 2>&1 || true

    local decode_tok_s prefill_tok_s
    # Parse eval time from various llama.cpp output formats:
    #   common_perf_print:        eval time =    2499.70 ms /   127 runs   (   19.68 ms per token,    50.81 tokens per second)
    #   llama_perf_context_print: eval time = ...
    #   llama_print_timings:      eval time = ...
    # Exclude lines containing "prompt eval" to get decode-only eval time.
    decode_tok_s=$(grep -E 'eval time' "$tmpfile" | grep -v 'prompt eval' | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    # Parse prompt eval time (prefill):
    #   common_perf_print: prompt eval time = ...
    prefill_tok_s=$(grep -E 'prompt eval time' "$tmpfile" | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)

    rm -f "$tmpfile"
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

    # Use temp file to avoid bash xrealloc issues with binary/unicode model output
    local tmpfile
    tmpfile=$(mktemp)

    "$LLAMA_SPECULATIVE" -m "$model" -md "$draft_model" \
        -p "$prompt" -n "$max_tokens" \
        --no-display-prompt ${extra_flags[@]+"${extra_flags[@]}"} > "$tmpfile" 2>&1 || true

    local decode_tok_s prefill_tok_s acceptance_pct
    decode_tok_s=$(grep -E 'eval time' "$tmpfile" | grep -v 'prompt eval' | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    prefill_tok_s=$(grep -E 'prompt eval time' "$tmpfile" | \
        grep -oE '[0-9]+\.?[0-9]* tokens per second' | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    # Parse: speculative: accept rate: 0.456
    acceptance_pct=$(grep -oE 'accept rate: [0-9]+\.?[0-9]*' "$tmpfile" | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    # Convert from 0-1 fraction to percentage if present
    if [[ -n "$acceptance_pct" ]]; then
        acceptance_pct=$(awk "BEGIN {printf \"%.1f\", $acceptance_pct * 100}")
    fi

    rm -f "$tmpfile"
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
