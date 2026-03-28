#!/usr/bin/env bash
# bench_context.sh — Context-length scaling sweep benchmark.
# Sourced by full_comparison.sh; not intended to be run standalone.
#
# Measures how decode and prefill throughput scale with increasing
# context (prompt) lengths on the 8B model.

set -euo pipefail

###############################################################################
# run_context_scaling
#   Benchmark decode throughput across context lengths 16..4096.
#   Results written to $RESULTS_DIR/context_scaling.jsonl
###############################################################################
run_context_scaling() {
    local outfile="$RESULTS_DIR/context_scaling.jsonl"
    : > "$outfile"

    local model="$MODEL_LLAMA_8B"
    local model_name="llama-8b"
    local decode_tokens=16
    local context_runs=3
    local context_lengths=(16 64 256 512 1024 2048 4096)

    echo "--- Context scaling: $model_name ---"

    for ctx_len in "${context_lengths[@]}"; do
        local prompt
        prompt=$(generate_prompt "$ctx_len")

        echo "  Context length=$ctx_len"

        # ── Vexel ──
        echo "    Vexel ($context_runs runs)..."
        for ((r = 1; r <= context_runs; r++)); do
            local result
            result=$(run_vexel_generate "$model" "$prompt" "$decode_tokens")
            local decode_tok_s prefill_tok_s
            decode_tok_s=$(echo "$result" | awk '{print $1}')
            prefill_tok_s=$(echo "$result" | awk '{print $2}')

            emit_jsonl "$outfile" "{\"engine\":\"vexel\",\"mode\":\"context_scaling\",\"model\":\"$model_name\",\"run\":$r,\"context_length\":$ctx_len,\"decode_tokens\":$decode_tokens,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":$prefill_tok_s}"
            echo "      run $r: decode=${decode_tok_s} tok/s, prefill=${prefill_tok_s} tok/s"
        done

        # ── llama.cpp ──
        if [[ -n "${LLAMA_CLI:-}" && "$LLAMA_CLI" != "[missing]" ]]; then
            echo "    llama.cpp ($context_runs runs)..."
            for ((r = 1; r <= context_runs; r++)); do
                local result
                result=$(run_llama_generate "$model" "$prompt" "$decode_tokens")
                local decode_tok_s prefill_tok_s
                decode_tok_s=$(echo "$result" | awk '{print $1}')
                prefill_tok_s=$(echo "$result" | awk '{print $2}')

                emit_jsonl "$outfile" "{\"engine\":\"llama.cpp\",\"mode\":\"context_scaling\",\"model\":\"$model_name\",\"run\":$r,\"context_length\":$ctx_len,\"decode_tokens\":$decode_tokens,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":$prefill_tok_s}"
                echo "      run $r: decode=${decode_tok_s} tok/s, prefill=${prefill_tok_s} tok/s"
            done
        else
            echo "    [SKIP] llama.cpp not available"
        fi
    done

    echo ""
    echo "Context scaling results: $outfile"
}

# When sourced by full_comparison.sh, the caller invokes run_context_scaling.
# When run standalone: uncomment the line below.
# run_context_scaling
