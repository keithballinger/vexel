#!/usr/bin/env bash
# bench_standard.sh — Standard single-stream decode throughput benchmark.
# Sourced by full_comparison.sh; not intended to be run standalone.
#
# Compares Vexel vs llama.cpp on standard (non-speculative) generation
# across multiple model sizes.

set -euo pipefail

###############################################################################
# run_standard_decode
#   Benchmark standard decode throughput for both engines and both models.
#   Results written to $RESULTS_DIR/standard_decode.jsonl
###############################################################################
run_standard_decode() {
    local outfile="$RESULTS_DIR/standard_decode.jsonl"
    : > "$outfile"

    local prompt
    prompt=$(generate_prompt 64)

    # Model name mapping: path -> short name
    declare -A model_names=(
        ["$MODEL_QWEN_05B"]="qwen-0.5b"
        ["$MODEL_LLAMA_8B"]="llama-8b"
    )

    for model_path in "$MODEL_QWEN_05B" "$MODEL_LLAMA_8B"; do
        local model_name="${model_names[$model_path]}"

        echo "--- Standard decode: $model_name ---"

        # ── Vexel ──
        echo "  Vexel warmup ($WARMUP runs)..."
        for ((w = 1; w <= WARMUP; w++)); do
            run_vexel_generate "$model_path" "$prompt" "$GEN_TOKENS" > /dev/null
        done

        echo "  Vexel measured ($RUNS runs)..."
        for ((r = 1; r <= RUNS; r++)); do
            local result
            result=$(run_vexel_generate "$model_path" "$prompt" "$GEN_TOKENS")
            local decode_tok_s prefill_tok_s
            decode_tok_s=$(echo "$result" | awk '{print $1}')
            prefill_tok_s=$(echo "$result" | awk '{print $2}')

            emit_jsonl "$outfile" "{\"engine\":\"vexel\",\"mode\":\"standard\",\"model\":\"$model_name\",\"run\":$r,\"gen_tokens\":$GEN_TOKENS,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":$prefill_tok_s}"
            echo "    run $r: decode=${decode_tok_s} tok/s, prefill=${prefill_tok_s} tok/s"
        done

        # ── llama.cpp ──
        if [[ -n "${LLAMA_CLI:-}" && "$LLAMA_CLI" != "[missing]" ]]; then
            echo "  llama.cpp warmup ($WARMUP runs)..."
            for ((w = 1; w <= WARMUP; w++)); do
                run_llama_generate "$model_path" "$prompt" "$GEN_TOKENS" > /dev/null
            done

            echo "  llama.cpp measured ($RUNS runs)..."
            for ((r = 1; r <= RUNS; r++)); do
                local result
                result=$(run_llama_generate "$model_path" "$prompt" "$GEN_TOKENS")
                local decode_tok_s prefill_tok_s
                decode_tok_s=$(echo "$result" | awk '{print $1}')
                prefill_tok_s=$(echo "$result" | awk '{print $2}')

                emit_jsonl "$outfile" "{\"engine\":\"llama.cpp\",\"mode\":\"standard\",\"model\":\"$model_name\",\"run\":$r,\"gen_tokens\":$GEN_TOKENS,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":$prefill_tok_s}"
                echo "    run $r: decode=${decode_tok_s} tok/s, prefill=${prefill_tok_s} tok/s"
            done
        else
            echo "  [SKIP] llama.cpp not available"
        fi

        echo ""
    done

    echo "Standard decode results: $outfile"
}

# Run the benchmark
run_standard_decode
