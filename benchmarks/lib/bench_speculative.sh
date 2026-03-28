#!/usr/bin/env bash
# bench_speculative.sh — Speculative decoding benchmark (Medusa + draft-model).
# Sourced by full_comparison.sh; not intended to be run standalone.
#
# Compares Vexel Medusa heads, Vexel draft-model, and llama.cpp speculative
# decoding on the 8B model. Speculation overhead is not meaningful on small
# models, so only the 8B model is tested.

set -euo pipefail

###############################################################################
# run_speculative
#   Benchmark speculative decoding variants for the 8B model.
#   Results written to $RESULTS_DIR/speculative.jsonl
###############################################################################
run_speculative() {
    local outfile="$RESULTS_DIR/speculative.jsonl"
    : > "$outfile"

    local model="$MODEL_LLAMA_8B"
    local model_name="llama-8b"
    local prompt
    prompt=$(generate_prompt 64)

    echo "--- Speculative decode: $model_name ---"

    # ── Vexel Medusa ──
    echo "  Vexel Medusa: warming up heads (50-token generation)..."
    run_vexel_medusa "$model" "$prompt" 50 > /dev/null

    echo "  Vexel Medusa warmup ($WARMUP runs)..."
    for ((w = 1; w <= WARMUP; w++)); do
        run_vexel_medusa "$model" "$prompt" "$GEN_TOKENS" > /dev/null
    done

    echo "  Vexel Medusa measured ($RUNS runs)..."
    for ((r = 1; r <= RUNS; r++)); do
        local result
        result=$(run_vexel_medusa "$model" "$prompt" "$GEN_TOKENS")
        local decode_tok_s prefill_tok_s acceptance_pct speedup
        decode_tok_s=$(echo "$result" | awk '{print $1}')
        prefill_tok_s=$(echo "$result" | awk '{print $2}')
        acceptance_pct=$(echo "$result" | awk '{print $3}')
        speedup=$(echo "$result" | awk '{print $4}')

        emit_jsonl "$outfile" "{\"engine\":\"vexel\",\"mode\":\"medusa\",\"model\":\"$model_name\",\"run\":$r,\"gen_tokens\":$GEN_TOKENS,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":$prefill_tok_s,\"acceptance_pct\":$acceptance_pct,\"speedup\":$speedup}"
        echo "    run $r: decode=${decode_tok_s} tok/s, acceptance=${acceptance_pct}%, speedup=${speedup}x"
    done

    # ── Vexel draft-model ──
    echo "  Vexel draft-model warmup ($WARMUP runs)..."
    for ((w = 1; w <= WARMUP; w++)); do
        run_vexel_draft "$model" "$MODEL_TINYLLAMA" "$prompt" "$GEN_TOKENS" > /dev/null
    done

    echo "  Vexel draft-model measured ($RUNS runs)..."
    for ((r = 1; r <= RUNS; r++)); do
        local result
        result=$(run_vexel_draft "$model" "$MODEL_TINYLLAMA" "$prompt" "$GEN_TOKENS")
        local decode_tok_s prefill_tok_s acceptance_pct speedup
        decode_tok_s=$(echo "$result" | awk '{print $1}')
        prefill_tok_s=$(echo "$result" | awk '{print $2}')
        acceptance_pct=$(echo "$result" | awk '{print $3}')
        speedup=$(echo "$result" | awk '{print $4}')

        emit_jsonl "$outfile" "{\"engine\":\"vexel\",\"mode\":\"draft\",\"model\":\"$model_name\",\"run\":$r,\"gen_tokens\":$GEN_TOKENS,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":$prefill_tok_s,\"acceptance_pct\":$acceptance_pct,\"speedup\":$speedup}"
        echo "    run $r: decode=${decode_tok_s} tok/s, acceptance=${acceptance_pct}%, speedup=${speedup}x"
    done

    # ── llama.cpp speculative ──
    if [[ -n "${LLAMA_SPECULATIVE:-}" && "$LLAMA_SPECULATIVE" != "[missing]" ]]; then
        echo "  llama.cpp speculative warmup ($WARMUP runs)..."
        for ((w = 1; w <= WARMUP; w++)); do
            run_llama_speculative "$model" "$MODEL_TINYLLAMA" "$prompt" "$GEN_TOKENS" > /dev/null
        done

        echo "  llama.cpp speculative measured ($RUNS runs)..."
        for ((r = 1; r <= RUNS; r++)); do
            local result
            result=$(run_llama_speculative "$model" "$MODEL_TINYLLAMA" "$prompt" "$GEN_TOKENS")
            local decode_tok_s prefill_tok_s acceptance_pct
            decode_tok_s=$(echo "$result" | awk '{print $1}')
            prefill_tok_s=$(echo "$result" | awk '{print $2}')
            acceptance_pct=$(echo "$result" | awk '{print $3}')

            emit_jsonl "$outfile" "{\"engine\":\"llama.cpp\",\"mode\":\"draft\",\"model\":\"$model_name\",\"run\":$r,\"gen_tokens\":$GEN_TOKENS,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":$prefill_tok_s,\"acceptance_pct\":$acceptance_pct}"
            echo "    run $r: decode=${decode_tok_s} tok/s, acceptance=${acceptance_pct}%"
        done
    else
        echo "  [SKIP] llama.cpp speculative not available"
    fi

    echo ""
    echo "Speculative decode results: $outfile"
}

# Run the benchmark
run_speculative
