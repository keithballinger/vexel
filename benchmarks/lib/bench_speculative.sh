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

    # ── Vexel Medusa (server mode — heads persist across requests) ──
    echo "  Vexel Medusa: starting server for online training..."
    local medusa_port=18086
    "$VEXEL_BIN" --model "$model" --medusa serve \
        --port "$medusa_port" --max-batch-size 1 > /tmp/vexel_medusa.log 2>&1 &
    local medusa_pid=$!

    # Wait for server
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:$medusa_port/health" > /dev/null 2>&1; then
            echo "    Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    if ! curl -sf "http://localhost:$medusa_port/health" > /dev/null 2>&1; then
        echo "    [ERROR] Medusa server failed to start"
        kill "$medusa_pid" 2>/dev/null || true
    else
        # Warmup: generate 600 tokens to collect ~500+ training samples
        echo "  Vexel Medusa: generating 600 warmup tokens..."
        curl -sf -X POST "http://localhost:$medusa_port/generate" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"'"$prompt"'","max_tokens":600,"temperature":0}' -o /dev/null 2>&1 || true

        # Wait for trainer to reach Hot phase
        echo "  Vexel Medusa: waiting 5s for heads to train..."
        sleep 5

        # Measured runs
        echo "  Vexel Medusa measured ($RUNS runs)..."
        for ((r = 1; r <= RUNS; r++)); do
            local start_time end_time wall_time
            start_time=$(python3 -c 'import time; print(time.time())')

            curl -sf -X POST "http://localhost:$medusa_port/generate" \
                -H "Content-Type: application/json" \
                -d '{"prompt":"'"$prompt"'","max_tokens":'"$GEN_TOKENS"',"temperature":0}' -o /dev/null 2>&1 || true

            end_time=$(python3 -c 'import time; print(time.time())')
            wall_time=$(python3 -c "print(round($end_time - $start_time, 3))")
            local decode_tok_s
            decode_tok_s=$(python3 -c "wt=$wall_time; print(round($GEN_TOKENS / wt, 1) if wt > 0.01 else 0)")

            emit_jsonl "$outfile" "{\"engine\":\"vexel\",\"mode\":\"medusa\",\"model\":\"$model_name\",\"run\":$r,\"gen_tokens\":$GEN_TOKENS,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":0,\"acceptance_pct\":0,\"speedup\":0}"
            echo "    run $r: decode=${decode_tok_s} tok/s (wall=${wall_time}s)"
        done

        kill "$medusa_pid" 2>/dev/null || true
        wait "$medusa_pid" 2>/dev/null || true
    fi
    sleep 1

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

# When sourced by full_comparison.sh, the caller invokes run_speculative.
# When run standalone: uncomment the line below.
# run_speculative
