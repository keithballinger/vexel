#!/usr/bin/env bash
# bench_batched.sh — Batched multi-client decode throughput benchmark.
# Sourced by full_comparison.sh; not intended to be run standalone.
#
# Measures aggregate throughput under concurrent load by spinning up
# a server and sending parallel requests at varying concurrency levels.

set -euo pipefail

###############################################################################
# Helpers
###############################################################################

# _timestamp — high-resolution timestamp in seconds
_timestamp() {
    python3 -c 'import time; print(time.time())'
}

# _kill_server <pid> — kill a server process and wait for it to exit
_kill_server() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

# _wait_for_health <url> <max_attempts> — poll a health endpoint
_wait_for_health() {
    local url="$1"
    local max_attempts="${2:-10}"
    for ((i = 1; i <= max_attempts; i++)); do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: health check failed after ${max_attempts}s at $url" >&2
    return 1
}

###############################################################################
# run_batched
#   Benchmark batched inference at concurrency levels 1, 2, 4, 8.
#   Results written to $RESULTS_DIR/batched.jsonl
###############################################################################
run_batched() {
    local outfile="$RESULTS_DIR/batched.jsonl"
    : > "$outfile"

    local model="$MODEL_LLAMA_8B"
    local model_name="llama-8b"
    local tokens_per_request=64
    local prompt
    prompt=$(generate_prompt 32)

    local concurrency_levels=(1 2 4 8)

    echo "--- Batched decode: $model_name ---"

    for concurrency in "${concurrency_levels[@]}"; do
        local total_tokens=$((concurrency * tokens_per_request))

        echo "  Concurrency=$concurrency (${total_tokens} total tokens)"

        # ── Vexel server ──
        local vexel_pid=""
        trap '_kill_server "${vexel_pid:-}" 2>/dev/null; _kill_server "${llama_pid:-}" 2>/dev/null' EXIT

        echo "    Starting Vexel server..."
        "$VEXEL_BIN" --model "$model" serve --port 18080 --max-batch-size "$concurrency" &
        vexel_pid=$!
        sleep 3

        if ! _wait_for_health "http://localhost:18080/health" 10; then
            echo "    [ERROR] Vexel server failed to start, skipping"
            _kill_server "$vexel_pid"
            vexel_pid=""
            continue
        fi

        echo "    Vexel measured ($RUNS runs)..."
        for ((r = 1; r <= RUNS; r++)); do
            local t_start t_end wall_time_s aggregate_tok_s
            local pids=()

            t_start=$(_timestamp)

            # Launch concurrent requests
            for ((c = 0; c < concurrency; c++)); do
                curl -sf http://localhost:18080/generate \
                    -H "Content-Type: application/json" \
                    -d "{\"prompt\":\"$prompt\",\"max_tokens\":$tokens_per_request}" \
                    -o /dev/null &
                pids+=($!)
            done

            # Wait for all requests
            for pid in "${pids[@]}"; do
                wait "$pid" 2>/dev/null || true
            done

            t_end=$(_timestamp)
            wall_time_s=$(python3 -c "print(round($t_end - $t_start, 4))")
            aggregate_tok_s=$(python3 -c "print(round($total_tokens / ($t_end - $t_start), 2))")

            emit_jsonl "$outfile" "{\"engine\":\"vexel\",\"mode\":\"batched\",\"model\":\"$model_name\",\"run\":$r,\"concurrency\":$concurrency,\"total_tokens\":$total_tokens,\"wall_time_s\":$wall_time_s,\"aggregate_tok_s\":$aggregate_tok_s}"
            echo "      run $r: wall=${wall_time_s}s, aggregate=${aggregate_tok_s} tok/s"
        done

        _kill_server "$vexel_pid"
        vexel_pid=""
        sleep 1

        # ── llama.cpp server ──
        local llama_pid=""

        if [[ -n "${LLAMA_SERVER:-}" && "$LLAMA_SERVER" != "[missing]" ]]; then
            echo "    Starting llama.cpp server..."
            "$LLAMA_SERVER" -m "$model" --port 18081 -np "$concurrency" &
            llama_pid=$!
            sleep 5

            if ! _wait_for_health "http://localhost:18081/health" 15; then
                echo "    [ERROR] llama.cpp server failed to start, skipping"
                _kill_server "$llama_pid"
                llama_pid=""
                continue
            fi

            echo "    llama.cpp measured ($RUNS runs)..."
            for ((r = 1; r <= RUNS; r++)); do
                local t_start t_end wall_time_s aggregate_tok_s
                local pids=()

                t_start=$(_timestamp)

                # Launch concurrent requests
                for ((c = 0; c < concurrency; c++)); do
                    curl -sf http://localhost:18081/completion \
                        -H "Content-Type: application/json" \
                        -d "{\"prompt\":\"$prompt\",\"n_predict\":$tokens_per_request}" \
                        -o /dev/null &
                    pids+=($!)
                done

                # Wait for all requests
                for pid in "${pids[@]}"; do
                    wait "$pid" 2>/dev/null || true
                done

                t_end=$(_timestamp)
                wall_time_s=$(python3 -c "print(round($t_end - $t_start, 4))")
                aggregate_tok_s=$(python3 -c "print(round($total_tokens / ($t_end - $t_start), 2))")

                emit_jsonl "$outfile" "{\"engine\":\"llama.cpp\",\"mode\":\"batched\",\"model\":\"$model_name\",\"run\":$r,\"concurrency\":$concurrency,\"total_tokens\":$total_tokens,\"wall_time_s\":$wall_time_s,\"aggregate_tok_s\":$aggregate_tok_s}"
                echo "      run $r: wall=${wall_time_s}s, aggregate=${aggregate_tok_s} tok/s"
            done

            _kill_server "$llama_pid"
            llama_pid=""
            sleep 1
        else
            echo "    [SKIP] llama.cpp server not available"
        fi

        # Reset trap since we cleaned up
        trap - EXIT
    done

    echo ""
    echo "Batched decode results: $outfile"
}

# When sourced by full_comparison.sh, the caller invokes run_batched.
# When run standalone: uncomment the line below.
# run_batched
