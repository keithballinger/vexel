# Medusa Cold-Start Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Medusa speculative decoding cold-start so heads begin speculating within ~500 tokens instead of 5,000, and update the benchmark to properly warm heads before measuring.

**Architecture:** Two changes: (1) Lower `WarmupSamples` from 5,000 to 500 in `DefaultOnlineConfig()` — 500 samples gives the ring buffer enough diversity for meaningful training while letting heads start speculating 10x sooner. Also lower `EvalInterval` from 5s to 1s so the PhaseWarming→PhaseHot transition happens faster. (2) Update the benchmark to generate 600 warmup tokens (enough to collect 500 samples + ~100 for initial training steps), then wait briefly for the trainer to transition to PhaseHot before measuring.

**Tech Stack:** Go 1.25.4, Bash (benchmark scripts)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/medusa/trainer.go` | Lower WarmupSamples, EvalInterval |
| Modify | `inference/medusa/gpu_trainer.go` | Lower minStepsForHot for faster transition |
| Modify | `benchmarks/lib/bench_speculative.sh` | Proper Medusa warmup pass |

---

### Task 1: Lower warmup thresholds

**Files:**
- Modify: `inference/medusa/trainer.go`
- Modify: `inference/medusa/gpu_trainer.go`

- [ ] **Step 1: Read DefaultOnlineConfig in trainer.go**

Read `inference/medusa/trainer.go` lines 96-108 to see the current defaults.

- [ ] **Step 2: Lower WarmupSamples and EvalInterval**

In `inference/medusa/trainer.go`, change `DefaultOnlineConfig()`:

```go
func DefaultOnlineConfig() OnlineConfig {
	return OnlineConfig{
		NumHeads:       4,
		BufferCapacity: 50000,
		WarmupSamples:  500,     // Was 5000 — 500 gives enough diversity for meaningful training
		MinAccuracy:    0.3,
		BatchSize:      64,
		LearningRate:   0.001,
		TrainInterval:  100 * time.Millisecond,
		EvalInterval:   1 * time.Second, // Was 5s — check for PhaseHot faster
	}
}
```

- [ ] **Step 3: Lower minStepsForHot in gpu_trainer.go**

In `inference/medusa/gpu_trainer.go`, find the PhaseWarming→PhaseHot transition (around line 227). Change `minStepsForHot` from 20 to 10:

```go
minStepsForHot := int64(10) // Was 20 — with 100ms train interval, 10 steps = 1 second
```

This means after collecting 500 samples, the trainer needs ~1 second (10 steps × 100ms) to train before checking loss, and the eval runs every 1s to check for the transition.

- [ ] **Step 4: Run medusa tests**

Run: `go test -v ./inference/medusa/`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add inference/medusa/trainer.go inference/medusa/gpu_trainer.go
git commit -m "perf(medusa): lower warmup threshold from 5000 to 500 samples for faster cold-start"
```

---

### Task 2: Fix benchmark Medusa warmup

**Files:**
- Modify: `benchmarks/lib/bench_speculative.sh`

- [ ] **Step 1: Read current bench_speculative.sh**

Read `benchmarks/lib/bench_speculative.sh` to see the current Medusa warmup (50 tokens — way too few even with the lowered threshold).

- [ ] **Step 2: Update Medusa warmup to 600 tokens + wait**

In `benchmarks/lib/bench_speculative.sh`, replace the Medusa warmup section:

Old:
```bash
echo "  Vexel Medusa: warming up heads (50-token generation)..."
run_vexel_medusa "$model" "$prompt" 50 > /dev/null
```

New:
```bash
echo "  Vexel Medusa: warming heads (600 tokens to collect training samples)..."
# Generate 600 tokens to collect ~500+ training samples.
# With WarmupSamples=500, this triggers Cold→Warming transition.
# Then wait 3s for training steps + Warming→Hot transition.
run_vexel_generate "$model" "$prompt" 600 --medusa > /dev/null 2>&1 || true
echo "  Vexel Medusa: waiting 3s for trainer to reach Hot phase..."
sleep 3
```

Note: We use `run_vexel_generate` with `--medusa` as extra flag (not `run_vexel_medusa`) because we don't need to parse the output — we just want the side effect of collecting training samples.

However, each `run_vexel_generate` call is a fresh process — the Medusa heads are trained in-process and lost when the process exits. This means **the warmup and measurement must happen in the same process**.

This is a fundamental issue with the benchmark approach. Each CLI invocation starts fresh — there's no state persistence between calls unless we use `--medusa-heads` to save/load.

- [ ] **Step 3: Restructure benchmark to use server mode for Medusa**

The fix: use Vexel in **server mode** for Medusa benchmarking, similar to the batched benchmark. Start the server with `--medusa`, send warmup requests, wait for heads to train, then send measured requests.

Replace the entire Medusa section in `bench_speculative.sh`:

```bash
# ── Vexel Medusa (via server — heads need persistent process for training) ──
echo "  Vexel Medusa: starting server for online training..."
local medusa_port=18086
"$VEXEL_BIN" --model "$model" --medusa serve \
    --port "$medusa_port" --max-batch-size 1 > /tmp/vexel_medusa.log 2>&1 &
local medusa_pid=$!

# Wait for server
for i in $(seq 1 15); do
    if curl -sf "http://localhost:$medusa_port/health" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:$medusa_port/health" > /dev/null 2>&1; then
    echo "    [ERROR] Medusa server failed to start"
    kill "$medusa_pid" 2>/dev/null || true
else
    # Warmup: generate 600 tokens to collect training samples
    echo "  Vexel Medusa: generating 600 warmup tokens..."
    curl -sf -X POST "http://localhost:$medusa_port/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"'"$prompt"'","max_tokens":600,"temperature":0}' -o /dev/null || true

    # Wait for trainer to reach Hot phase
    echo "  Vexel Medusa: waiting 5s for heads to train..."
    sleep 5

    # Measured runs
    echo "  Vexel Medusa measured ($RUNS runs)..."
    for ((r = 1; r <= RUNS; r++)); do
        local start_time end_time wall_time
        start_time=$(python3 -c 'import time; print(time.time())')

        local response
        response=$(curl -sf -X POST "http://localhost:$medusa_port/generate" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"'"$prompt"'","max_tokens":'"$GEN_TOKENS"',"temperature":0}' 2>&1) || true

        end_time=$(python3 -c 'import time; print(time.time())')
        wall_time=$(python3 -c "print(round($end_time - $start_time, 3))")

        local decode_tok_s
        decode_tok_s=$(python3 -c "print(round($GEN_TOKENS / $wall_time, 1))" 2>/dev/null || echo "0")

        emit_jsonl "$outfile" "{\"engine\":\"vexel\",\"mode\":\"medusa\",\"model\":\"$model_name\",\"run\":$r,\"gen_tokens\":$GEN_TOKENS,\"decode_tok_s\":$decode_tok_s,\"prefill_tok_s\":0,\"acceptance_pct\":0,\"speedup\":0}"
        echo "    run $r: decode=${decode_tok_s} tok/s (wall=${wall_time}s)"
    done

    kill "$medusa_pid" 2>/dev/null || true
    wait "$medusa_pid" 2>/dev/null || true
fi
sleep 1
```

- [ ] **Step 4: Verify the script has valid syntax**

```bash
bash -n benchmarks/lib/bench_speculative.sh
```

- [ ] **Step 5: Commit**

```bash
git add benchmarks/lib/bench_speculative.sh
git commit -m "fix(bench): use server mode for Medusa benchmark to persist trained heads"
```

---

### Task 3: Manual verification

- [ ] **Step 1: Build**

Run: `make build`

- [ ] **Step 2: Run speculative benchmark**

```bash
WARMUP=1 RUNS=2 bash benchmarks/full_comparison.sh speculative
```

Expected: Medusa decode tok/s should be non-zero and ideally > standard decode (if heads learned well).

- [ ] **Step 3: Check Medusa trainer logs**

```bash
grep -i "phase\|hot\|warming\|cold\|accept" /tmp/vexel_medusa.log | tail -20
```

Expected: Should show phase transitions: Cold → Warming → Hot.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(medusa): address cold-start issues from benchmark verification"
```
