# Remaining Model Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Gemma 2 decode performance, Phi-3.5 chat template, and Phi-3.5 prefill speed.

**Architecture:** Three independent fixes targeting different subsystems: (1) re-enable cross-layer batching for Gemma 2 by adding missing barriers in the FP16 softcap SDPA path, (2) fix Phi-3.5 chat template EOS detection, (3) enable batched Q5_K kernel for Phi-3.5 prefill.

**Tech Stack:** Go, Metal (MSL), CGo

---

### Task 1: Re-enable cross-layer batching for Gemma 2

The current workaround in `decode.go` disables cross-layer batching for models with `HasPostNorms`, dropping Gemma 2 decode from ~68 tok/s to ~20 tok/s. The fix is to add proper barriers in the FP16+softcap SDPA path and re-enable batching.

**Files:**
- Modify: `inference/runtime/block.go` — add barrier after F16→F32 conversion in softcap SDPA path
- Modify: `inference/runtime/decode.go` — remove HasPostNorms exception from cross-layer batching

- [ ] **Step 1: Add barrier in FP16 softcap SDPA path**

In `inference/runtime/block.go`, find the `FP16KVCache+SoftCap` decode SDPA path (around line 1613). After the two `ConvertF16ToF32` calls and before `SDPASoftCap`, add a barrier:

```go
} else if useFP16KVCache && b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
    // ... existing code ...
    b.fp16Ops.ConvertF16ToF32(fullKPtr, kF32Tmp, fullSeqLen*numKVHeads*headDim)
    b.fp16Ops.ConvertF16ToF32(fullVPtr, vF32Tmp, fullSeqLen*numKVHeads*headDim)
    barrier() // Ensure F16→F32 conversion completes before SDPA reads the F32 buffers
    effKVLen, kvStartPos := b.effectiveKVLen(layerIdx, fullSeqLen)
    // ... rest unchanged ...
```

- [ ] **Step 2: Re-enable cross-layer batching for HasPostNorms models**

In `inference/runtime/decode.go`, revert the `HasPostNorms` exception:

```go
// BEFORE:
if batchSize == 1 && !m.config.HasPostNorms {

// AFTER:
if batchSize == 1 {
```

- [ ] **Step 3: Test Gemma 2 decode**

```bash
# Should produce coherent output (identifies "Paris") at >50 tok/s decode
timeout 60 go run -tags metal ./inference/cmd/vexel \
  --model /Users/qeetbastudio/projects/llama.cpp/models/gemma-2-2b-it-Q4_K_M.gguf \
  --no-chat-template --verbose generate --prompt "The capital of France is" 2>&1 | tail -5
```

Expected: decode >50 tok/s, output contains "Paris" or coherent text (not spaces/colons).

- [ ] **Step 4: Verify other models still work**

```bash
# Quick smoke test all models
for model in qwen2.5-0.5b-instruct-q4_k_m Phi-3.5-mini-instruct-Q4_K_M Meta-Llama-3.1-8B-Instruct-Q4_K_M; do
  echo "=== $model ===" && timeout 30 go run -tags metal ./inference/cmd/vexel \
    --model /Users/qeetbastudio/projects/llama.cpp/models/${model}.gguf \
    --no-chat-template generate --prompt "1 2 3 4 5" 2>&1 | tail -2
done
```

Expected: All produce "6 7 8 9 10..."

- [ ] **Step 5: Commit**

```bash
git add inference/runtime/block.go inference/runtime/decode.go
git commit -m "perf(runtime): re-enable cross-layer batching for Gemma 2 decode

Add missing barrier after F16→F32 KV conversion in the softcap SDPA path.
This was the actual synchronization point preventing correct batched execution.
Remove the HasPostNorms workaround that disabled cross-layer batching entirely.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix Phi-3.5 chat template EOS detection

Phi-3.5 with chat template generates EOS immediately. The likely cause is the EOS token ID mismatch — Phi-3.5's chat template uses `<|im_end|>` (token 32007) as the stop token, but the model's EOS from GGUF metadata might be different (32000 `<|endoftext|>`).

**Files:**
- Modify: `inference/pkg/tokenizer/tokenizer.go` — add chat template stop tokens
- Modify: `inference/scheduler/scheduler.go` — check stop tokens during decode

- [ ] **Step 1: Investigate the actual EOS behavior**

```bash
# Check what EOS token Phi-3.5 uses
DEBUG_DECODE=1 timeout 30 go run -tags metal ./inference/cmd/vexel \
  --model /Users/qeetbastudio/projects/llama.cpp/models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  generate --prompt "Hi" 2>&1 | grep -E 'SAMPLE|EOS|eos_token|stop|Logits top|chatml' | head -10
```

Check: what token does the model generate that triggers EOS? Is it the GGUF `eos_token_id` or a chat template stop token like `<|im_end|>` (32007)?

- [ ] **Step 2: Check the scheduler EOS logic**

Read `inference/scheduler/scheduler.go` and find where EOS is detected. Check if it compares against just the GGUF EOS token or also checks for chat template stop tokens.

Read `inference/cmd/vexel/commands.go` to see how the generate command detects EOS when using chat templates.

- [ ] **Step 3: Implement the fix**

Based on findings from steps 1-2, the fix will be one of:
- A) Add chat template stop tokens to the EOS check in the scheduler
- B) Add the `<|im_end|>` token ID to the Phi-3 chat template definition
- C) Fix the GGUF EOS token parsing for Phi-3

The exact code depends on investigation results. The engineer should implement the minimal fix that makes `go run -tags metal ./inference/cmd/vexel --model .../Phi-3.5-mini-instruct-Q4_K_M.gguf generate --prompt "What is 2+2?"` produce a multi-token response.

- [ ] **Step 4: Test**

```bash
timeout 30 go run -tags metal ./inference/cmd/vexel \
  --model /Users/qeetbastudio/projects/llama.cpp/models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  generate --prompt "What is 2+2?" 2>&1 | tail -3
```

Expected: Multi-token coherent response (e.g., "4" or "The answer is 4").

- [ ] **Step 5: Commit**

```bash
git add inference/pkg/tokenizer/tokenizer.go inference/scheduler/scheduler.go inference/cmd/vexel/commands.go
git commit -m "fix(tokenizer): fix Phi-3.5 chat template EOS detection

[Description based on actual fix]

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Enable batched Q5_K kernel for Phi-3.5 prefill

Phi-3.5 prefill (78 tok/s) is slower than llama.cpp (113 tok/s) because Q5_K uses a looped M=1 kernel instead of a true batched kernel. The Q5_0 batched kernel already exists and works — adapt it for Q5_K format.

**Files:**
- Modify: `inference/backend/metal/metal_bridge_darwin.m` — add batched Q5_K kernel
- Modify: `inference/backend/metal/metal_bridge.h` — add C dispatch declaration
- Modify: `inference/backend/metal/backend.go` — add pipeline + dispatch, use batched for M>1

- [ ] **Step 1: Write the batched Q5_K kernel**

In `inference/backend/metal/metal_bridge_darwin.m`, add a new kernel `matmul_q5k_batched_f32` modeled on the existing `matmul_q5_0_batched_f32` kernel (line ~5408) but using Q5_K dequantization from `matvec_q5k_multi_output_f32` (line ~4870).

The kernel should:
- Take `uint2 gid [[threadgroup_position_in_grid]]` with `gid.y` for batch row
- Process one output row per simdgroup (same pattern as Q5_0 batched)
- Use the Q5_K block format: 176 bytes per 256 elements (d, dmin, scales[12], qh[32], qs[128])
- Have `if (m >= M) return;` bounds check

- [ ] **Step 2: Add the C dispatch function**

Add `void metal_matmul_q5k_batched_f32(...)` dispatch function matching the pattern of `metal_matmul_q5_0_batched_f32` (line ~12142). Grid: `(nTiles, M, 1)`.

- [ ] **Step 3: Add pipeline and Go dispatch**

In `backend.go`:
1. Add `matmulQ5KBatchedPipeline` field
2. Initialize it with `metal_create_pipeline(device, library, "matmul_q5k_batched_f32")`
3. In `MatMulQ5_K`, use the batched pipeline for M>1:

```go
func (b *Backend) MatMulQ5_K(a, bMat, out tensor.DevicePtr, m, n, k int) {
    if m == 1 {
        // existing M=1 path unchanged
    } else if b.matmulQ5KBatchedPipeline != nil {
        C.metal_matmul_q5k_batched_f32(b.queue, b.matmulQ5KBatchedPipeline, ...)
    } else {
        // existing looped fallback
    }
}
```

- [ ] **Step 4: Test correctness**

```bash
# Verify Phi-3.5 still produces correct output
go run -tags metal ./inference/cmd/vexel \
  --model /Users/qeetbastudio/projects/llama.cpp/models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  --no-chat-template generate --prompt "1 2 3 4 5" 2>&1 | tail -2

# Run Q5_K metal tests
go test -tags metal -v ./inference/backend/metal/ -run TestQ5K -timeout 60s
```

Expected: "6 7 8 9 10..." and tests pass.

- [ ] **Step 5: Benchmark**

```bash
timeout 60 go run -tags metal ./inference/cmd/vexel \
  --model /Users/qeetbastudio/projects/llama.cpp/models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  --no-chat-template --verbose generate --prompt "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox." 2>&1 | grep tok/s
```

Expected: prefill >100 tok/s (up from 78).

- [ ] **Step 6: Commit**

```bash
git add inference/backend/metal/backend.go inference/backend/metal/metal_bridge.h inference/backend/metal/metal_bridge_darwin.m
git commit -m "perf(metal): add batched Q5_K kernel for faster prefill (Phi-3.5)

Add matmul_q5k_batched_f32 Metal kernel that processes all prefill tokens
in a single GPU dispatch instead of looping M=1 per token. Uses the same
2D grid pattern as the Q5_0 batched kernel.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
