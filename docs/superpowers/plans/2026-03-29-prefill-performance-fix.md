# Prefill Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the prefill deadlock (>16 tokens on LLaMA 8B) and recover prefill throughput from 10-40 tok/s to near llama.cpp levels (70-1300 tok/s).

**Architecture:** The root cause is `finish_encode` calling `waitUntilCompleted` on every Metal dispatch. For prefill, this creates 576 individual command buffers with blocking waits. The fix enables per-layer command buffer batching for prefill: each layer's ~18 dispatches share one command buffer, committed without waiting. A single `Sync()` after all layers ensures completion before CPU reads the result. This reduces 576 blocking waits to 32 non-blocking commits + 1 sync.

**Tech Stack:** Metal GPU (Objective-C bridge), Go runtime

---

## File Structure

| File | Responsibility | Changes |
|------|---------------|---------|
| `inference/runtime/block.go` | Per-layer transformer execution | Enable batching for prefill (remove `prefillSeqLen == 1` guard) |
| `inference/runtime/decode.go` | Forward pass orchestration | Add Sync after layer loop for prefill |
| `inference/backend/metal/metal_bridge_darwin.m` | Metal command buffer management | Add `finish_encode_no_wait` variant for fire-and-forget |
| `inference/backend/metal/backend.go` | Go↔Metal bridge | No changes needed |

---

### Task 1: Enable per-layer batching for prefill in block.go

**Files:**
- Modify: `inference/runtime/block.go:1074-1077`

The current code disables batching for prefill (`prefillSeqLen == 1`). Each of the ~18 dispatches per layer creates its own command buffer with `waitUntilCompleted`. Removing this guard lets each layer batch its dispatches into one command buffer.

- [ ] **Step 1: Change the batching condition**

In `inference/runtime/block.go`, find line ~1074:
```go
	// Command buffer batching for decode only. For prefill (seqLen>1),
	// individual dispatches with finish_encode prevent GPU timeout issues.
	prefillSeqLen := x.Shape().NumElements() / b.HiddenSize
	useBatching := b.batcher != nil && !cachedGPUProfile && prefillSeqLen == 1
```

Replace with:
```go
	// Command buffer batching for both decode and prefill.
	// Decode: nested within outer batch from DecodeWithGPUKV (layers share one encoder).
	// Prefill: each layer gets its own batch (BeginBatch → ~18 dispatches → EndBatch → commit).
	// Per-layer batching reduces prefill from 576 individual waitUntilCompleted calls to
	// 32 non-blocking commits, fixing the deadlock for >16 token prompts on 32-layer models.
	useBatching := b.batcher != nil && !cachedGPUProfile
```

- [ ] **Step 2: Verify build**

```bash
make build
```
Expected: Clean compilation.

- [ ] **Step 3: Commit**

```bash
git add inference/runtime/block.go
git commit -m "perf(metal): enable per-layer batching for prefill dispatches"
```

---

### Task 2: Add Sync after layer loop for prefill in decode.go

**Files:**
- Modify: `inference/runtime/decode.go:765`

Per-layer batching commits without waiting (default `metal_end_batch` behavior). We need an explicit Sync after all layers to ensure the GPU has finished before we extract the last token's hidden state for logits computation.

- [ ] **Step 1: Add prefill Sync after the layer loop**

In `inference/runtime/decode.go`, find the end of the layer loop (after `for i, layer := range m.layers { ... }`), approximately line 765:
```go
	}

	// Timing: mark end of layer loop
```

Insert between:
```go
	}

	// For prefill (no outer batch), Sync to ensure all per-layer batches completed
	// before extracting the last token's hidden state for logits computation.
	// For decode, the deferred EndBatch from the outer cross-layer batch handles this.
	if batchSize > 1 {
		m.backend.Sync()
	}

	// Timing: mark end of layer loop
```

- [ ] **Step 2: Verify build**

```bash
make build
```

- [ ] **Step 3: Test short prompts (TinyLlama, 2-4 tokens)**

```bash
timeout 15 ./vexel --model benchmarks/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --no-chat-template generate --prompt "Hello" --max-tokens 5
```
Expected: `, World!` (or similar coherent output, no hang)

- [ ] **Step 4: Test medium prompts (TinyLlama, 10+ tokens)**

```bash
timeout 15 ./vexel --model benchmarks/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --no-chat-template generate --prompt "The sun is shining and the birds are singing" --max-tokens 5
```
Expected: Coherent output, no hang.

- [ ] **Step 5: Test the critical case (LLaMA 8B, 15+ tokens)**

```bash
timeout 60 ./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --no-chat-template generate \
  --prompt "The quick brown fox jumps over the lazy dog and runs through the field" \
  --max-tokens 5
```
Expected: Coherent output within 60 seconds. Previously hung indefinitely.

- [ ] **Step 6: Test LLaMA 8B decode accuracy**

```bash
timeout 60 ./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --no-chat-template generate --prompt "Water boils at" --max-tokens 8
```
Expected: `212°F (100°C)` (correct factual answer with proper UTF-8).

- [ ] **Step 7: Benchmark prefill speed improvement**

```bash
timeout 60 ./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --no-chat-template --verbose generate --prompt "Water boils at" --max-tokens 10 2>&1 | grep "tok/s"
```
Expected: Prefill should be significantly faster than previous 10-40 tok/s (closer to 100+ tok/s).

- [ ] **Step 8: Verify decode performance hasn't regressed**

```bash
timeout 60 ./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --no-chat-template --verbose generate --prompt "Hello" --max-tokens 50 2>&1 | grep "tok/s"
```
Expected: Decode ~65 tok/s (same as before — decode path unchanged).

- [ ] **Step 9: Commit**

```bash
git add inference/runtime/decode.go
git commit -m "perf(metal): add prefill Sync for per-layer batching correctness"
```

---

### Task 3: Handle CopyBufferBatched failure case

**Files:**
- Modify: `inference/runtime/gpu_kv_cache.go:207-238`

The KV cache blit fallback path (`CopyBufferBatched`) switches from compute to blit encoder within a batch. If this path is triggered during batched prefill, the compute encoder is ended and recreated. This is supported by `metal_copy_buffer_batched` (lines 11169-11181 in metal_bridge_darwin.m) but we should verify it doesn't cause issues. If the FP16 scatter path is always taken (which it should be), this task is just verification.

- [ ] **Step 1: Add a safety check**

In `inference/runtime/gpu_kv_cache.go`, find the blit fallback section (around line 207):
```go
	// Fall back to blit copies (less efficient but works for all cases)
```

Add a warning log before it:
```go
	// Fall back to blit copies — should not be reached in normal operation
	// (ScatterKV or ScatterKVF16 should handle all cases).
	if os.Getenv("DEBUG_DECODE") == "1" {
		fmt.Printf("[WARNING] KV cache AppendKV using blit fallback for layer %d\n", layerIdx)
	}
```

- [ ] **Step 2: Test that the blit fallback is NOT triggered**

```bash
timeout 60 env DEBUG_DECODE=1 ./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --no-chat-template generate --prompt "Water boils at" --max-tokens 3 2>&1 | grep "blit fallback"
```
Expected: No output (blit fallback not triggered).

- [ ] **Step 3: Commit**

```bash
git add inference/runtime/gpu_kv_cache.go
git commit -m "fix(kv): add diagnostic for unexpected blit fallback in AppendKV"
```

---

### Task 4: Update benchmarks with new prefill numbers

**Files:**
- Modify: `benchmarks/RESULTS.md`

- [ ] **Step 1: Run comprehensive benchmarks**

```bash
# TinyLlama decode (3 runs, best)
for i in 1 2 3; do timeout 30 ./vexel --model benchmarks/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --no-chat-template --verbose generate --prompt "Hello" --max-tokens 50 2>&1 | grep "tok/s"; done

# TinyLlama prefill (longer prompt)
timeout 30 ./vexel --model benchmarks/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --no-chat-template --verbose generate \
  --prompt "The sun is shining and the birds are singing in the beautiful morning sky" \
  --max-tokens 20 2>&1 | grep "tok/s"

# LLaMA 8B decode (3 runs, best)
for i in 1 2 3; do timeout 60 ./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --no-chat-template --verbose generate --prompt "Hello" --max-tokens 50 2>&1 | grep "tok/s"; done

# LLaMA 8B prefill (longer prompt, PREVIOUSLY HUNG)
timeout 60 ./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --no-chat-template --verbose generate \
  --prompt "The quick brown fox jumps over the lazy dog and runs through the field" \
  --max-tokens 10 2>&1 | grep "tok/s"

# llama.cpp comparison (decode)
for i in 1 2 3; do timeout 60 /Users/qeetbastudio/projects/llama.cpp/build/bin/llama-completion \
  -m benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p "Hello" -n 50 -no-cnv --temp 0 2>&1 | grep "eval time"; done
```

- [ ] **Step 2: Update RESULTS.md with new numbers**

Update the tables in `benchmarks/RESULTS.md`:
- Update decode tok/s if changed
- Add prefill tok/s numbers
- Update the "Known Limitations" section to remove the prefill deadlock note (if fixed)
- Keep the note about prefill being slower than llama.cpp (expected since llama.cpp batches more aggressively)

- [ ] **Step 3: Commit**

```bash
git add benchmarks/RESULTS.md
git commit -m "docs(benchmarks): update with batched prefill performance numbers"
```

---

### Task 5: If per-layer batching still hangs — fallback to fire-and-forget

**This task is only needed if Task 2 testing reveals the per-layer batching still causes hangs.**

**Files:**
- Modify: `inference/backend/metal/metal_bridge_darwin.m:11361-11376`
- Modify: `inference/runtime/decode.go`

The alternative approach: keep individual dispatches but remove `waitUntilCompleted`. Instead, limit the number of in-flight command buffers by syncing every N dispatches.

- [ ] **Step 1: Add a non-blocking finish_encode variant**

In `metal_bridge_darwin.m`, after the existing `finish_encode`, add:
```c
// Non-blocking variant: commit without waiting. Used for prefill where
// hundreds of dispatches would otherwise serialize on waitUntilCompleted.
// Caller MUST call metal_sync() before reading GPU results on CPU.
static inline void finish_encode_async(id<MTLComputeCommandEncoder> encoder,
                                        id<MTLCommandBuffer> cmdBuf, bool shouldCommit) {
    if (shouldCommit) {
        [encoder endEncoding];
        [cmdBuf commit];
        g_gpuBatchCount++;
    }
}
```

- [ ] **Step 2: Add a dispatch function that uses the async variant**

Create `dispatch_kernel_async` (similar to `dispatch_kernel` but using `finish_encode_async`).

- [ ] **Step 3: Add periodic sync in DecodeWithGPUKV layer loop**

In `decode.go`, add a sync every 8 layers during prefill to prevent command buffer queue overflow:
```go
if batchSize > 1 && (i+1)%8 == 0 {
    m.backend.Sync()
}
```

- [ ] **Step 4: Test and verify**

Same tests as Task 2 Steps 3-8.

---

## Self-Review Checklist

1. **Spec coverage**: All 3 issues addressed — prefill deadlock (Tasks 1-2), prefill speed (Tasks 1-2), benchmarks (Task 4). Scratch allocator Q4_K issue is out of scope (documented as known limitation).
2. **Placeholder scan**: All steps have concrete code, commands, and expected output.
3. **Type consistency**: `useBatching`, `batchSize`, `m.backend.Sync()` — all match existing codebase patterns.
