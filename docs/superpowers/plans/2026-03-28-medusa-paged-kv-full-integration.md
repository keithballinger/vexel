# Medusa + Paged KV Full Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Medusa speculative decoding with paged KV cache so serve mode gets both multi-client batching AND speculation.

**Architecture:** Create two new runtime methods: `DecodeWithPagedKVAndHidden()` (single-token decode with hidden state capture) and `VerifySpeculativeWithPagedKVAndHidden()` (multi-token verification with per-token hidden states). Then adapt the MedusaScheduler to use paged KV decode paths when GPU KV cache is not available. Finally, revert the serve mode workaround so `--medusa` uses paged KV.

**Tech Stack:** Go 1.25.4, Metal backend, paged KV cache

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/runtime/decode.go` | Add DecodeWithPagedKVAndHidden, VerifySpeculativeWithPagedKVAndHidden |
| Modify | `inference/scheduler/medusa_scheduler.go` | Use paged KV paths when available |
| Modify | `inference/cmd/vexel/commands.go` | Revert: serve --medusa uses paged KV |

---

### Task 1: Add DecodeWithPagedKVAndHidden

**Files:**
- Modify: `inference/runtime/decode.go`

- [ ] **Step 1: Read DecodeWithPagedKV and DecodeWithGPUKVAndHidden**

Read `inference/runtime/decode.go` to understand both methods. The hidden state is captured AFTER `applyFinalNorm` and BEFORE `outputHeadMatMul`. For single-token decode (batchSize=1), the entire state IS the hidden state.

- [ ] **Step 2: Create DecodeWithPagedKVAndHidden**

Add after `DecodeWithPagedKV` (around line 439). This is a copy of `DecodeWithPagedKV` with two changes:
1. Return signature adds `hidden []float32`
2. After `applyFinalNorm`, capture the hidden state to CPU

```go
// DecodeWithPagedKVAndHidden performs decode with paged KV and returns the post-norm hidden state.
// Used by Medusa scheduler for training sample collection and speculation.
func (m *ModelRuntime) DecodeWithPagedKVAndHidden(tokens []int, seqID int64, pos int) (tensor.Tensor, []float32, error) {
```

The hidden state capture code (insert between `applyFinalNorm` and `outputHeadMatMul`):

```go
// Capture post-norm hidden state for Medusa training.
// For batchSize > 1, extract only the last token's state.
var hiddenPtr tensor.DevicePtr
if batchSize == 1 {
    hiddenPtr = statePtr
} else {
    offset := uintptr((batchSize - 1) * hiddenSize * 4)
    hiddenPtr = tensor.DevicePtrOffset(statePtr, offset)
}
m.backend.Sync()
hiddenBytes := make([]byte, hiddenSize*4)
m.backend.ToHost(hiddenBytes, hiddenPtr)
hidden := bytesToFloat32(hiddenBytes)
```

Return: `return logits, hidden, nil`

- [ ] **Step 3: Verify build**

Run: `CGO_ENABLED=1 go build -tags metal ./inference/runtime/...`

- [ ] **Step 4: Commit**

```bash
git add inference/runtime/decode.go
git commit -m "feat(runtime): add DecodeWithPagedKVAndHidden for Medusa + paged KV"
```

---

### Task 2: Add VerifySpeculativeWithPagedKVAndHidden

**Files:**
- Modify: `inference/runtime/decode.go`

- [ ] **Step 1: Read VerifySpeculativeWithHidden**

Read `inference/runtime/decode.go` around line 1227. This method:
1. Processes multiple tokens through embedding + all layers + final norm
2. Returns logits [seqLen, vocabSize] and hidden states [][]float32

The key change: replace `layer.ExecuteWithGPUKV` with `layer.ExecuteWithPagedKV`, and replace `m.gpuCache` checks with paged cache.

- [ ] **Step 2: Create VerifySpeculativeWithPagedKVAndHidden**

Add after `VerifySpeculativeWithHidden`. This is a copy with:
1. Signature takes `seqID int64` instead of using implicit GPU cache position
2. Layer loop uses `ExecuteWithPagedKV(state, scratch, m.pagedCache, m.gpuPool, seqID, i, startPos)`
3. Checks `m.pagedCache != nil` instead of `m.gpuCache != nil`

```go
// VerifySpeculativeWithPagedKVAndHidden verifies speculative tokens using paged KV cache.
// Returns logits [seqLen, vocabSize] and per-token post-norm hidden states.
func (m *ModelRuntime) VerifySpeculativeWithPagedKVAndHidden(tokens []int, seqID int64, startPos int) (tensor.Tensor, [][]float32, error) {
```

- [ ] **Step 3: Verify build**

Run: `CGO_ENABLED=1 go build -tags metal ./inference/runtime/...`

- [ ] **Step 4: Commit**

```bash
git add inference/runtime/decode.go
git commit -m "feat(runtime): add VerifySpeculativeWithPagedKVAndHidden for Medusa + paged KV"
```

---

### Task 3: Adapt MedusaScheduler to use paged KV paths

**Files:**
- Modify: `inference/scheduler/medusa_scheduler.go`

- [ ] **Step 1: Update runDecodeStepWithTraining to support paged KV**

In `runDecodeStepWithTraining` (around line 244), the decode call selection currently only handles GPU KV:

```go
if useGPUCache && ms.trainer != nil {
    logits, hidden, err = ms.runtime.DecodeWithGPUKVAndHidden([]int{token}, pos)
} else if useGPUCache {
    logits, err = ms.runtime.DecodeWithGPUKV([]int{token}, pos)
} else {
    // generic fallback
}
```

Add paged KV paths:

```go
if useGPUCache && ms.trainer != nil {
    logits, hidden, err = ms.runtime.DecodeWithGPUKVAndHidden([]int{token}, pos)
} else if useGPUCache {
    logits, err = ms.runtime.DecodeWithGPUKV([]int{token}, pos)
} else if usePagedCache && ms.trainer != nil {
    logits, hidden, err = ms.runtime.DecodeWithPagedKVAndHidden([]int{token}, seq.KVSeqID(), pos)
} else if usePagedCache {
    logits, err = ms.runtime.DecodeWithPagedKV([]int{token}, seq.KVSeqID(), pos)
} else {
    inputs := runtime.NewBatchRuntimeInputsWithPos([]int{token}, []int{pos}, nil)
    logits, err = ms.runtime.DecodeStep(inputs)
}
```

Note: need to add `usePagedCache` check in this method (it already exists in the base scheduler's runDecodeStep).

- [ ] **Step 2: Update runMedusaDecodeStep to support paged KV**

In `runMedusaDecodeStep` (around line 466), the verification call:

```go
allLogits, hiddenStates, err := ms.runtime.VerifySpeculativeWithHidden(verifyTokens, startPos)
```

Add paged KV branch:

```go
var allLogits tensor.Tensor
var hiddenStates [][]float32
var err error

if usePagedCache {
    allLogits, hiddenStates, err = ms.runtime.VerifySpeculativeWithPagedKVAndHidden(verifyTokens, seq.KVSeqID(), startPos)
} else {
    allLogits, hiddenStates, err = ms.runtime.VerifySpeculativeWithHidden(verifyTokens, startPos)
}
```

Also update the cache truncation (around line 567):
```go
if cache != nil {
    newCacheLen := startPos + 1 + numAccepted
    cache.Truncate(newCacheLen)
}
// Add paged KV truncation:
if pool := ms.runtime.GetGPUBlockPool(); pool != nil && seq.KVSeqID() != 0 {
    newCacheLen := startPos + 1 + numAccepted
    pool.TruncateSequence(seq.KVSeqID(), newCacheLen)
}
```

Check if `GPUBlockPool` has a `TruncateSequence` method. If not, it needs to be added.

- [ ] **Step 3: Update runTreeMedusaDecodeStep similarly**

Same pattern as runMedusaDecodeStep — add paged KV branch for verification and truncation.

- [ ] **Step 4: Update prefill handling**

In `runDecodeStepWithTraining`, the prefill check calls `runGPUPrefill`. Add paged KV prefill:

```go
if useGPUCache {
    if err := ms.runGPUPrefill(seq); err != nil {
        return err
    }
} else if usePagedCache {
    if err := ms.runBatchedPrefill(seq); err != nil {
        return err
    }
}
```

- [ ] **Step 5: Run tests**

Run: `go test -v ./inference/scheduler/`

- [ ] **Step 6: Commit**

```bash
git add inference/scheduler/medusa_scheduler.go
git commit -m "feat(scheduler): adapt MedusaScheduler to use paged KV decode paths"
```

---

### Task 4: Revert serve workaround and enable paged KV for Medusa

**Files:**
- Modify: `inference/cmd/vexel/commands.go`

- [ ] **Step 1: Change usePaged back to true for Medusa**

In `runServe`, change:

```go
usePaged := !globals.Medusa && globals.DraftModel == ""
```

to:

```go
usePaged := globals.DraftModel == ""  // Medusa now works with paged KV; draft-model still needs GPU KV
```

Remove the Medusa warning log.

- [ ] **Step 2: Build and run tests**

Run: `make build && go test ./inference/cmd/vexel/`

- [ ] **Step 3: Commit**

```bash
git add inference/cmd/vexel/commands.go
git commit -m "feat(serve): enable paged KV for Medusa mode (multi-client + speculation)"
```

---

### Task 5: Manual verification

- [ ] **Step 1: Test Medusa serve with 8B model**

```bash
./vexel --model benchmarks/models/llama-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --medusa serve --port 18092 &
sleep 20
# Should see "Using paged KV cache" AND "Using Medusa speculative decoding"
curl -X POST http://localhost:18092/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":16,"temperature":0}'
kill %1
```

- [ ] **Step 2: Test concurrent Medusa requests**

```bash
# 2 concurrent requests should both complete (paged KV enables multi-sequence)
time (
  curl -sf -X POST http://localhost:18092/generate -H "Content-Type: application/json" \
    -d '{"prompt":"Hello","max_tokens":16,"temperature":0}' -o /dev/null &
  curl -sf -X POST http://localhost:18092/generate -H "Content-Type: application/json" \
    -d '{"prompt":"World","max_tokens":16,"temperature":0}' -o /dev/null &
  wait
)
```

- [ ] **Step 3: Commit fixes**
