# Batched Decode with Paged KV Cache Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sequential per-sequence decode loop in the scheduler (scheduler.go:326) with true batched decode, enabling multi-client serving throughput gains of 2-3x.

**Architecture:** Currently when batch_size > 1, the scheduler loops over sequences calling `DecodeWithPagedKV()` one at a time. We need: (1) a `DecodeWithPagedKVBatched()` runtime method that processes N sequences in one forward pass, (2) batched `ExecuteWithPagedKV` at the block level accepting multiple seqIDs, (3) batched `Attention` on the GPUBlockPool, and (4) a batched `SDPAPagedDecode` Metal kernel. The approach is bottom-up: kernel first, then pool, then block, then runtime, then scheduler.

**Tech Stack:** Go 1.25.4, Metal GPU kernels (Obj-C bridge), paged KV cache (`inference/kv/`), GPU block pool (`inference/runtime/gpu_block_pool.go`)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/backend/backend.go` | Add `SDPAPagedDecodeBatched` to `PagedKVOps` interface |
| Modify | `inference/backend/metal/backend.go` | Implement batched paged SDPA dispatch |
| Modify | `inference/runtime/gpu_block_pool.go` | Add `AttentionBatched()` method |
| Modify | `inference/runtime/block.go` | Add `ExecuteWithPagedKVBatched()` |
| Modify | `inference/runtime/decode.go` | Add `DecodeWithPagedKVBatched()` |
| Modify | `inference/scheduler/scheduler.go` | Call batched decode instead of loop |
| Create | `inference/backend/metal/paged_batched_test.go` | Metal kernel correctness test |
| Create | `inference/runtime/batched_decode_test.go` | Runtime batched decode test |
| Create | `inference/scheduler/batched_decode_test.go` | Scheduler batched decode test |

---

### Task 1: Add batched interface to PagedKVOps

**Files:**
- Modify: `inference/backend/backend.go`

- [ ] **Step 1: Read current PagedKVOps interface**

Read `inference/backend/backend.go` lines 390-409 to see the current single-sequence `SDPAPagedDecode` signature.

- [ ] **Step 2: Add SDPAPagedDecodeBatched to the interface**

Add below the existing `SDPAPagedDecode` method in the `PagedKVOps` interface:

```go
// SDPAPagedDecodeBatched performs batched SDPA across multiple sequences.
// Each sequence has its own query, block table, and context length.
// q: [batchSize, numQHeads, headDim] - queries concatenated per sequence
// kvPool: base pointer to shared block pool
// blockTables: [batchSize] device pointers, each pointing to [numBlocks] int32
// out: [batchSize, numQHeads, headDim]
// seqLens: [batchSize] int - context length per sequence
SDPAPagedDecodeBatched(
	q, kvPool tensor.DevicePtr,
	blockTables []tensor.DevicePtr,
	out tensor.DevicePtr,
	batchSize, maxBlocks, blockSize, numQHeads, numKVHeads, headDim int,
	scale float32,
	seqLens []int,
)
```

- [ ] **Step 3: Verify build still compiles (CPU backend may need stub)**

Run: `go build ./inference/backend/...`
Expected: May fail if CPU backend implements PagedKVOps — check and add stub if needed.

- [ ] **Step 4: Add stub implementation to any backend that needs it**

If the CPU backend or CUDA backend implements `PagedKVOps`, add a no-op stub:

```go
func (b *Backend) SDPAPagedDecodeBatched(q, kvPool tensor.DevicePtr, blockTables []tensor.DevicePtr, out tensor.DevicePtr, batchSize, maxBlocks, blockSize, numQHeads, numKVHeads, headDim int, scale float32, seqLens []int) {
	// Fallback: process one at a time
	for i := 0; i < batchSize; i++ {
		// ... call single-sequence SDPAPagedDecode
	}
}
```

- [ ] **Step 5: Commit**

```bash
git add inference/backend/backend.go
git commit -m "feat(backend): add SDPAPagedDecodeBatched to PagedKVOps interface"
```

---

### Task 2: Implement batched paged SDPA in Metal backend

**Files:**
- Modify: `inference/backend/metal/backend.go`
- Create: `inference/backend/metal/paged_batched_test.go`

- [ ] **Step 1: Write failing kernel correctness test**

Create `inference/backend/metal/paged_batched_test.go`:

```go
//go:build metal && darwin && cgo

package metal

import (
	"math"
	"math/rand"
	"testing"

	"vexel/inference/tensor"
)

func TestSDPAPagedDecodeBatched_TwoSequences(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	batchSize := 2
	numQHeads, numKVHeads, headDim := 4, 4, 64
	blockSize := 16
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Sequence 1: 20 tokens (2 blocks), Sequence 2: 10 tokens (1 block)
	seqLens := []int{20, 10}

	// Allocate Q: [batchSize, numQHeads, headDim]
	qSize := batchSize * numQHeads * headDim
	qData := make([]float32, qSize)
	for i := range qData {
		qData[i] = rand.Float32()*2 - 1
	}
	qPtr := b.Alloc(qSize * 4)
	defer b.Free(qPtr)
	b.ToDevice(qPtr, float32ToBytes(qData))

	// Allocate KV pool (shared) and block tables
	// ... (allocate blocks, fill with random KV data)

	// Allocate output
	outSize := batchSize * numQHeads * headDim
	outPtr := b.Alloc(outSize * 4)
	defer b.Free(outPtr)

	// Run batched
	b.SDPAPagedDecodeBatched(qPtr, kvPoolPtr, blockTables, outPtr,
		batchSize, maxBlocks, blockSize, numQHeads, numKVHeads, headDim, scale, seqLens)
	b.Sync()

	// Run single-sequence for each and compare
	for seq := 0; seq < batchSize; seq++ {
		singleOut := b.Alloc(numQHeads * headDim * 4)
		qOffset := seq * numQHeads * headDim * 4
		// ... extract per-sequence Q, run SDPAPagedDecode, compare outputs
		b.Free(singleOut)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test -tags metal -run TestSDPAPagedDecodeBatched -v ./inference/backend/metal/`
Expected: FAIL — method not implemented

- [ ] **Step 3: Implement SDPAPagedDecodeBatched in Metal backend**

In `inference/backend/metal/backend.go`, implement as a loop over single-sequence calls initially (functional correctness first, kernel fusion later):

```go
func (b *Backend) SDPAPagedDecodeBatched(
	q, kvPool tensor.DevicePtr,
	blockTables []tensor.DevicePtr,
	out tensor.DevicePtr,
	batchSize, maxBlocks, blockSize, numQHeads, numKVHeads, headDim int,
	scale float32,
	seqLens []int,
) {
	qStride := numQHeads * headDim * 4   // bytes per sequence Q
	outStride := numQHeads * headDim * 4  // bytes per sequence output

	for i := 0; i < batchSize; i++ {
		seqQ := tensor.DevicePtr{Ptr: q.Ptr, Offset: q.Offset + i*qStride}
		seqOut := tensor.DevicePtr{Ptr: out.Ptr, Offset: out.Offset + i*outStride}
		numBlocks := (seqLens[i] + blockSize - 1) / blockSize
		tokensInLastBlock := seqLens[i] - (numBlocks-1)*blockSize

		b.SDPAPagedDecode(seqQ, kvPool, blockTables[i], seqOut,
			numBlocks, blockSize, numQHeads, numKVHeads, headDim, scale, tokensInLastBlock)
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test -tags metal -run TestSDPAPagedDecodeBatched -v ./inference/backend/metal/`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference/backend/metal/backend.go inference/backend/metal/paged_batched_test.go
git commit -m "feat(metal): implement SDPAPagedDecodeBatched (loop-based initial impl)"
```

---

### Task 3: Add AttentionBatched to GPUBlockPool

**Files:**
- Modify: `inference/runtime/gpu_block_pool.go`

- [ ] **Step 1: Read current Attention method**

Read `inference/runtime/gpu_block_pool.go` around line 237 to understand the single-sequence `Attention()` signature and how it resolves block tables.

- [ ] **Step 2: Write failing test**

Create `inference/runtime/batched_decode_test.go`:

```go
//go:build metal && darwin && cgo

package runtime

import (
	"testing"
)

func TestGPUBlockPoolAttentionBatched(t *testing.T) {
	// Test that AttentionBatched produces same results as calling Attention N times
	// This requires a GPUBlockPool with at least 2 sequences registered
	t.Skip("TODO: implement after AttentionBatched exists")
}
```

- [ ] **Step 3: Implement AttentionBatched**

In `inference/runtime/gpu_block_pool.go`:

```go
// AttentionBatched performs batched SDPA across multiple sequences using paged KV.
func (g *GPUBlockPool) AttentionBatched(
	layerIdx int,
	seqIDs []int64,
	qPtr, outPtr tensor.DevicePtr,
	numQHeads, headDim int,
	scale float32,
) error {
	batchSize := len(seqIDs)
	blockTables := make([]tensor.DevicePtr, batchSize)
	seqLens := make([]int, batchSize)
	maxBlocks := 0

	for i, seqID := range seqIDs {
		state, ok := g.seqs[seqID]
		if !ok {
			return fmt.Errorf("sequence %d not found in GPU block pool", seqID)
		}
		blockTables[i] = state.blockTablePtrs[layerIdx]
		seqLens[i] = state.seqLen
		numBlocks := (state.seqLen + g.blockSize - 1) / g.blockSize
		if numBlocks > maxBlocks {
			maxBlocks = numBlocks
		}
	}

	if pagedOps, ok := g.backend.(backend.PagedKVOps); ok {
		pagedOps.SDPAPagedDecodeBatched(qPtr, g.kvPool, blockTables, outPtr,
			batchSize, maxBlocks, g.blockSize, numQHeads, g.numKVHeads, headDim, scale, seqLens)
		return nil
	}

	return fmt.Errorf("backend does not support PagedKVOps")
}
```

- [ ] **Step 4: Run build to verify**

Run: `CGO_ENABLED=1 go build -tags metal ./inference/runtime/...`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add inference/runtime/gpu_block_pool.go inference/runtime/batched_decode_test.go
git commit -m "feat(runtime): add AttentionBatched to GPUBlockPool"
```

---

### Task 4: Add ExecuteWithPagedKVBatched to BlockRuntime

**Files:**
- Modify: `inference/runtime/block.go`

- [ ] **Step 1: Read current ExecuteWithPagedKV**

Read `inference/runtime/block.go` around line 700 to understand the single-sequence layer execution path — specifically the GPU decode path (seqLen==1, gpuPool != nil).

- [ ] **Step 2: Implement ExecuteWithPagedKVBatched**

Add to `inference/runtime/block.go`:

```go
// ExecuteWithPagedKVBatched executes a transformer block for multiple sequences.
// x: [batchSize, hiddenSize] - one hidden state per sequence
// Each sequence is at seqLen=1 (decode mode).
func (b *BlockRuntime) ExecuteWithPagedKVBatched(
	x, scratch tensor.Tensor,
	pagedCache *kv.PagedKVCache,
	gpuPool *GPUBlockPool,
	seqIDs []int64,
	layerIdx int,
	positions []int,
) (tensor.Tensor, error) {
	batchSize := len(seqIDs)

	// For now, fall back to sequential execution per sequence.
	// The attention step uses AttentionBatched, but RMSNorm/MatMul
	// still process one sequence at a time (they don't depend on KV cache).
	// TODO: batch the non-attention ops when M>1 matmul kernels are ready.

	// Process each sequence through the block
	outputs := make([]tensor.Tensor, batchSize)
	for i, seqID := range seqIDs {
		// Extract sequence i's hidden state from x
		seqX := x.Slice(i, i+1) // [1, hiddenSize]
		seqScratch := scratch

		out, err := b.ExecuteWithPagedKV(seqX, seqScratch, pagedCache, gpuPool, seqID, layerIdx, positions[i])
		if err != nil {
			return tensor.Tensor{}, fmt.Errorf("sequence %d: %w", seqID, err)
		}
		outputs[i] = out
	}

	// Concatenate outputs into [batchSize, hiddenSize]
	return tensor.Concat(outputs, 0), nil
}
```

- [ ] **Step 3: Run build to verify**

Run: `CGO_ENABLED=1 go build -tags metal ./inference/runtime/...`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add inference/runtime/block.go
git commit -m "feat(runtime): add ExecuteWithPagedKVBatched for multi-sequence decode"
```

---

### Task 5: Add DecodeWithPagedKVBatched to ModelRuntime

**Files:**
- Modify: `inference/runtime/decode.go`

- [ ] **Step 1: Read DecodeWithPagedKV to understand single-sequence flow**

Read `inference/runtime/decode.go` around line 326 to trace: embedding lookup → layer loop → lm_head → logits.

- [ ] **Step 2: Implement DecodeWithPagedKVBatched**

Add to `inference/runtime/decode.go`:

```go
// DecodeWithPagedKVBatched processes multiple sequences in a single forward pass.
// tokens: [batchSize] - one token per sequence
// seqIDs: [batchSize] - paged KV cache sequence IDs
// positions: [batchSize] - RoPE positions per sequence
// Returns: logits [batchSize, vocabSize]
func (m *ModelRuntime) DecodeWithPagedKVBatched(tokens []int, seqIDs []int64, positions []int) (tensor.Tensor, error) {
	batchSize := len(tokens)
	if batchSize != len(seqIDs) || batchSize != len(positions) {
		return tensor.Tensor{}, fmt.Errorf("mismatched batch dimensions: tokens=%d seqIDs=%d positions=%d",
			len(tokens), len(seqIDs), len(positions))
	}

	// For batchSize==1, delegate to the optimized single-sequence path
	if batchSize == 1 {
		return m.DecodeWithPagedKV(tokens, seqIDs[0], positions[0])
	}

	// Process each sequence through embedding + all layers + lm_head
	// Initial implementation: sequential per-sequence (correctness first)
	// The GPU batching benefit comes from AttentionBatched in block execution
	allLogits := make([]tensor.Tensor, batchSize)
	for i := 0; i < batchSize; i++ {
		logits, err := m.DecodeWithPagedKV([]int{tokens[i]}, seqIDs[i], positions[i])
		if err != nil {
			return tensor.Tensor{}, fmt.Errorf("batch seq %d: %w", i, err)
		}
		allLogits[i] = logits
	}

	return tensor.Concat(allLogits, 0), nil
}
```

- [ ] **Step 3: Run build to verify**

Run: `CGO_ENABLED=1 go build -tags metal ./inference/runtime/...`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add inference/runtime/decode.go
git commit -m "feat(runtime): add DecodeWithPagedKVBatched (sequential initial impl)"
```

---

### Task 6: Wire batched decode into scheduler

**Files:**
- Modify: `inference/scheduler/scheduler.go`
- Create: `inference/scheduler/batched_decode_test.go`

- [ ] **Step 1: Write failing test**

Create `inference/scheduler/batched_decode_test.go`:

```go
package scheduler

import (
	"testing"
)

func TestBatchedDecodeCallsRuntime(t *testing.T) {
	// Verify that when 2+ sequences are in decode state,
	// the scheduler calls DecodeWithPagedKVBatched instead of looping.
	// Use a mock runtime that records which method was called.
	t.Skip("TODO: requires mock runtime interface")
}
```

- [ ] **Step 2: Replace the loop at scheduler.go:324-367 with batched call**

In `inference/scheduler/scheduler.go`, replace the multi-sequence fallback block:

```go
} else if usePagedCache {
	if len(decodeSeqs) == 1 {
		logits, err = s.runtime.DecodeWithPagedKV(tokens, seqIDs[0], positions[0])
	} else {
		// Batched decode for multiple sequences
		logits, err = s.runtime.DecodeWithPagedKVBatched(tokens, seqIDs, positions)
	}
}
```

This replaces the entire per-sequence loop (lines 324-367) with a single call. The sampling logic after this block (lines 378+) already handles batched logits — it iterates over sequences and extracts per-sequence logits from the batch.

- [ ] **Step 3: Verify the existing sampling path handles batched logits**

Read scheduler.go lines 378-420 to confirm the post-decode sampling handles `logits [batchSize, vocabSize]` correctly. The `getLogitsOnCPU` call and per-sequence sampling loop should already work since the single-sequence GPU path and the BatchRuntimeInputs path both produce batched logits.

- [ ] **Step 4: Run full test suite**

Run: `go test -tags metal -v ./inference/scheduler/...`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference/scheduler/scheduler.go inference/scheduler/batched_decode_test.go
git commit -m "feat(scheduler): replace sequential decode loop with DecodeWithPagedKVBatched"
```

---

### Task 7: Integration test with multiple concurrent sequences

**Files:**
- Modify: `inference/scheduler/batched_decode_test.go`

- [ ] **Step 1: Write integration test**

```go
func TestBatchedDecodeE2E(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		t.Skip("VEXEL_TEST_MODEL not set")
	}

	// Create scheduler with MaxBatchSize=4
	// Submit 3 sequences concurrently
	// Verify all 3 complete and produce valid output
	// Compare output against single-sequence runs for determinism (temp=0)
}
```

- [ ] **Step 2: Run test**

Run: `VEXEL_TEST_MODEL=models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf go test -tags metal -run TestBatchedDecodeE2E -v ./inference/scheduler/`
Expected: PASS (or SKIP if no model)

- [ ] **Step 3: Commit**

```bash
git add inference/scheduler/batched_decode_test.go
git commit -m "test(scheduler): add E2E integration test for batched decode"
```
