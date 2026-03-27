# Long-Context SDPA Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce decode throughput degradation from ~20% to <10% at context length 2048 (vs ctx=16 baseline), closing the gap to llama.cpp's ~2.7% degradation.

**Architecture:** Three-pronged approach: (1) Tune NWG kernel tile size from fixed 256 to adaptive based on kvLen, improving occupancy at medium contexts (256-1024). (2) Promote the experimental tiled split-K kernel to auto-selection for kvLen > 2048 with managed buffer lifecycle. (3) Add an F16 paged SDPA kernel (currently F32-only) to combine paged KV with the fastest decode path. All work is in the Metal backend kernel dispatch layer.

**Tech Stack:** Go 1.25.4, Metal GPU kernels (Obj-C via cgo), FlashAttention-2, NWG multi-threadgroup dispatch

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/backend/metal/backend.go` | SDPA dispatch logic, NWG tile tuning, tiled auto-selection |
| Modify | `inference/runtime/plan.go` | Add SDPA kernel selection thresholds |
| Create | `inference/backend/metal/sdpa_context_scaling_test.go` | Comprehensive context scaling benchmark |
| Modify | `inference/backend/metal/sdpa_flash_f16_nwg_test.go` | Add adaptive tile tests |
| Modify | `inference/backend/metal/sdpa_flash_f16_tiled_test.go` | Add auto-selection tests |

---

### Task 1: Establish context scaling baseline benchmark

**Files:**
- Create: `inference/backend/metal/sdpa_context_scaling_test.go`

- [ ] **Step 1: Write comprehensive benchmark that measures degradation across context lengths**

Create `inference/backend/metal/sdpa_context_scaling_test.go`:

```go
//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"vexel/inference/tensor"
)

// TestContextScalingBaseline measures decode SDPA throughput across context lengths
// to establish a degradation baseline before optimization.
func TestContextScalingBaseline(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	numQHeads, numKVHeads, headDim := 32, 32, 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvHeadStride := 4096 * headDim // maxSeqLen * headDim

	// Allocate Q: [numQHeads, headDim]
	qSize := numQHeads * headDim
	qData := make([]float32, qSize)
	for i := range qData {
		qData[i] = rand.Float32()*2 - 1
	}
	qPtr := b.Alloc(qSize * 4)
	defer b.Free(qPtr)
	b.ToDevice(qPtr, float32ToBytes(qData))

	// Allocate KV cache: [numKVHeads, maxSeqLen, headDim] in F16
	maxSeqLen := 4096
	kvBytes := numKVHeads * maxSeqLen * headDim * 2 // F16
	kPtr := b.Alloc(kvBytes)
	vPtr := b.Alloc(kvBytes)
	defer b.Free(kPtr)
	defer b.Free(vPtr)
	// Fill with random F16 data
	kvData := make([]byte, kvBytes)
	rand.Read(kvData)
	b.ToDevice(kPtr, kvData)
	b.ToDevice(vPtr, kvData)

	outPtr := b.Alloc(qSize * 4)
	defer b.Free(outPtr)

	contexts := []int{16, 32, 64, 128, 256, 512, 1024, 2048}
	numLayers := 32
	warmup := 5
	iters := 20

	baselineTokS := 0.0

	for _, kvLen := range contexts {
		// Warm up
		for i := 0; i < warmup; i++ {
			b.SDPAF16(qPtr, kPtr, vPtr, outPtr, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		}
		b.Sync()

		// Benchmark: simulate full model (numLayers SDPA calls per token)
		start := time.Now()
		for i := 0; i < iters; i++ {
			for l := 0; l < numLayers; l++ {
				b.SDPAF16(qPtr, kPtr, vPtr, outPtr, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			b.Sync()
		}
		elapsed := time.Since(start)

		tokS := float64(iters) / elapsed.Seconds()
		if kvLen == 16 {
			baselineTokS = tokS
		}
		degrad := 0.0
		if baselineTokS > 0 {
			degrad = (1.0 - tokS/baselineTokS) * 100
		}

		t.Logf("kvLen=%4d: %.1f tok/s (degradation: %.1f%%)", kvLen, tokS, degrad)
	}
}
```

- [ ] **Step 2: Run baseline benchmark**

Run: `go test -tags metal -run TestContextScalingBaseline -v -timeout 120s ./inference/backend/metal/`
Expected: Output shows degradation percentages. Record baseline numbers.

- [ ] **Step 3: Commit**

```bash
git add inference/backend/metal/sdpa_context_scaling_test.go
git commit -m "test(metal): add context scaling baseline benchmark for SDPA"
```

---

### Task 2: Tune NWG tile size for medium contexts

**Files:**
- Modify: `inference/backend/metal/backend.go`
- Modify: `inference/backend/metal/sdpa_flash_f16_nwg_test.go`

- [ ] **Step 1: Read current NWG dispatch in SDPAF16**

Read `inference/backend/metal/backend.go` around lines 1992-2034 to find the NWG dispatch. Look for the fixed tile size (256 positions per threadgroup) and the `kvLen > 64` activation threshold.

- [ ] **Step 2: Write test for adaptive tile sizing**

Add to `inference/backend/metal/sdpa_flash_f16_nwg_test.go`:

```go
func TestNWGAdaptiveTileSize(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	numQHeads, numKVHeads, headDim := 32, 32, 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Test that tile size adapts for different kvLen ranges
	// kvLen=128: should use smaller tiles (64 or 128) for better occupancy
	// kvLen=2048: should use larger tiles (256) for amortized merge overhead
	testCases := []struct {
		kvLen       int
		description string
	}{
		{128, "short context - smaller tiles preferred"},
		{512, "medium context - moderate tiles"},
		{2048, "long context - large tiles for merge amortization"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// Allocate, run, verify correctness against v3 reference
			// ... (standard SDPA test setup)
			// Key assertion: output matches v3 within tolerance
		})
	}
}
```

- [ ] **Step 3: Run test to verify current behavior**

Run: `go test -tags metal -run TestNWGAdaptiveTileSize -v ./inference/backend/metal/`

- [ ] **Step 4: Modify NWG dispatch to use adaptive tile size**

In `inference/backend/metal/backend.go`, modify the NWG dispatch section. Replace the fixed `tileSize := 256` with:

```go
// Adaptive tile sizing based on context length
// Short contexts: smaller tiles = more TGs = better occupancy
// Long contexts: larger tiles = fewer TGs = less merge overhead
var tileSize int
switch {
case kvLen <= 128:
	tileSize = 64
case kvLen <= 512:
	tileSize = 128
default:
	tileSize = 256
}
numTGs := (kvLen + tileSize - 1) / tileSize
```

Update the scratch buffer allocation to use the computed tileSize:

```go
partialsSize := numQHeads * numTGs * (2 + headDim) * 4
```

- [ ] **Step 5: Run context scaling benchmark to measure improvement**

Run: `go test -tags metal -run TestContextScalingBaseline -v -timeout 120s ./inference/backend/metal/`
Expected: Degradation at kvLen=256 and kvLen=512 should improve (target: <8% at 512)

- [ ] **Step 6: Run full SDPA test suite to verify correctness**

Run: `go test -tags metal -run "TestSDPA|TestNWG|TestFlash" -v ./inference/backend/metal/`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add inference/backend/metal/backend.go inference/backend/metal/sdpa_flash_f16_nwg_test.go
git commit -m "perf(metal): adaptive NWG tile sizing for improved medium-context SDPA throughput"
```

---

### Task 3: Promote tiled split-K kernel to auto-selection

**Files:**
- Modify: `inference/backend/metal/backend.go`
- Modify: `inference/backend/metal/sdpa_flash_f16_tiled_test.go`

- [ ] **Step 1: Read current tiled kernel dispatch**

Read `inference/backend/metal/backend.go` to find where `sdpa_flash_decode_f16_tiled` is dispatched. Currently it's NOT auto-selected — only available via `VEXEL_FORCE_SDPA` override.

- [ ] **Step 2: Read tiled kernel test to understand buffer management**

Read `inference/backend/metal/sdpa_flash_f16_tiled_test.go` lines 100-150 to see how partials buffers are allocated and the merge step works.

- [ ] **Step 3: Write test for auto-selection at kvLen > 2048**

Add to `inference/backend/metal/sdpa_flash_f16_tiled_test.go`:

```go
func TestTiledAutoSelectionAtLongContext(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	numQHeads, numKVHeads, headDim := 32, 32, 128
	kvLen := 4096 // Should trigger tiled auto-selection
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Run SDPAF16 at kvLen=4096 - should auto-select tiled kernel
	// Verify output matches NWG reference within tolerance
	// ... (standard SDPA setup with large KV cache)
}
```

- [ ] **Step 4: Add tiled auto-selection to SDPAF16 dispatch**

In `inference/backend/metal/backend.go`, modify the SDPAF16 dispatch chain to add tiled kernel selection for very long contexts:

```go
// Dispatch priority for decode SDPA:
// 1. kvLen > 2048 && tiled pipeline available → tiled split-K (best occupancy)
// 2. kvLen > 64 && NWG pipeline available → NWG multi-TG (good for 64-2048)
// 3. v3 pipeline available → chunk-based (default)
// 4. v1 fallback

if kvLen > 2048 && b.sdpaFlashDecodeF16TiledPipeline != nil && headDim%32 == 0 {
	// Tiled split-K: allocate partials from scratch, dispatch tiles, merge
	tileKV := 64
	numTiles := (kvLen + tileKV - 1) / tileKV
	partialsSize := numQHeads * numTiles * (2 + headDim) * 4

	partialsPtr := b.ScratchAlloc(partialsSize)
	// ... dispatch tiled kernel with partials
	// ... dispatch merge kernel
	return
}
```

- [ ] **Step 5: Manage partials buffer lifecycle**

The tiled kernel needs a temporary partials buffer. Use the scratch allocator:

```go
if scratchAlloc, ok := b.(backend.ScratchAllocator); ok {
	partialsPtr := scratchAlloc.ScratchAlloc(partialsSize)
	// dispatch with partialsPtr
} else {
	// Fallback: allocate, dispatch, free
	partialsPtr := b.Alloc(partialsSize)
	defer b.Free(partialsPtr)
}
```

- [ ] **Step 6: Run context scaling benchmark**

Run: `go test -tags metal -run TestContextScalingBaseline -v -timeout 120s ./inference/backend/metal/`
Expected: kvLen=2048 degradation should improve (target: <15%)

- [ ] **Step 7: Run full test suite**

Run: `go test -tags metal -v ./inference/backend/metal/`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add inference/backend/metal/backend.go inference/backend/metal/sdpa_flash_f16_tiled_test.go
git commit -m "perf(metal): auto-select tiled split-K SDPA for kvLen > 2048"
```

---

### Task 4: Update execution plan with SDPA thresholds

**Files:**
- Modify: `inference/runtime/plan.go`

- [ ] **Step 1: Read current SDPA kernel selection in plan.go**

Read `inference/runtime/plan.go` lines 375-389 to see how `SDPADecode` kernel is set in the execution plan.

- [ ] **Step 2: Add context-aware SDPA selection to plan**

The execution plan is static (set at model load time), but the actual kernel selection in `SDPAF16()` is dynamic based on kvLen. Document this in the plan config so the runtime knows which kernels are available:

```go
// In the plan's Kernels section, add dispatch thresholds:
plan.Kernels = KernelVariants{
	SDPADecode:  "sdpa_flash_decode_f16_adaptive", // Auto-selects based on kvLen
	SDPAPrefill: "flash_attention_2_f16",
	// ...existing entries...
}

// Add to TuningParams:
plan.Tuning.SDPANWGThreshold = 64    // Use NWG for kvLen > this
plan.Tuning.SDPATiledThreshold = 2048 // Use tiled for kvLen > this
plan.Tuning.SDPANWGTileShort = 64     // NWG tile size for short contexts
plan.Tuning.SDPANWGTileMedium = 128   // NWG tile size for medium contexts
plan.Tuning.SDPANWGTileLong = 256     // NWG tile size for long contexts
```

- [ ] **Step 3: Add env var override for tiled threshold**

In `applyEnvOverrides`:

```go
if v := os.Getenv("VEXEL_SDPA_TILED_THRESHOLD"); v != "" {
	if n, err := strconv.Atoi(v); err == nil {
		plan.Tuning.SDPATiledThreshold = n
	}
}
```

- [ ] **Step 4: Run build**

Run: `CGO_ENABLED=1 go build -tags metal ./inference/runtime/...`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add inference/runtime/plan.go
git commit -m "feat(runtime): add context-aware SDPA kernel thresholds to execution plan"
```

---

### Task 5: Add F16 paged SDPA kernel

**Files:**
- Modify: `inference/backend/backend.go`
- Modify: `inference/backend/metal/backend.go`

- [ ] **Step 1: Check current paged SDPA is F32**

Read `inference/backend/metal/backend.go` around lines 2298-2310 (SDPAPagedDecode) to confirm it uses F32 pipeline.

- [ ] **Step 2: Write failing test for F16 paged SDPA**

Add to `inference/backend/metal/paged_sdpa_test.go`:

```go
func TestSDPAPagedDecodeF16(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	// Test that paged SDPA with F16 KV produces results matching F32 path
	// within F16 tolerance (~1e-3)
	numQHeads, numKVHeads, headDim := 4, 4, 64
	blockSize := 16
	seqLen := 48 // 3 blocks
	// ... standard paged SDPA setup but verify F16 path is used
}
```

- [ ] **Step 3: Add SDPAPagedDecodeF16 to PagedKVOps interface**

In `inference/backend/backend.go`:

```go
// SDPAPagedDecodeF16 performs paged SDPA with F16 KV cache blocks.
// Same semantics as SDPAPagedDecode but operates on F16 data for 2x bandwidth.
SDPAPagedDecodeF16(q, kvPool, blockTable, out tensor.DevicePtr,
	numBlocks, blockSize, numQHeads, numKVHeads, headDim int,
	scale float32, tokensInLastBlock int)
```

- [ ] **Step 4: Implement in Metal backend**

In `inference/backend/metal/backend.go`, implement using the existing Flash F16 v3 kernel adapted for paged layout. Initial implementation can convert paged blocks to contiguous and call SDPAF16:

```go
func (b *Backend) SDPAPagedDecodeF16(q, kvPool, blockTable, out tensor.DevicePtr,
	numBlocks, blockSize, numQHeads, numKVHeads, headDim int,
	scale float32, tokensInLastBlock int) {
	// Gather paged KV into contiguous buffer, then call SDPAF16
	kvLen := (numBlocks-1)*blockSize + tokensInLastBlock
	kvHeadStride := kvLen * headDim

	// Allocate temporary contiguous K, V buffers
	kvSize := numKVHeads * kvLen * headDim * 2 // F16
	kContig := b.Alloc(kvSize)
	vContig := b.Alloc(kvSize)
	defer b.Free(kContig)
	defer b.Free(vContig)

	// Gather from paged blocks to contiguous (use existing scatter kernel in reverse)
	// ... dispatch gather kernel ...

	b.SDPAF16(q, kContig, vContig, out, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
}
```

- [ ] **Step 5: Run tests**

Run: `go test -tags metal -run TestSDPAPagedDecodeF16 -v ./inference/backend/metal/`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add inference/backend/backend.go inference/backend/metal/backend.go
git commit -m "feat(metal): add F16 paged SDPA kernel for improved bandwidth utilization"
```

---

### Task 6: Final benchmark and regression verification

- [ ] **Step 1: Run full context scaling benchmark**

Run: `go test -tags metal -run TestContextScalingBaseline -v -timeout 120s ./inference/backend/metal/`
Expected:
- kvLen=512: degradation < 8% (was ~10%)
- kvLen=2048: degradation < 15% (was ~20%)

- [ ] **Step 2: Run all SDPA-related tests**

Run: `go test -tags metal -run "TestSDPA|TestFlash|TestNWG|TestTiled|TestPaged|TestContext" -v ./inference/backend/metal/`
Expected: All pass

- [ ] **Step 3: Run full test suite for regressions**

Run: `make test-metal`
Expected: All pass

- [ ] **Step 4: Commit any final adjustments**

```bash
git add -A
git commit -m "perf(metal): context scaling optimization - reduce degradation at long contexts"
```
