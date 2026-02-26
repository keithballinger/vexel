//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// TestPagedKVRoundtripProfile measures the GPU↔CPU data transfer overhead
// in the current paged KV cache path vs GPU-native alternatives.
//
// Track 3: Paged KV Batching, Phase 1: Investigation & Profiling.
//
// Current ExecuteWithPagedKV path (per layer):
//
//	K,V on GPU → Sync → ToHost(16KB×2) → CPU cache → ToDevice(grows) → Sync → SDPA
//
// Target path (Phase 2):
//
//	K,V on GPU → ReshapePagedKV scatter → PagedSDPA (no CPU roundtrip)
func TestPagedKVRoundtripProfile(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		numKVHeads = 32
		headDim    = 128
		numQHeads  = 32
		numLayers  = 32
		blockSize  = 16
		iters      = 100
	)

	newKVElems := numKVHeads * headDim // single decode token per layer
	newKVBytes := newKVElems * 4       // 16 KB

	seqLens := []int{64, 256, 512, 1024, 2048}

	fmt.Println("\n[PAGED KV ROUNDTRIP PROFILE]")
	fmt.Println("Model: LLaMA 2 7B (32 KV heads, dim 128, 32 layers)")
	fmt.Printf("New K/V per layer: %d floats = %d KB × 2 (K+V)\n\n", newKVElems, newKVBytes/1024)

	for _, kvLen := range seqLens {
		fullKVElems := kvLen * numKVHeads * headDim
		fullKVBytes := fullKVElems * 4
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		kvStride := kvLen * headDim

		// GPU buffers
		qBuf := b.Alloc(numQHeads * headDim * 4)
		newKBuf := b.Alloc(newKVBytes)
		newVBuf := b.Alloc(newKVBytes)
		fullKBuf := b.Alloc(fullKVBytes)
		fullVBuf := b.Alloc(fullKVBytes)
		outBuf := b.Alloc(numQHeads * headDim * 4)

		b.Zero(qBuf, numQHeads*headDim)
		b.Zero(newKBuf, newKVElems)
		b.Zero(newVBuf, newKVElems)
		b.Zero(fullKBuf, fullKVElems)
		b.Zero(fullVBuf, fullKVElems)
		b.Sync()

		// CPU buffers for roundtrip simulation
		cpuNewK := make([]byte, newKVBytes)
		cpuNewV := make([]byte, newKVBytes)
		cpuFullK := make([]byte, fullKVBytes)
		cpuFullV := make([]byte, fullKVBytes)

		// --- Measure GPU→CPU: new K/V (matches block.go lines 648-656) ---
		b.Sync()
		start := time.Now()
		for i := 0; i < iters; i++ {
			b.Sync()
			b.ToHost(cpuNewK, newKBuf)
			b.ToHost(cpuNewV, newVBuf)
			b.Sync()
		}
		gpuToCPU := time.Since(start) / time.Duration(iters)

		// --- Measure CPU→GPU: full K/V (matches block.go lines 679-688) ---
		start = time.Now()
		for i := 0; i < iters; i++ {
			b.ToDevice(fullKBuf, cpuFullK)
			b.ToDevice(fullVBuf, cpuFullV)
			b.Sync()
		}
		cpuToGPU := time.Since(start) / time.Duration(iters)

		// --- Measure SDPA decode ---
		for i := 0; i < 5; i++ {
			b.SDPA(qBuf, fullKBuf, fullVBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvStride)
			b.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			b.SDPA(qBuf, fullKBuf, fullVBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvStride)
			b.Sync()
		}
		sdpaTime := time.Since(start) / time.Duration(iters)

		// --- Measure ReshapePagedKV scatter (existing kernel) ---
		numBlocks := (kvLen + blockSize - 1) / blockSize
		blockElems := blockSize * numKVHeads * headDim * 2
		poolBuf := b.Alloc(numBlocks * blockElems * 4)
		b.Zero(poolBuf, numBlocks*blockElems)
		ptBuf := b.Alloc(4)
		offBuf := b.Alloc(4)
		b.ToDevice(ptBuf, []byte{0, 0, 0, 0})
		b.ToDevice(offBuf, []byte{0, 0, 0, 0})
		b.Sync()

		for i := 0; i < 5; i++ {
			b.ReshapePagedKV(newKBuf, poolBuf, ptBuf, offBuf, 1, numKVHeads, headDim, blockSize, false)
			b.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			b.ReshapePagedKV(newKBuf, poolBuf, ptBuf, offBuf, 1, numKVHeads, headDim, blockSize, false)
			b.ReshapePagedKV(newVBuf, poolBuf, ptBuf, offBuf, 1, numKVHeads, headDim, blockSize, true)
			b.Sync()
		}
		reshapeTime := time.Since(start) / time.Duration(iters)

		// --- Per-token aggregation (×32 layers) ---
		roundtripPerLayer := gpuToCPU + cpuToGPU
		roundtripTotal := roundtripPerLayer * time.Duration(numLayers)
		sdpaTotal := sdpaTime * time.Duration(numLayers)
		reshapeTotal := reshapeTime * time.Duration(numLayers)
		currentPerToken := roundtripTotal + sdpaTotal
		targetPerToken := reshapeTotal + sdpaTotal

		fmt.Printf("--- kvLen=%d (full KV: %.1f MB K+V) ---\n", kvLen, float64(fullKVBytes*2)/1e6)
		fmt.Printf("  Single layer:\n")
		fmt.Printf("    GPU→CPU (32KB new KV):      %7v\n", gpuToCPU)
		fmt.Printf("    CPU→GPU (%.1fMB full KV):  %7v\n", float64(fullKVBytes*2)/1e6, cpuToGPU)
		fmt.Printf("    SDPA decode:                %7v\n", sdpaTime)
		fmt.Printf("    ReshapePagedKV (scatter):   %7v\n", reshapeTime)
		fmt.Printf("  Per-token (×%d layers):\n", numLayers)
		fmt.Printf("    Current (roundtrip+SDPA):   %7v\n", currentPerToken)
		fmt.Printf("    Target  (reshape+SDPA):     %7v\n", targetPerToken)
		fmt.Printf("    Roundtrip overhead:          %.0f%% of current\n",
			float64(roundtripTotal)/float64(currentPerToken)*100)
		if targetPerToken > 0 {
			fmt.Printf("    Projected speedup:          %.2f×\n",
				float64(currentPerToken)/float64(targetPerToken))
		}
		fmt.Println()

		b.Free(qBuf)
		b.Free(newKBuf)
		b.Free(newVBuf)
		b.Free(fullKBuf)
		b.Free(fullVBuf)
		b.Free(outBuf)
		b.Free(poolBuf)
		b.Free(ptBuf)
		b.Free(offBuf)
	}

	fmt.Println("NOTE: CPU cache store/gather adds additional overhead not measured here.")
	fmt.Println("Actual roundtrip overhead in ExecuteWithPagedKV is higher.")
}

// TestPagedAttentionKernelAudit documents existing Metal kernel capabilities
// and identifies the gap that Phase 2 must fill for GPU-native paged attention.
//
// Track 3: Paged KV Batching, Phase 1: Investigation & Profiling.
func TestPagedAttentionKernelAudit(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Functional verification: ReshapePagedKV scatter works correctly.
	// Confirms GPU-side block pool infrastructure is in place for Phase 2.
	numTokens := 4
	numKVHeads := 2
	headDim := 4
	blockSize := 4

	srcElems := numTokens * numKVHeads * headDim
	srcBuf := b.Alloc(srcElems * 4)
	defer b.Free(srcBuf)

	srcData := make([]float32, srcElems)
	for tok := 0; tok < numTokens; tok++ {
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < headDim; d++ {
				srcData[tok*numKVHeads*headDim+h*headDim+d] = float32(tok*100 + h*10 + d)
			}
		}
	}
	b.ToDevice(srcBuf, float32ToBytes(srcData))

	blockElems := blockSize * numKVHeads * headDim * 2 // K+V per block
	poolBuf := b.Alloc(2 * blockElems * 4)
	defer b.Free(poolBuf)
	b.Zero(poolBuf, 2*blockElems)

	// All 4 tokens → block 0, offsets 0-3
	ptData := make([]byte, numTokens*4) // zeros = block 0
	offData := make([]byte, numTokens*4)
	for i := 0; i < numTokens; i++ {
		offData[i*4] = byte(i)
	}
	ptBuf := b.Alloc(numTokens * 4)
	offBuf := b.Alloc(numTokens * 4)
	defer b.Free(ptBuf)
	defer b.Free(offBuf)
	b.ToDevice(ptBuf, ptData)
	b.ToDevice(offBuf, offData)
	b.Sync()

	b.ReshapePagedKV(srcBuf, poolBuf, ptBuf, offBuf, numTokens, numKVHeads, headDim, blockSize, false)
	b.Sync()

	poolOut := make([]byte, 2*blockElems*4)
	b.ToHost(poolOut, poolBuf)
	poolData := bytesToFloat32(poolOut)

	for tok := 0; tok < numTokens; tok++ {
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < headDim; d++ {
				expected := float32(tok*100 + h*10 + d)
				idx := tok*numKVHeads*headDim + h*headDim + d
				got := poolData[idx]
				if got != expected {
					t.Fatalf("K[tok=%d,h=%d,d=%d]: got %v, want %v", tok, h, d, got, expected)
				}
			}
		}
	}

	// --- Kernel Audit Report ---
	fmt.Println("\n[PAGED ATTENTION KERNEL AUDIT]")
	fmt.Println("===============================================")
	fmt.Println()
	fmt.Println("EXISTING Metal Kernels (verified):")
	fmt.Println("  ✓ reshape_paged_kv_f32")
	fmt.Println("      Scatter K/V from contiguous → GPU-resident paged blocks")
	fmt.Println("      Input:  src [numTokens, numKVHeads, headDim] contiguous")
	fmt.Println("      Output: block[blockID][offset] in GPU block pool")
	fmt.Println("      Layout: K [blockSize, numKVHeads, headDim] + V [blockSize, ...]")
	fmt.Println("      Via:    pageTable[tok] → blockID, blockOffsets[tok] → pos")
	fmt.Println()
	fmt.Println("  ✓ sdpa_gqa_f32           (decode, kvLen < 16)")
	fmt.Println("  ✓ sdpa_flash_decode_f32   (decode, kvLen >= 16)")
	fmt.Println("  ✓ sdpa_prefill_f32        (prefill, seqLen < 32)")
	fmt.Println("  ✓ flash_attention_2_f32   (prefill, seqLen >= 32)")
	fmt.Println("      ALL require CONTIGUOUS K/V buffers (no block table support)")
	fmt.Println()
	fmt.Println("MISSING Kernels (Phase 2 targets):")
	fmt.Println("  ✗ sdpa_paged_decode_f32")
	fmt.Println("      SDPA decode reading K/V directly from paged block pool")
	fmt.Println("      Q:       [numQHeads, headDim]")
	fmt.Println("      KV pool: base pointer, accessed via block_table")
	fmt.Println("      Params:  blockTable[numBlocks], numBlocks, blockSize")
	fmt.Println("      Strategy: flash-decode with online softmax across blocks")
	fmt.Println("      Each threadgroup handles one Q head, iterates blocks")
	fmt.Println()
	fmt.Println("  △ flash_attention_2_paged_f32 (optional, lower priority)")
	fmt.Println("      Paged prefill for multi-sequence batched prefill")
	fmt.Println("      Single-sequence prefill can gather into contiguous buffer")
	fmt.Println()
	fmt.Println("INTERFACE CONTRACT for sdpa_paged_decode_f32:")
	fmt.Println("  Buffers:")
	fmt.Println("    Q:           [numQHeads, headDim] float32")
	fmt.Println("    kvPool:      base pointer to block pool")
	fmt.Println("    blockTable:  [numBlocks] int32 (logical → physical)")
	fmt.Println("    out:         [numQHeads, headDim] float32")
	fmt.Println("  Scalars:")
	fmt.Println("    numBlocks, blockSize, numQHeads, numKVHeads, headDim")
	fmt.Println("    scale, tokensInLastBlock (for partial last block)")
	fmt.Println()
	fmt.Println("  Block pool layout (per physical block):")
	fmt.Println("    K: [blockSize, numKVHeads, headDim] float32")
	fmt.Println("    V: [blockSize, numKVHeads, headDim] float32")
	fmt.Println("    Total: 2 × blockSize × numKVHeads × headDim × 4 bytes")
	fmt.Println("    LLaMA 7B (bs=16): 2×16×32×128×4 = 512 KB/block")
	fmt.Println()
	fmt.Println("CURRENT vs TARGET data flow (per layer, decode):")
	fmt.Println("  Current: K,V [GPU] → Sync → ToHost → CPU cache →")
	fmt.Println("           ToDevice → Sync → SDPA [GPU]")
	fmt.Println("  Target:  K,V [GPU] → ReshapePagedKV → pool [GPU] →")
	fmt.Println("           PagedSDPA [GPU] (zero CPU involvement)")
	fmt.Println("===============================================")
}
