//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// TestSDPAFlashDecodeF16Tiled tests the tiled split-K SDPA F16 decode kernel
// against the CPU reference. The tiled kernel splits KV positions into tiles
// (TILE_KV=64) for better GPU occupancy, then merges partials.
func TestSDPAFlashDecodeF16Tiled(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16TiledPipeline == nil {
		t.Skip("Tiled F16 SDPA pipeline not available")
	}

	tests := []struct {
		name       string
		kvLen      int
		numQHeads  int
		numKVHeads int
		headDim    int
		tol        float64
	}{
		// Edge cases: fewer positions than one tile (64)
		{"kvLen=1_gqa4x_hd128", 1, 32, 8, 128, 5e-3},
		{"kvLen=4_gqa4x_hd128", 4, 32, 8, 128, 5e-3},
		{"kvLen=7_gqa4x_hd128", 7, 32, 8, 128, 5e-3},
		{"kvLen=16_gqa4x_hd128", 16, 32, 8, 128, 5e-3},
		// One full tile
		{"kvLen=64_gqa4x_hd128", 64, 32, 8, 128, 5e-3},
		// Multiple tiles
		{"kvLen=128_gqa4x_hd128", 128, 32, 8, 128, 5e-3},
		{"kvLen=256_gqa4x_hd128", 256, 32, 8, 128, 5e-3},
		{"kvLen=512_gqa4x_hd128", 512, 32, 8, 128, 5e-3},
		// Stress test
		{"kvLen=1024_gqa4x_hd128", 1024, 32, 8, 128, 5e-3},
		// Non-tile-aligned
		{"kvLen=100_gqa4x_hd128", 100, 32, 8, 128, 5e-3},
		{"kvLen=200_gqa4x_hd128", 200, 32, 8, 128, 5e-3},
		// No GQA
		{"kvLen=64_noGQA_hd128", 64, 32, 32, 128, 5e-3},
		// GQA 8:1
		{"kvLen=64_gqa8x_hd128", 64, 32, 4, 128, 5e-3},
		// headDim=64
		{"kvLen=128_gqa4x_hd64", 128, 8, 2, 64, 5e-3},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			kvLen := tc.kvLen
			numQHeads := tc.numQHeads
			numKVHeads := tc.numKVHeads
			headDim := tc.headDim
			scale := float32(1.0 / math.Sqrt(float64(headDim)))

			// Generate deterministic test data
			qData := make([]float32, numQHeads*headDim)
			kData := make([]float32, numKVHeads*kvLen*headDim)
			vData := make([]float32, numKVHeads*kvLen*headDim)

			for i := range qData {
				qData[i] = float32(i%17-8) * 0.05
			}
			for i := range kData {
				kData[i] = float32(i%13-6) * 0.05
			}
			for i := range vData {
				vData[i] = float32(i%11-5) * 0.05
			}

			// CPU reference
			expected := cpuSDPADecodeF32(qData, kData, vData, kvLen, numQHeads, numKVHeads, headDim, scale)

			// Allocate F32 staging buffers
			qF32 := backend.Alloc(len(qData) * 4)
			kF32 := backend.Alloc(len(kData) * 4)
			vF32 := backend.Alloc(len(vData) * 4)
			defer backend.Free(qF32)
			defer backend.Free(kF32)
			defer backend.Free(vF32)

			backend.ToDevice(qF32, float32ToBytes(qData))
			backend.ToDevice(kF32, float32ToBytes(kData))
			backend.ToDevice(vF32, float32ToBytes(vData))

			// Convert to F16
			qF16 := backend.Alloc(len(qData) * 2)
			kF16 := backend.Alloc(len(kData) * 2)
			vF16 := backend.Alloc(len(vData) * 2)
			outF16 := backend.Alloc(numQHeads * headDim * 2)
			defer backend.Free(qF16)
			defer backend.Free(kF16)
			defer backend.Free(vF16)
			defer backend.Free(outF16)

			backend.ConvertF32ToF16(qF32, qF16, len(qData))
			backend.ConvertF32ToF16(kF32, kF16, len(kData))
			backend.ConvertF32ToF16(vF32, vF16, len(vData))
			backend.Sync()

			// Allocate partials buffer: numQHeads * numTiles * (2 + headDim) * 4 bytes
			tileKV := 64
			numTiles := (kvLen + tileKV - 1) / tileKV
			partialsSize := numQHeads * numTiles * (2 + headDim) * 4
			partialsBuf := backend.Alloc(partialsSize)
			defer backend.Free(partialsBuf)

			kvHeadStride := kvLen * headDim
			backend.SDPAF16Tiled(qF16, kF16, vF16, outF16, partialsBuf,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Read back and compare
			outF32Buf := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(outF32Buf)
			backend.ConvertF16ToF32(outF16, outF32Buf, numQHeads*headDim)
			backend.Sync()

			outBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(outBytes, outF32Buf)
			result := bytesToFloat32(outBytes)

			maxDiff := 0.0
			maxDiffIdx := 0
			for i := range expected {
				if math.IsNaN(float64(result[i])) || math.IsInf(float64(result[i]), 0) {
					t.Fatalf("NaN/Inf at index %d (head=%d, dim=%d)", i, i/headDim, i%headDim)
				}
				diff := math.Abs(float64(result[i] - expected[i]))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
			}

			t.Logf("kvLen=%d: max_diff=%.6f at idx=%d (tol=%.4f)", kvLen, maxDiff, maxDiffIdx, tc.tol)
			if maxDiff > tc.tol {
				head := maxDiffIdx / headDim
				dim := maxDiffIdx % headDim
				t.Fatalf("Mismatch: diff=%.6f > tol=%.4f at head=%d dim=%d (expected=%.6f, got=%.6f)",
					maxDiff, tc.tol, head, dim, expected[maxDiffIdx], result[maxDiffIdx])
			}
		})
	}
}

// TestSDPAFlashDecodeF16Tiled_vs_NonTiled verifies the tiled kernel produces
// the same results as the existing non-tiled F16 SDPA kernel.
func TestSDPAFlashDecodeF16Tiled_vs_NonTiled(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16TiledPipeline == nil {
		t.Skip("Tiled F16 SDPA pipeline not available")
	}
	if backend.sdpaFlashDecodeF16Pipeline == nil {
		t.Skip("Flash F16 SDPA pipeline not available")
	}

	// LLaMA 2 7B config
	numQHeads := 32
	numKVHeads := 8
	headDim := 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	contextLengths := []int{16, 64, 128, 256, 512}

	for _, kvLen := range contextLengths {
		t.Run(fmt.Sprintf("ctx=%d", kvLen), func(t *testing.T) {
			// Generate data
			qData := make([]float32, numQHeads*headDim)
			kData := make([]float32, numKVHeads*kvLen*headDim)
			vData := make([]float32, numKVHeads*kvLen*headDim)

			for i := range qData {
				qData[i] = float32(i%17-8) * 0.05
			}
			for i := range kData {
				kData[i] = float32(i%13-6) * 0.05
			}
			for i := range vData {
				vData[i] = float32(i%11-5) * 0.05
			}

			// Allocate and convert
			qF32 := backend.Alloc(len(qData) * 4)
			kF32 := backend.Alloc(len(kData) * 4)
			vF32 := backend.Alloc(len(vData) * 4)
			defer backend.Free(qF32)
			defer backend.Free(kF32)
			defer backend.Free(vF32)

			backend.ToDevice(qF32, float32ToBytes(qData))
			backend.ToDevice(kF32, float32ToBytes(kData))
			backend.ToDevice(vF32, float32ToBytes(vData))

			qF16 := backend.Alloc(len(qData) * 2)
			kF16 := backend.Alloc(len(kData) * 2)
			vF16 := backend.Alloc(len(vData) * 2)
			defer backend.Free(qF16)
			defer backend.Free(kF16)
			defer backend.Free(vF16)

			backend.ConvertF32ToF16(qF32, qF16, len(qData))
			backend.ConvertF32ToF16(kF32, kF16, len(kData))
			backend.ConvertF32ToF16(vF32, vF16, len(vData))
			backend.Sync()

			kvHeadStride := kvLen * headDim

			// Non-tiled result
			outNonTiled := backend.Alloc(numQHeads * headDim * 2)
			defer backend.Free(outNonTiled)
			backend.SDPAF16(qF16, kF16, vF16, outNonTiled, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Tiled result
			outTiled := backend.Alloc(numQHeads * headDim * 2)
			defer backend.Free(outTiled)
			tileKV := 64
			numTiles := (kvLen + tileKV - 1) / tileKV
			partialsBuf := backend.Alloc(numQHeads * numTiles * (2 + headDim) * 4)
			defer backend.Free(partialsBuf)
			backend.SDPAF16Tiled(qF16, kF16, vF16, outTiled, partialsBuf,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Convert both to F32 for comparison
			ntF32 := backend.Alloc(numQHeads * headDim * 4)
			tiF32 := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(ntF32)
			defer backend.Free(tiF32)

			backend.ConvertF16ToF32(outNonTiled, ntF32, numQHeads*headDim)
			backend.ConvertF16ToF32(outTiled, tiF32, numQHeads*headDim)
			backend.Sync()

			ntBytes := make([]byte, numQHeads*headDim*4)
			tiBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(ntBytes, ntF32)
			backend.ToHost(tiBytes, tiF32)
			ntResult := bytesToFloat32(ntBytes)
			tiResult := bytesToFloat32(tiBytes)

			maxDiff := 0.0
			maxDiffIdx := 0
			for i := range ntResult {
				diff := math.Abs(float64(ntResult[i] - tiResult[i]))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
			}

			t.Logf("ctx=%d: tiled vs non-tiled max_diff=%.6f", kvLen, maxDiff)
			// Both are F16, so they should be very close (both use same F16 inputs)
			if maxDiff > 1e-3 {
				head := maxDiffIdx / headDim
				dim := maxDiffIdx % headDim
				t.Fatalf("Tiled diverges from non-tiled: diff=%.6f at head=%d dim=%d (non-tiled=%.6f, tiled=%.6f)",
					maxDiff, head, dim, ntResult[maxDiffIdx], tiResult[maxDiffIdx])
			}
		})
	}
}

// TestSDPAFlashDecodeF16Tiled_Throughput benchmarks the tiled kernel vs non-tiled
// at various context lengths to measure the occupancy improvement.
func TestSDPAFlashDecodeF16Tiled_Throughput(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16TiledPipeline == nil {
		t.Skip("Tiled F16 SDPA pipeline not available")
	}

	// LLaMA 2 7B config
	numQHeads := 32
	numKVHeads := 8
	headDim := 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	numLayers := 32
	warmup := 10
	iters := 100

	contextLengths := []int{16, 64, 128, 256, 512, 1024, 2048}

	type result struct {
		kvLen      int
		nonTiledUs float64  // batched 32-layer time (µs)
		tiledUs    float64
	}
	var results []result

	for _, kvLen := range contextLengths {
		kvHeadStride := kvLen * headDim

		qBuf := backend.Alloc(numQHeads * headDim * 2)
		kBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		vBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		outBuf := backend.Alloc(numQHeads * headDim * 2)

		tileKV := 64
		numTiles := (kvLen + tileKV - 1) / tileKV
		partialsBuf := backend.Alloc(numQHeads * numTiles * (2 + headDim) * 4)

		// === Non-tiled baseline ===
		for i := 0; i < warmup; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		nonTiledUs := float64(time.Since(start).Microseconds()) / float64(iters)

		// === Tiled ===
		for i := 0; i < warmup; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16Tiled(qBuf, kBuf, vBuf, outBuf, partialsBuf,
					kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16Tiled(qBuf, kBuf, vBuf, outBuf, partialsBuf,
					kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		tiledUs := float64(time.Since(start).Microseconds()) / float64(iters)

		results = append(results, result{kvLen, nonTiledUs, tiledUs})

		backend.Free(qBuf)
		backend.Free(kBuf)
		backend.Free(vBuf)
		backend.Free(outBuf)
		backend.Free(partialsBuf)
	}

	t.Logf("\n=== SDPA F16 Tiled vs Non-Tiled (32-layer batched, LLaMA 2 7B) ===")
	t.Logf("%-8s  %12s  %12s  %10s  %10s", "kvLen", "non-tiled µs", "tiled µs", "speedup", "TGs")
	for _, r := range results {
		speedup := r.nonTiledUs / r.tiledUs
		numTiles := (r.kvLen + 63) / 64
		tgs := numQHeads * numTiles
		t.Logf("%-8d  %12.1f  %12.1f  %10.2fx  %10d", r.kvLen, r.nonTiledUs, r.tiledUs, speedup, tgs)
	}

	// Context scaling comparison
	if len(results) >= 5 {
		ntBase := results[0].nonTiledUs
		ntCtx512 := results[4].nonTiledUs
		tiBase := results[0].tiledUs
		tiCtx512 := results[4].tiledUs
		t.Logf("\nContext scaling (32-layer, ctx=16→512):")
		t.Logf("  Non-tiled: %.1f → %.1f µs (%.1f%% increase)", ntBase, ntCtx512, (ntCtx512-ntBase)/ntBase*100)
		t.Logf("  Tiled:     %.1f → %.1f µs (%.1f%% increase)", tiBase, tiCtx512, (tiCtx512-tiBase)/tiBase*100)
	}
}
