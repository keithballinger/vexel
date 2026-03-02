//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// TestSDPAFlashDecodeF16V3 tests the chunk-based SDPA v3 kernel against CPU reference.
// Same test cases as v1 to ensure identical numerical behavior.
func TestSDPAFlashDecodeF16V3(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16V3Pipeline == nil {
		t.Skip("Flash Attention F16 V3 decode pipeline not available")
	}

	tests := []struct {
		name       string
		kvLen      int
		numQHeads  int
		numKVHeads int
		headDim    int
		tol        float64
	}{
		// Edge case: fewer positions than simdgroups (8)
		{"kvLen=1_noGQA_hd128", 1, 4, 4, 128, 5e-3},
		{"kvLen=4_noGQA_hd128", 4, 4, 4, 128, 5e-3},
		// Short context
		{"kvLen=16_gqa4x_hd128", 16, 32, 8, 128, 5e-3},
		{"kvLen=16_noGQA_hd64", 16, 8, 8, 64, 5e-3},
		// Medium context
		{"kvLen=64_gqa4x_hd128", 64, 32, 8, 128, 5e-3},
		{"kvLen=128_gqa4x_hd128", 128, 32, 8, 128, 5e-3},
		// Long context (where context scaling matters)
		{"kvLen=256_gqa4x_hd128", 256, 32, 8, 128, 5e-3},
		{"kvLen=512_gqa4x_hd128", 512, 32, 8, 128, 5e-3},
		// 1024 — stress test
		{"kvLen=1024_gqa4x_hd128", 1024, 32, 8, 128, 5e-3},
		// No GQA (LLaMA 2 7B: 32 Q heads, 32 KV heads)
		{"kvLen=64_noGQA_hd128", 64, 32, 32, 128, 5e-3},
		{"kvLen=512_noGQA_hd128", 512, 32, 32, 128, 5e-3},
		// GQA 8:1 (fewer KV heads)
		{"kvLen=64_gqa8x_hd128", 64, 32, 4, 128, 5e-3},
		// kvLen not divisible by chunk size (32) or simdgroups (8)
		{"kvLen=13_gqa4x_hd128", 13, 32, 8, 128, 5e-3},
		{"kvLen=33_gqa4x_hd128", 33, 32, 8, 128, 5e-3},
		{"kvLen=100_gqa4x_hd128", 100, 32, 8, 128, 5e-3},
		{"kvLen=255_gqa4x_hd128", 255, 32, 8, 128, 5e-3},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			kvLen := tc.kvLen
			numQHeads := tc.numQHeads
			numKVHeads := tc.numKVHeads
			headDim := tc.headDim
			scale := float32(1.0 / math.Sqrt(float64(headDim)))

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

			expected := cpuSDPADecodeF32(qData, kData, vData, kvLen, numQHeads, numKVHeads, headDim, scale)

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
			outF16 := backend.Alloc(numQHeads * headDim * 2)
			defer backend.Free(qF16)
			defer backend.Free(kF16)
			defer backend.Free(vF16)
			defer backend.Free(outF16)

			backend.ConvertF32ToF16(qF32, qF16, len(qData))
			backend.ConvertF32ToF16(kF32, kF16, len(kData))
			backend.ConvertF32ToF16(vF32, vF16, len(vData))
			backend.Sync()

			kvHeadStride := kvLen * headDim
			backend.SDPAF16V3(qF16, kF16, vF16, outF16, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			outF32 := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(outF32)
			backend.ConvertF16ToF32(outF16, outF32, numQHeads*headDim)
			backend.Sync()

			outBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(outBytes, outF32)
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
				t.Fatalf("Mismatch too large: diff=%.6f > tol=%.4f at head=%d dim=%d (expected=%.6f, got=%.6f)",
					maxDiff, tc.tol, head, dim, expected[maxDiffIdx], result[maxDiffIdx])
			}
		})
	}
}

// TestSDPAFlashDecodeF16V3_vs_V1 compares v3 output against v1 output directly.
// Both should produce identical results (within FP16 tolerance).
func TestSDPAFlashDecodeF16V3_vs_V1(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16Pipeline == nil || backend.sdpaFlashDecodeF16V3Pipeline == nil {
		t.Skip("Need both F16 v1 and v3 pipelines")
	}

	contextLengths := []int{1, 4, 16, 33, 64, 128, 255, 512, 1024}
	numQHeads := 32
	numKVHeads := 32 // LLaMA 2 7B MHA
	headDim := 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for _, kvLen := range contextLengths {
		t.Run(fmt.Sprintf("ctx=%d", kvLen), func(t *testing.T) {
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

			// Upload and convert to F16
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
			outV1F16 := backend.Alloc(numQHeads * headDim * 2)
			outV3F16 := backend.Alloc(numQHeads * headDim * 2)
			defer backend.Free(qF16)
			defer backend.Free(kF16)
			defer backend.Free(vF16)
			defer backend.Free(outV1F16)
			defer backend.Free(outV3F16)

			backend.ConvertF32ToF16(qF32, qF16, len(qData))
			backend.ConvertF32ToF16(kF32, kF16, len(kData))
			backend.ConvertF32ToF16(vF32, vF16, len(vData))
			backend.Sync()

			kvHeadStride := kvLen * headDim

			// Run v1
			backend.SDPAF16(qF16, kF16, vF16, outV1F16, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Run v3
			backend.SDPAF16V3(qF16, kF16, vF16, outV3F16, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Convert both to F32
			outV1F32 := backend.Alloc(numQHeads * headDim * 4)
			outV3F32 := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(outV1F32)
			defer backend.Free(outV3F32)

			backend.ConvertF16ToF32(outV1F16, outV1F32, numQHeads*headDim)
			backend.ConvertF16ToF32(outV3F16, outV3F32, numQHeads*headDim)
			backend.Sync()

			v1Bytes := make([]byte, numQHeads*headDim*4)
			v3Bytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(v1Bytes, outV1F32)
			backend.ToHost(v3Bytes, outV3F32)
			v1Result := bytesToFloat32(v1Bytes)
			v3Result := bytesToFloat32(v3Bytes)

			maxDiff := 0.0
			for i := range v1Result {
				diff := math.Abs(float64(v1Result[i] - v3Result[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			t.Logf("ctx=%d: v1 vs v3 max_diff=%.6f", kvLen, maxDiff)
			if maxDiff > 1e-3 {
				t.Fatalf("ctx=%d: v1 vs v3 diff=%.6f exceeds 1e-3", kvLen, maxDiff)
			}
		})
	}
}

// TestSDPAFlashDecodeF16V3_Throughput compares v3 throughput against v1.
// This is the key benchmark — v3 should show flatter context scaling.
func TestSDPAFlashDecodeF16V3_Throughput(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16V3Pipeline == nil {
		t.Skip("Flash Attention F16 V3 decode pipeline not available")
	}

	// LLaMA 2 7B MHA config
	numQHeads := 32
	numKVHeads := 32
	headDim := 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	numLayers := 32

	contextLengths := []int{16, 32, 64, 128, 256, 512, 1024}
	warmup := 10
	iters := 100

	type benchResult struct {
		kvLen     int
		v1Us     float64 // v1 batched 32 layers
		v3Us     float64 // v3 batched 32 layers
	}
	var results []benchResult

	for _, kvLen := range contextLengths {
		kvHeadStride := kvLen * headDim

		qBuf := backend.Alloc(numQHeads * headDim * 2)
		kBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		vBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		outBuf := backend.Alloc(numQHeads * headDim * 2)

		// === V1: batched 32 layers ===
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
		v1Us := float64(time.Since(start).Microseconds()) / float64(iters)

		// === V3: batched 32 layers ===
		for i := 0; i < warmup; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16V3(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16V3(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		v3Us := float64(time.Since(start).Microseconds()) / float64(iters)

		results = append(results, benchResult{kvLen, v1Us, v3Us})

		backend.Free(qBuf)
		backend.Free(kBuf)
		backend.Free(vBuf)
		backend.Free(outBuf)
	}

	// Report
	t.Logf("\n=== SDPA F16 V3 vs V1 Throughput (LLaMA 2 7B: 32Q/32KV, hd128) ===")
	t.Logf("%-8s  %12s  %12s  %10s  %12s  %12s", "kvLen", "v1 32L µs", "v3 32L µs", "speedup", "v1/layer µs", "v3/layer µs")
	for _, r := range results {
		speedup := r.v1Us / r.v3Us
		t.Logf("%-8d  %12.1f  %12.1f  %10.2fx  %12.1f  %12.1f",
			r.kvLen, r.v1Us, r.v3Us, speedup, r.v1Us/float64(numLayers), r.v3Us/float64(numLayers))
	}

	// Context scaling comparison
	if len(results) >= 5 {
		v1Base := results[0].v1Us
		v1Ctx512 := results[5].v1Us
		v3Base := results[0].v3Us
		v3Ctx512 := results[5].v3Us

		v1Deg := (v1Ctx512 - v1Base) / v1Base * 100
		v3Deg := (v3Ctx512 - v3Base) / v3Base * 100

		t.Logf("\nContext scaling ctx=16→512:")
		t.Logf("  V1: %.1f µs → %.1f µs (%.1f%% increase)", v1Base, v1Ctx512, v1Deg)
		t.Logf("  V3: %.1f µs → %.1f µs (%.1f%% increase)", v3Base, v3Ctx512, v3Deg)
		t.Logf("  V3 improvement: %.1f pp reduction in degradation", v1Deg-v3Deg)
	}
}
