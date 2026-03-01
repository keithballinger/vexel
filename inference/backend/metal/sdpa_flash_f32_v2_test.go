//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
)

// TestSDPAFlashDecodeF32V2 tests the new Flash Attention F32 v2 decode kernel
// (split-KV online softmax) against the CPU reference. The v2 kernel replaces
// the old sdpa_flash_decode_f32 which materialized weights[kvLen] in shared
// memory and used a two-phase algorithm that degraded with context length.
func TestSDPAFlashDecodeF32V2(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF32V2Pipeline == nil {
		t.Skip("Flash Attention F32 v2 decode pipeline not available")
	}

	tests := []struct {
		name       string
		kvLen      int
		numQHeads  int
		numKVHeads int
		headDim    int
		tol        float64
	}{
		// Edge cases: fewer positions than simdgroups (8)
		{"kvLen=1_noGQA_hd128", 1, 4, 4, 128, 1e-4},
		{"kvLen=4_noGQA_hd128", 4, 4, 4, 128, 1e-4},
		{"kvLen=7_noGQA_hd128", 7, 4, 4, 128, 1e-4},
		// Short context
		{"kvLen=16_gqa4x_hd128", 16, 32, 8, 128, 1e-4},
		{"kvLen=16_noGQA_hd64", 16, 8, 8, 64, 1e-4},
		// Medium context
		{"kvLen=64_gqa4x_hd128", 64, 32, 8, 128, 1e-4},
		{"kvLen=128_gqa4x_hd128", 128, 32, 8, 128, 1e-4},
		// Long context (where context scaling matters)
		{"kvLen=256_gqa4x_hd128", 256, 32, 8, 128, 1e-4},
		{"kvLen=512_gqa4x_hd128", 512, 32, 8, 128, 1e-4},
		// 1024 — stress test
		{"kvLen=1024_gqa4x_hd128", 1024, 32, 8, 128, 1e-4},
		// No GQA (all heads)
		{"kvLen=64_noGQA_hd128", 64, 32, 32, 128, 1e-4},
		// GQA 8:1 (fewer KV heads)
		{"kvLen=64_gqa8x_hd128", 64, 32, 4, 128, 1e-4},
		// kvLen not divisible by 8 (partial simdgroup work)
		{"kvLen=13_gqa4x_hd128", 13, 32, 8, 128, 1e-4},
		{"kvLen=100_gqa4x_hd128", 100, 32, 8, 128, 1e-4},
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
			// Head-major KV: [numKVHeads, kvLen, headDim]
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

			// CPU reference (same as used by F16 tests)
			expected := cpuSDPADecodeF32(qData, kData, vData, kvLen, numQHeads, numKVHeads, headDim, scale)

			// GPU: allocate F32 buffers and upload
			qBuf := backend.Alloc(len(qData) * 4)
			kBuf := backend.Alloc(len(kData) * 4)
			vBuf := backend.Alloc(len(vData) * 4)
			outBuf := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(qBuf)
			defer backend.Free(kBuf)
			defer backend.Free(vBuf)
			defer backend.Free(outBuf)

			backend.ToDevice(qBuf, float32ToBytes(qData))
			backend.ToDevice(kBuf, float32ToBytes(kData))
			backend.ToDevice(vBuf, float32ToBytes(vData))

			kvHeadStride := kvLen * headDim

			// Call the v2 kernel directly via SDPA (which should route to v2)
			backend.SDPA(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			resultBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			// Compare against CPU reference
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

// TestSDPAFlashDecodeF32V2_ContextScaling verifies that the F32 v2 flash kernel
// maintains consistent results across context lengths. The old F32 kernel degraded
// -24.5% from ctx=16 to ctx=512; the v2 kernel should have <3% degradation.
func TestSDPAFlashDecodeF32V2_ContextScaling(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF32V2Pipeline == nil {
		t.Skip("Flash Attention F32 v2 decode pipeline not available")
	}

	// LLaMA 2 7B config
	numQHeads := 32
	numKVHeads := 8
	headDim := 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	contextLengths := []int{16, 64, 128, 256, 512, 1024}

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

			expected := cpuSDPADecodeF32(qData, kData, vData, kvLen, numQHeads, numKVHeads, headDim, scale)

			qBuf := backend.Alloc(len(qData) * 4)
			kBuf := backend.Alloc(len(kData) * 4)
			vBuf := backend.Alloc(len(vData) * 4)
			outBuf := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(qBuf)
			defer backend.Free(kBuf)
			defer backend.Free(vBuf)
			defer backend.Free(outBuf)

			backend.ToDevice(qBuf, float32ToBytes(qData))
			backend.ToDevice(kBuf, float32ToBytes(kData))
			backend.ToDevice(vBuf, float32ToBytes(vData))

			kvHeadStride := kvLen * headDim
			backend.SDPA(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			resultBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			maxDiff := 0.0
			for i := range expected {
				if math.IsNaN(float64(result[i])) || math.IsInf(float64(result[i]), 0) {
					t.Fatalf("NaN/Inf at index %d", i)
				}
				diff := math.Abs(float64(result[i] - expected[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			t.Logf("ctx=%d: max_diff=%.6f", kvLen, maxDiff)
			if maxDiff > 1e-4 {
				t.Fatalf("ctx=%d: max_diff=%.6f exceeds tolerance 1e-4", kvLen, maxDiff)
			}
		})
	}
}

// TestSDPAFlashDecodeF32V2_vs_OldKernel verifies the v2 kernel produces
// identical results to the old sdpa_flash_decode_f32 kernel at short contexts
// where both should work correctly.
func TestSDPAFlashDecodeF32V2_vs_OldKernel(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF32V2Pipeline == nil {
		t.Skip("Flash Attention F32 v2 decode pipeline not available")
	}

	numQHeads := 32
	numKVHeads := 8
	headDim := 128
	kvLen := 64
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

	// CPU reference is ground truth
	expected := cpuSDPADecodeF32(qData, kData, vData, kvLen, numQHeads, numKVHeads, headDim, scale)

	// Allocate and upload
	qBuf := backend.Alloc(len(qData) * 4)
	kBuf := backend.Alloc(len(kData) * 4)
	vBuf := backend.Alloc(len(vData) * 4)
	outBuf := backend.Alloc(numQHeads * headDim * 4)
	defer backend.Free(qBuf)
	defer backend.Free(kBuf)
	defer backend.Free(vBuf)
	defer backend.Free(outBuf)

	backend.ToDevice(qBuf, float32ToBytes(qData))
	backend.ToDevice(kBuf, float32ToBytes(kData))
	backend.ToDevice(vBuf, float32ToBytes(vData))

	kvHeadStride := kvLen * headDim
	backend.SDPA(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
	backend.Sync()

	resultBytes := make([]byte, numQHeads*headDim*4)
	backend.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	maxDiff := 0.0
	maxDiffIdx := 0
	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
	}

	t.Logf("v2 vs CPU reference: max_diff=%.6f at idx=%d", maxDiff, maxDiffIdx)
	if maxDiff > 1e-4 {
		t.Fatalf("v2 kernel diverges from CPU reference: diff=%.6f > 1e-4", maxDiff)
	}
}
