//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// cpuSDPADecodeF32 computes single-query SDPA on CPU for reference.
// Q: [numQHeads, headDim], K/V: head-major [numKVHeads, kvLen, headDim]
// Returns: [numQHeads, headDim]
func cpuSDPADecodeF32(q, k, v []float32, kvLen, numQHeads, numKVHeads, headDim int, scale float32) []float32 {
	out := make([]float32, numQHeads*headDim)
	headsPerKV := numQHeads / numKVHeads

	for h := 0; h < numQHeads; h++ {
		kvHead := h / headsPerKV
		kvBase := kvHead * kvLen * headDim

		// Compute scores: Q[h] dot K[kvHead, pos] for each position
		scores := make([]float64, kvLen)
		maxScore := math.Inf(-1)
		for pos := 0; pos < kvLen; pos++ {
			dot := float64(0)
			for d := 0; d < headDim; d++ {
				dot += float64(q[h*headDim+d]) * float64(k[kvBase+pos*headDim+d])
			}
			scores[pos] = dot * float64(scale)
			if scores[pos] > maxScore {
				maxScore = scores[pos]
			}
		}

		// Softmax
		sumExp := float64(0)
		weights := make([]float64, kvLen)
		for pos := 0; pos < kvLen; pos++ {
			weights[pos] = math.Exp(scores[pos] - maxScore)
			sumExp += weights[pos]
		}
		for pos := 0; pos < kvLen; pos++ {
			weights[pos] /= sumExp
		}

		// Weighted sum of V
		for d := 0; d < headDim; d++ {
			sum := float64(0)
			for pos := 0; pos < kvLen; pos++ {
				sum += weights[pos] * float64(v[kvBase+pos*headDim+d])
			}
			out[h*headDim+d] = float32(sum)
		}
	}
	return out
}

// TestSDPAFlashDecodeF16 tests the Flash Attention F16 decode kernel against CPU reference.
// Exercises multiple context lengths, GQA ratios, and edge cases.
func TestSDPAFlashDecodeF16(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16Pipeline == nil {
		t.Skip("Flash Attention F16 decode pipeline not available")
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
		// No GQA (all heads)
		{"kvLen=64_noGQA_hd128", 64, 32, 32, 128, 5e-3},
		// GQA 8:1 (fewer KV heads)
		{"kvLen=64_gqa8x_hd128", 64, 32, 4, 128, 5e-3},
		// kvLen not divisible by 8 (partial simdgroup work)
		{"kvLen=13_gqa4x_hd128", 13, 32, 8, 128, 5e-3},
		{"kvLen=100_gqa4x_hd128", 100, 32, 8, 128, 5e-3},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			kvLen := tc.kvLen
			numQHeads := tc.numQHeads
			numKVHeads := tc.numKVHeads
			headDim := tc.headDim
			scale := float32(1.0 / math.Sqrt(float64(headDim)))

			// Generate deterministic test data with values in F16-safe range
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

			// CPU reference
			expected := cpuSDPADecodeF32(qData, kData, vData, kvLen, numQHeads, numKVHeads, headDim, scale)

			// GPU: allocate F32 buffers, upload, convert to F16
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

			// Run SDPAF16 (should route to flash kernel for headDim % 32 == 0)
			kvHeadStride := kvLen * headDim
			backend.SDPAF16(qF16, kF16, vF16, outF16, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Convert output back to F32
			outF32 := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(outF32)
			backend.ConvertF16ToF32(outF16, outF32, numQHeads*headDim)
			backend.Sync()

			outBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(outBytes, outF32)
			result := bytesToFloat32(outBytes)

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

// TestSDPAFlashDecodeF16_ContextScaling verifies that flash attention maintains
// consistent throughput characteristics across different context lengths.
// This is a smoke test — the performance benchmark is in throughput_bench_test.go.
func TestSDPAFlashDecodeF16_ContextScaling(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16Pipeline == nil {
		t.Skip("Flash Attention F16 decode pipeline not available")
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
			backend.SDPAF16(qF16, kF16, vF16, outF16, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			outF32Buf := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(outF32Buf)
			backend.ConvertF16ToF32(outF16, outF32Buf, numQHeads*headDim)
			backend.Sync()

			outBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(outBytes, outF32Buf)
			result := bytesToFloat32(outBytes)

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
			if maxDiff > 5e-3 {
				t.Fatalf("ctx=%d: max_diff=%.6f exceeds tolerance 5e-3", kvLen, maxDiff)
			}
		})
	}
}

// TestSDPAFlashDecodeF16_Throughput measures SDPA F16 decode latency at various
// context lengths. Uses batch mode to simulate real inference (many dispatches
// per command buffer) and isolates SDPA scaling from matvec overhead.
func TestSDPAFlashDecodeF16_Throughput(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16Pipeline == nil {
		t.Skip("Flash Attention F16 decode pipeline not available")
	}

	// LLaMA 2 7B config: 32 Q heads, 8 KV heads (GQA 4:1), headDim=128
	numQHeads := 32
	numKVHeads := 8
	headDim := 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	numLayers := 32 // simulate full model

	contextLengths := []int{16, 32, 64, 128, 256, 512, 1024, 2048}
	warmup := 10
	iters := 100

	type result struct {
		kvLen       int
		avgUs       float64 // per SDPA call (single layer)
		batchedUs   float64 // 32 layers batched, per iteration
	}
	var results []result

	for _, kvLen := range contextLengths {
		kvHeadStride := kvLen * headDim

		// Allocate FP16 buffers
		qBuf := backend.Alloc(numQHeads * headDim * 2)
		kBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		vBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		outBuf := backend.Alloc(numQHeads * headDim * 2)

		// === Measurement 1: single-dispatch (with Sync) ===
		for i := 0; i < warmup; i++ {
			backend.SDPAF16(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			backend.SDPAF16(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()
		}
		singleUs := float64(time.Since(start).Microseconds()) / float64(iters)

		// === Measurement 2: batched (32 layers per command buffer) ===
		for i := 0; i < warmup; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		batchedUs := float64(time.Since(start).Microseconds()) / float64(iters)

		results = append(results, result{kvLen, singleUs, batchedUs})

		backend.Free(qBuf)
		backend.Free(kBuf)
		backend.Free(vBuf)
		backend.Free(outBuf)
	}

	// Report
	t.Logf("\n=== SDPA F16 Decode Throughput (LLaMA 2 7B: 32Q/8KV, hd128) ===")
	t.Logf("%-8s  %10s  %14s  %14s  %12s", "kvLen", "single µs", "32-layer µs", "per-layer µs", "eff BW GB/s")
	for _, r := range results {
		perLayer := r.batchedUs / float64(numLayers)
		// KV data read per call: numKVHeads * kvLen * headDim * 2 bytes * 2 (K+V)
		// But GQA: 4 Q heads share same KV head, actual unique reads depend on cache
		kvBytes := float64(numKVHeads) * float64(r.kvLen) * float64(headDim) * 2.0 * 2.0
		effBW := kvBytes / (perLayer * 1e-6) / 1e9
		t.Logf("%-8d  %10.1f  %14.1f  %14.1f  %12.1f", r.kvLen, r.avgUs, r.batchedUs, perLayer, effBW)
	}

	// Context scaling
	if len(results) >= 6 {
		base := results[0].batchedUs
		ctx512 := results[5].batchedUs
		overhead := ctx512 - base
		t.Logf("\nBatched 32-layer overhead ctx=16→512: %.1f µs (%.1f%% increase)", overhead, (overhead/base)*100)
		t.Logf("As fraction of decode budget (14ms): %.1f%%", (overhead/14000)*100)
	}
}
