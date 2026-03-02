//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// nwgBufferSizes returns the required buffer sizes for NWG kernel scratch space.
func nwgBufferSizes(numQHeads, numTGsPerHead, headDim int) (partialsBytes, countersBytes int) {
	// Each TG writes: [max(float32), sum(float32), acc[headDim](float32)]
	partialsBytes = numQHeads * numTGsPerHead * (2 + headDim) * 4
	// One atomic_uint per Q head
	countersBytes = numQHeads * 4
	return
}

func nwgNumTGs(kvLen int) int {
	n := (kvLen + 255) / 256
	if n < 1 {
		n = 1
	}
	return n
}

// TestSDPAFlashDecodeF16NWG tests the NWG kernel against CPU reference.
func TestSDPAFlashDecodeF16NWG(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16NWGPipeline == nil {
		t.Skip("Flash Attention F16 NWG decode pipeline not available")
	}

	tests := []struct {
		name       string
		kvLen      int
		numQHeads  int
		numKVHeads int
		headDim    int
		tol        float64
	}{
		// Single TG (numTGs=1, fallback to v3-like behavior)
		{"kvLen=1_1TG", 1, 4, 4, 128, 5e-3},
		{"kvLen=16_1TG", 16, 32, 32, 128, 5e-3},
		{"kvLen=64_1TG", 64, 32, 8, 128, 5e-3},
		{"kvLen=128_1TG", 128, 32, 8, 128, 5e-3},
		{"kvLen=255_1TG", 255, 32, 8, 128, 5e-3},
		{"kvLen=256_1TG", 256, 32, 8, 128, 5e-3},
		// 2 TGs (edge case: second TG has 1 position)
		{"kvLen=257_2TG", 257, 32, 8, 128, 5e-3},
		// 2 TGs (balanced)
		{"kvLen=512_2TG", 512, 32, 32, 128, 5e-3},
		// 3 TGs
		{"kvLen=600_3TG", 600, 32, 8, 128, 5e-3},
		// 4 TGs (1024 positions)
		{"kvLen=1024_4TG", 1024, 32, 32, 128, 5e-3},
		// 8 TGs (2048 positions — target context)
		{"kvLen=2048_8TG", 2048, 32, 32, 128, 5e-3},
		// Non-power-of-2
		{"kvLen=500_2TG", 500, 32, 8, 128, 5e-3},
		{"kvLen=1000_4TG", 1000, 32, 32, 128, 5e-3},
		// GQA configs
		{"kvLen=512_gqa8x", 512, 32, 4, 128, 5e-3},
		// headDim=64
		{"kvLen=512_hd64", 512, 32, 32, 64, 5e-3},
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
			outF16 := backend.Alloc(numQHeads * headDim * 2)
			defer backend.Free(qF16)
			defer backend.Free(kF16)
			defer backend.Free(vF16)
			defer backend.Free(outF16)

			backend.ConvertF32ToF16(qF32, qF16, len(qData))
			backend.ConvertF32ToF16(kF32, kF16, len(kData))
			backend.ConvertF32ToF16(vF32, vF16, len(vData))
			backend.Sync()

			// Allocate NWG scratch buffers
			numTGs := nwgNumTGs(kvLen)
			partialsBytes, countersBytes := nwgBufferSizes(numQHeads, numTGs, headDim)
			partialsBuf := backend.Alloc(partialsBytes)
			countersBuf := backend.Alloc(countersBytes)
			defer backend.Free(partialsBuf)
			defer backend.Free(countersBuf)

			// Zero counters before dispatch
			backend.Zero(countersBuf, numQHeads)
			backend.Sync()

			kvHeadStride := kvLen * headDim
			backend.SDPAF16NWG(qF16, kF16, vF16, outF16, partialsBuf, countersBuf,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
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

			t.Logf("kvLen=%d numTGs=%d: max_diff=%.6f at idx=%d (tol=%.4f)",
				kvLen, numTGs, maxDiff, maxDiffIdx, tc.tol)
			if maxDiff > tc.tol {
				head := maxDiffIdx / headDim
				dim := maxDiffIdx % headDim
				t.Fatalf("Mismatch too large: diff=%.6f > tol=%.4f at head=%d dim=%d (expected=%.6f, got=%.6f)",
					maxDiff, tc.tol, head, dim, expected[maxDiffIdx], result[maxDiffIdx])
			}
		})
	}
}

// TestSDPAFlashDecodeF16NWG_vs_V3 compares NWG output against V3 directly.
// Both should produce identical results (within FP16 tolerance).
func TestSDPAFlashDecodeF16NWG_vs_V3(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16NWGPipeline == nil || backend.sdpaFlashDecodeF16V3Pipeline == nil {
		t.Skip("Need both F16 NWG and v3 pipelines")
	}

	contextLengths := []int{1, 16, 64, 128, 255, 256, 257, 512, 1024, 2048}
	numQHeads := 32
	numKVHeads := 32
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
			outV3F16 := backend.Alloc(numQHeads * headDim * 2)
			outNWGF16 := backend.Alloc(numQHeads * headDim * 2)
			defer backend.Free(qF16)
			defer backend.Free(kF16)
			defer backend.Free(vF16)
			defer backend.Free(outV3F16)
			defer backend.Free(outNWGF16)

			backend.ConvertF32ToF16(qF32, qF16, len(qData))
			backend.ConvertF32ToF16(kF32, kF16, len(kData))
			backend.ConvertF32ToF16(vF32, vF16, len(vData))
			backend.Sync()

			kvHeadStride := kvLen * headDim

			// Run V3
			backend.SDPAF16V3(qF16, kF16, vF16, outV3F16, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Run NWG (allocate scratch, zero counters)
			numTGs := nwgNumTGs(kvLen)
			partialsBytes, countersBytes := nwgBufferSizes(numQHeads, numTGs, headDim)
			partialsBuf := backend.Alloc(partialsBytes)
			countersBuf := backend.Alloc(countersBytes)
			defer backend.Free(partialsBuf)
			defer backend.Free(countersBuf)

			backend.Zero(countersBuf, numQHeads)
			backend.Sync()

			backend.SDPAF16NWG(qF16, kF16, vF16, outNWGF16, partialsBuf, countersBuf,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()

			// Convert both to F32
			outV3F32 := backend.Alloc(numQHeads * headDim * 4)
			outNWGF32 := backend.Alloc(numQHeads * headDim * 4)
			defer backend.Free(outV3F32)
			defer backend.Free(outNWGF32)

			backend.ConvertF16ToF32(outV3F16, outV3F32, numQHeads*headDim)
			backend.ConvertF16ToF32(outNWGF16, outNWGF32, numQHeads*headDim)
			backend.Sync()

			v3Bytes := make([]byte, numQHeads*headDim*4)
			nwgBytes := make([]byte, numQHeads*headDim*4)
			backend.ToHost(v3Bytes, outV3F32)
			backend.ToHost(nwgBytes, outNWGF32)
			v3Result := bytesToFloat32(v3Bytes)
			nwgResult := bytesToFloat32(nwgBytes)

			maxDiff := 0.0
			for i := range v3Result {
				diff := math.Abs(float64(v3Result[i] - nwgResult[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			numTGsUsed := nwgNumTGs(kvLen)
			t.Logf("ctx=%d numTGs=%d: v3 vs NWG max_diff=%.6f", kvLen, numTGsUsed, maxDiff)
			if maxDiff > 2e-3 {
				t.Fatalf("ctx=%d: v3 vs NWG diff=%.6f exceeds 2e-3", kvLen, maxDiff)
			}
		})
	}
}

// TestSDPAFlashDecodeF16NWG_Throughput compares NWG throughput against V3.
func TestSDPAFlashDecodeF16NWG_Throughput(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.sdpaFlashDecodeF16NWGPipeline == nil || backend.sdpaFlashDecodeF16V3Pipeline == nil {
		t.Skip("Need both F16 NWG and v3 pipelines")
	}

	numQHeads := 32
	numKVHeads := 32
	headDim := 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	numLayers := 32

	contextLengths := []int{16, 64, 128, 256, 512, 1024, 2048}
	warmup := 10
	iters := 100

	type benchResult struct {
		kvLen  int
		numTGs int
		v3Us   float64
		nwgUs  float64
	}
	var results []benchResult

	for _, kvLen := range contextLengths {
		kvHeadStride := kvLen * headDim
		numTGs := nwgNumTGs(kvLen)

		qBuf := backend.Alloc(numQHeads * headDim * 2)
		kBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		vBuf := backend.Alloc(numKVHeads * kvLen * headDim * 2)
		outBuf := backend.Alloc(numQHeads * headDim * 2)

		partialsBytes, countersBytes := nwgBufferSizes(numQHeads, numTGs, headDim)
		partialsBuf := backend.Alloc(partialsBytes)
		countersBuf := backend.Alloc(countersBytes)

		// === V3: batched 32 layers ===
		for i := 0; i < warmup; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16V3(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16V3(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		v3Us := float64(time.Since(start).Microseconds()) / float64(iters)

		// === NWG: batched 32 layers ===
		for i := 0; i < warmup; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.Zero(countersBuf, numQHeads)
				backend.SDPAF16NWG(qBuf, kBuf, vBuf, outBuf, partialsBuf, countersBuf,
					kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.Zero(countersBuf, numQHeads)
				backend.SDPAF16NWG(qBuf, kBuf, vBuf, outBuf, partialsBuf, countersBuf,
					kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			}
			backend.EndBatch()
			backend.Sync()
		}
		nwgUs := float64(time.Since(start).Microseconds()) / float64(iters)

		results = append(results, benchResult{kvLen, numTGs, v3Us, nwgUs})

		backend.Free(qBuf)
		backend.Free(kBuf)
		backend.Free(vBuf)
		backend.Free(outBuf)
		backend.Free(partialsBuf)
		backend.Free(countersBuf)
	}

	t.Logf("\n=== SDPA F16 NWG vs V3 Throughput (LLaMA 2 7B: 32Q/32KV, hd128) ===")
	t.Logf("%-8s  %5s  %12s  %12s  %10s  %12s  %12s",
		"kvLen", "nTGs", "v3 32L µs", "NWG 32L µs", "speedup", "v3/layer µs", "NWG/layer µs")
	for _, r := range results {
		speedup := r.v3Us / r.nwgUs
		t.Logf("%-8d  %5d  %12.1f  %12.1f  %10.2fx  %12.1f  %12.1f",
			r.kvLen, r.numTGs, r.v3Us, r.nwgUs, speedup, r.v3Us/float64(numLayers), r.nwgUs/float64(numLayers))
	}

	// Context scaling comparison
	if len(results) >= 7 {
		v3Base := results[0].v3Us    // ctx=16
		v3Max := results[6].v3Us     // ctx=2048
		nwgBase := results[0].nwgUs  // ctx=16
		nwgMax := results[6].nwgUs   // ctx=2048

		v3Deg := (v3Max - v3Base) / v3Base * 100
		nwgDeg := (nwgMax - nwgBase) / nwgBase * 100

		t.Logf("\nContext scaling ctx=16→2048:")
		t.Logf("  V3:  %.1f µs → %.1f µs (%.1f%% increase)", v3Base, v3Max, v3Deg)
		t.Logf("  NWG: %.1f µs → %.1f µs (%.1f%% increase)", nwgBase, nwgMax, nwgDeg)
		t.Logf("  NWG improvement: %.1f pp reduction in overhead", v3Deg-nwgDeg)
	}
}
