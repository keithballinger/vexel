//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"
	"time"
)

// cpuFlashAttention computes causal attention for a single head.
func cpuFlashAttention(q, k, v []float32, seqLen, headDim int, scale float32) []float32 {
	out := make([]float32, seqLen*headDim)
	for t := 0; t < seqLen; t++ {
		// Compute logits up to t (causal)
		maxLogit := float32(math.Inf(-1))
		logits := make([]float32, t+1)
		for s := 0; s <= t; s++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += q[t*headDim+d] * k[s*headDim+d]
			}
			logit := dot * scale
			logits[s] = logit
			if logit > maxLogit {
				maxLogit = logit
			}
		}

		// Softmax with max subtraction for stability
		var denom float32
		for s := 0; s <= t; s++ {
			logits[s] = float32(math.Exp(float64(logits[s] - maxLogit)))
			denom += logits[s]
		}

		// Weighted sum of V
		for d := 0; d < headDim; d++ {
			var sum float32
			for s := 0; s <= t; s++ {
				sum += logits[s] / denom * v[s*headDim+d]
			}
			out[t*headDim+d] = sum
		}
	}
	return out
}

func TestFlashAttention2_NumericalStability_HighMagnitude(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	seqLen := 40
	headDim := 16
	scale := float32(0.25)

	q := make([]float32, seqLen*headDim)
	k := make([]float32, seqLen*headDim)
	v := make([]float32, seqLen*headDim)

	// Large-magnitude alternating values to stress softmax stability.
	for i := 0; i < seqLen*headDim; i++ {
		sign := float32(1)
		if i%2 == 1 {
			sign = -1
		}
		q[i] = sign * 48.0
		k[i] = sign * 46.0
		v[i] = sign * 0.5
	}

	expected := cpuFlashAttention(q, k, v, seqLen, headDim, scale)

	qBuf := backend.Alloc(len(q) * 4)
	kBuf := backend.Alloc(len(k) * 4)
	vBuf := backend.Alloc(len(v) * 4)
	outBuf := backend.Alloc(len(expected) * 4)
	defer backend.Free(qBuf)
	defer backend.Free(kBuf)
	defer backend.Free(vBuf)
	defer backend.Free(outBuf)

	backend.ToDevice(qBuf, float32ToBytes(q))
	backend.ToDevice(kBuf, float32ToBytes(k))
	backend.ToDevice(vBuf, float32ToBytes(v))

	backend.FlashAttention2(qBuf, kBuf, vBuf, outBuf, seqLen, 1, 1, headDim, scale)
	backend.Sync()

	outBytes := make([]byte, len(expected)*4)
	backend.ToHost(outBytes, outBuf)
	out := bytesToFloat32(outBytes)

	for i := range expected {
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("output contains NaN/Inf at %d: %v", i, out[i])
		}
		diff := math.Abs(float64(out[i] - expected[i]))
		if diff > 1e-3 {
			t.Fatalf("mismatch at %d: gpu=%f cpu=%f diff=%f", i, out[i], expected[i], diff)
		}
	}
}

func TestFlashAttention2_NumericalStability_LongSequence(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	seqLen := 64
	headDim := 8
	scale := float32(0.25)

	q := make([]float32, seqLen*headDim)
	k := make([]float32, seqLen*headDim)
	v := make([]float32, seqLen*headDim)

	// Moderately varied values to simulate real logits and avoid overflow.
	for t := 0; t < seqLen; t++ {
		for d := 0; d < headDim; d++ {
			val := float32((t+d)%13 - 6)
			q[t*headDim+d] = val * 0.1
			k[t*headDim+d] = val * 0.08
			v[t*headDim+d] = val * 0.05
		}
	}

	qBuf := backend.Alloc(len(q) * 4)
	kBuf := backend.Alloc(len(k) * 4)
	vBuf := backend.Alloc(len(v) * 4)
	outFA := backend.Alloc(seqLen * headDim * 4)
	outStd := backend.Alloc(seqLen * headDim * 4)
	defer backend.Free(qBuf)
	defer backend.Free(kBuf)
	defer backend.Free(vBuf)
	defer backend.Free(outFA)
	defer backend.Free(outStd)

	backend.ToDevice(qBuf, float32ToBytes(q))
	backend.ToDevice(kBuf, float32ToBytes(k))
	backend.ToDevice(vBuf, float32ToBytes(v))

	// Baseline using standard prefill kernel
	backend.SDPAPrefillStandard(qBuf, kBuf, vBuf, outStd, seqLen, 1, 1, headDim, scale)
	backend.Sync()

	// Flash Attention 2 path
	backend.FlashAttention2(qBuf, kBuf, vBuf, outFA, seqLen, 1, 1, headDim, scale)
	backend.Sync()

	stdBytes := make([]byte, seqLen*headDim*4)
	faBytes := make([]byte, seqLen*headDim*4)
	backend.ToHost(stdBytes, outStd)
	backend.ToHost(faBytes, outFA)
	std := bytesToFloat32(stdBytes)
	out := bytesToFloat32(faBytes)

	var maxDiff float64
	var sumDiff float64
	for i := range std {
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("output contains NaN/Inf at %d: %v", i, out[i])
		}
		diff := math.Abs(float64(out[i] - std[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		sumDiff += diff
	}
	meanDiff := sumDiff / float64(len(std))
	if maxDiff > 1.5e-1 || meanDiff > 2e-2 {
		t.Fatalf("flash vs standard drift: maxDiff=%f meanDiff=%f", maxDiff, meanDiff)
	}
	t.Logf("max diff: %f mean diff: %f", maxDiff, meanDiff)
}

// Benchmark-style test to report prefill throughput for Flash Attention 2.
func TestFlashAttention2_PrefillThroughput(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	seqLen := 512
	numHeads := 32
	numKVHeads := 4
	headDim := 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	iterations := 10

	q := make([]float32, seqLen*numHeads*headDim)
	k := make([]float32, seqLen*numKVHeads*headDim)
	v := make([]float32, seqLen*numKVHeads*headDim)

	qBuf := backend.Alloc(len(q) * 4)
	kBuf := backend.Alloc(len(k) * 4)
	vBuf := backend.Alloc(len(v) * 4)
	outBuf := backend.Alloc(len(q) * 4)
	defer backend.Free(qBuf)
	defer backend.Free(kBuf)
	defer backend.Free(vBuf)
	defer backend.Free(outBuf)

	backend.ToDevice(qBuf, float32ToBytes(q))
	backend.ToDevice(kBuf, float32ToBytes(k))
	backend.ToDevice(vBuf, float32ToBytes(v))

	start := time.Now()
	for i := 0; i < iterations; i++ {
		backend.FlashAttention2(qBuf, kBuf, vBuf, outBuf, seqLen, numHeads, numKVHeads, headDim, scale)
	}
	backend.Sync()
	elapsed := time.Since(start).Seconds() / float64(iterations)
	tokPerSec := float64(seqLen) / elapsed

	t.Logf("Flash Attention 2 prefill: seqLen=%d heads=%d headDim=%d -> %.1f tok/s (avg over %d iters)",
		seqLen, numHeads, headDim, tokPerSec, iterations)
}

// cpuFlashAttentionMultiHead computes causal attention with multi-head and GQA support.
// Q layout: [seqLen, numQHeads, headDim]
// K/V layout: [seqLen, numKVHeads, headDim]
// GQA: each KV head is shared by numQHeads/numKVHeads Q heads.
func cpuFlashAttentionMultiHead(q, k, v []float32, seqLen, numQHeads, numKVHeads, headDim int, scale float32) []float32 {
	out := make([]float32, seqLen*numQHeads*headDim)
	headsPerKV := numQHeads / numKVHeads

	for t := 0; t < seqLen; t++ {
		for qh := 0; qh < numQHeads; qh++ {
			kvh := qh / headsPerKV

			// Q·K dot products up to t (causal)
			maxLogit := float32(math.Inf(-1))
			logits := make([]float32, t+1)
			for s := 0; s <= t; s++ {
				var dot float32
				qOff := t*numQHeads*headDim + qh*headDim
				kOff := s*numKVHeads*headDim + kvh*headDim
				for d := 0; d < headDim; d++ {
					dot += q[qOff+d] * k[kOff+d]
				}
				logit := dot * scale
				logits[s] = logit
				if logit > maxLogit {
					maxLogit = logit
				}
			}

			// Softmax
			var denom float32
			for s := 0; s <= t; s++ {
				logits[s] = float32(math.Exp(float64(logits[s] - maxLogit)))
				denom += logits[s]
			}

			// Weighted sum of V
			outOff := t*numQHeads*headDim + qh*headDim
			for d := 0; d < headDim; d++ {
				var sum float32
				for s := 0; s <= t; s++ {
					vOff := s*numKVHeads*headDim + kvh*headDim
					sum += logits[s] / denom * v[vOff+d]
				}
				out[outOff+d] = sum
			}
		}
	}
	return out
}

// TestFA2V2_Correctness_vsCPU tests the new FA2 v2 kernel against CPU reference
// across multiple configurations including GQA.
func TestFA2V2_Correctness_vsCPU(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	configs := []struct {
		name       string
		seqLen     int
		numQHeads  int
		numKVHeads int
		headDim    int
	}{
		// Basic: single head, small
		{"1h_seq16_hd16", 16, 1, 1, 16},
		{"1h_seq32_hd64", 32, 1, 1, 64},
		{"1h_seq64_hd128", 64, 1, 1, 128},
		// Multi-head no GQA
		{"4h_seq32_hd64", 32, 4, 4, 64},
		{"32h_seq32_hd128", 32, 32, 32, 128},
		// GQA: LLaMA 2 7B config (32Q/32KV = no GQA, but test the path)
		{"32q32kv_seq64_hd128", 64, 32, 32, 128},
		// GQA: 32Q/8KV (like LLaMA 2 70B / Mistral)
		{"32q8kv_seq32_hd128", 32, 32, 8, 128},
		{"32q8kv_seq64_hd128", 64, 32, 8, 128},
		// GQA: 32Q/4KV
		{"32q4kv_seq32_hd128", 32, 32, 4, 128},
		// Edge case: seqLen not multiple of tileQ
		{"1h_seq17_hd128", 17, 1, 1, 128},
		{"1h_seq33_hd128", 33, 1, 1, 128},
		// LLaMA 2 7B full config
		{"llama2_7b_seq128_hd128", 128, 32, 32, 128},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			seqLen := cfg.seqLen
			numQHeads := cfg.numQHeads
			numKVHeads := cfg.numKVHeads
			headDim := cfg.headDim
			scale := float32(1.0 / math.Sqrt(float64(headDim)))

			qSize := seqLen * numQHeads * headDim
			kvSize := seqLen * numKVHeads * headDim

			q := make([]float32, qSize)
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)

			// Deterministic pseudo-random fill
			for i := range q {
				q[i] = float32((i*7+3)%17-8) * 0.1
			}
			for i := range k {
				k[i] = float32((i*11+5)%19-9) * 0.08
			}
			for i := range v {
				v[i] = float32((i*13+7)%23-11) * 0.05
			}

			// CPU reference
			expected := cpuFlashAttentionMultiHead(q, k, v, seqLen, numQHeads, numKVHeads, headDim, scale)

			// GPU buffers
			qBuf := backend.Alloc(qSize * 4)
			kBuf := backend.Alloc(kvSize * 4)
			vBuf := backend.Alloc(kvSize * 4)
			outBuf := backend.Alloc(qSize * 4)
			defer backend.Free(qBuf)
			defer backend.Free(kBuf)
			defer backend.Free(vBuf)
			defer backend.Free(outBuf)

			backend.ToDevice(qBuf, float32ToBytes(q))
			backend.ToDevice(kBuf, float32ToBytes(k))
			backend.ToDevice(vBuf, float32ToBytes(v))

			// Run FA2 v2
			backend.SDPAPrefillFA2V2(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale)
			backend.Sync()

			outBytes := make([]byte, qSize*4)
			backend.ToHost(outBytes, outBuf)
			out := bytesToFloat32(outBytes)

			// Compare
			var maxDiff float64
			var sumDiff float64
			maxDiffIdx := 0
			for i := range expected {
				if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
					t.Fatalf("output NaN/Inf at %d: %v", i, out[i])
				}
				diff := math.Abs(float64(out[i] - expected[i]))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
				sumDiff += diff
			}
			meanDiff := sumDiff / float64(len(expected))
			t.Logf("max diff: %e (at %d, gpu=%f cpu=%f) mean diff: %e",
				maxDiff, maxDiffIdx, out[maxDiffIdx], expected[maxDiffIdx], meanDiff)

			if maxDiff > 1e-2 {
				t.Fatalf("FAIL: max diff %e exceeds 1e-2", maxDiff)
			}
			if meanDiff > 1e-3 {
				t.Fatalf("FAIL: mean diff %e exceeds 1e-3", meanDiff)
			}
		})
	}
}

// TestFA2V2_vs_V1 compares FA2 v2 output against the original FA2 kernel.
// Both should produce near-identical results (both implement the same algorithm,
// just different tiling strategies).
func TestFA2V2_vs_V1(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	configs := []struct {
		name       string
		seqLen     int
		numQHeads  int
		numKVHeads int
		headDim    int
	}{
		{"1h_seq64_hd128", 64, 1, 1, 128},
		{"32h_seq64_hd128", 64, 32, 32, 128},
		{"32q8kv_seq64_hd128", 64, 32, 8, 128},
		{"llama2_7b_seq128", 128, 32, 32, 128},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			seqLen := cfg.seqLen
			numQHeads := cfg.numQHeads
			numKVHeads := cfg.numKVHeads
			headDim := cfg.headDim
			scale := float32(1.0 / math.Sqrt(float64(headDim)))

			qSize := seqLen * numQHeads * headDim
			kvSize := seqLen * numKVHeads * headDim

			q := make([]float32, qSize)
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for i := range q {
				q[i] = float32((i*7+3)%17-8) * 0.1
			}
			for i := range k {
				k[i] = float32((i*11+5)%19-9) * 0.08
			}
			for i := range v {
				v[i] = float32((i*13+7)%23-11) * 0.05
			}

			qBuf := backend.Alloc(qSize * 4)
			kBuf := backend.Alloc(kvSize * 4)
			vBuf := backend.Alloc(kvSize * 4)
			outV1 := backend.Alloc(qSize * 4)
			outV2 := backend.Alloc(qSize * 4)
			defer backend.Free(qBuf)
			defer backend.Free(kBuf)
			defer backend.Free(vBuf)
			defer backend.Free(outV1)
			defer backend.Free(outV2)

			backend.ToDevice(qBuf, float32ToBytes(q))
			backend.ToDevice(kBuf, float32ToBytes(k))
			backend.ToDevice(vBuf, float32ToBytes(v))

			// Run FA2 v1
			backend.SDPAPrefillFA2V1(qBuf, kBuf, vBuf, outV1, seqLen, numQHeads, numKVHeads, headDim, scale)
			backend.Sync()

			// Run FA2 v2
			backend.SDPAPrefillFA2V2(qBuf, kBuf, vBuf, outV2, seqLen, numQHeads, numKVHeads, headDim, scale)
			backend.Sync()

			v1Bytes := make([]byte, qSize*4)
			v2Bytes := make([]byte, qSize*4)
			backend.ToHost(v1Bytes, outV1)
			backend.ToHost(v2Bytes, outV2)
			v1Out := bytesToFloat32(v1Bytes)
			v2Out := bytesToFloat32(v2Bytes)

			var maxDiff float64
			var sumDiff float64
			for i := range v1Out {
				if math.IsNaN(float64(v2Out[i])) || math.IsInf(float64(v2Out[i]), 0) {
					t.Fatalf("v2 output NaN/Inf at %d: %v", i, v2Out[i])
				}
				diff := math.Abs(float64(v1Out[i] - v2Out[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				sumDiff += diff
			}
			meanDiff := sumDiff / float64(len(v1Out))
			t.Logf("v1 vs v2: max diff: %e mean diff: %e", maxDiff, meanDiff)

			if maxDiff > 1e-2 {
				t.Fatalf("FAIL: max diff %e exceeds 1e-2", maxDiff)
			}
		})
	}
}

// TestFA2V2_Throughput benchmarks FA2 v2 vs v1 for LLaMA 2 7B config.
func TestFA2V2_Throughput(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	configs := []struct {
		name       string
		seqLen     int
		numQHeads  int
		numKVHeads int
		headDim    int
	}{
		{"llama2_7b_seq32", 32, 32, 32, 128},
		{"llama2_7b_seq64", 64, 32, 32, 128},
		{"llama2_7b_seq128", 128, 32, 32, 128},
		{"llama2_7b_seq256", 256, 32, 32, 128},
		{"gqa_32q8kv_seq128", 128, 32, 8, 128},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			seqLen := cfg.seqLen
			numQHeads := cfg.numQHeads
			numKVHeads := cfg.numKVHeads
			headDim := cfg.headDim
			scale := float32(1.0 / math.Sqrt(float64(headDim)))
			iterations := 20

			qSize := seqLen * numQHeads * headDim
			kvSize := seqLen * numKVHeads * headDim

			q := make([]float32, qSize)
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)

			qBuf := backend.Alloc(qSize * 4)
			kBuf := backend.Alloc(kvSize * 4)
			vBuf := backend.Alloc(kvSize * 4)
			outBuf := backend.Alloc(qSize * 4)
			defer backend.Free(qBuf)
			defer backend.Free(kBuf)
			defer backend.Free(vBuf)
			defer backend.Free(outBuf)

			backend.ToDevice(qBuf, float32ToBytes(q))
			backend.ToDevice(kBuf, float32ToBytes(k))
			backend.ToDevice(vBuf, float32ToBytes(v))

			// Warmup
			for i := 0; i < 3; i++ {
				backend.SDPAPrefillFA2V1(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale)
			}
			backend.Sync()

			// Benchmark FA2 v1
			start := time.Now()
			for i := 0; i < iterations; i++ {
				backend.SDPAPrefillFA2V1(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale)
			}
			backend.Sync()
			v1Time := time.Since(start).Seconds() / float64(iterations)

			// Warmup v2
			for i := 0; i < 3; i++ {
				backend.SDPAPrefillFA2V2(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale)
			}
			backend.Sync()

			// Benchmark FA2 v2
			start = time.Now()
			for i := 0; i < iterations; i++ {
				backend.SDPAPrefillFA2V2(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale)
			}
			backend.Sync()
			v2Time := time.Since(start).Seconds() / float64(iterations)

			speedup := v1Time / v2Time
			t.Logf("FA2 v1: %.3f ms | FA2 v2: %.3f ms | speedup: %.2fx",
				v1Time*1000, v2Time*1000, speedup)
		})
	}
}
