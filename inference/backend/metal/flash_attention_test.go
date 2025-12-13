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
