//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"
)

func TestFlashAttention2_F16_Correctness(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	// Parameters
	seqLen := 64 // > FA2 threshold (32)
	numQHeads := 4
	numKVHeads := 2 // GQA 2:1 ratio
	headDim := 32
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Generate random data
	q := make([]float32, seqLen*numQHeads*headDim)
	k := make([]float32, seqLen*numKVHeads*headDim)
	v := make([]float32, seqLen*numKVHeads*headDim)

	for i := range q {
		q[i] = float32(i%17-8) * 0.1
	}
	for i := range k {
		k[i] = float32(i%13-6) * 0.1
	}
	for i := range v {
		v[i] = float32(i%11-5) * 0.1
	}

	// Compute expected output on CPU (in FP32)
	// We need to adapt cpuFlashAttention to handle multiple heads and GQA
	expected := make([]float32, seqLen*numQHeads*headDim)
	headsPerKV := numQHeads / numKVHeads

	for h := 0; h < numQHeads; h++ {
		kvHead := h / headsPerKV
		
		// Extract Q head
		qHead := make([]float32, seqLen*headDim)
		for t := 0; t < seqLen; t++ {
			copy(qHead[t*headDim:(t+1)*headDim], q[t*numQHeads*headDim+h*headDim:])
		}

		// Extract K/V head
		kHead := make([]float32, seqLen*headDim)
		vHead := make([]float32, seqLen*headDim)
		for t := 0; t < seqLen; t++ {
			copy(kHead[t*headDim:(t+1)*headDim], k[t*numKVHeads*headDim+kvHead*headDim:])
			copy(vHead[t*headDim:(t+1)*headDim], v[t*numKVHeads*headDim+kvHead*headDim:])
		}

		// Compute attention for this head
		outHead := cpuFlashAttention(qHead, kHead, vHead, seqLen, headDim, scale)

		// Copy back to output
		for t := 0; t < seqLen; t++ {
			copy(expected[t*numQHeads*headDim+h*headDim:], outHead[t*headDim:(t+1)*headDim])
		}
	}

	// Allocate and copy buffers to GPU (F32)
	qBufF32 := backend.Alloc(len(q) * 4)
	kBufF32 := backend.Alloc(len(k) * 4)
	vBufF32 := backend.Alloc(len(v) * 4)
	defer backend.Free(qBufF32)
	defer backend.Free(kBufF32)
	defer backend.Free(vBufF32)

	backend.ToDevice(qBufF32, float32ToBytes(q))
	backend.ToDevice(kBufF32, float32ToBytes(k))
	backend.ToDevice(vBufF32, float32ToBytes(v))

	// Convert to F16
	qBufF16 := backend.Alloc(len(q) * 2)
	kBufF16 := backend.Alloc(len(k) * 2)
	vBufF16 := backend.Alloc(len(v) * 2)
	outBufF16 := backend.Alloc(len(expected) * 2)
	defer backend.Free(qBufF16)
	defer backend.Free(kBufF16)
	defer backend.Free(vBufF16)
	defer backend.Free(outBufF16)

	backend.ConvertF32ToF16(qBufF32, qBufF16, len(q))
	backend.ConvertF32ToF16(kBufF32, kBufF16, len(k))
	backend.ConvertF32ToF16(vBufF32, vBufF16, len(v))

	// Run Flash Attention 2 F16
	backend.SDPAPrefillF16(qBufF16, kBufF16, vBufF16, outBufF16, seqLen, numQHeads, numKVHeads, headDim, scale)
	backend.Sync()

	// Convert output back to F32
	outBufF32 := backend.Alloc(len(expected) * 4)
	defer backend.Free(outBufF32)
	backend.ConvertF16ToF32(outBufF16, outBufF32, len(expected))
	backend.Sync()

	// Copy to host
	outBytes := make([]byte, len(expected)*4)
	backend.ToHost(outBytes, outBufF32)
	out := bytesToFloat32(outBytes)

	// Compare
	maxDiff := 0.0
	for i := range expected {
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("NaN/Inf at %d", i)
		}
		diff := math.Abs(float64(out[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// F16 has lower precision, so tolerance is looser
	t.Logf("Max difference: %f", maxDiff)
	if maxDiff > 1e-2 { // slightly looser
		t.Fatalf("Mismatch too large: %f", maxDiff)
	}
}
