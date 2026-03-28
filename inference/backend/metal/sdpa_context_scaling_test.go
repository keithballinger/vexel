//go:build metal && darwin && cgo

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestContextScalingBaseline measures decode SDPA throughput across context lengths.
func TestContextScalingBaseline(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	numQHeads, numKVHeads, headDim := 32, 32, 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvHeadStride := 4096 * headDim

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
	kvBytes := numKVHeads * maxSeqLen * headDim * 2
	kPtr := b.Alloc(kvBytes)
	vPtr := b.Alloc(kvBytes)
	defer b.Free(kPtr)
	defer b.Free(vPtr)
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
		for i := 0; i < warmup; i++ {
			b.SDPAF16(qPtr, kPtr, vPtr, outPtr, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		}
		b.Sync()

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
