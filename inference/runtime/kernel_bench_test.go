//go:build metal && darwin && cgo

package runtime

import (
	"testing"
	"time"
	"unsafe"

	"vexel/inference/backend/metal"
)

func TestW2KernelMicrobench(t *testing.T) {
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	// TinyLlama dimensions for W2: down_proj [2048, 5632]
	k := 5632 // input dim
	n := 2048 // output dim

	// Allocate buffers
	inputSize := k * 4  // F32
	outputSize := n * 4 // F32

	// Q4_0 weight size: 18 bytes per 32 elements
	numBlocks := (k + 31) / 32
	weightRowBytes := numBlocks * 18
	weightSize := n * weightRowBytes

	inputBuf := b.Alloc(inputSize)
	outputBuf := b.Alloc(outputSize)
	weightBuf := b.Alloc(weightSize)

	// Initialize with dummy data
	input := make([]float32, k)
	for i := range input {
		input[i] = float32(i%100) * 0.01
	}
	// Convert float32 slice to bytes
	inputBytes := make([]byte, inputSize)
	for i, v := range input {
		bits := *(*uint32)(unsafe.Pointer(&v))
		inputBytes[i*4+0] = byte(bits)
		inputBytes[i*4+1] = byte(bits >> 8)
		inputBytes[i*4+2] = byte(bits >> 16)
		inputBytes[i*4+3] = byte(bits >> 24)
	}
	b.ToDevice(inputBuf, inputBytes)

	weights := make([]byte, weightSize)
	b.ToDevice(weightBuf, weights)

	b.Sync()

	// Warmup
	for i := 0; i < 100; i++ {
		b.MatMulQ4_0(inputBuf, weightBuf, outputBuf, 1, n, k)
	}
	b.Sync()

	// Benchmark
	iterations := 1000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		b.MatMulQ4_0(inputBuf, weightBuf, outputBuf, 1, n, k)
	}
	b.Sync()
	elapsed := time.Since(start)

	usPerCall := float64(elapsed.Microseconds()) / float64(iterations)

	// Calculate bandwidth
	weightBytesTotal := float64(n * numBlocks * 18)
	inputBytesTotal := float64(k * 4 * (n / 8)) // 8 outputs per threadgroup reads input
	outputBytesTotal := float64(n * 4)
	totalBytesTransfer := weightBytesTotal + inputBytesTotal + outputBytesTotal
	gbPerSec := totalBytesTransfer / (usPerCall * 1000.0)

	t.Logf("W2 MatMulQ4_0 [1, %d] x [%d, %d]:", k, n, k)
	t.Logf("  Time per call: %.2f µs", usPerCall)
	t.Logf("  Weight bytes: %.2f MB", weightBytesTotal/1e6)
	t.Logf("  Total bytes (incl repeated input): %.2f MB", totalBytesTransfer/1e6)
	t.Logf("  Effective bandwidth: %.2f GB/s", gbPerSec)

	// Also test Wo dimensions
	t.Log("")
	t.Log("Testing Wo dimensions [2048, 2048]...")

	kWo := 2048
	nWo := 2048
	numBlocksWo := (kWo + 31) / 32
	weightRowBytesWo := numBlocksWo * 18
	weightSizeWo := nWo * weightRowBytesWo

	inputBufWo := b.Alloc(kWo * 4)
	outputBufWo := b.Alloc(nWo * 4)
	weightBufWo := b.Alloc(weightSizeWo)

	inputWo := make([]float32, kWo)
	for i := range inputWo {
		inputWo[i] = float32(i%100) * 0.01
	}
	inputBytesWo := make([]byte, kWo*4)
	for i, v := range inputWo {
		bits := *(*uint32)(unsafe.Pointer(&v))
		inputBytesWo[i*4+0] = byte(bits)
		inputBytesWo[i*4+1] = byte(bits >> 8)
		inputBytesWo[i*4+2] = byte(bits >> 16)
		inputBytesWo[i*4+3] = byte(bits >> 24)
	}
	b.ToDevice(inputBufWo, inputBytesWo)
	weightsWo := make([]byte, weightSizeWo)
	b.ToDevice(weightBufWo, weightsWo)
	b.Sync()

	// Warmup
	for i := 0; i < 100; i++ {
		b.MatMulQ4_0(inputBufWo, weightBufWo, outputBufWo, 1, nWo, kWo)
	}
	b.Sync()

	start = time.Now()
	for i := 0; i < iterations; i++ {
		b.MatMulQ4_0(inputBufWo, weightBufWo, outputBufWo, 1, nWo, kWo)
	}
	b.Sync()
	elapsed = time.Since(start)

	usPerCallWo := float64(elapsed.Microseconds()) / float64(iterations)
	weightBytesWo := float64(nWo * numBlocksWo * 18)
	gbPerSecWo := weightBytesWo / (usPerCallWo * 1000.0)

	t.Logf("Wo MatMulQ4_0 [1, %d] x [%d, %d]:", kWo, nWo, kWo)
	t.Logf("  Time per call: %.2f µs", usPerCallWo)
	t.Logf("  Weight bytes: %.2f MB", weightBytesWo/1e6)
	t.Logf("  Effective bandwidth: %.2f GB/s (weights only)", gbPerSecWo)

	b.Free(inputBuf)
	b.Free(outputBuf)
	b.Free(weightBuf)
	b.Free(inputBufWo)
	b.Free(outputBufWo)
	b.Free(weightBufWo)

	// Test full layer simulation (22 layers worth of decode matmuls)
	t.Log("")
	t.Log("=== Full Decode Token Simulation (22 layers) ===")

	numLayers := 22
	hiddenSize := 2048
	intermediateSize := 5632

	// Allocate all weight buffers (simulating full model)
	numBlocksHidden := (hiddenSize + 31) / 32
	numBlocksInter := (intermediateSize + 31) / 32

	// Per layer: Wo [2048,2048], W2 [2048,5632]
	// (QKV and GateUp are fused, but let's add their equivalent for comparison)
	woSize := hiddenSize * numBlocksHidden * 18
	w2Size := hiddenSize * numBlocksInter * 18

	woBuf := b.Alloc(woSize)
	w2Buf := b.Alloc(w2Size)
	hidden := b.Alloc(hiddenSize * 4)
	inter := b.Alloc(intermediateSize * 4)
	outHidden := b.Alloc(hiddenSize * 4)

	// Initialize
	woBytes := make([]byte, woSize)
	w2Bytes := make([]byte, w2Size)
	b.ToDevice(woBuf, woBytes)
	b.ToDevice(w2Buf, w2Bytes)
	b.Sync()

	// Warmup
	for i := 0; i < 100; i++ {
		// Wo
		b.MatMulQ4_0(hidden, woBuf, outHidden, 1, hiddenSize, hiddenSize)
		// W2 (down_proj: intermediate -> hidden)
		b.MatMulQ4_0(inter, w2Buf, outHidden, 1, hiddenSize, intermediateSize)
	}
	b.Sync()

	// Benchmark: simulate 22 layers of decode
	tokenIterations := 100
	start = time.Now()
	for i := 0; i < tokenIterations; i++ {
		for layer := 0; layer < numLayers; layer++ {
			// Wo and W2 (the unfused matmuls)
			b.MatMulQ4_0(hidden, woBuf, outHidden, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(inter, w2Buf, outHidden, 1, hiddenSize, intermediateSize)
		}
	}
	b.Sync()
	elapsed = time.Since(start)

	usPerToken := float64(elapsed.Microseconds()) / float64(tokenIterations)
	tokPerSec := 1e6 / usPerToken

	t.Logf("22 layers × (Wo + W2) only:")
	t.Logf("  Time per token: %.0f µs = %.2f ms", usPerToken, usPerToken/1000.0)
	t.Logf("  Implied tok/s: %.1f", tokPerSec)
	t.Logf("  (This excludes QKV, GateUp, SDPA, RoPE, etc.)")

	b.Free(woBuf)
	b.Free(w2Buf)
	b.Free(hidden)
	b.Free(inter)
	b.Free(outHidden)
}

func TestFullLayerMicrobench(t *testing.T) {
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	// TinyLlama dimensions
	numLayers := 22
	hiddenSize := 2048
	intermediateSize := 5632
	numQHeads := 32
	numKVHeads := 4 // GQA
	headDim := 64

	numBlocksHidden := (hiddenSize + 31) / 32
	numBlocksInter := (intermediateSize + 31) / 32

	// Allocate weight buffers
	woSize := hiddenSize * numBlocksHidden * 18
	w2Size := hiddenSize * numBlocksInter * 18

	// For fused kernels, we'd need full QKV and GateUp weights
	// QKV: [2560, 2048] (Q=2048 + K=256 + V=256)
	// GateUp: [11264, 2048] (gate=5632 + up=5632)
	qkvDim := hiddenSize + 2*hiddenSize/8 // Q + K + V for GQA
	gateUpDim := intermediateSize * 2
	numBlocksQKV := (hiddenSize + 31) / 32
	numBlocksGateUp := (hiddenSize + 31) / 32

	qkvSize := qkvDim * numBlocksQKV * 18
	gateUpSize := gateUpDim * numBlocksGateUp * 18

	woBuf := b.Alloc(woSize)
	w2Buf := b.Alloc(w2Size)
	qkvBuf := b.Alloc(qkvSize)
	gateUpBuf := b.Alloc(gateUpSize)

	// Activation buffers
	hidden := b.Alloc(hiddenSize * 4)
	qOut := b.Alloc(hiddenSize * 4)           // Q [numQHeads, headDim]
	kOut := b.Alloc(numKVHeads * headDim * 4) // K [numKVHeads, headDim]
	vOut := b.Alloc(numKVHeads * headDim * 4) // V
	attnOut := b.Alloc(hiddenSize * 4)
	ffnHidden := b.Alloc(intermediateSize * 4)
	outHidden := b.Alloc(hiddenSize * 4)

	// RMSNorm weights
	normWeight := b.Alloc(hiddenSize * 4)

	// KV cache (simulate short context)
	kvLen := 64
	kvHeadStride := kvLen * headDim
	kCache := b.Alloc(kvLen * numKVHeads * headDim * 4)
	vCache := b.Alloc(kvLen * numKVHeads * headDim * 4)

	// Initialize
	b.ToDevice(woBuf, make([]byte, woSize))
	b.ToDevice(w2Buf, make([]byte, w2Size))
	b.ToDevice(qkvBuf, make([]byte, qkvSize))
	b.ToDevice(gateUpBuf, make([]byte, gateUpSize))
	b.ToDevice(normWeight, make([]byte, hiddenSize*4))
	b.Sync()

	scale := 1.0 / float32(headDim)

	// Warmup
	for i := 0; i < 50; i++ {
		b.RMSNorm(hidden, normWeight, outHidden, 1, hiddenSize, 1e-5)
		b.MatMulQ4_0(outHidden, qkvBuf, qOut, 1, qkvDim, hiddenSize)
		b.SDPA(qOut, kCache, vCache, attnOut, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		b.MatMulQ4_0(attnOut, woBuf, outHidden, 1, hiddenSize, hiddenSize)
		b.RMSNorm(hidden, normWeight, outHidden, 1, hiddenSize, 1e-5)
		b.MatMulQ4_0(outHidden, gateUpBuf, ffnHidden, 1, gateUpDim, hiddenSize)
		b.SiLUMul(ffnHidden, ffnHidden, ffnHidden, intermediateSize)
		b.MatMulQ4_0(ffnHidden, w2Buf, outHidden, 1, hiddenSize, intermediateSize)
	}
	b.Sync()

	// Benchmark full layer
	iterations := 100
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for layer := 0; layer < numLayers; layer++ {
			// Attention block
			b.RMSNorm(hidden, normWeight, outHidden, 1, hiddenSize, 1e-5)
			b.MatMulQ4_0(outHidden, qkvBuf, qOut, 1, qkvDim, hiddenSize)
			// Skip RoPE for simplicity
			b.SDPA(qOut, kCache, vCache, attnOut, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			b.MatMulQ4_0(attnOut, woBuf, outHidden, 1, hiddenSize, hiddenSize)
			// Skip residual add

			// FFN block
			b.RMSNorm(hidden, normWeight, outHidden, 1, hiddenSize, 1e-5)
			b.MatMulQ4_0(outHidden, gateUpBuf, ffnHidden, 1, gateUpDim, hiddenSize)
			b.SiLUMul(ffnHidden, ffnHidden, ffnHidden, intermediateSize)
			b.MatMulQ4_0(ffnHidden, w2Buf, outHidden, 1, hiddenSize, intermediateSize)
			// Skip residual add
		}
	}
	b.Sync()
	elapsed := time.Since(start)

	usPerToken := float64(elapsed.Microseconds()) / float64(iterations)
	tokPerSec := 1e6 / usPerToken

	t.Log("=== Full Layer Simulation (22 layers, simplified) ===")
	t.Logf("Context length: %d", kvLen)
	t.Logf("Time per token: %.0f µs = %.2f ms", usPerToken, usPerToken/1000.0)
	t.Logf("Implied tok/s: %.1f", tokPerSec)
	t.Log("(Excludes: RoPE, residual adds, LM head)")

	// Cleanup
	b.Free(woBuf)
	b.Free(w2Buf)
	b.Free(qkvBuf)
	b.Free(gateUpBuf)
	b.Free(hidden)
	b.Free(qOut)
	b.Free(kOut)
	b.Free(vOut)
	b.Free(attnOut)
	b.Free(ffnHidden)
	b.Free(outHidden)
	b.Free(normWeight)
	b.Free(kCache)
	b.Free(vCache)
}

func TestLMHeadMicrobench(t *testing.T) {
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	// LM head dimensions: [32000, 2048] -> output 32000 logits
	hiddenSize := 2048
	vocabSize := 32000

	// Q6_K: 210 bytes per 256 elements
	numElements := vocabSize * hiddenSize
	numBlocks := (numElements + 255) / 256
	lmHeadSize := numBlocks * 210

	lmHeadBuf := b.Alloc(lmHeadSize)
	inputBuf := b.Alloc(hiddenSize * 4)
	outputBuf := b.Alloc(vocabSize * 4)

	// Initialize
	b.ToDevice(lmHeadBuf, make([]byte, lmHeadSize))
	b.ToDevice(inputBuf, make([]byte, hiddenSize*4))
	b.Sync()

	// Warmup
	for i := 0; i < 50; i++ {
		b.MatMulQ6_K(inputBuf, lmHeadBuf, outputBuf, 1, vocabSize, hiddenSize)
	}
	b.Sync()

	// Benchmark
	iterations := 500
	start := time.Now()
	for i := 0; i < iterations; i++ {
		b.MatMulQ6_K(inputBuf, lmHeadBuf, outputBuf, 1, vocabSize, hiddenSize)
	}
	b.Sync()
	elapsed := time.Since(start)

	usPerCall := float64(elapsed.Microseconds()) / float64(iterations)

	// Q6_K: 6.5625 bits per element = 0.8203125 bytes/element
	weightBytes := float64(numBlocks * 210)
	gbPerSec := weightBytes / (usPerCall * 1000.0)

	t.Log("=== LM Head (Q6_K) Microbench ===")
	t.Logf("Dimensions: [1, %d] x [%d, %d]", hiddenSize, vocabSize, hiddenSize)
	t.Logf("Time per call: %.2f µs", usPerCall)
	t.Logf("Weight bytes: %.2f MB", weightBytes/1e6)
	t.Logf("Effective bandwidth: %.2f GB/s", gbPerSec)

	b.Free(lmHeadBuf)
	b.Free(inputBuf)
	b.Free(outputBuf)
}
