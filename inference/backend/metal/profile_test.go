//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
	"unsafe"

	"vexel/inference/pkg/gguf"
)

func TestProfileOperations(t *testing.T) {
	modelPath := "../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}
	fmt.Printf("Metal: %s\n", backend.DeviceName())

	// Load a Q4_0 weight for benchmarking
	loader, err := gguf.NewTensorLoader(modelPath)
	if err != nil {
		t.Skipf("Model not found: %v", err)
	}
	wqRaw, _, _, _ := loader.LoadTensorRaw("blk.0.attn_q.weight")

	// Dimensions for single token decode
	M := 1
	K := 2048 // hidden size
	N := 2048 // output size
	iterations := 100

	// Allocate GPU buffers
	inputBytes := make([]byte, M*K*4)
	for i := 0; i < M*K; i++ {
		*(*float32)(unsafe.Pointer(&inputBytes[i*4])) = 0.01
	}

	inputPtr := backend.Alloc(M * K * 4)
	wqPtr := backend.Alloc(len(wqRaw))
	outPtr := backend.Alloc(M * N * 4)

	backend.ToDevice(inputPtr, inputBytes)
	backend.ToDevice(wqPtr, wqRaw)
	backend.Sync()

	fmt.Println("\n=== Per-Operation Timing (decode, M=1) ===")

	// Q4_0 matmul (single row)
	backend.Sync()
	start := time.Now()
	for i := 0; i < iterations; i++ {
		backend.MatMulQ4_0(inputPtr, wqPtr, outPtr, M, N, K)
	}
	backend.Sync()
	q4Time := time.Since(start) / time.Duration(iterations)
	fmt.Printf("MatMulQ4_0 [1,2048] x [2048,2048]: %.3f ms\n", float64(q4Time.Microseconds())/1000)

	// Prefill case: M=32 tokens
	M = 32
	prefillInputPtr := backend.Alloc(M * K * 4)
	prefillOutPtr := backend.Alloc(M * N * 4)

	prefillInputBytes := make([]byte, M*K*4)
	backend.ToDevice(prefillInputPtr, prefillInputBytes)
	backend.Sync()

	start = time.Now()
	for i := 0; i < iterations; i++ {
		backend.MatMulQ4_0(prefillInputPtr, wqPtr, prefillOutPtr, M, N, K)
	}
	backend.Sync()
	q4PrefillTime := time.Since(start) / time.Duration(iterations)
	fmt.Printf("MatMulQ4_0 [32,2048] x [2048,2048]: %.3f ms (%.1f rows/ms)\n",
		float64(q4PrefillTime.Microseconds())/1000, 32.0/(float64(q4PrefillTime.Microseconds())/1000))

	// RMSNorm
	M = 1
	normWeight := backend.Alloc(K * 4)
	normOut := backend.Alloc(K * 4)
	backend.Sync()

	start = time.Now()
	for i := 0; i < iterations; i++ {
		backend.RMSNorm(inputPtr, normWeight, normOut, M, K, 1e-6)
	}
	backend.Sync()
	normTime := time.Since(start) / time.Duration(iterations)
	fmt.Printf("RMSNorm [1,2048]: %.3f ms\n", float64(normTime.Microseconds())/1000)

	// RoPE
	headDim := 64
	numHeads := 32
	numKVHeads := 4
	qSize := M * numHeads * headDim * 4
	kSize := M * numKVHeads * headDim * 4
	qPtr := backend.Alloc(qSize)
	kPtr := backend.Alloc(kSize)
	backend.Sync()

	start = time.Now()
	ropeDim := headDim // Use full RoPE
	for i := 0; i < iterations; i++ {
		backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, M, 0, ropeDim, 10000.0, false)
	}
	backend.Sync()
	ropeTime := time.Since(start) / time.Duration(iterations)
	fmt.Printf("RoPE [1, 32 heads]: %.3f ms\n", float64(ropeTime.Microseconds())/1000)

	// SDPA decode
	kvLen := 64
	kvSize := kvLen * numKVHeads * headDim * 4
	qSDPA := backend.Alloc(numHeads * headDim * 4)
	kSDPA := backend.Alloc(kvSize)
	vSDPA := backend.Alloc(kvSize)
	outSDPA := backend.Alloc(numHeads * headDim * 4)
	backend.Sync()

	maxSeqLen := 2048 // Typical max seq len for KV cache
	kvHeadStride := maxSeqLen * headDim
	start = time.Now()
	for i := 0; i < iterations; i++ {
		backend.SDPA(qSDPA, kSDPA, vSDPA, outSDPA, kvLen, numHeads, numKVHeads, headDim, 0.125, kvHeadStride)
	}
	backend.Sync()
	sdpaTime := time.Since(start) / time.Duration(iterations)
	fmt.Printf("SDPA decode [kvLen=64]: %.3f ms\n", float64(sdpaTime.Microseconds())/1000)

	// SDPA with longer KV
	kvLen = 256
	kvSize = kvLen * numKVHeads * headDim * 4
	kSDPA256 := backend.Alloc(kvSize)
	vSDPA256 := backend.Alloc(kvSize)
	backend.Sync()

	start = time.Now()
	for i := 0; i < iterations; i++ {
		backend.SDPA(qSDPA, kSDPA256, vSDPA256, outSDPA, kvLen, numHeads, numKVHeads, headDim, 0.125, kvHeadStride)
	}
	backend.Sync()
	sdpa256Time := time.Since(start) / time.Duration(iterations)
	fmt.Printf("SDPA decode [kvLen=256]: %.3f ms\n", float64(sdpa256Time.Microseconds())/1000)

	// ToHost/ToDevice overhead (simulating KV cache roundtrip)
	kvBytes := kvLen * numKVHeads * headDim * 4
	hostBuf := make([]byte, kvBytes)

	start = time.Now()
	for i := 0; i < iterations; i++ {
		backend.ToHost(hostBuf, kSDPA256)
		backend.ToDevice(kSDPA256, hostBuf)
	}
	backend.Sync()
	copyTime := time.Since(start) / time.Duration(iterations)
	fmt.Printf("ToHost+ToDevice [%d bytes]: %.3f ms\n", kvBytes, float64(copyTime.Microseconds())/1000)

	// =========================================================================
	// PREFILL PROFILING (M=512)
	// =========================================================================
	fmt.Println("\n=== Per-Operation Timing (Prefill, M=512) ===")
	M = 512
	prefillIter := 20

	// 1. MatMulQ4_0 (Simdgroup)
	prefillInputPtr = backend.Alloc(M * K * 4)
	prefillOutPtr = backend.Alloc(M * N * 4)
	backend.Sync()

	start = time.Now()
	for i := 0; i < prefillIter; i++ {
		backend.MatMulQ4_0(prefillInputPtr, wqPtr, prefillOutPtr, M, N, K)
	}
	backend.Sync()
	matmulPrefillTime := time.Since(start) / time.Duration(prefillIter)
	fmt.Printf("MatMulQ4_0 [512,2048] x [2048,2048]: %.3f ms (%.1f GB/s effective)\n",
		float64(matmulPrefillTime.Microseconds())/1000,
		float64(M*N*K*2)/float64(matmulPrefillTime.Nanoseconds())) // approximate FLOPs/bandwidth? Weights: N*K/2 bytes. Read weights once per batch?
	// Weights 2048*2048*0.5625 bytes = 2.3MB. Input 512*2048*4 = 4MB. Output 4MB.
	// Total IO = 10MB per call.
	// Throughput = 10MB / time.

	// 2. FlashAttention2 F16
	// Convert Q, K, V to F16 first
	qSize = M * numHeads * headDim
	kSize = M * numKVHeads * headDim
	qF32 := backend.Alloc(qSize * 4)
	kF32 := backend.Alloc(kSize * 4)
	// vF32 unused in timing loop
	qF16 := backend.Alloc(qSize * 2)
	kF16 := backend.Alloc(kSize * 2)
	vF16 := backend.Alloc(kSize * 2)
	outF16 := backend.Alloc(qSize * 2)

	// Measure F16 conversion time
	start = time.Now()
	for i := 0; i < prefillIter; i++ {
		backend.ConvertF32ToF16(qF32, qF16, qSize)
	}
	backend.Sync()
	convertTime := time.Since(start) / time.Duration(prefillIter)
	fmt.Printf("ConvertF32ToF16 [Q size]: %.3f ms\n", float64(convertTime.Microseconds())/1000)

	// Measure FA2 F16
	start = time.Now()
	for i := 0; i < prefillIter; i++ {
		backend.SDPAPrefillF16(qF16, kF16, vF16, outF16, M, numHeads, numKVHeads, headDim, 0.125)
	}
	backend.Sync()
	fa2Time := time.Since(start) / time.Duration(prefillIter)
	fmt.Printf("FlashAttention2 F16 [seqLen=512]: %.3f ms\n", float64(fa2Time.Microseconds())/1000)

	// Estimate total prefill time
	fmt.Println("\n=== Estimated Prefill Time (22 layers, 512 tokens) ===")
	// Matmul: 7 per layer
	// Attention: 1 FA2 + conversion overhead
	// RoPE, RMSNorm: scaled by 512 (approx)

	// Measure RoPE/RMSNorm for M=512
	start = time.Now()
	for i := 0; i < prefillIter; i++ {
		backend.RMSNorm(prefillInputPtr, normWeight, prefillInputPtr, M, K, 1e-6)
	}
	backend.Sync()
	normPrefillTime := time.Since(start) / time.Duration(prefillIter)

	start = time.Now()
	prefillRopeDim := headDim // Use full RoPE
	for i := 0; i < prefillIter; i++ {
		backend.RoPE(qF32, kF32, headDim, numHeads, numKVHeads, M, 0, prefillRopeDim, 10000.0, false)
	}
	backend.Sync()
	ropePrefillTime := time.Since(start) / time.Duration(prefillIter)

	layerTime := 7*matmulPrefillTime + fa2Time + convertTime*3 + 2*normPrefillTime + ropePrefillTime
	totalPrefill := layerTime * 22
	tokPerSec := 512.0 / totalPrefill.Seconds()

	fmt.Printf("Per Layer: %.2f ms\n", float64(layerTime.Microseconds())/1000)
	fmt.Printf("  MatMul (x7): %.2f ms\n", float64(7*matmulPrefillTime.Microseconds())/1000)
	fmt.Printf("  Attn (FA2+Conv): %.2f ms\n", float64((fa2Time+3*convertTime).Microseconds())/1000)
	fmt.Printf("  Norm+RoPE: %.2f ms\n", float64((2*normPrefillTime+ropePrefillTime).Microseconds())/1000)
	fmt.Printf("Total (22 layers): %.2f ms\n", float64(totalPrefill.Microseconds())/1000)
	fmt.Printf("Estimated Throughput: %.1f tok/s\n", tokPerSec)

	// Estimate decode step time
	fmt.Println("\n=== Estimated Decode Time (22 layers) ===")
	// Per layer: 2 RMSNorm, 4 Q4 matmul (Q,K,V,O), RoPE, SDPA, 3 FFN matmul
	perLayerMatmul := 7 * q4Time
	perLayerOther := 2*normTime + ropeTime + sdpaTime
	perLayerKVCopy := 2 * copyTime // K and V roundtrip
	perLayer := perLayerMatmul + perLayerOther + perLayerKVCopy

	fmt.Printf("Per layer: %.2f ms (matmul: %.2f, other: %.2f, KV copy: %.2f)\n",
		float64(perLayer.Microseconds())/1000,
		float64(perLayerMatmul.Microseconds())/1000,
		float64(perLayerOther.Microseconds())/1000,
		float64(perLayerKVCopy.Microseconds())/1000)

	total := 22 * perLayer
	fmt.Printf("Total 22 layers: %.1f ms\n", float64(total.Microseconds())/1000)
	fmt.Printf("Estimated tok/s: %.1f\n", 1000000.0/float64(total.Microseconds()))
}

func BenchmarkMatMulQ4_0_Decode(b *testing.B) {
	backend, err := NewBackend(0)
	if err != nil {
		b.Skip("Metal not available")
	}

	modelPath := "../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
	loader, err := gguf.NewTensorLoader(modelPath)
	if err != nil {
		b.Skip("Model not found")
	}
	wqRaw, _, _, _ := loader.LoadTensorRaw("blk.0.attn_q.weight")

	K := 2048
	N := 2048

	inputPtr := backend.Alloc(K * 4)
	wqPtr := backend.Alloc(len(wqRaw))
	outPtr := backend.Alloc(N * 4)
	backend.ToDevice(wqPtr, wqRaw)
	backend.Sync()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backend.MatMulQ4_0(inputPtr, wqPtr, outPtr, 1, N, K)
		backend.Sync()
	}
}

func TestRMSNormCorrectness(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}

	dim := 2048
	rows := 1

	// Create test data - use non-trivial values
	input := make([]float32, rows*dim)
	weight := make([]float32, dim)
	for i := range input {
		input[i] = float32(i%100)/100.0 - 0.5 // Values from -0.5 to 0.49
	}
	for i := range weight {
		weight[i] = 1.0 + float32(i%10)/100.0 // Values around 1.0
	}

	// Compute expected result on CPU
	eps := float32(1e-6)
	expected := make([]float32, rows*dim)
	for row := 0; row < rows; row++ {
		base := row * dim
		// Sum of squares
		sumSq := float64(0.0)
		for i := 0; i < dim; i++ {
			sumSq += float64(input[base+i] * input[base+i])
		}
		// RMS = 1 / sqrt(sumSq/dim + eps)
		rms := float32(1.0 / math.Sqrt(sumSq/float64(dim)+float64(eps)))
		for i := 0; i < dim; i++ {
			expected[base+i] = input[base+i] * rms * weight[i]
		}
	}

	// GPU computation
	inputBytes := make([]byte, len(input)*4)
	for i, v := range input {
		*(*float32)(unsafe.Pointer(&inputBytes[i*4])) = v
	}
	weightBytes := make([]byte, len(weight)*4)
	for i, v := range weight {
		*(*float32)(unsafe.Pointer(&weightBytes[i*4])) = v
	}

	inputPtr := backend.Alloc(len(inputBytes))
	weightPtr := backend.Alloc(len(weightBytes))
	outPtr := backend.Alloc(rows * dim * 4)

	backend.ToDevice(inputPtr, inputBytes)
	backend.ToDevice(weightPtr, weightBytes)
	backend.RMSNorm(inputPtr, weightPtr, outPtr, rows, dim, eps)
	backend.Sync()

	// Read back result
	outBytes := make([]byte, rows*dim*4)
	backend.ToHost(outBytes, outPtr)

	// Compare
	maxDiff := float32(0.0)
	for i := 0; i < rows*dim; i++ {
		got := *(*float32)(unsafe.Pointer(&outBytes[i*4]))
		diff := got - expected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 0.01 {
			t.Errorf("Mismatch at %d: got %f, expected %f (diff %f)", i, got, expected[i], diff)
			if i > 10 {
				break
			}
		}
	}
	fmt.Printf("RMSNorm max diff: %e\n", maxDiff)
}

func BenchmarkSDPA_Decode(b *testing.B) {
	backend, err := NewBackend(0)
	if err != nil {
		b.Skip("Metal not available")
	}

	numHeads := 32
	numKVHeads := 4
	headDim := 64
	kvLen := 128

	qPtr := backend.Alloc(numHeads * headDim * 4)
	kPtr := backend.Alloc(kvLen * numKVHeads * headDim * 4)
	vPtr := backend.Alloc(kvLen * numKVHeads * headDim * 4)
	outPtr := backend.Alloc(numHeads * headDim * 4)
	backend.Sync()

	benchMaxSeqLen := 2048
	benchKVHeadStride := benchMaxSeqLen * headDim
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backend.SDPA(qPtr, kPtr, vPtr, outPtr, kvLen, numHeads, numKVHeads, headDim, 0.125, benchKVHeadStride)
		backend.Sync()
	}
}
