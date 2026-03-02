//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
	"unsafe"

	"vexel/inference/pkg/gguf"
	"vexel/inference/tensor"
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

// TestDecodeMatVecBandwidth measures the effective memory bandwidth of the M=1
// matvec kernel at all LLaMA 2 7B dimensions used during decode.
// Uses data-dependency chains (output feeds next input via Add) to prevent
// the GPU from optimizing away dispatches.
func TestDecodeMatVecBandwidth(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skip("Metal not available")
	}
	defer b.Close()

	configs := []struct {
		name string
		n, k int
	}{
		{"Wo_4096x4096", 4096, 4096},
		{"W2_4096x11008", 4096, 11008},
		{"FusedMLP_11008x4096", 11008, 4096},
		{"lm_head_32000x4096", 32000, 4096},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			n, k := cfg.n, cfg.k
			blockSize := 32
			bytesPerBlock := 18
			numBlocks := (k + blockSize - 1) / blockSize

			// Allocate buffers
			aPtr := b.Alloc(k * 4) // FP32 input
			wPtr := b.Alloc(n * numBlocks * bytesPerBlock)
			oPtr := b.Alloc(n * 4) // FP32 output
			// For data dependency: add a small fraction of output back into input
			// This requires an intermediate buffer of size K (matching input dim)
			tmpPtr := b.Alloc(k * 4) // temp for creating dependency
			defer b.Free(aPtr)
			defer b.Free(wPtr)
			defer b.Free(oPtr)
			defer b.Free(tmpPtr)

			weightBytes := float64(n) * float64(numBlocks) * float64(bytesPerBlock)

			// Single-dispatch measurement: serialize with Sync
			// This gives accurate per-dispatch timing including launch overhead.
			iters := 30
			// Warmup
			for i := 0; i < 5; i++ {
				b.MatMulQ4_0(aPtr, wPtr, oPtr, 1, n, k)
				b.Sync()
			}

			start := time.Now()
			for i := 0; i < iters; i++ {
				b.MatMulQ4_0(aPtr, wPtr, oPtr, 1, n, k)
				b.Sync()
			}
			elapsed := time.Since(start)

			perCallSolo := elapsed.Seconds() / float64(iters)
			gbpsSolo := weightBytes / perCallSolo / 1e9

			// Batched measurement: chain 32 dispatches (like one decode layer set)
			// using Add to create data dependency: aPtr += scale * oPtr[:K]
			itersB := 10
			batchSize := 32
			for i := 0; i < 3; i++ {
				b.BeginBatch()
				for j := 0; j < batchSize; j++ {
					b.MatMulQ4_0(aPtr, wPtr, oPtr, 1, n, k)
					// Create data dependency: add first K elements of output back to input
					b.Add(aPtr, oPtr, aPtr, k)
				}
				b.EndBatch()
			}

			start = time.Now()
			for i := 0; i < itersB; i++ {
				b.BeginBatch()
				for j := 0; j < batchSize; j++ {
					b.MatMulQ4_0(aPtr, wPtr, oPtr, 1, n, k)
					// Data dependency: aPtr reads oPtr which was just written
					b.Add(aPtr, oPtr, aPtr, k)
				}
				b.EndBatch()
			}
			elapsedB := time.Since(start)

			perCallBatch := elapsedB.Seconds() / float64(itersB*batchSize)
			gbpsBatch := weightBytes / perCallBatch / 1e9

			t.Logf("%s:", cfg.name)
			t.Logf("  Solo (per-Sync): %.1f µs, %.1f GB/s", perCallSolo*1e6, gbpsSolo)
			t.Logf("  Batched (32/batch): %.1f µs, %.1f GB/s (includes Add overhead)", perCallBatch*1e6, gbpsBatch)
			t.Logf("  Weight read: %.1f MB", weightBytes/1e6)
		})
	}

	// Measure empty Sync() overhead
	syncIters := 100
	// Warmup
	for i := 0; i < 10; i++ {
		b.Sync()
	}
	syncStart := time.Now()
	for i := 0; i < syncIters; i++ {
		b.Sync()
	}
	syncElapsed := time.Since(syncStart)
	syncOverhead := syncElapsed.Seconds() / float64(syncIters)
	t.Logf("")
	t.Logf("Empty Sync() overhead: %.1f µs", syncOverhead*1e6)

	// Measure BeginBatch + EndBatch overhead (empty)
	for i := 0; i < 10; i++ {
		b.BeginBatch()
		b.EndBatch()
	}
	batchStart := time.Now()
	for i := 0; i < syncIters; i++ {
		b.BeginBatch()
		b.EndBatch()
	}
	batchElapsed := time.Since(batchStart)
	batchOverhead := batchElapsed.Seconds() / float64(syncIters)
	t.Logf("Empty BeginBatch/EndBatch overhead: %.1f µs", batchOverhead*1e6)

	// Measure BeginBatch + single Add + EndBatch (minimal kernel)
	aSmall := b.Alloc(4096 * 4)
	bSmall := b.Alloc(4096 * 4)
	cSmall := b.Alloc(4096 * 4)
	defer b.Free(aSmall)
	defer b.Free(bSmall)
	defer b.Free(cSmall)
	for i := 0; i < 10; i++ {
		b.BeginBatch()
		b.Add(aSmall, bSmall, cSmall, 4096)
		b.EndBatch()
	}
	addStart := time.Now()
	for i := 0; i < syncIters; i++ {
		b.BeginBatch()
		b.Add(aSmall, bSmall, cSmall, 4096)
		b.EndBatch()
	}
	addElapsed := time.Since(addStart)
	addOverhead := addElapsed.Seconds() / float64(syncIters)
	t.Logf("BeginBatch + 1 Add + EndBatch: %.1f µs (kernel ≈ %.1f µs)", addOverhead*1e6, (addOverhead-batchOverhead)*1e6)

	t.Logf("")
	t.Logf("M3 Max peak bandwidth = 400 GB/s")
	t.Logf("Decode token: 14.3ms wall clock, 3.5GB weight → 245 GB/s effective")
	t.Logf("Subtracting Sync overhead from Solo measurements:")
	t.Logf("Note: Solo includes %.0fµs of Sync+commit overhead", syncOverhead*1e6)
}

// TestCGOOverhead measures the cost of CGO round-trips for barriers and dispatches.
// This helps quantify how much time per decode token is spent crossing the Go→C boundary.
func TestCGOOverhead(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skip("Metal not available")
	}

	// Allocate small buffers for a tiny dispatch
	aPtr := backend.Alloc(4096 * 4) // small activation
	bPtr := backend.Alloc(4096 * 4) // dummy
	cPtr := backend.Alloc(4096 * 4)

	const N = 10000

	// 1. Measure barrier-only CGO calls (in batch mode)
	backend.BeginBatch()
	start := time.Now()
	for i := 0; i < N; i++ {
		backend.MemoryBarrier()
	}
	backend.EndBatch()
	barrierTotal := time.Since(start)
	barrierPer := float64(barrierTotal.Nanoseconds()) / float64(N)
	t.Logf("Barrier CGO: %.0f ns/call (%.2f µs) over %d calls", barrierPer, barrierPer/1000, N)

	// 2. Measure dispatch CGO calls (using Add as a minimal kernel)
	backend.BeginBatch()
	start = time.Now()
	for i := 0; i < N; i++ {
		backend.Add(aPtr, bPtr, cPtr, 4096)
	}
	backend.EndBatch()
	dispatchTotal := time.Since(start)
	dispatchPer := float64(dispatchTotal.Nanoseconds()) / float64(N)
	t.Logf("Dispatch CGO (Add): %.0f ns/call (%.2f µs) over %d calls", dispatchPer, dispatchPer/1000, N)

	// 3. Measure barrier + dispatch combined
	backend.BeginBatch()
	start = time.Now()
	for i := 0; i < N; i++ {
		backend.MemoryBarrier()
		backend.Add(aPtr, bPtr, cPtr, 4096)
	}
	backend.EndBatch()
	combinedTotal := time.Since(start)
	combinedPer := float64(combinedTotal.Nanoseconds()) / float64(N)
	t.Logf("Barrier+Dispatch CGO: %.0f ns/call (%.2f µs) over %d calls", combinedPer, combinedPer/1000, N)

	// 4. Compute estimated overhead for decode token
	dispatchesPerToken := 227
	barriersPerToken := 192
	totalCGOCalls := dispatchesPerToken + barriersPerToken
	estimatedOverheadMs := float64(totalCGOCalls) * combinedPer / 2 / 1e6 // average of barrier and dispatch
	t.Logf("")
	t.Logf("Estimated decode overhead:")
	t.Logf("  %d dispatches × %.0f ns = %.2f ms", dispatchesPerToken, dispatchPer, float64(dispatchesPerToken)*dispatchPer/1e6)
	t.Logf("  %d barriers × %.0f ns = %.2f ms", barriersPerToken, barrierPer, float64(barriersPerToken)*barrierPer/1e6)
	t.Logf("  Total CGO overhead: %.2f ms (%.0f calls × avg %.0f ns)", estimatedOverheadMs*2, float64(totalCGOCalls), (dispatchPer+barrierPer)/2)
}

// TestRealisticDecodeSimulation simulates the full LLaMA 2 7B Q4_0 fused decode
// pipeline with DIFFERENT weights per layer (no L2 cache reuse across layers).
// This measures the true effective memory bandwidth of the decode pipeline.
func TestRealisticDecodeSimulation(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skip("Metal not available")
	}
	defer b.Close()

	// LLaMA 2 7B Q4_0 config
	numLayers := 32
	hiddenSize := 4096
	numKVHeads := 8
	headDim := 128
	intermediateSize := 11008

	blockSize := 32
	bytesPerBlock := 18
	numBlocks4096 := (hiddenSize + blockSize - 1) / blockSize
	numBlocks11008 := (intermediateSize + blockSize - 1) / blockSize

	// Per-layer weight sizes in Q4_0
	type layerWeights struct {
		wq, wk, wv, wo, w1, w3, w2 tensor.DevicePtr
	}

	// Allocate DIFFERENT weights for each layer (forces device memory reads)
	layers := make([]layerWeights, numLayers)
	var totalWeightBytes int64
	for l := 0; l < numLayers; l++ {
		// Wq [4096, 4096]: N=4096, K=4096
		wqSize := hiddenSize * numBlocks4096 * bytesPerBlock
		layers[l].wq = b.Alloc(wqSize)
		totalWeightBytes += int64(wqSize)

		// Wk [1024, 4096]: N=numKVHeads*headDim=1024, K=4096
		kvDim := numKVHeads * headDim
		wkSize := kvDim * numBlocks4096 * bytesPerBlock
		layers[l].wk = b.Alloc(wkSize)
		totalWeightBytes += int64(wkSize)

		// Wv [1024, 4096]
		layers[l].wv = b.Alloc(wkSize)
		totalWeightBytes += int64(wkSize)

		// Wo [4096, 4096]
		layers[l].wo = b.Alloc(wqSize)
		totalWeightBytes += int64(wqSize)

		// W1 [11008, 4096]: N=11008, K=4096
		w1Size := intermediateSize * numBlocks4096 * bytesPerBlock
		layers[l].w1 = b.Alloc(w1Size)
		totalWeightBytes += int64(w1Size)

		// W3 [11008, 4096]
		layers[l].w3 = b.Alloc(w1Size)
		totalWeightBytes += int64(w1Size)

		// W2 [4096, 11008]: N=4096, K=11008
		w2Size := hiddenSize * numBlocks11008 * bytesPerBlock
		layers[l].w2 = b.Alloc(w2Size)
		totalWeightBytes += int64(w2Size)
	}

	// LM head [32000, 4096]
	lmHeadSize := 32000 * numBlocks4096 * bytesPerBlock
	lmHead := b.Alloc(lmHeadSize)
	totalWeightBytes += int64(lmHeadSize)

	// Activation buffers (reused across layers)
	x := b.Alloc(hiddenSize * 4)           // main hidden state [4096] F32
	gate := b.Alloc(intermediateSize * 4)   // MLP intermediate [11008] F32
	mlpOut := b.Alloc(hiddenSize * 4)       // MLP output [4096] F32
	lmOut := b.Alloc(32000 * 4)             // logits

	t.Logf("Total weight memory: %.2f GB (%d layers + LM head)", float64(totalWeightBytes)/1e9, numLayers)

	// Benchmark: Dispatch matvec-only pipeline (skip RoPE, SDPA, etc.)
	// This isolates the matvec bandwidth which is ~75% of total decode time.
	warmup := 3
	iters := 20

	kvDim := numKVHeads * headDim

	// Warmup
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			// 3 attention projections
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			// Wo
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			// W1 + W3 (MLP)
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			// W2
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		// LM head
		b.MatMulQ4_0(x, lmHead, lmOut, 1, 32000, hiddenSize)
		b.EndBatch()
		b.Sync()
	}

	// Measure
	start := time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		b.MatMulQ4_0(x, lmHead, lmOut, 1, 32000, hiddenSize)
		b.EndBatch()
		b.Sync()
	}
	elapsed := time.Since(start)

	perToken := elapsed.Seconds() / float64(iters)
	gbps := float64(totalWeightBytes) / perToken / 1e9
	toksPerSec := 1.0 / perToken

	t.Logf("\n=== Realistic Decode Simulation (LLaMA 2 7B Q4_0, matvec-only) ===")
	t.Logf("  Per-token time: %.2f ms", perToken*1e3)
	t.Logf("  Effective BW:   %.1f GB/s (%.1f%% of 400 GB/s peak)", gbps, gbps/400*100)
	t.Logf("  Throughput:     %.1f tok/s (matvec-limited ceiling)", toksPerSec)
	t.Logf("  Weight data:    %.2f GB per token", float64(totalWeightBytes)/1e9)

	// How much non-matvec overhead we can afford
	targetToksPerSec := 76.3 // llama.cpp Q4_0
	targetPerToken := 1.0 / targetToksPerSec
	nonMatvecBudget := targetPerToken - perToken
	t.Logf("\n  llama.cpp target: 76.3 tok/s = %.2f ms/token", targetPerToken*1e3)
	if nonMatvecBudget > 0 {
		t.Logf("  Non-matvec budget: %.2f ms (to beat llama.cpp)", nonMatvecBudget*1e3)
	} else {
		t.Logf("  ⚠ Matvec alone exceeds llama.cpp budget by %.2f ms", -nonMatvecBudget*1e3)
	}
}

// TestNonMatvecOverheadBreakdown measures every non-matvec operation at LLaMA 2 7B
// decode dimensions. This identifies exactly where the 3+ ms of non-matvec overhead goes.
//
// Per layer, the fused decode path dispatches:
//   1. FusedRMSNorm+QKV (3 matvec dispatches — counted in matvec budget)
//   2. barrier
//   3. RoPE+ScatterKV (1 dispatch)
//   4. barrier
//   5. SDPA F16 (1 dispatch)
//   6. barrier
//   7. Wo (1 matvec — counted in matvec budget)
//   8. barrier
//   9. AddRMSNorm (1 dispatch)
//  10. barrier
//  11. FusedMLP (1 matvec — counted in matvec budget)
//  12. barrier
//  13. W2+Add2 (1 matvec — counted in matvec budget)
//
// Non-matvec per layer: RoPE+ScatterKV + SDPA + AddRMSNorm + 6 barriers
func TestNonMatvecOverheadBreakdown(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skip("Metal not available")
	}
	defer b.Close()

	// LLaMA 2 7B config
	hiddenSize := 4096
	numQHeads := 32
	numKVHeads := 8
	headDim := 128
	kvLen := 16 // ctx=16 baseline

	// Allocate activation buffers at realistic dimensions
	// Q after projection: [numQHeads * headDim] F16 = [4096] = 8192 bytes
	qF16 := b.Alloc(numQHeads * headDim * 2)
	// K,V src after projection: [numKVHeads * headDim] F16
	kSrcF16 := b.Alloc(numKVHeads * headDim * 2)
	vSrcF16 := b.Alloc(numKVHeads * headDim * 2)
	// KV cache: [numKVHeads, maxSeqLen, headDim] F16
	maxSeqLen := 2048
	kvHeadStride := maxSeqLen * headDim
	kCacheF16 := b.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	vCacheF16 := b.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	// SDPA output: [numQHeads * headDim] F16
	sdpaOut := b.Alloc(numQHeads * headDim * 2)
	// Hidden state: [4096] F32
	x := b.Alloc(hiddenSize * 4)
	residual := b.Alloc(hiddenSize * 4)
	normWeight := b.Alloc(hiddenSize * 4)
	normOut := b.Alloc(hiddenSize * 4)

	iters := 50
	warmup := 10
	theta := float32(10000.0)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// =====================================================================
	// 1. RoPE+ScatterKV F16
	// =====================================================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		b.RoPEScatterKVF16(qF16, kSrcF16, kCacheF16, vSrcF16, vCacheF16,
			numQHeads, numKVHeads, headDim, 0, headDim, theta, maxSeqLen, 0)
		b.EndBatch()
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		b.RoPEScatterKVF16(qF16, kSrcF16, kCacheF16, vSrcF16, vCacheF16,
			numQHeads, numKVHeads, headDim, i%maxSeqLen, headDim, theta, maxSeqLen, i%maxSeqLen)
		b.EndBatch()
	}
	ropeScatterTime := time.Since(start).Seconds() / float64(iters)

	// =====================================================================
	// 2. SDPA F16 at ctx=16 and ctx=512
	// =====================================================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		b.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		b.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		b.EndBatch()
	}
	sdpa16Time := time.Since(start).Seconds() / float64(iters)

	kvLen512 := 512
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		b.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen512, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		b.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen512, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		b.EndBatch()
	}
	sdpa512Time := time.Since(start).Seconds() / float64(iters)

	// =====================================================================
	// 3. AddRMSNorm
	// =====================================================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		b.AddRMSNorm(x, residual, normWeight, normOut, 1, hiddenSize, 1e-6)
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		b.AddRMSNorm(x, residual, normWeight, normOut, 1, hiddenSize, 1e-6)
		b.EndBatch()
	}
	addRMSNormTime := time.Since(start).Seconds() / float64(iters)

	// =====================================================================
	// 4. Memory Barrier overhead (batched)
	// =====================================================================
	barriersPerLayer := 6
	totalBarriers := barriersPerLayer * 32
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for j := 0; j < totalBarriers; j++ {
			b.MemoryBarrier()
		}
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for j := 0; j < totalBarriers; j++ {
			b.MemoryBarrier()
		}
		b.EndBatch()
	}
	barrierTotalTime := time.Since(start).Seconds() / float64(iters)
	barrierPerTime := barrierTotalTime / float64(totalBarriers)

	// =====================================================================
	// 5. BeginBatch/EndBatch overhead
	// =====================================================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters*5; i++ {
		b.BeginBatch()
		b.EndBatch()
	}
	batchOverhead := time.Since(start).Seconds() / float64(iters*5)

	// =====================================================================
	// Report
	// =====================================================================
	t.Logf("\n=== Non-MatVec Overhead Breakdown (LLaMA 2 7B Q4_0 decode) ===")
	t.Logf("")
	t.Logf("Per-dispatch latency (including BeginBatch/EndBatch):")
	t.Logf("  RoPE+ScatterKV F16:  %.1f µs", ropeScatterTime*1e6)
	t.Logf("  SDPA F16 (ctx=16):   %.1f µs", sdpa16Time*1e6)
	t.Logf("  SDPA F16 (ctx=512):  %.1f µs", sdpa512Time*1e6)
	t.Logf("  AddRMSNorm:          %.1f µs", addRMSNormTime*1e6)
	t.Logf("  MemoryBarrier:       %.2f µs", barrierPerTime*1e6)
	t.Logf("  BeginBatch/EndBatch: %.1f µs", batchOverhead*1e6)
	t.Logf("")

	// Per-token costs (× 32 layers)
	numLayers := 32
	ropeTotal := ropeScatterTime * float64(numLayers)
	sdpa16Total := sdpa16Time * float64(numLayers)
	sdpa512Total := sdpa512Time * float64(numLayers)
	addNormTotal := addRMSNormTime * float64(numLayers)
	barrierTotal := barrierPerTime * float64(totalBarriers)
	batchTotal := batchOverhead // 1 batch per token

	total16 := ropeTotal + sdpa16Total + addNormTotal + barrierTotal + batchTotal
	total512 := ropeTotal + sdpa512Total + addNormTotal + barrierTotal + batchTotal

	t.Logf("Per-token costs (× %d layers):", numLayers)
	t.Logf("  RoPE+ScatterKV:  %.2f ms (×%d = %.1f µs each)", ropeTotal*1e3, numLayers, ropeScatterTime*1e6)
	t.Logf("  SDPA F16@ctx16:  %.2f ms", sdpa16Total*1e3)
	t.Logf("  SDPA F16@ctx512: %.2f ms", sdpa512Total*1e3)
	t.Logf("  AddRMSNorm:      %.2f ms", addNormTotal*1e3)
	t.Logf("  Barriers (%d):   %.2f ms", totalBarriers, barrierTotal*1e3)
	t.Logf("  Batch overhead:  %.2f ms", batchTotal*1e3)
	t.Logf("  ─────────────────────────")
	t.Logf("  Total @ctx16:    %.2f ms", total16*1e3)
	t.Logf("  Total @ctx512:   %.2f ms", total512*1e3)
	t.Logf("")
	t.Logf("Budget analysis (matvec ceiling from TestRealisticDecodeSimulation ≈ 10.86ms):")
	matvecMs := 10.86
	t.Logf("  Matvec:          %.2f ms", matvecMs)
	t.Logf("  + Non-matvec:    %.2f ms (ctx=16)", total16*1e3)
	t.Logf("  = Predicted:     %.2f ms → %.1f tok/s", matvecMs+total16*1e3, 1000.0/(matvecMs+total16*1e3))
	t.Logf("  Actual decode:   ~14.3 ms → ~70 tok/s")
	t.Logf("  llama.cpp:       13.11 ms → 76.3 tok/s")
	unexplained := 14.3 - matvecMs - total16*1e3
	if unexplained > 0 {
		t.Logf("  Unexplained gap: %.2f ms (CGO overhead, batch encoding, pipeline switches?)", unexplained)
	}

	// =====================================================================
	// 6. Batched non-matvec: all 32 layers in one command buffer
	// This measures realistic amortized cost with GPU pipelining.
	// =====================================================================
	t.Logf("")
	t.Logf("=== Batched Non-MatVec (32 layers in single command buffer) ===")

	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			b.MemoryBarrier()
			b.RoPEScatterKVF16(qF16, kSrcF16, kCacheF16, vSrcF16, vCacheF16,
				numQHeads, numKVHeads, headDim, i%maxSeqLen, headDim, theta, maxSeqLen, i%maxSeqLen)
			b.MemoryBarrier()
			b.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			b.MemoryBarrier()
			// (Wo matvec would go here — skipped)
			b.MemoryBarrier()
			b.AddRMSNorm(x, residual, normWeight, normOut, 1, hiddenSize, 1e-6)
			b.MemoryBarrier()
			// (FusedMLP matvec would go here — skipped)
			b.MemoryBarrier()
			// (W2+Add2 matvec would go here — skipped)
		}
		b.EndBatch()
	}

	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			b.MemoryBarrier()
			b.RoPEScatterKVF16(qF16, kSrcF16, kCacheF16, vSrcF16, vCacheF16,
				numQHeads, numKVHeads, headDim, i%maxSeqLen, headDim, theta, maxSeqLen, i%maxSeqLen)
			b.MemoryBarrier()
			b.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			b.MemoryBarrier()
			b.MemoryBarrier()
			b.AddRMSNorm(x, residual, normWeight, normOut, 1, hiddenSize, 1e-6)
			b.MemoryBarrier()
			b.MemoryBarrier()
		}
		b.EndBatch()
	}
	batchedNonMatvec16 := time.Since(start).Seconds() / float64(iters)

	// Same for ctx=512
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			b.MemoryBarrier()
			b.RoPEScatterKVF16(qF16, kSrcF16, kCacheF16, vSrcF16, vCacheF16,
				numQHeads, numKVHeads, headDim, i%maxSeqLen, headDim, theta, maxSeqLen, i%maxSeqLen)
			b.MemoryBarrier()
			b.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen512, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			b.MemoryBarrier()
			b.MemoryBarrier()
			b.AddRMSNorm(x, residual, normWeight, normOut, 1, hiddenSize, 1e-6)
			b.MemoryBarrier()
			b.MemoryBarrier()
		}
		b.EndBatch()
	}
	batchedNonMatvec512 := time.Since(start).Seconds() / float64(iters)

	t.Logf("  Batched non-matvec @ctx16:   %.2f ms (vs %.2f ms individual sum)", batchedNonMatvec16*1e3, total16*1e3)
	t.Logf("  Batched non-matvec @ctx512:  %.2f ms (vs %.2f ms individual sum)", batchedNonMatvec512*1e3, total512*1e3)
	t.Logf("")
	t.Logf("  Predicted decode @ctx16:  %.2f ms (matvec) + %.2f ms (non-matvec) = %.2f ms → %.1f tok/s",
		matvecMs, batchedNonMatvec16*1e3, matvecMs+batchedNonMatvec16*1e3, 1000.0/(matvecMs+batchedNonMatvec16*1e3))
	t.Logf("  Predicted decode @ctx512: %.2f ms (matvec) + %.2f ms (non-matvec) = %.2f ms → %.1f tok/s",
		matvecMs, batchedNonMatvec512*1e3, matvecMs+batchedNonMatvec512*1e3, 1000.0/(matvecMs+batchedNonMatvec512*1e3))
}

// TestFusedVsPlainKernels benchmarks fused kernels against equivalent plain matvecs.
// This measures the per-kernel overhead of fused operations.
func TestFusedVsPlainKernels(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skip("Metal not available")
	}
	defer b.Close()

	// LLaMA 2 7B dims
	hiddenSize := 4096
	numKVHeads := 8
	headDim := 128
	kvDim := numKVHeads * headDim // 1024
	intermediateSize := 11008
	numLayers := 32

	blockSize := 32
	bytesPerBlock := 18
	numBlocks4096 := (hiddenSize + blockSize - 1) / blockSize
	numBlocks11008 := (intermediateSize + blockSize - 1) / blockSize

	// DIFFERENT weights per layer to force device memory reads (no L2 cache reuse)
	type layerW struct {
		wq, wk, wv, wo, w1, w3, w2 tensor.DevicePtr
		normWeight                  tensor.DevicePtr
	}
	layers := make([]layerW, numLayers)
	for l := 0; l < numLayers; l++ {
		layers[l].wq = b.Alloc(hiddenSize * numBlocks4096 * bytesPerBlock)
		layers[l].wk = b.Alloc(kvDim * numBlocks4096 * bytesPerBlock)
		layers[l].wv = b.Alloc(kvDim * numBlocks4096 * bytesPerBlock)
		layers[l].wo = b.Alloc(hiddenSize * numBlocks4096 * bytesPerBlock)
		layers[l].w1 = b.Alloc(intermediateSize * numBlocks4096 * bytesPerBlock)
		layers[l].w3 = b.Alloc(intermediateSize * numBlocks4096 * bytesPerBlock)
		layers[l].w2 = b.Alloc(hiddenSize * numBlocks11008 * bytesPerBlock)
		layers[l].normWeight = b.Alloc(hiddenSize * 4)
	}

	// Activations
	x := b.Alloc(hiddenSize * 4)
	qOut := b.Alloc(hiddenSize * 4) // F32
	kOut := b.Alloc(kvDim * 4)      // F32
	vOut := b.Alloc(kvDim * 4)      // F32
	qF16 := b.Alloc(hiddenSize * 2) // F16
	kF16 := b.Alloc(kvDim * 2)      // F16
	vF16 := b.Alloc(kvDim * 2)      // F16
	gate := b.Alloc(intermediateSize * 4)
	mlpOut := b.Alloc(hiddenSize * 4)
	residual := b.Alloc(hiddenSize * 4)
	normOut := b.Alloc(hiddenSize * 4)

	iters := 20
	warmup := 5

	// =====================================================================
	// 1. FusedRMSNormQKV_F16 (1 dispatch) vs 3× MatMulQ4_0 (3 dispatches)
	// =====================================================================
	qkvWeightBytes := float64(hiddenSize+kvDim+kvDim) * float64(numBlocks4096) * float64(bytesPerBlock)

	// Plain: 3 separate MatMulQ4_0
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.wq, qOut, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, kOut, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, vOut, 1, kvDim, hiddenSize)
		}
		b.EndBatch()
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.wq, qOut, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, kOut, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, vOut, 1, kvDim, hiddenSize)
		}
		b.EndBatch()
	}
	plainQKVTime := time.Since(start).Seconds() / float64(iters)
	plainQKVPerLayer := plainQKVTime / float64(numLayers)

	// Fused: FusedRMSNormQKV_F16 (1 dispatch)
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0_FusedRMSNormQKV_F16(x, lw.normWeight, lw.wq, lw.wk, lw.wv, qF16, kF16, vF16,
				hiddenSize, kvDim, hiddenSize, 1e-6)
		}
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0_FusedRMSNormQKV_F16(x, lw.normWeight, lw.wq, lw.wk, lw.wv, qF16, kF16, vF16,
				hiddenSize, kvDim, hiddenSize, 1e-6)
		}
		b.EndBatch()
	}
	fusedQKVTime := time.Since(start).Seconds() / float64(iters)
	fusedQKVPerLayer := fusedQKVTime / float64(numLayers)

	t.Logf("=== FusedRMSNormQKV_F16 vs 3× plain MatMulQ4_0 (×%d layers, DIFFERENT weights) ===", numLayers)
	t.Logf("  3× plain MatMulQ4_0:       %.2f ms total, %.1f µs/layer (%.1f GB/s)",
		plainQKVTime*1e3, plainQKVPerLayer*1e6, qkvWeightBytes/plainQKVPerLayer/1e9)
	t.Logf("  FusedRMSNormQKV_F16:       %.2f ms total, %.1f µs/layer (%.1f GB/s)",
		fusedQKVTime*1e3, fusedQKVPerLayer*1e6, qkvWeightBytes/fusedQKVPerLayer/1e9)
	t.Logf("  Ratio: %.2fx (fused/plain)", fusedQKVPerLayer/plainQKVPerLayer)

	// =====================================================================
	// 2. FusedMLP (1 dispatch) vs 2× MatMulQ4_0 (2 dispatches)
	// =====================================================================
	mlpWeightBytes := float64(2*intermediateSize) * float64(numBlocks4096) * float64(bytesPerBlock)

	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
		}
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
		}
		b.EndBatch()
	}
	plainMLPTime := time.Since(start).Seconds() / float64(iters)
	plainMLPPerLayer := plainMLPTime / float64(numLayers)

	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0_FusedMLP(x, lw.w1, lw.w3, gate, 1, intermediateSize, hiddenSize)
		}
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0_FusedMLP(x, lw.w1, lw.w3, gate, 1, intermediateSize, hiddenSize)
		}
		b.EndBatch()
	}
	fusedMLPTime := time.Since(start).Seconds() / float64(iters)
	fusedMLPPerLayer := fusedMLPTime / float64(numLayers)

	t.Logf("")
	t.Logf("=== FusedMLP vs 2× plain MatMulQ4_0 (×%d layers, DIFFERENT weights) ===", numLayers)
	t.Logf("  2× plain MatMulQ4_0:  %.2f ms total, %.1f µs/layer (%.1f GB/s)",
		plainMLPTime*1e3, plainMLPPerLayer*1e6, mlpWeightBytes/plainMLPPerLayer/1e9)
	t.Logf("  FusedMLP:             %.2f ms total, %.1f µs/layer (%.1f GB/s)",
		fusedMLPTime*1e3, fusedMLPPerLayer*1e6, mlpWeightBytes/fusedMLPPerLayer/1e9)
	t.Logf("  Ratio: %.2fx (fused/plain)", fusedMLPPerLayer/plainMLPPerLayer)

	// =====================================================================
	// 3. W2+Add2 (1 dispatch) vs plain MatMulQ4_0 (1 dispatch)
	// =====================================================================
	w2WeightBytes := float64(hiddenSize) * float64(numBlocks11008) * float64(bytesPerBlock)

	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		b.EndBatch()
	}
	plainW2Time := time.Since(start).Seconds() / float64(iters)
	plainW2PerLayer := plainW2Time / float64(numLayers)

	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0_Add(gate, lw.w2, mlpOut, hiddenSize, intermediateSize)
		}
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0_Add(gate, lw.w2, mlpOut, hiddenSize, intermediateSize)
		}
		b.EndBatch()
	}
	fusedW2Time := time.Since(start).Seconds() / float64(iters)
	fusedW2PerLayer := fusedW2Time / float64(numLayers)

	t.Logf("")
	t.Logf("=== W2+Add2 vs plain MatMulQ4_0 (×%d layers, DIFFERENT weights) ===", numLayers)
	t.Logf("  plain MatMulQ4_0:  %.2f ms total, %.1f µs/layer (%.1f GB/s)",
		plainW2Time*1e3, plainW2PerLayer*1e6, w2WeightBytes/plainW2PerLayer/1e9)
	t.Logf("  W2+Add2 (fused):   %.2f ms total, %.1f µs/layer (%.1f GB/s)",
		fusedW2Time*1e3, fusedW2PerLayer*1e6, w2WeightBytes/fusedW2PerLayer/1e9)
	t.Logf("  Ratio: %.2fx (fused/plain)", fusedW2PerLayer/plainW2PerLayer)

	// =====================================================================
	// 4. AddRMSNorm timing (batched across layers)
	// =====================================================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.AddRMSNorm(x, residual, lw.normWeight, normOut, 1, hiddenSize, 1e-6)
		}
		b.EndBatch()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.AddRMSNorm(x, residual, lw.normWeight, normOut, 1, hiddenSize, 1e-6)
		}
		b.EndBatch()
	}
	addNormTime := time.Since(start).Seconds() / float64(iters)
	addNormPerLayer := addNormTime / float64(numLayers)

	t.Logf("")
	t.Logf("=== AddRMSNorm (×%d layers batched) ===", numLayers)
	t.Logf("  Total: %.2f ms, %.1f µs/layer", addNormTime*1e3, addNormPerLayer*1e6)

	// =====================================================================
	// Summary: estimated cost of fused path overhead
	// =====================================================================
	fusedOverheadQKV := (fusedQKVPerLayer - plainQKVPerLayer) * float64(numLayers)
	fusedOverheadMLP := (fusedMLPPerLayer - plainMLPPerLayer) * float64(numLayers)
	fusedOverheadW2 := (fusedW2PerLayer - plainW2PerLayer) * float64(numLayers)
	totalFusedOverhead := fusedOverheadQKV + fusedOverheadMLP + fusedOverheadW2

	t.Logf("")
	t.Logf("=== Fused Kernel Overhead Summary ===")
	t.Logf("  QKV fused overhead:  %+.2f ms (%+.1f µs/layer)", fusedOverheadQKV*1e3, (fusedQKVPerLayer-plainQKVPerLayer)*1e6)
	t.Logf("  MLP fused overhead:  %+.2f ms (%+.1f µs/layer)", fusedOverheadMLP*1e3, (fusedMLPPerLayer-plainMLPPerLayer)*1e6)
	t.Logf("  W2+Add2 overhead:    %+.2f ms (%+.1f µs/layer)", fusedOverheadW2*1e3, (fusedW2PerLayer-plainW2PerLayer)*1e6)
	t.Logf("  Total fused overhead: %+.2f ms per token", totalFusedOverhead*1e3)
}

// TestFullPipelineSimulation simulates the COMPLETE fused decode pipeline:
// matvec + non-matvec + barriers interleaved exactly as in the real pipeline.
// This captures the pipeline stall cost that simple sum-of-parts misses.
func TestFullPipelineSimulation(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skip("Metal not available")
	}
	defer backend.Close()

	// LLaMA 2 7B Q4_0 config
	numLayers := 32
	hiddenSize := 4096
	numKVHeads := 32 // LLaMA 2 7B is MHA (not GQA)
	numQHeads := 32
	headDim := 128
	intermediateSize := 11008

	blockSize := 32
	bytesPerBlock := 18
	numBlocks4096 := (hiddenSize + blockSize - 1) / blockSize
	numBlocks11008 := (intermediateSize + blockSize - 1) / blockSize

	// Per-layer weights (DIFFERENT per layer to prevent L2 reuse)
	type layerW struct {
		wq, wk, wv, wo, w1, w3, w2 tensor.DevicePtr
	}

	layers := make([]layerW, numLayers)
	var totalWeightBytes int64
	kvDim := numKVHeads * headDim
	for l := 0; l < numLayers; l++ {
		wqSize := hiddenSize * numBlocks4096 * bytesPerBlock
		wkSize := kvDim * numBlocks4096 * bytesPerBlock
		w1Size := intermediateSize * numBlocks4096 * bytesPerBlock
		w2Size := hiddenSize * numBlocks11008 * bytesPerBlock

		layers[l].wq = backend.Alloc(wqSize)
		layers[l].wk = backend.Alloc(wkSize)
		layers[l].wv = backend.Alloc(wkSize)
		layers[l].wo = backend.Alloc(wqSize)
		layers[l].w1 = backend.Alloc(w1Size)
		layers[l].w3 = backend.Alloc(w1Size)
		layers[l].w2 = backend.Alloc(w2Size)
		totalWeightBytes += int64(wqSize*2 + wkSize*2 + w1Size*2 + w2Size)
	}
	lmHead := backend.Alloc(32000 * numBlocks4096 * bytesPerBlock)
	totalWeightBytes += int64(32000 * numBlocks4096 * bytesPerBlock)

	// Activations
	x := backend.Alloc(hiddenSize * 4)
	residual := backend.Alloc(hiddenSize * 4)
	normWeight := backend.Alloc(hiddenSize * 4)
	normOut := backend.Alloc(hiddenSize * 4)
	qF16 := backend.Alloc(numQHeads * headDim * 2)
	kSrcF16 := backend.Alloc(numKVHeads * headDim * 2)
	vSrcF16 := backend.Alloc(numKVHeads * headDim * 2)
	maxSeqLen := 2048
	kvHeadStride := maxSeqLen * headDim
	kCacheF16 := backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	vCacheF16 := backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	sdpaOut := backend.Alloc(numQHeads * headDim * 2)
	gate := backend.Alloc(intermediateSize * 4)
	mlpOut := backend.Alloc(hiddenSize * 4)
	lmOut := backend.Alloc(32000 * 4)

	theta := float32(10000.0)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	warmup := 3
	iters := 20

	t.Logf("Total weight memory: %.2f GB", float64(totalWeightBytes)/1e9)

	for _, kvLen := range []int{16, 512} {
		name := fmt.Sprintf("ctx=%d", kvLen)
		t.Run(name, func(t *testing.T) {
			// Full pipeline dispatch sequence per layer (matching real fused path):
			//  1. FusedRMSNormQKV_F16 (1 dispatch — norm + Q/K/V)
			//  2. barrier
			//  3. RoPE+ScatterKV
			//  4. barrier
			//  5. SDPA F16
			//  6. barrier
			//  7. Wo (matvec)
			//  8. barrier
			//  9. AddRMSNorm
			// 10. barrier
			// 11. FusedMLP (1 dispatch — W1+W3+SiLU)
			// 12. barrier
			// 13. W2+Add2 (1 dispatch — matvec + residual add)

			dispatch := func(iter int) {
				backend.BeginBatch()
				for l := 0; l < numLayers; l++ {
					lw := layers[l]
					// Phase 1: FusedRMSNormQKV_F16 (1 dispatch for norm + Q + K + V)
					backend.MatMulQ4_0_FusedRMSNormQKV_F16(x, normWeight,
						lw.wq, lw.wk, lw.wv, qF16, kSrcF16, vSrcF16,
						hiddenSize, kvDim, hiddenSize, 1e-6)
					// barrier
					backend.MemoryBarrier()
					// Phase 2: RoPE+ScatterKV
					backend.RoPEScatterKVF16(qF16, kSrcF16, kCacheF16, vSrcF16, vCacheF16,
						numQHeads, numKVHeads, headDim, iter%maxSeqLen, headDim, theta, maxSeqLen, iter%maxSeqLen)
					// barrier
					backend.MemoryBarrier()
					// Phase 3: SDPA
					backend.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
					// barrier
					backend.MemoryBarrier()
					// Phase 4: Wo (plain matvec — real pipeline may use F16In variant)
					backend.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
					// barrier
					backend.MemoryBarrier()
					// Phase 5: AddRMSNorm
					backend.AddRMSNorm(x, residual, normWeight, normOut, 1, hiddenSize, 1e-6)
					// barrier
					backend.MemoryBarrier()
					// Phase 6: FusedMLP (W1+W3+SiLU in one dispatch)
					backend.MatMulQ4_0_FusedMLP(x, lw.w1, lw.w3, gate, 1, intermediateSize, hiddenSize)
					// barrier
					backend.MemoryBarrier()
					// Phase 7: W2+Add2 (matvec + residual add)
					backend.MatMulQ4_0_Add(gate, lw.w2, mlpOut, hiddenSize, intermediateSize)
				}
				// LM head
				backend.MatMulQ4_0(x, lmHead, lmOut, 1, 32000, hiddenSize)
				backend.EndBatch()
				backend.Sync()
			}

			// Warmup
			for i := 0; i < warmup; i++ {
				dispatch(i)
			}

			// Measure
			start := time.Now()
			for i := 0; i < iters; i++ {
				dispatch(i)
			}
			elapsed := time.Since(start)

			perToken := elapsed.Seconds() / float64(iters)
			gbps := float64(totalWeightBytes) / perToken / 1e9
			toksPerSec := 1.0 / perToken

			t.Logf("\n=== Full Pipeline Simulation (LLaMA 2 7B Q4_0, %s) ===", name)
			t.Logf("  Per-token time: %.2f ms", perToken*1e3)
			t.Logf("  Effective BW:   %.1f GB/s (%.1f%% of 400 GB/s)", gbps, gbps/400*100)
			t.Logf("  Throughput:     %.1f tok/s", toksPerSec)
			t.Logf("")
			t.Logf("  llama.cpp target: 76.3 tok/s = 13.11 ms")
			gap := 13.11 - perToken*1e3
			if gap > 0 {
				t.Logf("  ✓ Ahead by %.2f ms (%.1f tok/s surplus)", gap, toksPerSec-76.3)
			} else {
				t.Logf("  ✗ Behind by %.2f ms", -gap)
			}
			t.Logf("  MLX target: 83.5 tok/s = 11.98 ms")
		})
	}
}

// TestPipelineSimulationAccurate is a more realistic simulation that matches the actual
// decode pipeline exactly: per-layer KV caches, F16In Wo kernel, extra inter-layer barriers,
// and final RMSNorm. Compares with the simpler simulation to identify GPU-side overhead.
func TestPipelineSimulationAccurate(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer backend.Close()

	// LLaMA 2 7B Q4_0 config
	numLayers := 32
	hiddenSize := 4096
	numQHeads := 32
	numKVHeads := 32
	headDim := 128
	intermediateSize := 11008

	// Q4_0 sizes
	bytesPerBlock := 18
	elemsPerBlock := 32
	numBlocks4096 := hiddenSize / elemsPerBlock
	numBlocks11008 := intermediateSize / elemsPerBlock

	// Per-layer weights
	type layerW struct {
		wq, wk, wv, wo, w1, w3, w2 tensor.DevicePtr
	}
	layers := make([]layerW, numLayers)
	kvDim := numKVHeads * headDim
	for l := 0; l < numLayers; l++ {
		wqSize := hiddenSize * numBlocks4096 * bytesPerBlock
		wkSize := kvDim * numBlocks4096 * bytesPerBlock
		w1Size := intermediateSize * numBlocks4096 * bytesPerBlock
		w2Size := hiddenSize * numBlocks11008 * bytesPerBlock
		layers[l].wq = backend.Alloc(wqSize)
		layers[l].wk = backend.Alloc(wkSize)
		layers[l].wv = backend.Alloc(wkSize)
		layers[l].wo = backend.Alloc(wqSize)
		layers[l].w1 = backend.Alloc(w1Size)
		layers[l].w3 = backend.Alloc(w1Size)
		layers[l].w2 = backend.Alloc(w2Size)
	}
	lmHead := backend.Alloc(32000 * numBlocks4096 * bytesPerBlock)

	// Activations
	x := backend.Alloc(hiddenSize * 4)
	residual := backend.Alloc(hiddenSize * 4)
	normWeight := backend.Alloc(hiddenSize * 4)
	normOut := backend.Alloc(hiddenSize * 4)
	woOut := backend.Alloc(hiddenSize * 4)
	qF16 := backend.Alloc(numQHeads * headDim * 2)
	kSrcF16 := backend.Alloc(numKVHeads * headDim * 2)
	vSrcF16 := backend.Alloc(numKVHeads * headDim * 2)
	sdpaOut := backend.Alloc(numQHeads * headDim * 2)
	gate := backend.Alloc(intermediateSize * 4)
	mlpOut := backend.Alloc(hiddenSize * 4)
	lmOut := backend.Alloc(32000 * 4)

	// Per-layer KV caches (matching actual decode — 32 separate KV cache pairs)
	maxSeqLen := 2048
	kvHeadStride := maxSeqLen * headDim
	type layerKV struct {
		kCache, vCache tensor.DevicePtr
	}
	kvCaches := make([]layerKV, numLayers)
	for l := 0; l < numLayers; l++ {
		kvCaches[l].kCache = backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
		kvCaches[l].vCache = backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	}

	theta := float32(10000.0)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	kvLen := 16
	warmup := 3
	iters := 20

	// Simple simulation (same as TestFullPipelineSimulation)
	simpleDispatch := func(iter int) {
		backend.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			backend.MatMulQ4_0_FusedRMSNormQKV_F16(x, normWeight,
				lw.wq, lw.wk, lw.wv, qF16, kSrcF16, vSrcF16,
				hiddenSize, kvDim, hiddenSize, 1e-6)
			backend.MemoryBarrier()
			backend.RoPEScatterKVF16(qF16, kSrcF16, kvCaches[0].kCache, vSrcF16, kvCaches[0].vCache,
				numQHeads, numKVHeads, headDim, iter%maxSeqLen, headDim, theta, maxSeqLen, iter%maxSeqLen)
			backend.MemoryBarrier()
			backend.SDPAF16(qF16, kvCaches[0].kCache, kvCaches[0].vCache, sdpaOut,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.MemoryBarrier()
			backend.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			backend.MemoryBarrier()
			backend.AddRMSNorm(x, residual, normWeight, normOut, 1, hiddenSize, 1e-6)
			backend.MemoryBarrier()
			backend.MatMulQ4_0_FusedMLP(x, lw.w1, lw.w3, gate, 1, intermediateSize, hiddenSize)
			backend.MemoryBarrier()
			backend.MatMulQ4_0_Add(gate, lw.w2, mlpOut, hiddenSize, intermediateSize)
		}
		backend.MatMulQ4_0(x, lmHead, lmOut, 1, 32000, hiddenSize)
		backend.EndBatch()
		backend.Sync()
	}

	// Accurate simulation (matching actual decode pipeline)
	accurateDispatch := func(iter int) {
		backend.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			lkv := kvCaches[l] // Per-layer KV cache (like actual decode)

			backend.MatMulQ4_0_FusedRMSNormQKV_F16(x, normWeight,
				lw.wq, lw.wk, lw.wv, qF16, kSrcF16, vSrcF16,
				hiddenSize, kvDim, hiddenSize, 1e-6)
			backend.MemoryBarrier()
			backend.RoPEScatterKVF16(qF16, kSrcF16, lkv.kCache, vSrcF16, lkv.vCache,
				numQHeads, numKVHeads, headDim, iter%maxSeqLen, headDim, theta, maxSeqLen, iter%maxSeqLen)
			backend.MemoryBarrier()
			backend.SDPAF16(qF16, lkv.kCache, lkv.vCache, sdpaOut,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.MemoryBarrier()
			// Use F16In Wo (matching actual decode FP16 path)
			backend.MatMulQ4_0_F16In(sdpaOut, lw.wo, woOut, hiddenSize, numQHeads*headDim)
			backend.MemoryBarrier()
			backend.AddRMSNorm(woOut, x, normWeight, normOut, 1, hiddenSize, 1e-6)
			backend.MemoryBarrier()
			backend.MatMulQ4_0_FusedMLP(normOut, lw.w1, lw.w3, gate, 1, intermediateSize, hiddenSize)
			backend.MemoryBarrier()
			backend.MatMulQ4_0_Add(gate, lw.w2, x, hiddenSize, intermediateSize)
			// Extra barrier (from nested EndBatch in actual decode)
			backend.MemoryBarrier()
		}
		// Final RMSNorm (matching actual decode)
		backend.RMSNorm(x, normWeight, normOut, 1, hiddenSize, 1e-6)
		backend.MemoryBarrier()
		backend.MatMulQ4_0(normOut, lmHead, lmOut, 1, 32000, hiddenSize)
		backend.EndBatch()
		backend.Sync()
	}

	// Warmup both
	for i := 0; i < warmup; i++ {
		simpleDispatch(i)
		accurateDispatch(i)
	}

	// Measure simple
	start := time.Now()
	for i := 0; i < iters; i++ {
		simpleDispatch(i)
	}
	simpleMs := float64(time.Since(start).Microseconds()) / 1000.0 / float64(iters)

	// Measure accurate
	start = time.Now()
	for i := 0; i < iters; i++ {
		accurateDispatch(i)
	}
	accurateMs := float64(time.Since(start).Microseconds()) / 1000.0 / float64(iters)

	t.Logf("\n=== Pipeline Simulation Comparison (ctx=%d) ===", kvLen)
	t.Logf("  Simple (shared KV, FP32 Wo):    %.2f ms (%.1f tok/s)", simpleMs, 1000.0/simpleMs)
	t.Logf("  Accurate (per-layer KV, F16In):  %.2f ms (%.1f tok/s)", accurateMs, 1000.0/accurateMs)
	t.Logf("  Delta: %.2f ms", accurateMs-simpleMs)
	if accurateMs > simpleMs {
		t.Logf("  Per-layer KV + accurate pipeline costs %.2f ms extra", accurateMs-simpleMs)
	}
}

// TestSDPAContextScaling benchmarks the SDPA flash decode F16 kernel in isolation
// at various context lengths to identify if context degradation is from SDPA or matvec.
func TestSDPAContextScaling(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer backend.Close()

	numQHeads := 32
	numKVHeads := 32
	headDim := 128
	maxSeqLen := 2048
	kvHeadStride := maxSeqLen * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	qF16 := backend.Alloc(numQHeads * headDim * 2)
	kCacheF16 := backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	vCacheF16 := backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	sdpaOut := backend.Alloc(numQHeads * headDim * 2)

	warmup := 5
	iters := 100

	for _, kvLen := range []int{1, 16, 64, 128, 256, 512, 1024} {
		// Warmup
		for i := 0; i < warmup; i++ {
			backend.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()
		}

		// Measure
		start := time.Now()
		for i := 0; i < iters; i++ {
			backend.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.Sync()
		}
		elapsed := time.Since(start)
		usPerCall := float64(elapsed.Microseconds()) / float64(iters)

		t.Logf("  kvLen=%4d: %.1f µs (%.3f ms) per SDPA call  [×32 layers = %.2f ms total]",
			kvLen, usPerCall, usPerCall/1000, usPerCall*32/1000)
	}
}

// TestSDPAContextScalingBatched measures SDPA F16 scaling within a batched command buffer,
// matching how it runs in the actual decode pipeline (32 calls batched together).
// Compares the standard flash decode vs the tiled split-K variant at each context length.
func TestSDPAContextScalingBatched(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer backend.Close()

	numQHeads := 32
	numKVHeads := 32
	headDim := 128
	maxSeqLen := 2048
	kvHeadStride := maxSeqLen * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	numLayers := 32

	qF16 := backend.Alloc(numQHeads * headDim * 2)
	kCacheF16 := backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	vCacheF16 := backend.Alloc(numKVHeads * maxSeqLen * headDim * 2)
	sdpaOut := backend.Alloc(numQHeads * headDim * 2)

	// Tiled partials: numQHeads * ceil(maxKVLen/64) * (2 + headDim) * 4 bytes
	maxTiles := (2048 + 63) / 64
	partialsSize := numQHeads * maxTiles * (2 + headDim) * 4
	partials := backend.Alloc(partialsSize)

	warmup := 5
	iters := 30

	type result struct {
		kvLen              int
		standardMs, tiledMs float64
	}
	var results []result

	for _, kvLen := range []int{16, 64, 128, 256, 512, 1024} {
		// Standard SDPA
		dispatchStd := func() {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16(qF16, kCacheF16, vCacheF16, sdpaOut,
					kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
				backend.MemoryBarrier()
			}
			backend.EndBatch()
			backend.Sync()
		}

		// Tiled SDPA
		dispatchTiled := func() {
			backend.BeginBatch()
			for l := 0; l < numLayers; l++ {
				backend.SDPAF16Tiled(qF16, kCacheF16, vCacheF16, sdpaOut, partials,
					kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
				backend.MemoryBarrier()
			}
			backend.EndBatch()
			backend.Sync()
		}

		// Warmup
		for i := 0; i < warmup; i++ {
			dispatchStd()
			dispatchTiled()
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			dispatchStd()
		}
		stdMs := float64(time.Since(start).Microseconds()) / 1000.0 / float64(iters)

		start = time.Now()
		for i := 0; i < iters; i++ {
			dispatchTiled()
		}
		tiledMs := float64(time.Since(start).Microseconds()) / 1000.0 / float64(iters)

		results = append(results, result{kvLen, stdMs, tiledMs})
		t.Logf("  kvLen=%4d: standard=%.2f ms  tiled=%.2f ms  delta=%.2f ms (%.1f%%)",
			kvLen, stdMs, tiledMs, stdMs-tiledMs,
			(stdMs-tiledMs)/stdMs*100)
	}

	// Print summary with context degradation
	t.Log("\n--- Context degradation (reference: ctx=16) ---")
	ref := results[0]
	for _, r := range results {
		stdDeg := (r.standardMs - ref.standardMs) / ref.standardMs * 100
		tiledDeg := (r.tiledMs - ref.tiledMs) / ref.tiledMs * 100
		t.Logf("  kvLen=%4d: standard +%.1f%%  tiled +%.1f%%",
			r.kvLen, stdDeg, tiledDeg)
	}
}

// TestKVConsolidationSimulation validates that consolidating 64 per-layer KV Metal buffers
// into 2 contiguous buffers (one for all K, one for all V) recovers the ~1.8ms overhead
// caused by Metal buffer management with many allocated buffer objects.
func TestKVConsolidationSimulation(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer backend.Close()

	// LLaMA 2 7B Q4_0 config
	numLayers := 32
	hiddenSize := 4096
	numQHeads := 32
	numKVHeads := 32
	headDim := 128
	intermediateSize := 11008

	// Q4_0 sizes
	bytesPerBlock := 18
	elemsPerBlock := 32
	numBlocks4096 := hiddenSize / elemsPerBlock
	numBlocks11008 := intermediateSize / elemsPerBlock

	// Per-layer weights (same allocation as actual model — 291 buffers)
	type layerW struct {
		wq, wk, wv, wo, w1, w3, w2 tensor.DevicePtr
	}
	layers := make([]layerW, numLayers)
	kvDim := numKVHeads * headDim
	for l := 0; l < numLayers; l++ {
		wqSize := hiddenSize * numBlocks4096 * bytesPerBlock
		wkSize := kvDim * numBlocks4096 * bytesPerBlock
		w1Size := intermediateSize * numBlocks4096 * bytesPerBlock
		w2Size := hiddenSize * numBlocks11008 * bytesPerBlock
		layers[l].wq = backend.Alloc(wqSize)
		layers[l].wk = backend.Alloc(wkSize)
		layers[l].wv = backend.Alloc(wkSize)
		layers[l].wo = backend.Alloc(wqSize)
		layers[l].w1 = backend.Alloc(w1Size)
		layers[l].w3 = backend.Alloc(w1Size)
		layers[l].w2 = backend.Alloc(w2Size)
	}
	lmHead := backend.Alloc(32000 * numBlocks4096 * bytesPerBlock)

	// Activations
	x := backend.Alloc(hiddenSize * 4)
	residual := backend.Alloc(hiddenSize * 4)
	normWeight := backend.Alloc(hiddenSize * 4)
	normOut := backend.Alloc(hiddenSize * 4)
	woOut := backend.Alloc(hiddenSize * 4)
	qF16 := backend.Alloc(numQHeads * headDim * 2)
	kSrcF16 := backend.Alloc(numKVHeads * headDim * 2)
	vSrcF16 := backend.Alloc(numKVHeads * headDim * 2)
	sdpaOut := backend.Alloc(numQHeads * headDim * 2)
	gate := backend.Alloc(intermediateSize * 4)
	lmOut := backend.Alloc(32000 * 4)

	maxSeqLen := 2048
	kvHeadStride := maxSeqLen * headDim
	layerKVSize := numKVHeads * maxSeqLen * headDim * 2 // FP16

	// ---- Variant A: 64 separate KV buffers (current approach) ----
	type layerKV struct {
		kCache, vCache tensor.DevicePtr
	}
	kvSeparate := make([]layerKV, numLayers)
	for l := 0; l < numLayers; l++ {
		kvSeparate[l].kCache = backend.Alloc(layerKVSize)
		kvSeparate[l].vCache = backend.Alloc(layerKVSize)
	}

	// ---- Variant B: 2 consolidated KV buffers with per-layer offsets ----
	kContig := backend.AllocPermanent(numLayers * layerKVSize)
	vContig := backend.AllocPermanent(numLayers * layerKVSize)
	kvConsolidated := make([]layerKV, numLayers)
	for l := 0; l < numLayers; l++ {
		off := uintptr(l * layerKVSize)
		kvConsolidated[l].kCache = tensor.DevicePtrOffset(kContig, off)
		kvConsolidated[l].vCache = tensor.DevicePtrOffset(vContig, off)
	}

	theta := float32(10000.0)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	kvLen := 16
	warmup := 5
	iters := 30

	dispatch := func(kv []layerKV, iter int) {
		backend.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			lkv := kv[l]

			backend.MatMulQ4_0_FusedRMSNormQKV_F16(x, normWeight,
				lw.wq, lw.wk, lw.wv, qF16, kSrcF16, vSrcF16,
				hiddenSize, kvDim, hiddenSize, 1e-6)
			backend.MemoryBarrier()
			backend.RoPEScatterKVF16(qF16, kSrcF16, lkv.kCache, vSrcF16, lkv.vCache,
				numQHeads, numKVHeads, headDim, iter%maxSeqLen, headDim, theta, maxSeqLen, iter%maxSeqLen)
			backend.MemoryBarrier()
			backend.SDPAF16(qF16, lkv.kCache, lkv.vCache, sdpaOut,
				kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
			backend.MemoryBarrier()
			backend.MatMulQ4_0_F16In(sdpaOut, lw.wo, woOut, hiddenSize, numQHeads*headDim)
			backend.MemoryBarrier()
			backend.AddRMSNorm(woOut, x, normWeight, normOut, 1, hiddenSize, 1e-6)
			backend.MemoryBarrier()
			backend.MatMulQ4_0_FusedMLP(normOut, lw.w1, lw.w3, gate, 1, intermediateSize, hiddenSize)
			backend.MemoryBarrier()
			backend.MatMulQ4_0_Add(gate, lw.w2, x, hiddenSize, intermediateSize)
			backend.MemoryBarrier()
		}
		backend.RMSNorm(x, normWeight, normOut, 1, hiddenSize, 1e-6)
		backend.MemoryBarrier()
		backend.MatMulQ4_0(normOut, lmHead, lmOut, 1, 32000, hiddenSize)
		backend.EndBatch()
		backend.Sync()
	}

	_ = residual // suppress unused

	// Warmup both
	for i := 0; i < warmup; i++ {
		dispatch(kvSeparate, i)
		dispatch(kvConsolidated, i)
	}

	// Measure separate KV (current approach)
	start := time.Now()
	for i := 0; i < iters; i++ {
		dispatch(kvSeparate, i)
	}
	separateMs := float64(time.Since(start).Microseconds()) / 1000.0 / float64(iters)

	// Measure consolidated KV (proposed approach)
	start = time.Now()
	for i := 0; i < iters; i++ {
		dispatch(kvConsolidated, i)
	}
	consolidatedMs := float64(time.Since(start).Microseconds()) / 1000.0 / float64(iters)

	t.Logf("\n=== KV Cache Consolidation Test (ctx=%d) ===", kvLen)
	t.Logf("  Separate KV (64 Metal buffers):      %.2f ms (%.1f tok/s)", separateMs, 1000.0/separateMs)
	t.Logf("  Consolidated KV (2 Metal buffers):    %.2f ms (%.1f tok/s)", consolidatedMs, 1000.0/consolidatedMs)
	t.Logf("  Delta: %.2f ms (%.1f%% improvement)", separateMs-consolidatedMs,
		(separateMs-consolidatedMs)/separateMs*100)

	if consolidatedMs < separateMs {
		t.Logf("  ✓ Consolidation recovers %.2f ms per token!", separateMs-consolidatedMs)
	} else {
		t.Logf("  ✗ Consolidation did not help (%.2f ms overhead)", consolidatedMs-separateMs)
	}
}

// TestBarrierOverhead measures the cost of memory barriers in a realistic pipeline.
// Compares matvec-only dispatch (no barriers) vs with barriers between each dispatch group
// as in the real fused decode path (6 barriers per layer × 32 layers = 192 barriers).
func TestBarrierOverhead(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skip("Metal not available")
	}
	defer b.Close()

	// LLaMA 2 7B Q4_0 config
	numLayers := 32
	hiddenSize := 4096
	numKVHeads := 8
	headDim := 128
	intermediateSize := 11008

	blockSize := 32
	bytesPerBlock := 18
	numBlocks4096 := (hiddenSize + blockSize - 1) / blockSize
	numBlocks11008 := (intermediateSize + blockSize - 1) / blockSize

	// Per-layer weights (different per layer to prevent L2 caching)
	type layerWeights struct {
		wq, wk, wv, wo, w1, w3, w2 tensor.DevicePtr
	}
	layers := make([]layerWeights, numLayers)
	var totalWeightBytes int64
	for l := 0; l < numLayers; l++ {
		kvDim := numKVHeads * headDim
		wqSize := hiddenSize * numBlocks4096 * bytesPerBlock
		wkSize := kvDim * numBlocks4096 * bytesPerBlock
		w1Size := intermediateSize * numBlocks4096 * bytesPerBlock
		w2Size := hiddenSize * numBlocks11008 * bytesPerBlock
		layers[l].wq = b.Alloc(wqSize)
		layers[l].wk = b.Alloc(wkSize)
		layers[l].wv = b.Alloc(wkSize)
		layers[l].wo = b.Alloc(wqSize)
		layers[l].w1 = b.Alloc(w1Size)
		layers[l].w3 = b.Alloc(w1Size)
		layers[l].w2 = b.Alloc(w2Size)
		totalWeightBytes += int64(wqSize + wkSize*2 + wqSize + w1Size*2 + w2Size)
	}

	// Activations
	x := b.Alloc(hiddenSize * 4)
	gate := b.Alloc(intermediateSize * 4)
	mlpOut := b.Alloc(hiddenSize * 4)

	kvDim := numKVHeads * headDim
	warmup := 3
	iters := 20

	// ============================================
	// A. No barriers (existing matvec ceiling)
	// ============================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		b.EndBatch()
		b.Sync()
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		b.EndBatch()
		b.Sync()
	}
	noBarrierTime := time.Since(start).Seconds() / float64(iters)

	// ============================================
	// B. With 6 barriers per layer (real pipeline pattern)
	// Barrier after each dispatch group:
	//   QKV(3 dispatches) → barrier → Wo(1) → barrier → (skip non-matvec) → barrier → MLP(2) → barrier → W2(1)
	// Simplified: barrier between each of the 4 matvec groups per layer
	// ============================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			// Group 1: QKV
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			b.MemoryBarrier()
			// Group 2: Wo
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			b.MemoryBarrier()
			// Group 3: MLP (W1+W3)
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			b.MemoryBarrier()
			// Group 4: W2
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
			b.MemoryBarrier()
		}
		b.EndBatch()
		b.Sync()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			b.MemoryBarrier()
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			b.MemoryBarrier()
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			b.MemoryBarrier()
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
			b.MemoryBarrier()
		}
		b.EndBatch()
		b.Sync()
	}
	barrier4Time := time.Since(start).Seconds() / float64(iters)

	// ============================================
	// C. With 6 barriers per layer (matches real fused path exactly)
	// FusedQKV → barrier → RoPE(skip) → barrier → SDPA(skip) → barrier → Wo → barrier → AddRMSNorm(skip) → barrier → FusedMLP → barrier → W2+Add2
	// 6 barriers between 7 dispatch slots (4 matvec + 3 non-matvec)
	// ============================================
	for i := 0; i < warmup; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			// FusedQKV
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			b.MemoryBarrier() // 1: after QKV, before RoPE
			b.MemoryBarrier() // 2: after RoPE, before SDPA (no RoPE dispatch here)
			b.MemoryBarrier() // 3: after SDPA, before Wo (no SDPA dispatch here)
			// Wo
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			b.MemoryBarrier() // 4: after Wo, before AddRMSNorm
			b.MemoryBarrier() // 5: after AddRMSNorm, before FusedMLP (no AddRMSNorm here)
			// FusedMLP (W1 + W3)
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			b.MemoryBarrier() // 6: after FusedMLP, before W2
			// W2
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		b.EndBatch()
		b.Sync()
	}
	start = time.Now()
	for i := 0; i < iters; i++ {
		b.BeginBatch()
		for l := 0; l < numLayers; l++ {
			lw := layers[l]
			b.MatMulQ4_0(x, lw.wq, x, 1, hiddenSize, hiddenSize)
			b.MatMulQ4_0(x, lw.wk, x, 1, kvDim, hiddenSize)
			b.MatMulQ4_0(x, lw.wv, x, 1, kvDim, hiddenSize)
			b.MemoryBarrier()
			b.MemoryBarrier()
			b.MemoryBarrier()
			b.MatMulQ4_0(x, lw.wo, x, 1, hiddenSize, hiddenSize)
			b.MemoryBarrier()
			b.MemoryBarrier()
			b.MatMulQ4_0(x, lw.w1, gate, 1, intermediateSize, hiddenSize)
			b.MatMulQ4_0(x, lw.w3, gate, 1, intermediateSize, hiddenSize)
			b.MemoryBarrier()
			b.MatMulQ4_0(gate, lw.w2, mlpOut, 1, hiddenSize, intermediateSize)
		}
		b.EndBatch()
		b.Sync()
	}
	barrier6Time := time.Since(start).Seconds() / float64(iters)

	numBarriers4 := 4 * numLayers
	numBarriers6 := 6 * numLayers
	barrierCost4 := (barrier4Time - noBarrierTime) / float64(numBarriers4)
	barrierCost6 := (barrier6Time - noBarrierTime) / float64(numBarriers6)

	t.Logf("\n=== Barrier Overhead Analysis (LLaMA 2 7B Q4_0, %d layers) ===", numLayers)
	t.Logf("  No barriers:    %.2f ms (%.1f tok/s)", noBarrierTime*1e3, 1.0/noBarrierTime)
	t.Logf("  4 barriers/layer (%d total): %.2f ms (%.1f tok/s)", numBarriers4, barrier4Time*1e3, 1.0/barrier4Time)
	t.Logf("  6 barriers/layer (%d total): %.2f ms (%.1f tok/s)", numBarriers6, barrier6Time*1e3, 1.0/barrier6Time)
	t.Logf("")
	t.Logf("  Barrier overhead (4/layer): %.2f ms total, %.1f µs/barrier",
		(barrier4Time-noBarrierTime)*1e3, barrierCost4*1e6)
	t.Logf("  Barrier overhead (6/layer): %.2f ms total, %.1f µs/barrier",
		(barrier6Time-noBarrierTime)*1e3, barrierCost6*1e6)
	t.Logf("")
	t.Logf("  Real decode @ctx16: ~14.34 ms (69.7 tok/s)")
	t.Logf("  Gap (6-barrier vs real): %.2f ms (non-matvec ops + CGo + fused overhead)",
		(14.34-barrier6Time*1e3))
}
