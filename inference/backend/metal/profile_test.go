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
