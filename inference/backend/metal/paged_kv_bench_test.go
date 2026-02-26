//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// BenchmarkPagedSDPAvsContiguous measures throughput of paged SDPA decode
// against contiguous SDPA for realistic LLaMA-like dimensions.
//
// Track 3: Paged KV Batching, Phase 3 Verification.
func BenchmarkPagedSDPAvsContiguous(b *testing.B) {
	be, err := NewBackend(0)
	if err != nil {
		b.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	configs := []struct {
		name       string
		numQHeads  int
		numKVHeads int
		headDim    int
		blockSize  int
		kvLen      int
	}{
		{"llama7b_kv256", 32, 32, 128, 16, 256},
		{"llama7b_kv512", 32, 32, 128, 16, 512},
		{"llama7b_kv1024", 32, 32, 128, 16, 1024},
		{"llama7b_kv2048", 32, 32, 128, 16, 2048},
		{"gqa_kv1024", 32, 8, 128, 16, 1024},
	}

	for _, cfg := range configs {
		qSize := cfg.numQHeads * cfg.headDim
		kvElems := cfg.kvLen * cfg.numKVHeads * cfg.headDim
		numBlocks := (cfg.kvLen + cfg.blockSize - 1) / cfg.blockSize
		tokensInLastBlock := cfg.kvLen - (numBlocks-1)*cfg.blockSize
		scale := float32(1.0 / math.Sqrt(float64(cfg.headDim)))
		kvStride := cfg.kvLen * cfg.headDim

		// Allocate contiguous buffers
		qBuf := be.Alloc(qSize * 4)
		kContigBuf := be.Alloc(kvElems * 4)
		vContigBuf := be.Alloc(kvElems * 4)
		outContigBuf := be.Alloc(qSize * 4)

		// Fill with random data
		qData := make([]float32, qSize)
		for i := range qData {
			qData[i] = (rand.Float32() - 0.5) * 0.2
		}
		be.ToDevice(qBuf, float32ToBytes(qData))
		be.Zero(kContigBuf, kvElems)
		be.Zero(vContigBuf, kvElems)
		be.Sync()

		// Allocate paged buffers
		blockElems := cfg.blockSize * cfg.numKVHeads * cfg.headDim * 2
		poolBuf := be.Alloc(numBlocks * blockElems * 4)
		be.Zero(poolBuf, numBlocks*blockElems)

		blockTableData := make([]int32, numBlocks)
		for i := range blockTableData {
			blockTableData[i] = int32(i)
		}
		btBuf := be.Alloc(numBlocks * 4)
		be.ToDevice(btBuf, int32ToBytes(blockTableData))
		outPagedBuf := be.Alloc(qSize * 4)
		be.Sync()

		// Benchmark contiguous SDPA
		b.Run("contiguous/"+cfg.name, func(b *testing.B) {
			// Warmup
			for i := 0; i < 5; i++ {
				be.SDPA(qBuf, kContigBuf, vContigBuf, outContigBuf, cfg.kvLen,
					cfg.numQHeads, cfg.numKVHeads, cfg.headDim, scale, kvStride)
				be.Sync()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				be.SDPA(qBuf, kContigBuf, vContigBuf, outContigBuf, cfg.kvLen,
					cfg.numQHeads, cfg.numKVHeads, cfg.headDim, scale, kvStride)
				be.Sync()
			}
		})

		// Benchmark paged SDPA
		b.Run("paged/"+cfg.name, func(b *testing.B) {
			for i := 0; i < 5; i++ {
				be.SDPAPagedDecode(qBuf, poolBuf, btBuf, outPagedBuf,
					numBlocks, cfg.blockSize, cfg.numQHeads, cfg.numKVHeads,
					cfg.headDim, scale, tokensInLastBlock)
				be.Sync()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				be.SDPAPagedDecode(qBuf, poolBuf, btBuf, outPagedBuf,
					numBlocks, cfg.blockSize, cfg.numQHeads, cfg.numKVHeads,
					cfg.headDim, scale, tokensInLastBlock)
				be.Sync()
			}
		})

		be.Free(qBuf)
		be.Free(kContigBuf)
		be.Free(vContigBuf)
		be.Free(outContigBuf)
		be.Free(poolBuf)
		be.Free(btBuf)
		be.Free(outPagedBuf)
	}
}

// TestPagedVsContiguousE2E is the Phase 3 end-to-end correctness verification.
// It measures both correctness AND performance of the full GPU-native paged path
// (ReshapePagedKV + SDPAPagedDecode) vs the current CPU roundtrip path
// (ToHost + CPU cache + ToDevice + contiguous SDPA).
func TestPagedVsContiguousE2E(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	const (
		numQHeads  = 32
		numKVHeads = 32
		headDim    = 128
		blockSize  = 16
		numLayers  = 32
		iters      = 50
	)

	seqLens := []int{128, 512, 1024, 2048}

	fmt.Println("\n[PAGED vs CONTIGUOUS E2E COMPARISON]")
	fmt.Println("Model: LLaMA 2 7B dimensions, 32 layers")
	fmt.Printf("Benchmark: single decode token, %d iterations per measurement\n\n", iters)

	for _, kvLen := range seqLens {
		qSize := numQHeads * headDim
		kvElems := kvLen * numKVHeads * headDim
		newKVElems := numKVHeads * headDim
		numBlocks := (kvLen + blockSize - 1) / blockSize
		tokensInLastBlock := kvLen - (numBlocks-1)*blockSize
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		kvStride := kvLen * headDim

		// Random Q and K/V
		qData := make([]float32, qSize)
		kData := make([]float32, kvElems)
		vData := make([]float32, kvElems)
		for i := range qData {
			qData[i] = (rand.Float32() - 0.5) * 0.1
		}
		for i := range kData {
			kData[i] = (rand.Float32() - 0.5) * 0.1
			vData[i] = (rand.Float32() - 0.5) * 0.1
		}

		// Head-major layout for contiguous SDPA reference
		kHM := make([]float32, kvElems)
		vHM := make([]float32, kvElems)
		for pos := 0; pos < kvLen; pos++ {
			for h := 0; h < numKVHeads; h++ {
				for d := 0; d < headDim; d++ {
					src := pos*numKVHeads*headDim + h*headDim + d
					dst := h*kvLen*headDim + pos*headDim + d
					kHM[dst] = kData[src]
					vHM[dst] = vData[src]
				}
			}
		}

		qBuf := be.Alloc(qSize * 4)
		be.ToDevice(qBuf, float32ToBytes(qData))

		// === Path A: CPU roundtrip simulation ===
		// Upload new K/V (1 token), download full K/V to CPU, re-upload, SDPA
		newKBuf := be.Alloc(newKVElems * 4)
		newVBuf := be.Alloc(newKVElems * 4)
		be.ToDevice(newKBuf, float32ToBytes(kData[:newKVElems]))
		be.ToDevice(newVBuf, float32ToBytes(vData[:newKVElems]))

		fullKBuf := be.Alloc(kvElems * 4)
		fullVBuf := be.Alloc(kvElems * 4)
		be.ToDevice(fullKBuf, float32ToBytes(kHM))
		be.ToDevice(fullVBuf, float32ToBytes(vHM))
		outCPU := be.Alloc(qSize * 4)

		cpuNewK := make([]byte, newKVElems*4)
		cpuNewV := make([]byte, newKVElems*4)
		cpuFullK := float32ToBytes(kHM)
		cpuFullV := float32ToBytes(vHM)

		be.Sync()

		// Warmup
		for i := 0; i < 5; i++ {
			be.Sync()
			be.ToHost(cpuNewK, newKBuf)
			be.ToHost(cpuNewV, newVBuf)
			be.ToDevice(fullKBuf, cpuFullK)
			be.ToDevice(fullVBuf, cpuFullV)
			be.SDPA(qBuf, fullKBuf, fullVBuf, outCPU, kvLen, numQHeads, numKVHeads, headDim, scale, kvStride)
			be.Sync()
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			be.Sync()
			be.ToHost(cpuNewK, newKBuf)
			be.ToHost(cpuNewV, newVBuf)
			be.Sync()
			be.ToDevice(fullKBuf, cpuFullK)
			be.ToDevice(fullVBuf, cpuFullV)
			be.SDPA(qBuf, fullKBuf, fullVBuf, outCPU, kvLen, numQHeads, numKVHeads, headDim, scale, kvStride)
			be.Sync()
		}
		cpuPathTime := time.Since(start) / time.Duration(iters)

		refBytes := make([]byte, qSize*4)
		be.ToHost(refBytes, outCPU)
		refOut := bytesToFloat32(refBytes)

		// === Path B: GPU-native paged ===
		blockElems := blockSize * numKVHeads * headDim * 2
		poolBuf := be.Alloc(numBlocks * blockElems * 4)
		be.Zero(poolBuf, numBlocks*blockElems)

		// Scatter all K/V into blocks
		pageTable := make([]int32, kvLen)
		blockOffsets := make([]int32, kvLen)
		for i := 0; i < kvLen; i++ {
			pageTable[i] = int32(i / blockSize)
			blockOffsets[i] = int32(i % blockSize)
		}
		ptBuf := be.Alloc(kvLen * 4)
		offBuf := be.Alloc(kvLen * 4)
		be.ToDevice(ptBuf, int32ToBytes(pageTable))
		be.ToDevice(offBuf, int32ToBytes(blockOffsets))

		kSrcBuf := be.Alloc(kvElems * 4)
		vSrcBuf := be.Alloc(kvElems * 4)
		be.ToDevice(kSrcBuf, float32ToBytes(kData))
		be.ToDevice(vSrcBuf, float32ToBytes(vData))

		be.ReshapePagedKV(kSrcBuf, poolBuf, ptBuf, offBuf, kvLen, numKVHeads, headDim, blockSize, false)
		be.ReshapePagedKV(vSrcBuf, poolBuf, ptBuf, offBuf, kvLen, numKVHeads, headDim, blockSize, true)
		be.Sync()

		blockTableData := make([]int32, numBlocks)
		for i := range blockTableData {
			blockTableData[i] = int32(i)
		}
		btBuf := be.Alloc(numBlocks * 4)
		be.ToDevice(btBuf, int32ToBytes(blockTableData))
		outGPU := be.Alloc(qSize * 4)

		// For the benchmark, simulate single-token decode: scatter 1 token + paged SDPA
		singlePtBuf := be.Alloc(4)
		singleOffBuf := be.Alloc(4)

		// Warmup
		for i := 0; i < 5; i++ {
			be.ToDevice(singlePtBuf, []byte{0, 0, 0, 0})
			be.ToDevice(singleOffBuf, []byte{0, 0, 0, 0})
			be.ReshapePagedKV(newKBuf, poolBuf, singlePtBuf, singleOffBuf, 1, numKVHeads, headDim, blockSize, false)
			be.ReshapePagedKV(newVBuf, poolBuf, singlePtBuf, singleOffBuf, 1, numKVHeads, headDim, blockSize, true)
			be.SDPAPagedDecode(qBuf, poolBuf, btBuf, outGPU,
				numBlocks, blockSize, numQHeads, numKVHeads, headDim, scale, tokensInLastBlock)
			be.Sync()
		}

		start = time.Now()
		for i := 0; i < iters; i++ {
			be.ToDevice(singlePtBuf, []byte{0, 0, 0, 0})
			be.ToDevice(singleOffBuf, []byte{0, 0, 0, 0})
			be.ReshapePagedKV(newKBuf, poolBuf, singlePtBuf, singleOffBuf, 1, numKVHeads, headDim, blockSize, false)
			be.ReshapePagedKV(newVBuf, poolBuf, singlePtBuf, singleOffBuf, 1, numKVHeads, headDim, blockSize, true)
			be.SDPAPagedDecode(qBuf, poolBuf, btBuf, outGPU,
				numBlocks, blockSize, numQHeads, numKVHeads, headDim, scale, tokensInLastBlock)
			be.Sync()
		}
		gpuPathTime := time.Since(start) / time.Duration(iters)

		// Correctness: compare outputs
		gpuBytes := make([]byte, qSize*4)
		be.ToHost(gpuBytes, outGPU)
		gpuOut := bytesToFloat32(gpuBytes)

		maxDiff := float32(0)
		for i := 0; i < qSize; i++ {
			diff := float32(math.Abs(float64(refOut[i] - gpuOut[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		tol := float32(1e-3)
		status := "✓"
		if maxDiff > tol {
			status = "✗ FAIL"
			t.Errorf("kvLen=%d: max diff %v exceeds tolerance %v", kvLen, maxDiff, tol)
		}

		speedup := float64(cpuPathTime) / float64(gpuPathTime)
		perTokenCPU := cpuPathTime * time.Duration(numLayers)
		perTokenGPU := gpuPathTime * time.Duration(numLayers)

		fmt.Printf("--- kvLen=%d ---\n", kvLen)
		fmt.Printf("  Per-layer:  CPU roundtrip=%7v  GPU native=%7v  speedup=%.2f×\n",
			cpuPathTime, gpuPathTime, speedup)
		fmt.Printf("  Per-token (×%d layers): CPU=%7v  GPU=%7v\n",
			numLayers, perTokenCPU, perTokenGPU)
		fmt.Printf("  Correctness: max_diff=%.6f %s\n", maxDiff, status)
		fmt.Println()

		// Cleanup
		be.Free(qBuf)
		be.Free(newKBuf)
		be.Free(newVBuf)
		be.Free(fullKBuf)
		be.Free(fullVBuf)
		be.Free(outCPU)
		be.Free(poolBuf)
		be.Free(ptBuf)
		be.Free(offBuf)
		be.Free(kSrcBuf)
		be.Free(vSrcBuf)
		be.Free(btBuf)
		be.Free(outGPU)
		be.Free(singlePtBuf)
		be.Free(singleOffBuf)
	}
}
