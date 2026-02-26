//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestSDPAPagedDecode verifies that paged SDPA produces the same output
// as contiguous SDPA when given identical K/V data.
//
// Track 3: Paged KV Batching, Phase 2.
func TestSDPAPagedDecode(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	tests := []struct {
		name       string
		numQHeads  int
		numKVHeads int
		headDim    int
		blockSize  int
		kvLen      int
	}{
		{"small_exact", 4, 4, 8, 4, 8},       // 2 full blocks
		{"small_partial", 4, 4, 8, 4, 6},     // 1 full + 1 partial block
		{"gqa_2x", 8, 4, 8, 4, 8},            // GQA: 2 Q heads per KV head
		{"medium", 32, 32, 128, 16, 64},       // 4 full blocks, LLaMA-like dims
		{"medium_partial", 32, 32, 128, 16, 50}, // 3 full + 1 partial
		{"gqa_4x", 32, 8, 128, 16, 64},       // GQA 4:1
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			numBlocks := (tc.kvLen + tc.blockSize - 1) / tc.blockSize
			tokensInLastBlock := tc.kvLen - (numBlocks-1)*tc.blockSize

			// Generate random Q [numQHeads, headDim]
			qSize := tc.numQHeads * tc.headDim
			qData := make([]float32, qSize)
			for i := range qData {
				qData[i] = (rand.Float32() - 0.5) * 0.2
			}

			// Generate random K/V [kvLen, numKVHeads, headDim]
			kvElems := tc.kvLen * tc.numKVHeads * tc.headDim
			kData := make([]float32, kvElems)
			vData := make([]float32, kvElems)
			for i := range kData {
				kData[i] = (rand.Float32() - 0.5) * 0.2
				vData[i] = (rand.Float32() - 0.5) * 0.2
			}

			// === Path A: Contiguous SDPA (reference) ===
			// Reshape K/V to head-major [numKVHeads, kvLen, headDim]
			kHeadMajor := make([]float32, kvElems)
			vHeadMajor := make([]float32, kvElems)
			for pos := 0; pos < tc.kvLen; pos++ {
				for h := 0; h < tc.numKVHeads; h++ {
					for d := 0; d < tc.headDim; d++ {
						src := pos*tc.numKVHeads*tc.headDim + h*tc.headDim + d
						dst := h*tc.kvLen*tc.headDim + pos*tc.headDim + d
						kHeadMajor[dst] = kData[src]
						vHeadMajor[dst] = vData[src]
					}
				}
			}

			qBuf := b.Alloc(qSize * 4)
			kContigBuf := b.Alloc(kvElems * 4)
			vContigBuf := b.Alloc(kvElems * 4)
			outRefBuf := b.Alloc(qSize * 4)
			defer b.Free(qBuf)
			defer b.Free(kContigBuf)
			defer b.Free(vContigBuf)
			defer b.Free(outRefBuf)

			b.ToDevice(qBuf, float32ToBytes(qData))
			b.ToDevice(kContigBuf, float32ToBytes(kHeadMajor))
			b.ToDevice(vContigBuf, float32ToBytes(vHeadMajor))

			scale := float32(1.0 / math.Sqrt(float64(tc.headDim)))
			kvStride := tc.kvLen * tc.headDim

			b.SDPA(qBuf, kContigBuf, vContigBuf, outRefBuf, tc.kvLen,
				tc.numQHeads, tc.numKVHeads, tc.headDim, scale, kvStride)
			b.Sync()

			refOutBytes := make([]byte, qSize*4)
			b.ToHost(refOutBytes, outRefBuf)
			refOut := bytesToFloat32(refOutBytes)

			// === Path B: Paged SDPA ===
			// Scatter K/V into block pool using ReshapePagedKV
			blockElems := tc.blockSize * tc.numKVHeads * tc.headDim * 2
			maxPhysBlocks := numBlocks + 4 // extra space
			poolBuf := b.Alloc(maxPhysBlocks * blockElems * 4)
			defer b.Free(poolBuf)
			b.Zero(poolBuf, maxPhysBlocks*blockElems)

			// Build page table and offsets for all kvLen tokens
			pageTable := make([]int32, tc.kvLen)
			blockOffsets := make([]int32, tc.kvLen)
			// Logical block i → physical block i (simple 1:1 mapping)
			for i := 0; i < tc.kvLen; i++ {
				pageTable[i] = int32(i / tc.blockSize)
				blockOffsets[i] = int32(i % tc.blockSize)
			}

			ptBuf := b.Alloc(tc.kvLen * 4)
			offBuf := b.Alloc(tc.kvLen * 4)
			defer b.Free(ptBuf)
			defer b.Free(offBuf)
			b.ToDevice(ptBuf, int32ToBytes(pageTable))
			b.ToDevice(offBuf, int32ToBytes(blockOffsets))

			// Scatter K and V into pool
			kSrcBuf := b.Alloc(kvElems * 4)
			vSrcBuf := b.Alloc(kvElems * 4)
			defer b.Free(kSrcBuf)
			defer b.Free(vSrcBuf)
			b.ToDevice(kSrcBuf, float32ToBytes(kData))
			b.ToDevice(vSrcBuf, float32ToBytes(vData))

			b.ReshapePagedKV(kSrcBuf, poolBuf, ptBuf, offBuf, tc.kvLen,
				tc.numKVHeads, tc.headDim, tc.blockSize, false)
			b.ReshapePagedKV(vSrcBuf, poolBuf, ptBuf, offBuf, tc.kvLen,
				tc.numKVHeads, tc.headDim, tc.blockSize, true)
			b.Sync()

			// Build block table for SDPA: [numBlocks] mapping logical → physical
			blockTableData := make([]int32, numBlocks)
			for i := 0; i < numBlocks; i++ {
				blockTableData[i] = int32(i) // 1:1 mapping
			}
			btBuf := b.Alloc(numBlocks * 4)
			defer b.Free(btBuf)
			b.ToDevice(btBuf, int32ToBytes(blockTableData))

			outPagedBuf := b.Alloc(qSize * 4)
			defer b.Free(outPagedBuf)

			// Call paged SDPA
			b.SDPAPagedDecode(qBuf, poolBuf, btBuf, outPagedBuf,
				numBlocks, tc.blockSize, tc.numQHeads, tc.numKVHeads,
				tc.headDim, scale, tokensInLastBlock)
			b.Sync()

			pagedOutBytes := make([]byte, qSize*4)
			b.ToHost(pagedOutBytes, outPagedBuf)
			pagedOut := bytesToFloat32(pagedOutBytes)

			// Compare outputs
			maxDiff := float32(0)
			for i := 0; i < qSize; i++ {
				diff := float32(math.Abs(float64(refOut[i] - pagedOut[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			// Allow small numerical differences from different access patterns
			tol := float32(1e-3)
			if maxDiff > tol {
				t.Errorf("max difference %v exceeds tolerance %v", maxDiff, tol)
				// Print first few values for debugging
				for i := 0; i < min(8, qSize); i++ {
					fmt.Printf("  [%d] ref=%f paged=%f diff=%f\n",
						i, refOut[i], pagedOut[i], refOut[i]-pagedOut[i])
				}
			} else {
				fmt.Printf("  %s: max_diff=%.6f (tol=%.4f) ✓\n", tc.name, maxDiff, tol)
			}
		})
	}
}

// TestSDPAPagedDecodeScattered verifies paged SDPA with non-contiguous
// physical block assignments (blocks scattered in pool).
func TestSDPAPagedDecodeScattered(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	numQHeads := 4
	numKVHeads := 4
	headDim := 8
	blockSize := 4
	kvLen := 12 // 3 blocks

	numBlocks := 3
	tokensInLastBlock := 4

	qSize := numQHeads * headDim
	kvElems := kvLen * numKVHeads * headDim

	// Random data
	qData := make([]float32, qSize)
	kData := make([]float32, kvElems)
	vData := make([]float32, kvElems)
	for i := range qData {
		qData[i] = (rand.Float32() - 0.5) * 0.2
	}
	for i := range kData {
		kData[i] = (rand.Float32() - 0.5) * 0.2
		vData[i] = (rand.Float32() - 0.5) * 0.2
	}

	// Reference: contiguous SDPA
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

	qBuf := b.Alloc(qSize * 4)
	kBuf := b.Alloc(kvElems * 4)
	vBuf := b.Alloc(kvElems * 4)
	outRef := b.Alloc(qSize * 4)
	defer b.Free(qBuf)
	defer b.Free(kBuf)
	defer b.Free(vBuf)
	defer b.Free(outRef)

	b.ToDevice(qBuf, float32ToBytes(qData))
	b.ToDevice(kBuf, float32ToBytes(kHM))
	b.ToDevice(vBuf, float32ToBytes(vHM))

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	b.SDPA(qBuf, kBuf, vBuf, outRef, kvLen, numQHeads, numKVHeads, headDim, scale, kvLen*headDim)
	b.Sync()

	refBytes := make([]byte, qSize*4)
	b.ToHost(refBytes, outRef)
	refOut := bytesToFloat32(refBytes)

	// Paged SDPA with scattered physical blocks
	// Logical block 0 → physical 5, block 1 → physical 2, block 2 → physical 7
	physMap := []int32{5, 2, 7}
	maxPhysBlock := 10
	blockElems := blockSize * numKVHeads * headDim * 2
	poolBuf := b.Alloc(maxPhysBlock * blockElems * 4)
	defer b.Free(poolBuf)
	b.Zero(poolBuf, maxPhysBlock*blockElems)

	// Build per-token page table and offsets mapping to scattered physical blocks
	pageTable := make([]int32, kvLen)
	blockOffsets := make([]int32, kvLen)
	for i := 0; i < kvLen; i++ {
		logicalBlock := i / blockSize
		pageTable[i] = physMap[logicalBlock]
		blockOffsets[i] = int32(i % blockSize)
	}

	ptBuf := b.Alloc(kvLen * 4)
	offBuf := b.Alloc(kvLen * 4)
	defer b.Free(ptBuf)
	defer b.Free(offBuf)
	b.ToDevice(ptBuf, int32ToBytes(pageTable))
	b.ToDevice(offBuf, int32ToBytes(blockOffsets))

	kSrc := b.Alloc(kvElems * 4)
	vSrc := b.Alloc(kvElems * 4)
	defer b.Free(kSrc)
	defer b.Free(vSrc)
	b.ToDevice(kSrc, float32ToBytes(kData))
	b.ToDevice(vSrc, float32ToBytes(vData))

	b.ReshapePagedKV(kSrc, poolBuf, ptBuf, offBuf, kvLen, numKVHeads, headDim, blockSize, false)
	b.ReshapePagedKV(vSrc, poolBuf, ptBuf, offBuf, kvLen, numKVHeads, headDim, blockSize, true)
	b.Sync()

	// Block table for SDPA uses the scattered physical mapping
	btBuf := b.Alloc(numBlocks * 4)
	defer b.Free(btBuf)
	b.ToDevice(btBuf, int32ToBytes(physMap))

	outPaged := b.Alloc(qSize * 4)
	defer b.Free(outPaged)

	b.SDPAPagedDecode(qBuf, poolBuf, btBuf, outPaged,
		numBlocks, blockSize, numQHeads, numKVHeads, headDim, scale, tokensInLastBlock)
	b.Sync()

	pagedBytes := make([]byte, qSize*4)
	b.ToHost(pagedBytes, outPaged)
	pagedOut := bytesToFloat32(pagedBytes)

	maxDiff := float32(0)
	for i := 0; i < qSize; i++ {
		diff := float32(math.Abs(float64(refOut[i] - pagedOut[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tol := float32(1e-3)
	if maxDiff > tol {
		t.Errorf("scattered blocks: max difference %v exceeds tolerance %v", maxDiff, tol)
		for i := 0; i < min(8, qSize); i++ {
			fmt.Printf("  [%d] ref=%f paged=%f diff=%f\n",
				i, refOut[i], pagedOut[i], refOut[i]-pagedOut[i])
		}
	} else {
		fmt.Printf("  scattered: max_diff=%.6f (tol=%.4f) ✓\n", maxDiff, tol)
	}
}

// int32ToBytes converts a slice of int32 to little-endian bytes.
func int32ToBytes(in []int32) []byte {
	out := make([]byte, len(in)*4)
	for i, v := range in {
		u := uint32(v)
		out[i*4] = byte(u)
		out[i*4+1] = byte(u >> 8)
		out[i*4+2] = byte(u >> 16)
		out[i*4+3] = byte(u >> 24)
	}
	return out
}
