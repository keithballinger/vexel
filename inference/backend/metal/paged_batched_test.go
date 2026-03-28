//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"vexel/inference/tensor"
)

// TestSDPAPagedDecodeBatched verifies that SDPAPagedDecodeBatched produces
// identical output to calling SDPAPagedDecode individually per sequence.
func TestSDPAPagedDecodeBatched(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	numQHeads := 4
	numKVHeads := 4
	headDim := 8
	blockSize := 4
	batchSize := 2

	// Two sequences with different context lengths.
	seqLens := []int{8, 6} // seq0: 2 full blocks, seq1: 1 full + 1 partial

	qSize := numQHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Generate random Q for both sequences [batchSize * numQHeads * headDim].
	totalQ := batchSize * qSize
	qData := make([]float32, totalQ)
	for i := range qData {
		qData[i] = (rand.Float32() - 0.5) * 0.2
	}

	// Allocate a shared KV pool large enough for both sequences.
	// Seq0 uses physical blocks 0,1; Seq1 uses physical blocks 2,3.
	maxPhysBlocks := 4
	blockElems := blockSize * numKVHeads * headDim * 2 // K + V interleaved
	poolBuf := b.Alloc(maxPhysBlocks * blockElems * 4)
	defer b.Free(poolBuf)
	b.Zero(poolBuf, maxPhysBlocks*blockElems)

	// For each sequence, generate K/V data and scatter into the pool.
	type seqSetup struct {
		numBlocks        int
		tokensInLastBlock int
		blockTableData   []int32
	}
	seqs := make([]seqSetup, batchSize)

	physBlockIdx := 0
	for s := 0; s < batchSize; s++ {
		kvLen := seqLens[s]
		numBlocks := (kvLen + blockSize - 1) / blockSize
		tokensInLastBlock := kvLen - (numBlocks-1)*blockSize

		// Physical block assignments for this sequence.
		bt := make([]int32, numBlocks)
		for i := 0; i < numBlocks; i++ {
			bt[i] = int32(physBlockIdx)
			physBlockIdx++
		}

		seqs[s] = seqSetup{
			numBlocks:        numBlocks,
			tokensInLastBlock: tokensInLastBlock,
			blockTableData:   bt,
		}

		// Generate random K/V data for this sequence.
		kvElems := kvLen * numKVHeads * headDim
		kData := make([]float32, kvElems)
		vData := make([]float32, kvElems)
		for i := range kData {
			kData[i] = (rand.Float32() - 0.5) * 0.2
			vData[i] = (rand.Float32() - 0.5) * 0.2
		}

		// Build per-token page table and offsets for ReshapePagedKV.
		pageTable := make([]int32, kvLen)
		blockOffsets := make([]int32, kvLen)
		for i := 0; i < kvLen; i++ {
			logicalBlock := i / blockSize
			pageTable[i] = bt[logicalBlock]
			blockOffsets[i] = int32(i % blockSize)
		}

		ptBuf := b.Alloc(kvLen * 4)
		offBuf := b.Alloc(kvLen * 4)
		kSrcBuf := b.Alloc(kvElems * 4)
		vSrcBuf := b.Alloc(kvElems * 4)

		b.ToDevice(ptBuf, int32ToBytes(pageTable))
		b.ToDevice(offBuf, int32ToBytes(blockOffsets))
		b.ToDevice(kSrcBuf, float32ToBytes(kData))
		b.ToDevice(vSrcBuf, float32ToBytes(vData))

		b.ReshapePagedKV(kSrcBuf, poolBuf, ptBuf, offBuf, kvLen,
			numKVHeads, headDim, blockSize, false)
		b.ReshapePagedKV(vSrcBuf, poolBuf, ptBuf, offBuf, kvLen,
			numKVHeads, headDim, blockSize, true)

		b.Free(ptBuf)
		b.Free(offBuf)
		b.Free(kSrcBuf)
		b.Free(vSrcBuf)
	}
	b.Sync()

	// Upload Q data.
	qBuf := b.Alloc(totalQ * 4)
	defer b.Free(qBuf)
	b.ToDevice(qBuf, float32ToBytes(qData))

	// Build block table device buffers for each sequence.
	maxBlocks := 0
	blockTables := make([]tensor.DevicePtr, batchSize)
	for s := 0; s < batchSize; s++ {
		if seqs[s].numBlocks > maxBlocks {
			maxBlocks = seqs[s].numBlocks
		}
		btBuf := b.Alloc(seqs[s].numBlocks * 4)
		defer b.Free(btBuf)
		b.ToDevice(btBuf, int32ToBytes(seqs[s].blockTableData))
		blockTables[s] = btBuf
	}

	// === Path A: SDPAPagedDecodeBatched ===
	outBatchedBuf := b.Alloc(totalQ * 4)
	defer b.Free(outBatchedBuf)

	b.SDPAPagedDecodeBatched(qBuf, poolBuf, blockTables, outBatchedBuf,
		batchSize, maxBlocks, blockSize, numQHeads, numKVHeads, headDim, scale, seqLens)
	b.Sync()

	batchedOutBytes := make([]byte, totalQ*4)
	b.ToHost(batchedOutBytes, outBatchedBuf)
	batchedOut := bytesToFloat32(batchedOutBytes)

	// === Path B: Individual SDPAPagedDecode per sequence ===
	outIndivBuf := b.Alloc(totalQ * 4)
	defer b.Free(outIndivBuf)

	stride := uintptr(numQHeads * headDim * 4)
	for s := 0; s < batchSize; s++ {
		seqQ := tensor.DevicePtrOffset(qBuf, uintptr(s)*stride)
		seqOut := tensor.DevicePtrOffset(outIndivBuf, uintptr(s)*stride)
		b.SDPAPagedDecode(seqQ, poolBuf, blockTables[s], seqOut,
			seqs[s].numBlocks, blockSize, numQHeads, numKVHeads, headDim, scale, seqs[s].tokensInLastBlock)
	}
	b.Sync()

	indivOutBytes := make([]byte, totalQ*4)
	b.ToHost(indivOutBytes, outIndivBuf)
	indivOut := bytesToFloat32(indivOutBytes)

	// Compare outputs.
	maxDiff := float32(0)
	for i := 0; i < totalQ; i++ {
		diff := float32(math.Abs(float64(batchedOut[i] - indivOut[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tol := float32(1e-5) // Should be exact or near-exact since same code path
	if maxDiff > tol {
		t.Errorf("max difference %v exceeds tolerance %v", maxDiff, tol)
		for s := 0; s < batchSize; s++ {
			base := s * qSize
			fmt.Printf("  seq %d (len=%d):\n", s, seqLens[s])
			for i := 0; i < min(4, qSize); i++ {
				fmt.Printf("    [%d] batched=%f indiv=%f diff=%f\n",
					i, batchedOut[base+i], indivOut[base+i], batchedOut[base+i]-indivOut[base+i])
			}
		}
	} else {
		fmt.Printf("  SDPAPagedDecodeBatched: max_diff=%.8f (tol=%.6f) OK\n", maxDiff, tol)
	}
}
