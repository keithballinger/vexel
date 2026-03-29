//go:build metal && darwin && cgo

package runtime

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"vexel/inference/backend"
	metalBackend "vexel/inference/backend/metal"
)

// TestGPUBlockPoolStoreAndAttend verifies that GPUBlockPool correctly
// stores K/V data and produces identical attention output as contiguous SDPA.
//
// Track 3: Paged KV Batching, Phase 2 Task 2.
func TestGPUBlockPoolStoreAndAttend(t *testing.T) {
	b, err := metalBackend.NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	var be backend.Backend = b
	pagedOps, ok := be.(backend.PagedKVOps)
	if !ok {
		t.Skipf("Backend does not support PagedKVOps")
	}

	tests := []struct {
		name       string
		numQHeads  int
		numKVHeads int
		headDim    int
		kvLen      int
	}{
		{"small", 4, 4, 8, 8},
		{"gqa", 8, 4, 8, 12},
		{"medium", 32, 32, 128, 48},
		{"gqa_4x", 32, 8, 128, 32},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			numLayers := 1
			blockSize := 16
			maxBlocks := (tc.kvLen + blockSize - 1) / blockSize + 2

			pool := NewGPUBlockPool(b, pagedOps, numLayers, tc.numKVHeads, tc.headDim, blockSize, maxBlocks, false)
			defer pool.Close()

			seqID := int64(42)
			pool.CreateSequence(seqID)
			defer pool.DeleteSequence(seqID)

			qSize := tc.numQHeads * tc.headDim
			kvElems := tc.kvLen * tc.numKVHeads * tc.headDim

			// Random Q, K, V
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

			// === Store K/V into pool token by token (simulating decode) ===
			for pos := 0; pos < tc.kvLen; pos++ {
				tokenKVElems := tc.numKVHeads * tc.headDim
				kSlice := kData[pos*tokenKVElems : (pos+1)*tokenKVElems]
				vSlice := vData[pos*tokenKVElems : (pos+1)*tokenKVElems]

				kBuf := b.Alloc(tokenKVElems * 4)
				vBuf := b.Alloc(tokenKVElems * 4)
				b.ToDevice(kBuf, float32ToBytes(kSlice))
				b.ToDevice(vBuf, float32ToBytes(vSlice))

				err := pool.StoreKV(0, seqID, pos, kBuf, vBuf, 1)
				if err != nil {
					t.Fatalf("StoreKV pos=%d: %v", pos, err)
				}
				b.Free(kBuf)
				b.Free(vBuf)
			}
			b.Sync()

			// === Paged attention ===
			qBuf := b.Alloc(qSize * 4)
			outPagedBuf := b.Alloc(qSize * 4)
			defer b.Free(qBuf)
			defer b.Free(outPagedBuf)
			b.ToDevice(qBuf, float32ToBytes(qData))

			scale := float32(1.0 / math.Sqrt(float64(tc.headDim)))
			err := pool.Attention(0, seqID, qBuf, outPagedBuf, tc.numQHeads, tc.headDim, scale)
			if err != nil {
				t.Fatalf("Attention: %v", err)
			}
			b.Sync()

			pagedOutBytes := make([]byte, qSize*4)
			b.ToHost(pagedOutBytes, outPagedBuf)
			pagedOut := bytesToFloat32(pagedOutBytes)

			// === Reference: contiguous SDPA ===
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

			kContigBuf := b.Alloc(kvElems * 4)
			vContigBuf := b.Alloc(kvElems * 4)
			outRefBuf := b.Alloc(qSize * 4)
			defer b.Free(kContigBuf)
			defer b.Free(vContigBuf)
			defer b.Free(outRefBuf)

			b.ToDevice(kContigBuf, float32ToBytes(kHeadMajor))
			b.ToDevice(vContigBuf, float32ToBytes(vHeadMajor))

			kvStride := tc.kvLen * tc.headDim
			b.SDPA(qBuf, kContigBuf, vContigBuf, outRefBuf, tc.kvLen,
				tc.numQHeads, tc.numKVHeads, tc.headDim, scale, kvStride)
			b.Sync()

			refOutBytes := make([]byte, qSize*4)
			b.ToHost(refOutBytes, outRefBuf)
			refOut := bytesToFloat32(refOutBytes)

			// === Compare ===
			maxDiff := float32(0)
			for i := 0; i < qSize; i++ {
				diff := float32(math.Abs(float64(refOut[i] - pagedOut[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			tol := float32(1e-3)
			if maxDiff > tol {
				t.Errorf("max difference %v exceeds tolerance %v", maxDiff, tol)
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

// TestGPUBlockPoolPrefillThenDecode verifies the full prefill→decode flow
// through GPUBlockPool matches contiguous SDPA reference.
func TestGPUBlockPoolPrefillThenDecode(t *testing.T) {
	b, err := metalBackend.NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	var be backend.Backend = b
	pagedOps, ok := be.(backend.PagedKVOps)
	if !ok {
		t.Skipf("Backend does not support PagedKVOps")
	}

	numQHeads := 4
	numKVHeads := 4
	headDim := 8
	prefillLen := 6 // prefill tokens
	numLayers := 1
	blockSize := 4
	maxBlocks := 8

	pool := NewGPUBlockPool(b, pagedOps, numLayers, numKVHeads, headDim, blockSize, maxBlocks, false)
	defer pool.Close()

	seqID := int64(99)
	pool.CreateSequence(seqID)
	defer pool.DeleteSequence(seqID)

	qSize := numQHeads * headDim
	tokenKVElems := numKVHeads * headDim
	prefillKVElems := prefillLen * tokenKVElems

	// Random data
	allKData := make([]float32, (prefillLen+1)*tokenKVElems) // prefill + 1 decode token
	allVData := make([]float32, (prefillLen+1)*tokenKVElems)
	for i := range allKData {
		allKData[i] = (rand.Float32() - 0.5) * 0.2
		allVData[i] = (rand.Float32() - 0.5) * 0.2
	}

	// Prefill: store all prefill tokens at once
	prefillKBuf := b.Alloc(prefillKVElems * 4)
	prefillVBuf := b.Alloc(prefillKVElems * 4)
	defer b.Free(prefillKBuf)
	defer b.Free(prefillVBuf)
	b.ToDevice(prefillKBuf, float32ToBytes(allKData[:prefillKVElems]))
	b.ToDevice(prefillVBuf, float32ToBytes(allVData[:prefillKVElems]))

	err = pool.StoreKV(0, seqID, 0, prefillKBuf, prefillVBuf, prefillLen)
	if err != nil {
		t.Fatalf("StoreKV prefill: %v", err)
	}

	// Decode: store one more token
	decodeKBuf := b.Alloc(tokenKVElems * 4)
	decodeVBuf := b.Alloc(tokenKVElems * 4)
	defer b.Free(decodeKBuf)
	defer b.Free(decodeVBuf)
	b.ToDevice(decodeKBuf, float32ToBytes(allKData[prefillKVElems:]))
	b.ToDevice(decodeVBuf, float32ToBytes(allVData[prefillKVElems:]))

	err = pool.StoreKV(0, seqID, prefillLen, decodeKBuf, decodeVBuf, 1)
	if err != nil {
		t.Fatalf("StoreKV decode: %v", err)
	}
	b.Sync()

	// Query for decode position
	qData := make([]float32, qSize)
	for i := range qData {
		qData[i] = (rand.Float32() - 0.5) * 0.2
	}
	qBuf := b.Alloc(qSize * 4)
	outPagedBuf := b.Alloc(qSize * 4)
	defer b.Free(qBuf)
	defer b.Free(outPagedBuf)
	b.ToDevice(qBuf, float32ToBytes(qData))

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	err = pool.Attention(0, seqID, qBuf, outPagedBuf, numQHeads, headDim, scale)
	if err != nil {
		t.Fatalf("Attention: %v", err)
	}
	b.Sync()

	pagedOutBytes := make([]byte, qSize*4)
	b.ToHost(pagedOutBytes, outPagedBuf)
	pagedOut := bytesToFloat32(pagedOutBytes)

	// Reference: contiguous SDPA over all (prefill+1) tokens
	totalKVLen := prefillLen + 1
	totalKVElems := totalKVLen * tokenKVElems

	kHM := make([]float32, totalKVElems)
	vHM := make([]float32, totalKVElems)
	for pos := 0; pos < totalKVLen; pos++ {
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < headDim; d++ {
				src := pos*numKVHeads*headDim + h*headDim + d
				dst := h*totalKVLen*headDim + pos*headDim + d
				kHM[dst] = allKData[src]
				vHM[dst] = allVData[src]
			}
		}
	}

	kContigBuf := b.Alloc(totalKVElems * 4)
	vContigBuf := b.Alloc(totalKVElems * 4)
	outRefBuf := b.Alloc(qSize * 4)
	defer b.Free(kContigBuf)
	defer b.Free(vContigBuf)
	defer b.Free(outRefBuf)
	b.ToDevice(kContigBuf, float32ToBytes(kHM))
	b.ToDevice(vContigBuf, float32ToBytes(vHM))

	kvStride := totalKVLen * headDim
	b.SDPA(qBuf, kContigBuf, vContigBuf, outRefBuf, totalKVLen,
		numQHeads, numKVHeads, headDim, scale, kvStride)
	b.Sync()

	refOutBytes := make([]byte, qSize*4)
	b.ToHost(refOutBytes, outRefBuf)
	refOut := bytesToFloat32(refOutBytes)

	maxDiff := float32(0)
	for i := 0; i < qSize; i++ {
		diff := float32(math.Abs(float64(refOut[i] - pagedOut[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tol := float32(1e-3)
	if maxDiff > tol {
		t.Errorf("prefill→decode: max difference %v exceeds tolerance %v", maxDiff, tol)
		for i := 0; i < min(8, qSize); i++ {
			fmt.Printf("  [%d] ref=%f paged=%f diff=%f\n",
				i, refOut[i], pagedOut[i], refOut[i]-pagedOut[i])
		}
	} else {
		fmt.Printf("  prefill→decode: max_diff=%.6f (tol=%.4f) ✓\n", maxDiff, tol)
	}
}

// TestGPUBlockPoolPrefixSharing verifies that shared prefix blocks produce
// correct attention output for both the source and destination sequences.
func TestGPUBlockPoolPrefixSharing(t *testing.T) {
	b, err := metalBackend.NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	var be backend.Backend = b
	pagedOps, ok := be.(backend.PagedKVOps)
	if !ok {
		t.Skipf("Backend does not support PagedKVOps")
	}

	numQHeads := 4
	numKVHeads := 4
	headDim := 8
	prefixLen := 8 // shared prefix
	numLayers := 1
	blockSize := 4
	maxBlocks := 16

	pool := NewGPUBlockPool(b, pagedOps, numLayers, numKVHeads, headDim, blockSize, maxBlocks, false)
	defer pool.Close()

	tokenKVElems := numKVHeads * headDim
	qSize := numQHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Generate shared prefix K/V
	prefixKData := make([]float32, prefixLen*tokenKVElems)
	prefixVData := make([]float32, prefixLen*tokenKVElems)
	for i := range prefixKData {
		prefixKData[i] = (rand.Float32() - 0.5) * 0.2
		prefixVData[i] = (rand.Float32() - 0.5) * 0.2
	}

	// Sequence A: prefill prefix, then decode one token
	seqA := int64(1)
	pool.CreateSequence(seqA)
	defer pool.DeleteSequence(seqA)

	prefKBuf := b.Alloc(prefixLen * tokenKVElems * 4)
	prefVBuf := b.Alloc(prefixLen * tokenKVElems * 4)
	defer b.Free(prefKBuf)
	defer b.Free(prefVBuf)
	b.ToDevice(prefKBuf, float32ToBytes(prefixKData))
	b.ToDevice(prefVBuf, float32ToBytes(prefixVData))

	err = pool.StoreKV(0, seqA, 0, prefKBuf, prefVBuf, prefixLen)
	if err != nil {
		t.Fatalf("StoreKV seqA prefix: %v", err)
	}

	// Add one unique decode token to seqA
	decKA := make([]float32, tokenKVElems)
	decVA := make([]float32, tokenKVElems)
	for i := range decKA {
		decKA[i] = (rand.Float32() - 0.5) * 0.2
		decVA[i] = (rand.Float32() - 0.5) * 0.2
	}
	decKABuf := b.Alloc(tokenKVElems * 4)
	decVABuf := b.Alloc(tokenKVElems * 4)
	defer b.Free(decKABuf)
	defer b.Free(decVABuf)
	b.ToDevice(decKABuf, float32ToBytes(decKA))
	b.ToDevice(decVABuf, float32ToBytes(decVA))
	err = pool.StoreKV(0, seqA, prefixLen, decKABuf, decVABuf, 1)
	if err != nil {
		t.Fatalf("StoreKV seqA decode: %v", err)
	}
	b.Sync()

	// Sequence B: share prefix blocks from A, then decode a different token
	seqB := int64(2)
	pool.CreateSequence(seqB)
	defer pool.DeleteSequence(seqB)

	err = pool.ShareBlocks(seqA, seqB, prefixLen)
	if err != nil {
		t.Fatalf("ShareBlocks: %v", err)
	}

	// Add different decode token to seqB
	decKB := make([]float32, tokenKVElems)
	decVB := make([]float32, tokenKVElems)
	for i := range decKB {
		decKB[i] = (rand.Float32() - 0.5) * 0.2
		decVB[i] = (rand.Float32() - 0.5) * 0.2
	}
	decKBBuf := b.Alloc(tokenKVElems * 4)
	decVBBuf := b.Alloc(tokenKVElems * 4)
	defer b.Free(decKBBuf)
	defer b.Free(decVBBuf)
	b.ToDevice(decKBBuf, float32ToBytes(decKB))
	b.ToDevice(decVBBuf, float32ToBytes(decVB))
	err = pool.StoreKV(0, seqB, prefixLen, decKBBuf, decVBBuf, 1)
	if err != nil {
		t.Fatalf("StoreKV seqB decode: %v", err)
	}
	b.Sync()

	// Verify block stats show shared blocks
	total, free, shared := pool.BlockStats()
	t.Logf("Block stats: total=%d free=%d shared=%d", total, free, shared)
	if shared == 0 {
		t.Error("Expected shared blocks > 0 after ShareBlocks")
	}

	// Query: same Q for both sequences
	qData := make([]float32, qSize)
	for i := range qData {
		qData[i] = (rand.Float32() - 0.5) * 0.2
	}
	qBuf := b.Alloc(qSize * 4)
	defer b.Free(qBuf)
	b.ToDevice(qBuf, float32ToBytes(qData))

	// Get attention output for seqA
	outA := b.Alloc(qSize * 4)
	defer b.Free(outA)
	err = pool.Attention(0, seqA, qBuf, outA, numQHeads, headDim, scale)
	if err != nil {
		t.Fatalf("Attention seqA: %v", err)
	}
	b.Sync()

	// Get attention output for seqB
	outB := b.Alloc(qSize * 4)
	defer b.Free(outB)
	err = pool.Attention(0, seqB, qBuf, outB, numQHeads, headDim, scale)
	if err != nil {
		t.Fatalf("Attention seqB: %v", err)
	}
	b.Sync()

	outABytes := make([]byte, qSize*4)
	outBBytes := make([]byte, qSize*4)
	b.ToHost(outABytes, outA)
	b.ToHost(outBBytes, outB)
	outAData := bytesToFloat32(outABytes)
	outBData := bytesToFloat32(outBBytes)

	// seqA and seqB have the same prefix but different decode tokens,
	// so their attention outputs should be different
	allSame := true
	for i := 0; i < qSize; i++ {
		if outAData[i] != outBData[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("seqA and seqB have identical outputs despite different decode tokens")
	}

	// Now verify seqA's attention matches a reference computed from scratch
	// (prefix K/V + decode K/V for seqA, all contiguous)
	totalKVLen := prefixLen + 1
	totalKVElems := totalKVLen * tokenKVElems
	allKA := append(append([]float32{}, prefixKData...), decKA...)
	allVA := append(append([]float32{}, prefixVData...), decVA...)

	kHM := make([]float32, totalKVElems)
	vHM := make([]float32, totalKVElems)
	for pos := 0; pos < totalKVLen; pos++ {
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < headDim; d++ {
				src := pos*numKVHeads*headDim + h*headDim + d
				dst := h*totalKVLen*headDim + pos*headDim + d
				kHM[dst] = allKA[src]
				vHM[dst] = allVA[src]
			}
		}
	}

	kRefBuf := b.Alloc(totalKVElems * 4)
	vRefBuf := b.Alloc(totalKVElems * 4)
	outRefBuf := b.Alloc(qSize * 4)
	defer b.Free(kRefBuf)
	defer b.Free(vRefBuf)
	defer b.Free(outRefBuf)
	b.ToDevice(kRefBuf, float32ToBytes(kHM))
	b.ToDevice(vRefBuf, float32ToBytes(vHM))

	kvStride := totalKVLen * headDim
	b.SDPA(qBuf, kRefBuf, vRefBuf, outRefBuf, totalKVLen,
		numQHeads, numKVHeads, headDim, scale, kvStride)
	b.Sync()

	refOutBytes := make([]byte, qSize*4)
	b.ToHost(refOutBytes, outRefBuf)
	refOut := bytesToFloat32(refOutBytes)

	maxDiff := float32(0)
	for i := 0; i < qSize; i++ {
		diff := float32(math.Abs(float64(refOut[i] - outAData[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tol := float32(1e-3)
	if maxDiff > tol {
		t.Errorf("prefix sharing: seqA max diff %v exceeds tolerance %v", maxDiff, tol)
		for i := 0; i < min(8, qSize); i++ {
			fmt.Printf("  [%d] ref=%f seqA=%f diff=%f\n",
				i, refOut[i], outAData[i], refOut[i]-outAData[i])
		}
	} else {
		fmt.Printf("  prefix_sharing: seqA max_diff=%.6f (tol=%.4f) ✓\n", maxDiff, tol)
	}
}
