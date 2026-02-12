//go:build metal && darwin && cgo

package metal

import (
	"math/rand"
	"testing"
)

func TestReshapePagedKV(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Config
	numTokens := 32
	numKVHeads := 2
	headDim := 64
	blockSize := 16
	
	// Create Source Data [numTokens, numKVHeads, headDim]
	srcSize := numTokens * numKVHeads * headDim
	srcFloats := make([]float32, srcSize)
	for i := range srcFloats {
		srcFloats[i] = rand.Float32()
	}
	srcBuf := b.Alloc(srcSize * 4)
	defer b.Free(srcBuf)
	b.ToDevice(srcBuf, float32ToBytes(srcFloats))

	// Create Page Table and Offsets
	// We'll map tokens to 2 blocks: 
	// Tokens 0-15 -> Block 5
	// Tokens 16-31 -> Block 2
	pageTable := make([]int32, numTokens)
	blockOffsets := make([]int32, numTokens)
	
	for i := 0; i < numTokens; i++ {
		if i < 16 {
			pageTable[i] = 5
			blockOffsets[i] = int32(i)
		} else {
			pageTable[i] = 2
			blockOffsets[i] = int32(i - 16)
		}
	}
	
	ptBuf := b.Alloc(numTokens * 4)
	defer b.Free(ptBuf)
	// Convert int32 to bytes manually since we don't have a helper
	ptBytes := make([]byte, numTokens*4)
	for i, v := range pageTable {
		// Little endian
		u := uint32(v)
		ptBytes[i*4] = byte(u)
		ptBytes[i*4+1] = byte(u >> 8)
		ptBytes[i*4+2] = byte(u >> 16)
		ptBytes[i*4+3] = byte(u >> 24)
	}
	b.ToDevice(ptBuf, ptBytes)

	offBuf := b.Alloc(numTokens * 4)
	defer b.Free(offBuf)
	offBytes := make([]byte, numTokens*4)
	for i, v := range blockOffsets {
		u := uint32(v)
		offBytes[i*4] = byte(u)
		offBytes[i*4+1] = byte(u >> 8)
		offBytes[i*4+2] = byte(u >> 16)
		offBytes[i*4+3] = byte(u >> 24)
	}
	b.ToDevice(offBuf, offBytes)

	// Create Block Pool (Destination)
	// Max blocks = 10. Size = 10 * blockSize * numKVHeads * headDim * 2 (K+V)
	maxBlocks := 10
	blockElements := blockSize * numKVHeads * headDim * 2
	poolSize := maxBlocks * blockElements
	dstBuf := b.Alloc(poolSize * 4)
	defer b.Free(dstBuf)
	
	// Clear destination
	b.Zero(dstBuf, poolSize)

	// Execute Reshape (Write to K part, isValue=false)
	b.ReshapePagedKV(srcBuf, dstBuf, ptBuf, offBuf, numTokens, numKVHeads, headDim, blockSize, false)
	b.Sync()

	// Verify Data
	dstBytes := make([]byte, poolSize*4)
	b.ToHost(dstBytes, dstBuf)
	dstFloats := bytesToFloat32(dstBytes)

	// Check Block 5 (Tokens 0-15)
	// K part is at start of block
	block5Base := 5 * blockElements
	for i := 0; i < 16; i++ { // Token 0-15
		// Source index: i * heads * dim + h * dim + d
		// Dest index: blockBase + i * heads * dim + h * dim + d
		// They should match exactly since layout within block (for K) matches source
		// BUT dest is interleaved K/V block-wise.
		// K is first half of block.
		for j := 0; j < numKVHeads*headDim; j++ {
			srcVal := srcFloats[i*numKVHeads*headDim + j]
			dstVal := dstFloats[block5Base + i*numKVHeads*headDim + j]
			if srcVal != dstVal {
				t.Errorf("Mismatch at token %d (Block 5), element %d: %f != %f", i, j, srcVal, dstVal)
				t.FailNow()
			}
		}
	}

	// Check Block 2 (Tokens 16-31)
	block2Base := 2 * blockElements
	for i := 0; i < 16; i++ { // Token 16-31 (relative 0-15 in block)
		srcIdx := (16+i)*numKVHeads*headDim 
		for j := 0; j < numKVHeads*headDim; j++ {
			srcVal := srcFloats[srcIdx + j]
			dstVal := dstFloats[block2Base + i*numKVHeads*headDim + j]
			if srcVal != dstVal {
				t.Errorf("Mismatch at token %d (Block 2), element %d: %f != %f", 16+i, j, srcVal, dstVal)
				t.FailNow()
			}
		}
	}
}
