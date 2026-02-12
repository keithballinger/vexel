//go:build metal && darwin && cgo

package metal

import (
	"testing"
)

func BenchmarkContiguousWrite(b_bench *testing.B) {
	b, err := NewBackend(0)
	if err != nil {
		b_bench.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	numTokens := 128
	numKVHeads := 8
	headDim := 128
	totalElements := numTokens * numKVHeads * headDim
	
	srcBuf := b.Alloc(totalElements * 4)
	dstBuf := b.Alloc(totalElements * 4) // Contiguous
	defer b.Free(srcBuf)
	defer b.Free(dstBuf)

	// Fill with zeros to avoid NaNs
	b.Zero(srcBuf, totalElements)

	b_bench.ResetTimer()
	for i := 0; i < b_bench.N; i++ {
		// Use simple CopyBuffer for contiguous write simulation
		b.CopyBuffer(srcBuf, 0, dstBuf, 0, totalElements * 4)
		b.Sync()
	}
}

func BenchmarkPagedWrite(b_bench *testing.B) {
	b, err := NewBackend(0)
	if err != nil {
		b_bench.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	numTokens := 128
	numKVHeads := 8
	headDim := 128
	blockSize := 16
	
	totalElements := numTokens * numKVHeads * headDim
	
	srcBuf := b.Alloc(totalElements * 4)
	// Paged pool: slightly larger to simulate scattered blocks
	dstBuf := b.Alloc(totalElements * 4 * 2) 
	
	defer b.Free(srcBuf)
	defer b.Free(dstBuf)
	b.Zero(srcBuf, totalElements)

	// Setup Page Table
	pageTable := make([]int32, numTokens)
	blockOffsets := make([]int32, numTokens)
	
	// Map to scattered blocks
	for i := 0; i < numTokens; i++ {
		blockIdx := i / blockSize
		// Scatter: 0->0, 1->2, 2->4...
		pageTable[i] = int32(blockIdx * 2) 
		blockOffsets[i] = int32(i % blockSize)
	}
	
	ptBuf := b.Alloc(numTokens * 4)
	offBuf := b.Alloc(numTokens * 4)
	defer b.Free(ptBuf)
	defer b.Free(offBuf)
	
	// Helper to convert int32 slice to bytes
	i32ToBytes := func(in []int32) []byte {
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
	
	b.ToDevice(ptBuf, i32ToBytes(pageTable))
	b.ToDevice(offBuf, i32ToBytes(blockOffsets))

	b_bench.ResetTimer()
	for i := 0; i < b_bench.N; i++ {
		b.ReshapePagedKV(srcBuf, dstBuf, ptBuf, offBuf, numTokens, numKVHeads, headDim, blockSize, false)
		b.Sync()
	}
}
