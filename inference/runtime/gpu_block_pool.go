package runtime

import (
	"fmt"
	"sync"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// GPUBlockPool manages GPU-resident paged KV cache blocks.
// It provides block allocation and K/V scatter/attention operations
// entirely on the GPU, eliminating the CPU roundtrip in ExecuteWithPagedKV.
//
// Block pool layout per physical block:
//
//	K: [blockSize, numKVHeads, headDim] float32
//	V: [blockSize, numKVHeads, headDim] float32
//	Total: 2 * blockSize * numKVHeads * headDim * sizeof(float32)
type GPUBlockPool struct {
	be       backend.Backend
	pagedOps backend.PagedKVOps

	// allocPerm allocates GPU memory that survives ResetPool cycles.
	// All GPUBlockPool buffers must use this instead of be.Alloc.
	allocPerm func(bytes int) tensor.DevicePtr

	numLayers  int
	numKVHeads int
	headDim    int
	blockSize  int
	maxBlocks  int // per layer

	// Per-layer GPU pool buffer (single large allocation)
	pools []tensor.DevicePtr

	// Per-layer free list
	freeLists [][]int32

	// Per-layer reference counts: blockID → ref count
	refCounts []map[int32]int

	// Per-sequence state
	mu   sync.Mutex
	seqs map[int64]*gpuSeqState

	// Pre-allocated buffers for single-token scatter (decode path)
	ptBuf1  tensor.DevicePtr
	offBuf1 tensor.DevicePtr
}

type gpuSeqState struct {
	// Per-layer: [logicalBlock] → physicalBlockID
	blockTables [][]int32
	seqLen      int

	// Cached GPU block table buffers per layer
	btGPU   []tensor.DevicePtr
	btDirty []bool
}

// NewGPUBlockPool creates a GPU-resident block pool.
// maxBlocks is the maximum number of physical blocks per layer.
func NewGPUBlockPool(be backend.Backend, pagedOps backend.PagedKVOps,
	numLayers, numKVHeads, headDim, blockSize, maxBlocks int) *GPUBlockPool {

	// All GPUBlockPool allocations must be permanent (survive ResetPool cycles).
	// The pool allocator's ResetPool() recycles all Alloc'd buffers, but GPUBlockPool
	// retains buffers across forward passes. Using recyclable Alloc causes SIGSEGV
	// when buffers get repurposed by a later forward pass.
	type permanentAllocator interface {
		AllocPermanent(bytes int) tensor.DevicePtr
	}
	var allocPerm func(int) tensor.DevicePtr
	if pa, ok := be.(permanentAllocator); ok {
		allocPerm = pa.AllocPermanent
	} else {
		allocPerm = be.Alloc
	}

	g := &GPUBlockPool{
		be:        be,
		pagedOps:  pagedOps,
		allocPerm: allocPerm,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		blockSize:  blockSize,
		maxBlocks:  maxBlocks,
		pools:      make([]tensor.DevicePtr, numLayers),
		freeLists:  make([][]int32, numLayers),
		refCounts:  make([]map[int32]int, numLayers),
		seqs:       make(map[int64]*gpuSeqState),
	}

	alloc := allocPerm

	elementsPerBlock := 2 * blockSize * numKVHeads * headDim
	poolBytes := maxBlocks * elementsPerBlock * 4

	for i := 0; i < numLayers; i++ {
		g.pools[i] = alloc(poolBytes)
		g.freeLists[i] = make([]int32, maxBlocks)
		g.refCounts[i] = make(map[int32]int)
		for j := 0; j < maxBlocks; j++ {
			g.freeLists[i][j] = int32(j)
		}
	}

	g.ptBuf1 = alloc(4)
	g.offBuf1 = alloc(4)

	return g
}

// CreateSequence initializes tracking for a new sequence.
func (g *GPUBlockPool) CreateSequence(seqID int64) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.seqs[seqID] = &gpuSeqState{
		blockTables: make([][]int32, g.numLayers),
		btGPU:       make([]tensor.DevicePtr, g.numLayers),
		btDirty:     make([]bool, g.numLayers),
	}
}

// DeleteSequence releases all blocks for a sequence.
func (g *GPUBlockPool) DeleteSequence(seqID int64) {
	g.mu.Lock()
	defer g.mu.Unlock()

	seq, ok := g.seqs[seqID]
	if !ok {
		return
	}

	for layer := 0; layer < g.numLayers; layer++ {
		for _, blockID := range seq.blockTables[layer] {
			g.releaseBlock(layer, blockID)
		}
		if !seq.btGPU[layer].IsNil() {
			g.be.Free(seq.btGPU[layer])
		}
	}

	delete(g.seqs, seqID)
}

// TruncateSequence reduces a sequence's length, releasing any blocks that are
// no longer needed. This is used after speculative verification to discard
// KV entries for rejected draft tokens.
func (g *GPUBlockPool) TruncateSequence(seqID int64, newSeqLen int) {
	g.mu.Lock()
	defer g.mu.Unlock()

	seq, ok := g.seqs[seqID]
	if !ok || newSeqLen >= seq.seqLen {
		return
	}

	// Number of blocks needed after truncation
	newNumBlocks := 0
	if newSeqLen > 0 {
		newNumBlocks = (newSeqLen-1)/g.blockSize + 1
	}

	for layer := 0; layer < g.numLayers; layer++ {
		bt := seq.blockTables[layer]
		// Release blocks beyond what's needed
		for i := newNumBlocks; i < len(bt); i++ {
			g.releaseBlock(layer, bt[i])
		}
		if newNumBlocks < len(bt) {
			seq.blockTables[layer] = bt[:newNumBlocks]
			seq.btDirty[layer] = true
		}
	}

	seq.seqLen = newSeqLen
}

// HasSequence returns true if the sequence is tracked.
func (g *GPUBlockPool) HasSequence(seqID int64) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	_, ok := g.seqs[seqID]
	return ok
}

func (g *GPUBlockPool) allocBlock(layer int) (int32, error) {
	fl := g.freeLists[layer]
	if len(fl) == 0 {
		return 0, fmt.Errorf("no free blocks for layer %d (max=%d)", layer, g.maxBlocks)
	}
	blockID := fl[len(fl)-1]
	g.freeLists[layer] = fl[:len(fl)-1]
	g.refCounts[layer][blockID] = 1
	return blockID, nil
}

func (g *GPUBlockPool) releaseBlock(layer int, blockID int32) {
	rc := g.refCounts[layer][blockID]
	if rc <= 1 {
		delete(g.refCounts[layer], blockID)
		g.freeLists[layer] = append(g.freeLists[layer], blockID)
	} else {
		g.refCounts[layer][blockID] = rc - 1
	}
}

// StoreKV scatters new K/V data into the GPU block pool.
// kPtr, vPtr are GPU pointers to [seqLen, numKVHeads, headDim] data.
func (g *GPUBlockPool) StoreKV(layerIdx int, seqID int64, startPos int,
	kPtr, vPtr tensor.DevicePtr, seqLen int) error {

	g.mu.Lock()
	defer g.mu.Unlock()

	seq, ok := g.seqs[seqID]
	if !ok {
		return fmt.Errorf("sequence %d not found", seqID)
	}

	endPos := startPos + seqLen

	// Ensure blocks allocated for all positions
	lastBlock := (endPos - 1) / g.blockSize
	for len(seq.blockTables[layerIdx]) <= lastBlock {
		blockID, err := g.allocBlock(layerIdx)
		if err != nil {
			return err
		}
		seq.blockTables[layerIdx] = append(seq.blockTables[layerIdx], blockID)
		seq.btDirty[layerIdx] = true
	}

	if seqLen == 1 {
		// Fast path: single token (decode), use pre-allocated buffers
		pos := startPos
		logicalBlock := pos / g.blockSize
		physBlock := seq.blockTables[layerIdx][logicalBlock]
		offset := int32(pos % g.blockSize)

		ptBytes := [4]byte{byte(uint32(physBlock)), byte(uint32(physBlock) >> 8),
			byte(uint32(physBlock) >> 16), byte(uint32(physBlock) >> 24)}
		offBytes := [4]byte{byte(uint32(offset)), byte(uint32(offset) >> 8),
			byte(uint32(offset) >> 16), byte(uint32(offset) >> 24)}

		g.be.ToDevice(g.ptBuf1, ptBytes[:])
		g.be.ToDevice(g.offBuf1, offBytes[:])

		g.pagedOps.ReshapePagedKV(kPtr, g.pools[layerIdx], g.ptBuf1, g.offBuf1,
			1, g.numKVHeads, g.headDim, g.blockSize, false)
		g.pagedOps.ReshapePagedKV(vPtr, g.pools[layerIdx], g.ptBuf1, g.offBuf1,
			1, g.numKVHeads, g.headDim, g.blockSize, true)
	} else {
		// Multi-token path: scatter one token at a time.
		// The batched ReshapePagedKV kernel hangs at certain dispatch sizes
		// (Metal kernel bug). Per-token scatter is reliable and correct.
		kvElems := g.numKVHeads * g.headDim
		kvBytes := kvElems * 4
		tokenBuf := g.allocPerm(kvBytes)
		copier, _ := g.be.(backend.BufferCopier)
		for i := 0; i < seqLen; i++ {
			pos := startPos + i
			logicalBlock := pos / g.blockSize
			physBlock := seq.blockTables[layerIdx][logicalBlock]
			offset := int32(pos % g.blockSize)

			ptBytes := [4]byte{byte(uint32(physBlock)), byte(uint32(physBlock) >> 8),
				byte(uint32(physBlock) >> 16), byte(uint32(physBlock) >> 24)}
			offBytes := [4]byte{byte(uint32(offset)), byte(uint32(offset) >> 8),
				byte(uint32(offset) >> 16), byte(uint32(offset) >> 24)}
			g.be.ToDevice(g.ptBuf1, ptBytes[:])
			g.be.ToDevice(g.offBuf1, offBytes[:])

			if copier != nil {
				copier.CopyBuffer(kPtr, i*kvBytes, tokenBuf, 0, kvBytes)
			}
			g.pagedOps.ReshapePagedKV(tokenBuf, g.pools[layerIdx], g.ptBuf1, g.offBuf1,
				1, g.numKVHeads, g.headDim, g.blockSize, false)
			if copier != nil {
				copier.CopyBuffer(vPtr, i*kvBytes, tokenBuf, 0, kvBytes)
			}
			g.pagedOps.ReshapePagedKV(tokenBuf, g.pools[layerIdx], g.ptBuf1, g.offBuf1,
				1, g.numKVHeads, g.headDim, g.blockSize, true)
		}
		g.be.Free(tokenBuf)
	}

	if endPos > seq.seqLen {
		seq.seqLen = endPos
	}

	return nil
}

// Attention performs paged SDPA decode for a single query token.
// Uses seq.seqLen to determine the number of KV tokens.
func (g *GPUBlockPool) Attention(layerIdx int, seqID int64,
	qPtr, outPtr tensor.DevicePtr, numQHeads, headDim int, scale float32) error {
	return g.AttentionWithKVLen(layerIdx, seqID, qPtr, outPtr, numQHeads, headDim, scale, 0)
}

// AttentionWithKVLen performs paged SDPA decode with an explicit KV length.
// If kvLen <= 0, seq.seqLen is used (normal decode). If kvLen > 0, it overrides
// seq.seqLen for computing tokensInLastBlock. This is needed when processing
// multiple tokens sequentially across layers — seq.seqLen may reflect tokens
// stored by a prior layer, not the current layer.
func (g *GPUBlockPool) AttentionWithKVLen(layerIdx int, seqID int64,
	qPtr, outPtr tensor.DevicePtr, numQHeads, headDim int, scale float32, kvLen int) error {

	g.mu.Lock()
	seq, ok := g.seqs[seqID]
	if !ok {
		g.mu.Unlock()
		return fmt.Errorf("sequence %d not found", seqID)
	}

	bt := seq.blockTables[layerIdx]
	numBlocks := len(bt)
	if numBlocks == 0 {
		g.mu.Unlock()
		return fmt.Errorf("no blocks for seq %d layer %d", seqID, layerIdx)
	}

	effectiveLen := seq.seqLen
	if kvLen > 0 {
		effectiveLen = kvLen
	}
	// When kvLen overrides seq.seqLen, limit numBlocks to what's actually valid
	effectiveBlocks := (effectiveLen + g.blockSize - 1) / g.blockSize
	if effectiveBlocks < numBlocks {
		numBlocks = effectiveBlocks
	}
	tokensInLastBlock := effectiveLen - (numBlocks-1)*g.blockSize

	// Sync block table to GPU if dirty
	if seq.btDirty[layerIdx] || seq.btGPU[layerIdx].IsNil() {
		if !seq.btGPU[layerIdx].IsNil() {
			g.be.Free(seq.btGPU[layerIdx])
		}
		seq.btGPU[layerIdx] = g.allocPerm(numBlocks * 4)
		g.be.ToDevice(seq.btGPU[layerIdx], gpuInt32ToBytes(bt))
		seq.btDirty[layerIdx] = false
	}

	btGPU := seq.btGPU[layerIdx]
	g.mu.Unlock()

	g.pagedOps.SDPAPagedDecode(qPtr, g.pools[layerIdx], btGPU, outPtr,
		numBlocks, g.blockSize, numQHeads, g.numKVHeads, headDim,
		scale, tokensInLastBlock)

	return nil
}

// AttentionMultiquery performs multi-query paged SDPA in a single dispatch.
// qPtr: [querySeqLen, numQHeads, headDim], outPtr: [querySeqLen, numQHeads, headDim]
// kvLens: per-query KV lengths for causal masking (query i attends to 0..kvLens[i]-1).
func (g *GPUBlockPool) AttentionMultiquery(layerIdx int, seqID int64,
	qPtr, outPtr tensor.DevicePtr, numQHeads, headDim int, scale float32,
	querySeqLen int, kvLens []int) error {

	g.mu.Lock()
	seq, ok := g.seqs[seqID]
	if !ok {
		g.mu.Unlock()
		return fmt.Errorf("sequence %d not found", seqID)
	}

	bt := seq.blockTables[layerIdx]
	numBlocks := len(bt)
	if numBlocks == 0 {
		g.mu.Unlock()
		return fmt.Errorf("no blocks for seq %d layer %d", seqID, layerIdx)
	}

	// Sync block table to GPU if dirty
	if seq.btDirty[layerIdx] || seq.btGPU[layerIdx].IsNil() {
		if !seq.btGPU[layerIdx].IsNil() {
			g.be.Free(seq.btGPU[layerIdx])
		}
		seq.btGPU[layerIdx] = g.allocPerm(numBlocks * 4)
		g.be.ToDevice(seq.btGPU[layerIdx], gpuInt32ToBytes(bt))
		seq.btDirty[layerIdx] = false
	}

	btGPU := seq.btGPU[layerIdx]
	g.mu.Unlock()

	// Upload kvLens to GPU
	kvLens32 := make([]int32, querySeqLen)
	for i, l := range kvLens {
		kvLens32[i] = int32(l)
	}
	kvLensBuf := g.allocPerm(querySeqLen * 4)
	g.be.ToDevice(kvLensBuf, gpuInt32ToBytes(kvLens32))

	// Check if backend supports multi-query paged SDPA
	type multiqueryOps interface {
		SDPAPagedDecodeMultiquery(q, kvPool, blockTable, out, kvLens tensor.DevicePtr,
			numBlocks, blockSize, numQHeads, numKVHeads, headDim int,
			scale float32, querySeqLen int)
	}
	if mq, ok := g.be.(multiqueryOps); ok {
		mq.SDPAPagedDecodeMultiquery(qPtr, g.pools[layerIdx], btGPU, outPtr, kvLensBuf,
			numBlocks, g.blockSize, numQHeads, g.numKVHeads, g.headDim, scale, querySeqLen)
	} else {
		// Fallback: per-query sequential decode
		qStride := numQHeads * headDim
		for i := 0; i < querySeqLen; i++ {
			tokenQ := tensor.DevicePtrOffset(qPtr, uintptr(i*qStride*4))
			tokenOut := tensor.DevicePtrOffset(outPtr, uintptr(i*qStride*4))
			tokensInLastBlock := kvLens[i] - ((kvLens[i]-1)/g.blockSize)*g.blockSize
			nBlocks := (kvLens[i] + g.blockSize - 1) / g.blockSize
			g.pagedOps.SDPAPagedDecode(tokenQ, g.pools[layerIdx], btGPU, tokenOut,
				nBlocks, g.blockSize, numQHeads, g.numKVHeads, g.headDim, scale, tokensInLastBlock)
		}
	}

	g.be.Sync()
	g.be.Free(kvLensBuf)
	return nil
}

// AttentionBatched performs batched SDPA across multiple sequences using paged KV.
func (g *GPUBlockPool) AttentionBatched(
	layerIdx int,
	seqIDs []int64,
	qPtr, outPtr tensor.DevicePtr,
	numQHeads, headDim int,
	scale float32,
) error {
	batchSize := len(seqIDs)
	blockTables := make([]tensor.DevicePtr, batchSize)
	seqLens := make([]int, batchSize)
	maxBlocks := 0

	g.mu.Lock()
	for i, seqID := range seqIDs {
		seq, ok := g.seqs[seqID]
		if !ok {
			g.mu.Unlock()
			return fmt.Errorf("sequence %d not found in GPU block pool", seqID)
		}

		bt := seq.blockTables[layerIdx]
		numBlocks := len(bt)
		if numBlocks == 0 {
			g.mu.Unlock()
			return fmt.Errorf("no blocks for seq %d layer %d", seqID, layerIdx)
		}

		// Sync block table to GPU if dirty
		if seq.btDirty[layerIdx] || seq.btGPU[layerIdx].IsNil() {
			if !seq.btGPU[layerIdx].IsNil() {
				g.be.Free(seq.btGPU[layerIdx])
			}
			seq.btGPU[layerIdx] = g.allocPerm(numBlocks * 4)
			g.be.ToDevice(seq.btGPU[layerIdx], gpuInt32ToBytes(bt))
			seq.btDirty[layerIdx] = false
		}

		blockTables[i] = seq.btGPU[layerIdx]
		seqLens[i] = seq.seqLen
		if numBlocks > maxBlocks {
			maxBlocks = numBlocks
		}
	}
	g.mu.Unlock()

	g.pagedOps.SDPAPagedDecodeBatched(qPtr, g.pools[layerIdx], blockTables, outPtr,
		batchSize, maxBlocks, g.blockSize, numQHeads, g.numKVHeads, headDim, scale, seqLens)

	return nil
}

// GatherKV reads K/V from scattered blocks into contiguous GPU buffers.
// Returns pointers to contiguous K [totalTokens, numKVHeads, headDim] and
// V [totalTokens, numKVHeads, headDim] buffers. Caller must free the returned buffers.
// This is used for multi-token verification where SDPAPrefill needs contiguous K/V.
func (g *GPUBlockPool) GatherKV(layerIdx int, seqID int64, totalTokens int) (kOut, vOut tensor.DevicePtr, err error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	seq, ok := g.seqs[seqID]
	if !ok {
		return tensor.DevicePtr{}, tensor.DevicePtr{}, fmt.Errorf("sequence %d not found", seqID)
	}

	copier, ok := g.be.(backend.BufferCopier)
	if !ok {
		return tensor.DevicePtr{}, tensor.DevicePtr{}, fmt.Errorf("backend doesn't support BufferCopier")
	}

	kvElems := g.numKVHeads * g.headDim           // elements per token
	blockKVBytes := g.blockSize * kvElems * 4      // bytes for K (or V) in one block
	poolBlockBytes := 2 * blockKVBytes             // K + V per physical block
	outBytes := totalTokens * kvElems * 4

	kOut = g.allocPerm(outBytes)
	vOut = g.allocPerm(outBytes)

	bt := seq.blockTables[layerIdx]
	remaining := totalTokens

	for logicalBlock := 0; logicalBlock < len(bt) && remaining > 0; logicalBlock++ {
		physBlock := bt[logicalBlock]
		tokensInBlock := g.blockSize
		if tokensInBlock > remaining {
			tokensInBlock = remaining
		}
		copyBytes := tokensInBlock * kvElems * 4

		// Source offset in pool: physBlock * poolBlockBytes
		srcBase := int(physBlock) * poolBlockBytes
		// K is at offset 0, V is at offset blockKVBytes within each physical block
		dstOffset := logicalBlock * g.blockSize * kvElems * 4

		copier.CopyBuffer(g.pools[layerIdx], srcBase, kOut, dstOffset, copyBytes)
		copier.CopyBuffer(g.pools[layerIdx], srcBase+blockKVBytes, vOut, dstOffset, copyBytes)

		remaining -= tokensInBlock
	}

	return kOut, vOut, nil
}

// ShareBlocks shares the first numBlocks logical blocks from srcSeq to dstSeq
// across all layers. The shared physical blocks have their ref count incremented.
// This is used for prefix caching: when a new sequence shares a prefix with
// an existing one, the GPU blocks containing the prefix K/V are reused.
func (g *GPUBlockPool) ShareBlocks(srcSeqID, dstSeqID int64, numTokens int) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	src, ok := g.seqs[srcSeqID]
	if !ok {
		return fmt.Errorf("source sequence %d not found", srcSeqID)
	}
	dst, ok := g.seqs[dstSeqID]
	if !ok {
		return fmt.Errorf("destination sequence %d not found", dstSeqID)
	}

	numBlocks := (numTokens + g.blockSize - 1) / g.blockSize

	for layer := 0; layer < g.numLayers; layer++ {
		if numBlocks > len(src.blockTables[layer]) {
			return fmt.Errorf("source seq %d layer %d has %d blocks, need %d",
				srcSeqID, layer, len(src.blockTables[layer]), numBlocks)
		}

		// Share physical blocks by copying block table entries and incrementing ref counts
		for i := 0; i < numBlocks; i++ {
			blockID := src.blockTables[layer][i]
			dst.blockTables[layer] = append(dst.blockTables[layer], blockID)
			g.refCounts[layer][blockID]++
		}
		dst.btDirty[layer] = true
	}

	if numTokens > dst.seqLen {
		dst.seqLen = numTokens
	}

	return nil
}

// BlockStats returns allocation statistics for monitoring.
func (g *GPUBlockPool) BlockStats() (totalBlocks, freeBlocks, sharedBlocks int) {
	g.mu.Lock()
	defer g.mu.Unlock()

	totalBlocks = g.maxBlocks * g.numLayers
	for layer := 0; layer < g.numLayers; layer++ {
		freeBlocks += len(g.freeLists[layer])
		for _, rc := range g.refCounts[layer] {
			if rc > 1 {
				sharedBlocks++
			}
		}
	}
	return
}

// MemoryStats returns GPU memory usage in megabytes based on block pool allocations.
func (g *GPUBlockPool) MemoryStats() (totalMB, usedMB, freeMB float64) {
	total, free, _ := g.BlockStats()
	used := total - free
	bytesPerBlock := 2 * g.blockSize * g.numKVHeads * g.headDim * 4 // K+V, float32
	totalMB = float64(total*bytesPerBlock) / (1024 * 1024)
	usedMB = float64(used*bytesPerBlock) / (1024 * 1024)
	freeMB = float64(free*bytesPerBlock) / (1024 * 1024)
	return
}

// Close releases all GPU resources.
func (g *GPUBlockPool) Close() {
	for _, pool := range g.pools {
		if !pool.IsNil() {
			g.be.Free(pool)
		}
	}
	if !g.ptBuf1.IsNil() {
		g.be.Free(g.ptBuf1)
	}
	if !g.offBuf1.IsNil() {
		g.be.Free(g.offBuf1)
	}
	for _, seq := range g.seqs {
		for _, bt := range seq.btGPU {
			if !bt.IsNil() {
				g.be.Free(bt)
			}
		}
	}
}

// gpuInt32ToBytes converts int32 slice to little-endian bytes.
func gpuInt32ToBytes(in []int32) []byte {
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
