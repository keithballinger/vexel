package kv

import (
	"sync"
)

// PagedKVCache is the main KV cache using paged attention.
type PagedKVCache struct {
	mu        sync.RWMutex
	config    PagedKVConfig
	allocator *BlockAllocator
	fragments *FragmentCache

	// Per-sequence block tables (seqID -> BlockTable)
	sequences map[int64]*BlockTable
	nextSeqID int64
}

// NewPagedKVCache creates a new paged KV cache.
func NewPagedKVCache(config PagedKVConfig) *PagedKVCache {
	allocator := NewBlockAllocator(config)
	return &PagedKVCache{
		config:    config,
		allocator: allocator,
		fragments: NewFragmentCache(allocator),
		sequences: make(map[int64]*BlockTable),
		nextSeqID: 1,
	}
}

// Config returns the cache configuration.
func (c *PagedKVCache) Config() PagedKVConfig {
	return c.config
}

// Allocator returns the block allocator.
func (c *PagedKVCache) Allocator() *BlockAllocator {
	return c.allocator
}

// Fragments returns the fragment cache.
func (c *PagedKVCache) Fragments() *FragmentCache {
	return c.fragments
}

// CreateSequence creates a new sequence and returns its ID.
func (c *PagedKVCache) CreateSequence() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()

	seqID := c.nextSeqID
	c.nextSeqID++

	c.sequences[seqID] = NewBlockTable(c.config.NumLayers)
	return seqID
}

// DeleteSequence removes a sequence and frees its blocks.
func (c *PagedKVCache) DeleteSequence(seqID int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, exists := c.sequences[seqID]
	if !exists {
		return
	}

	// Free all blocks for this sequence
	for layer := 0; layer < c.config.NumLayers; layer++ {
		for _, blockID := range table.BlocksForLayer(layer) {
			c.allocator.Free(layer, blockID)
		}
	}

	delete(c.sequences, seqID)
}

// GetSequence returns the block table for a sequence.
func (c *PagedKVCache) GetSequence(seqID int64) *BlockTable {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.sequences[seqID]
}

// AllocateBlock allocates a new block for a sequence at a given layer.
func (c *PagedKVCache) AllocateBlock(seqID int64, layer int) (*Block, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, exists := c.sequences[seqID]
	if !exists {
		table = NewBlockTable(c.config.NumLayers)
		c.sequences[seqID] = table
	}

	block, err := c.allocator.Allocate(layer)
	if err != nil {
		return nil, err
	}

	table.AppendBlock(layer, block.ID)
	return block, nil
}

// StoreKV stores K and V for a token at a given position in a sequence.
// Allocates new blocks as needed.
func (c *PagedKVCache) StoreKV(seqID int64, layer, pos int, k, v []float32) error {
	blockSize := c.config.BlockSize
	blockIdx := pos / blockSize
	offsetInBlock := pos % blockSize

	c.mu.Lock()
	table := c.sequences[seqID]
	if table == nil {
		table = NewBlockTable(c.config.NumLayers)
		c.sequences[seqID] = table
	}
	c.mu.Unlock()

	// Ensure we have enough blocks
	for table.NumBlocks(layer) <= blockIdx {
		_, err := c.AllocateBlock(seqID, layer)
		if err != nil {
			return err
		}
	}

	blockID := table.GetBlockID(layer, blockIdx)
	block := c.allocator.GetBlock(layer, blockID)
	if block == nil {
		return nil // Should not happen
	}

	// Write K and V into block
	kvSize := c.config.NumKVHeads * c.config.HeadDim
	kOffset := offsetInBlock * kvSize
	vOffset := c.config.KVSizeFloats() + offsetInBlock*kvSize

	copy(block.Data[kOffset:kOffset+kvSize], k)
	copy(block.Data[vOffset:vOffset+kvSize], v)

	if offsetInBlock+1 > block.NumTokens {
		block.NumTokens = offsetInBlock + 1
	}

	// Update sequence length
	if pos+1 > table.SeqLen() {
		table.SetSeqLen(pos + 1)
	}

	return nil
}

// StoreKVBatch stores K and V for multiple tokens at consecutive positions.
// k and v should have shape [numTokens, numKVHeads*headDim] flattened.
// startPos is the position of the first token.
func (c *PagedKVCache) StoreKVBatch(seqID int64, layer, startPos int, k, v []float32, numTokens int) error {
	kvSize := c.config.NumKVHeads * c.config.HeadDim

	// Store each token
	for i := 0; i < numTokens; i++ {
		pos := startPos + i
		kStart := i * kvSize
		vStart := i * kvSize
		err := c.StoreKV(seqID, layer, pos, k[kStart:kStart+kvSize], v[vStart:vStart+kvSize])
		if err != nil {
			return err
		}
	}
	return nil
}

// GetKVSlice returns K and V data for positions [0, endPos] for a layer.
// Returns slices pointing into block memory (zero-copy when possible).
// For paged memory, this gathers from multiple blocks.
func (c *PagedKVCache) GetKVSlice(seqID int64, layer, endPos int) (k, v []float32) {
	c.mu.RLock()
	table := c.sequences[seqID]
	c.mu.RUnlock()

	if table == nil {
		return nil, nil
	}

	numTokens := endPos + 1
	kvSize := c.config.NumKVHeads * c.config.HeadDim
	totalSize := numTokens * kvSize

	// Gather K and V from blocks
	k = make([]float32, totalSize)
	v = make([]float32, totalSize)

	blockSize := c.config.BlockSize
	blockKVSize := c.config.KVSizeFloats()

	for pos := 0; pos < numTokens; pos++ {
		blockIdx := pos / blockSize
		offsetInBlock := pos % blockSize

		blockID := table.GetBlockID(layer, blockIdx)
		if blockID == InvalidBlockID {
			continue
		}

		block := c.allocator.GetBlock(layer, blockID)
		if block == nil {
			continue
		}

		srcKOffset := offsetInBlock * kvSize
		srcVOffset := blockKVSize + offsetInBlock*kvSize
		dstOffset := pos * kvSize

		copy(k[dstOffset:dstOffset+kvSize], block.Data[srcKOffset:srcKOffset+kvSize])
		copy(v[dstOffset:dstOffset+kvSize], block.Data[srcVOffset:srcVOffset+kvSize])
	}

	return k, v
}

// GetKVSliceForAttention returns K and V data with RoPE shifts applied for fragments.
// This is the method to use for attention computation when fragments may be present.
// k is modified in place to apply necessary RoPE shifts.
// ropeTheta: the RoPE theta parameter from model config
// applyShift: function to apply RoPE shift (typically backend.RoPEShift)
func (c *PagedKVCache) GetKVSliceForAttention(
	seqID int64,
	layer, endPos int,
	ropeTheta float32,
	applyShift func(k []float32, headDim, numKVHeads, numTokens, shift int, theta float32),
) (k, v []float32) {
	// Get raw K,V data
	k, v = c.GetKVSlice(seqID, layer, endPos)
	if k == nil || v == nil {
		return nil, nil
	}

	// Get fragment ranges
	c.mu.RLock()
	table := c.sequences[seqID]
	c.mu.RUnlock()

	if table == nil {
		return k, v
	}

	fragmentRanges := table.FragmentRanges()
	if len(fragmentRanges) == 0 {
		return k, v // No fragments, no shifts needed
	}

	numKVHeads := c.config.NumKVHeads
	headDim := c.config.HeadDim
	kvSize := numKVHeads * headDim

	// Apply RoPE shifts for each fragment range
	for _, fr := range fragmentRanges {
		if fr.EndPos > endPos {
			continue // Fragment extends beyond current query position
		}
		if fr.ShiftBy == 0 {
			continue // No shift needed
		}

		// Extract the K vectors for this fragment range and apply shift
		startIdx := fr.StartPos * kvSize
		numTokensInRange := fr.NumTokens
		if fr.StartPos+numTokensInRange-1 > endPos {
			numTokensInRange = endPos - fr.StartPos + 1
		}

		// Apply shift to this range of K vectors
		kSlice := k[startIdx : startIdx+numTokensInRange*kvSize]
		applyShift(kSlice, headDim, numKVHeads, numTokensInRange, fr.ShiftBy, ropeTheta)
	}

	return k, v
}

// InsertFragment inserts a cached fragment into a sequence at the given position.
// The fragment's blocks are shared (ref count incremented).
// Returns the number of tokens inserted.
// Note: RoPE position shift will be applied when reading K via GetKVSliceForAttention.
func (c *PagedKVCache) InsertFragment(seqID int64, fragmentName string, insertPos int) (int, error) {
	fragment, ok := c.fragments.Get(fragmentName)
	if !ok {
		return 0, nil // Fragment not found, insert nothing
	}

	c.mu.Lock()
	table := c.sequences[seqID]
	if table == nil {
		table = NewBlockTable(c.config.NumLayers)
		c.sequences[seqID] = table
	}
	c.mu.Unlock()

	// Share the fragment's blocks with this sequence
	for layer, blockIDs := range fragment.Blocks {
		for _, blockID := range blockIDs {
			c.allocator.AddRef(blockID)
			table.AppendBlock(layer, blockID)
		}
	}

	// Record fragment range for RoPE shift tracking
	// The fragment was cached with RoPE at positions 0..N-1
	// When inserted at position insertPos, we need to shift by insertPos
	table.AddFragmentRange(insertPos, fragment.NumTokens, insertPos)

	newSeqLen := insertPos + fragment.NumTokens
	if newSeqLen > table.SeqLen() {
		table.SetSeqLen(newSeqLen)
	}

	return fragment.NumTokens, nil
}

// CacheContent creates a cached fragment from a completed sequence.
// This takes ownership of the blocks (they become shared).
func (c *PagedKVCache) CacheContent(name string, seqID int64) (*CachedFragment, error) {
	c.mu.RLock()
	table := c.sequences[seqID]
	c.mu.RUnlock()

	if table == nil {
		return nil, nil
	}

	// Copy block references
	blocks := make([][]BlockID, c.config.NumLayers)
	for layer := 0; layer < c.config.NumLayers; layer++ {
		blocks[layer] = make([]BlockID, len(table.BlocksForLayer(layer)))
		copy(blocks[layer], table.BlocksForLayer(layer))
	}

	return c.fragments.CacheContent(name, table.SeqLen(), blocks)
}

// FreeBlocks returns the number of free blocks per layer (uses layer 0 as representative).
func (c *PagedKVCache) FreeBlocks() int {
	return c.allocator.FreeCount(0)
}

// BlockSize returns the number of tokens per block.
func (c *PagedKVCache) BlockSize() int {
	return c.config.BlockSize
}
