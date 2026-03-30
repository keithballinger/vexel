package kv

import (
	"fmt"
	"sync"
)

// BlockID uniquely identifies a physical block in the cache.
type BlockID int32

const InvalidBlockID BlockID = -1

// Block holds KV data for a fixed number of tokens at one layer.
// Layout: K [BlockSize, NumKVHeads, HeadDim] followed by V [BlockSize, NumKVHeads, HeadDim]
type Block struct {
	ID        BlockID
	LayerIdx  int
	Data      []float32 // K and V concatenated
	NumTokens int       // How many token slots are filled (0 to BlockSize)
}

// PagedKVConfig defines the configuration for paged attention.
type PagedKVConfig struct {
	NumLayers  int
	NumKVHeads int
	HeadDim    int
	BlockSize  int // Tokens per block (e.g., 16)
	MaxBlocks  int // Maximum blocks in the pool
}

// BlockSizeFloats returns the number of float32s in one block (K + V).
func (c PagedKVConfig) BlockSizeFloats() int {
	// K: [BlockSize, NumKVHeads, HeadDim] + V: same
	return 2 * c.BlockSize * c.NumKVHeads * c.HeadDim
}

// KVSizeFloats returns floats for just K or just V in one block.
func (c PagedKVConfig) KVSizeFloats() int {
	return c.BlockSize * c.NumKVHeads * c.HeadDim
}

// BlockAllocator manages a pool of physical blocks.
type BlockAllocator struct {
	mu        sync.Mutex
	config    PagedKVConfig
	blocks    [][]*Block      // blocks[layer][blockIdx]
	freeList  [][]BlockID     // freeList[layer] = stack of free block IDs
	refCounts map[BlockID]int // Reference counts for shared blocks
}

// NewBlockAllocator creates a new block allocator.
func NewBlockAllocator(config PagedKVConfig) *BlockAllocator {
	blocks := make([][]*Block, config.NumLayers)
	freeList := make([][]BlockID, config.NumLayers)

	blockFloats := config.BlockSizeFloats()

	for layer := 0; layer < config.NumLayers; layer++ {
		blocks[layer] = make([]*Block, config.MaxBlocks)
		freeList[layer] = make([]BlockID, config.MaxBlocks)

		// Pre-allocate all blocks and add to free list
		for i := 0; i < config.MaxBlocks; i++ {
			blocks[layer][i] = &Block{
				ID:        BlockID(i),
				LayerIdx:  layer,
				Data:      make([]float32, blockFloats),
				NumTokens: 0,
			}
			// Push to free list (reversed so we pop 0 first)
			freeList[layer][i] = BlockID(config.MaxBlocks - 1 - i)
		}
	}

	return &BlockAllocator{
		config:    config,
		blocks:    blocks,
		freeList:  freeList,
		refCounts: make(map[BlockID]int),
	}
}

// Allocate returns a free block for the given layer, or error if none available.
func (a *BlockAllocator) Allocate(layer int) (*Block, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if layer >= len(a.freeList) {
		return nil, fmt.Errorf("invalid layer %d", layer)
	}

	if len(a.freeList[layer]) == 0 {
		return nil, fmt.Errorf("no free blocks for layer %d", layer)
	}

	// Pop from free list
	n := len(a.freeList[layer])
	blockID := a.freeList[layer][n-1]
	a.freeList[layer] = a.freeList[layer][:n-1]

	block := a.blocks[layer][blockID]
	block.NumTokens = 0 // Reset
	a.refCounts[blockID] = 1

	return block, nil
}

// Free returns a block to the free list.
func (a *BlockAllocator) Free(layer int, blockID BlockID) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if layer >= len(a.freeList) || blockID < 0 || int(blockID) >= len(a.blocks[layer]) {
		return
	}

	// Decrement ref count
	a.refCounts[blockID]--
	if a.refCounts[blockID] <= 0 {
		delete(a.refCounts, blockID)
		a.freeList[layer] = append(a.freeList[layer], blockID)
	}
}

// AddRef increments the reference count for a block (for sharing).
func (a *BlockAllocator) AddRef(blockID BlockID) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.refCounts[blockID]++
}

// GetBlock returns the block for a given layer and ID.
func (a *BlockAllocator) GetBlock(layer int, blockID BlockID) *Block {
	if layer >= len(a.blocks) || blockID < 0 || int(blockID) >= len(a.blocks[layer]) {
		return nil
	}
	return a.blocks[layer][blockID]
}

// FreeCount returns the number of free blocks for a layer.
func (a *BlockAllocator) FreeCount(layer int) int {
	a.mu.Lock()
	defer a.mu.Unlock()
	if layer >= len(a.freeList) {
		return 0
	}
	return len(a.freeList[layer])
}

// Config returns the allocator configuration.
func (a *BlockAllocator) Config() PagedKVConfig {
	return a.config
}

// FragmentRange tracks a range of positions that came from a cached fragment.
// These positions need RoPE shift applied when used in attention.
type FragmentRange struct {
	StartPos  int // First position in sequence
	EndPos    int // Last position (inclusive)
	ShiftBy   int // RoPE position shift to apply (typically = StartPos)
	NumTokens int // Number of tokens from fragment
}

// BlockTable maps logical token positions to physical blocks for one sequence.
// Each layer has its own block table.
type BlockTable struct {
	// blocks[layer] = list of BlockIDs in logical order
	blocks [][]BlockID
	// Length of the sequence (number of tokens)
	seqLen int
	// fragmentRanges tracks positions from cached fragments needing RoPE shift
	fragmentRanges []FragmentRange
}

// NewBlockTable creates an empty block table for a sequence.
func NewBlockTable(numLayers int) *BlockTable {
	blocks := make([][]BlockID, numLayers)
	for i := range blocks {
		blocks[i] = make([]BlockID, 0)
	}
	return &BlockTable{
		blocks: blocks,
		seqLen: 0,
	}
}

// AppendBlock adds a block to the end of the table for a layer.
func (t *BlockTable) AppendBlock(layer int, blockID BlockID) {
	if layer < len(t.blocks) {
		t.blocks[layer] = append(t.blocks[layer], blockID)
	}
}

// GetBlockID returns the block ID for a given layer and logical block index.
func (t *BlockTable) GetBlockID(layer, blockIdx int) BlockID {
	if layer >= len(t.blocks) || blockIdx >= len(t.blocks[layer]) {
		return InvalidBlockID
	}
	return t.blocks[layer][blockIdx]
}

// NumBlocks returns the number of blocks for a layer.
func (t *BlockTable) NumBlocks(layer int) int {
	if layer >= len(t.blocks) {
		return 0
	}
	return len(t.blocks[layer])
}

// SetSeqLen sets the sequence length.
func (t *BlockTable) SetSeqLen(n int) {
	t.seqLen = n
}

// SeqLen returns the sequence length.
func (t *BlockTable) SeqLen() int {
	return t.seqLen
}

// BlocksForLayer returns all block IDs for a layer.
func (t *BlockTable) BlocksForLayer(layer int) []BlockID {
	if layer >= len(t.blocks) {
		return nil
	}
	return t.blocks[layer]
}

// AddFragmentRange records that positions [startPos, startPos+numTokens) came from a fragment.
// shiftBy is the RoPE position shift to apply (typically = startPos).
func (t *BlockTable) AddFragmentRange(startPos, numTokens, shiftBy int) {
	t.fragmentRanges = append(t.fragmentRanges, FragmentRange{
		StartPos:  startPos,
		EndPos:    startPos + numTokens - 1,
		ShiftBy:   shiftBy,
		NumTokens: numTokens,
	})
}

// FragmentRanges returns all fragment ranges for this sequence.
func (t *BlockTable) FragmentRanges() []FragmentRange {
	return t.fragmentRanges
}

// TruncateBlocks removes blocks beyond what is needed for newSeqLen tokens.
// Returns the list of freed block IDs per layer.
func (t *BlockTable) TruncateBlocks(newSeqLen int, blockSize int) [][]BlockID {
	freed := make([][]BlockID, len(t.blocks))

	// How many blocks do we need?
	neededBlocks := 0
	if newSeqLen > 0 {
		neededBlocks = (newSeqLen + blockSize - 1) / blockSize
	}

	for layer := range t.blocks {
		if neededBlocks < len(t.blocks[layer]) {
			freed[layer] = t.blocks[layer][neededBlocks:]
			t.blocks[layer] = t.blocks[layer][:neededBlocks]
		}
	}

	t.seqLen = newSeqLen
	return freed
}

// GetRoPEShiftForPos returns the RoPE shift needed for a given position.
// Returns 0 if the position is not from a fragment.
func (t *BlockTable) GetRoPEShiftForPos(pos int) int {
	for _, fr := range t.fragmentRanges {
		if pos >= fr.StartPos && pos <= fr.EndPos {
			return fr.ShiftBy
		}
	}
	return 0
}
