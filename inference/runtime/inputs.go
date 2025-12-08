package runtime

import "vexel/inference/kv"

// BatchRuntimeInputs holds the inputs for a single decode step.
type BatchRuntimeInputs struct {
	tokens    []int
	positions []int   // Position in sequence for each token (for RoPE)
	seqIDs    []int64 // Sequence IDs for PagedKVCache (one per token)
	kvHandles []*kv.SeqKVHandle
}

// NewBatchRuntimeInputs creates a new batch input container.
// If positions is nil, defaults to 0 for all tokens (legacy behavior).
func NewBatchRuntimeInputs(tokens []int, handles []*kv.SeqKVHandle) BatchRuntimeInputs {
	return BatchRuntimeInputs{
		tokens:    tokens,
		positions: nil, // Will default to 0 in runtime
		kvHandles: handles,
	}
}

// NewBatchRuntimeInputsWithPos creates a batch input with explicit positions.
func NewBatchRuntimeInputsWithPos(tokens []int, positions []int, handles []*kv.SeqKVHandle) BatchRuntimeInputs {
	return BatchRuntimeInputs{
		tokens:    tokens,
		positions: positions,
		kvHandles: handles,
	}
}

// NewBatchRuntimeInputsFull creates a batch input with positions and sequence IDs for PagedKVCache.
func NewBatchRuntimeInputsFull(tokens []int, positions []int, seqIDs []int64) BatchRuntimeInputs {
	return BatchRuntimeInputs{
		tokens:    tokens,
		positions: positions,
		seqIDs:    seqIDs,
	}
}

// Tokens returns the input tokens for the batch.
func (b BatchRuntimeInputs) Tokens() []int {
	return b.tokens
}

// Positions returns the position for each token (for RoPE).
// Returns nil if positions weren't set.
func (b BatchRuntimeInputs) Positions() []int {
	return b.positions
}

// SeqIDs returns the sequence IDs for PagedKVCache.
// Returns nil if not using PagedKVCache.
func (b BatchRuntimeInputs) SeqIDs() []int64 {
	return b.seqIDs
}

// KVHandles returns the KV cache handles for the batch.
func (b BatchRuntimeInputs) KVHandles() []*kv.SeqKVHandle {
	return b.kvHandles
}
