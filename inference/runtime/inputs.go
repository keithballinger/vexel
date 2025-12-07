package runtime

import "vexel/inference/kv"

// BatchRuntimeInputs holds the inputs for a single decode step.
type BatchRuntimeInputs struct {
	tokens    []int
	kvHandles []*kv.SeqKVHandle
}

// NewBatchRuntimeInputs creates a new batch input container.
func NewBatchRuntimeInputs(tokens []int, handles []*kv.SeqKVHandle) BatchRuntimeInputs {
	return BatchRuntimeInputs{
		tokens:    tokens,
		kvHandles: handles,
	}
}

// Tokens returns the input tokens for the batch.
func (b BatchRuntimeInputs) Tokens() []int {
	return b.tokens
}

// KVHandles returns the KV cache handles for the batch.
func (b BatchRuntimeInputs) KVHandles() []*kv.SeqKVHandle {
	return b.kvHandles
}
