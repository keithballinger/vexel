package runtime

import (
	"vexel/inference/backend/cpu"
	"vexel/inference/ir"
	"vexel/inference/tensor"
)

// BlockRuntime represents a single transformer layer (Attention + MLP).
type BlockRuntime struct {
	backend cpu.Backend
	graph   *ir.BlockIR
	
	// Weights
	AttnNorm   tensor.Tensor
	Wq, Wk, Wv tensor.Tensor
	Wo         tensor.Tensor
	
	FFNNorm    tensor.Tensor
	W1, W2, W3 tensor.Tensor // Gate, Down, Up
}

// NewBlockRuntime creates a new block runtime.
func NewBlockRuntime(backend cpu.Backend) *BlockRuntime {
	return &BlockRuntime{
		backend: backend,
	}
}

// Execute performs the forward pass for this block.
func (b *BlockRuntime) Execute(x tensor.Tensor) (tensor.Tensor, error) {
	// TODO: Call backend kernels
	// 1. RMSNorm
	// 2. Attention (QKV proj, RoPE, SDPA, O proj)
	// 3. Residual Add
	// 4. RMSNorm
	// 5. MLP (Gate/Up proj, SiLU, Down proj)
	// 6. Residual Add
	
	return x, nil // Passthrough for now
}