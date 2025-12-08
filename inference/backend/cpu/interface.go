package cpu

import "vexel/inference/tensor"

// Backend represents a compute backend (CPU, CUDA, Metal).
// It manages execution streams and device-specific resources.
type Backend interface {
	// CreateStream creates a new execution stream (or command queue).
	// Returns an opaque handle (interface{}).
	CreateStream() (interface{}, error)

	// Device returns the device associated with this backend.
	Device() tensor.Device

	// Compute Kernels
	Matmul(a, b, out []float32, m, n, k int)
	MatmulTransposeB(a, b, out []float32, m, n, k int)
	RMSNorm(x, weight, out []float32, rows, cols int, eps float32)
	RoPE(q, k []float32, headDim, numHeads, seqLen, startPos int, theta float32)
	SiLU(x, out []float32, n int)
	Embedding(ids []int, table []float32, out []float32, dim int)
	Softmax(x, out []float32, rows, cols int)

	// SDPA performs Scaled Dot-Product Attention for a single query position.
	// Q: [numQHeads, headDim] - query for current position
	// K: [kvLen, numKVHeads, headDim] - cached keys for all positions
	// V: [kvLen, numKVHeads, headDim] - cached values for all positions
	// out: [numQHeads, headDim] - attention output
	// kvLen: number of KV positions (typically pos+1 for causal attention)
	// numQHeads: number of query heads
	// numKVHeads: number of KV heads (for GQA, numQHeads >= numKVHeads)
	// headDim: dimension per head
	// scale: typically 1/sqrt(headDim)
	SDPA(q, k, v, out []float32, kvLen, numQHeads, numKVHeads, headDim int, scale float32)

	// RoPEShift applies a uniform RoPE position shift to K vectors.
	// This is used for fragment caching: when cached K has RoPE(0,1,2,...) applied,
	// and we insert at position N, we apply RoPE(N) to shift all positions.
	// RoPE is multiplicative: RoPE(a+b) = RoPE(a) * RoPE(b), so applying RoPE(shift)
	// transforms RoPE(p) -> RoPE(p+shift).
	// k: [numTokens, numKVHeads, headDim] - K vectors to shift (modified in place)
	// shift: position offset to apply
	RoPEShift(k []float32, headDim, numKVHeads, numTokens, shift int, theta float32)
}