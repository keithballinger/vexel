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
	RoPE(q, k []float32, headDim, seqLen, startPos int, theta float32)
	SiLU(x, out []float32, n int)
	Embedding(ids []int, table []float32, out []float32, dim int)
	Softmax(x, out []float32, rows, cols int)
}