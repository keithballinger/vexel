// Package backend defines the unified interface for compute backends (CPU, Metal, CUDA).
package backend

import "vexel/inference/tensor"

// PoolResetter is an optional interface for backends that support buffer pooling.
// Call ResetPool at the start of each forward pass to reuse temporary buffers.
type PoolResetter interface {
	ResetPool()
}

// Batcher is an optional interface for backends that support command buffer batching.
// Batching reduces dispatch overhead by combining multiple operations into one commit.
type Batcher interface {
	// BeginBatch starts a batch - subsequent operations share a command buffer.
	BeginBatch()
	// EndBatch commits all batched operations.
	EndBatch()
}

// BufferCopier is an optional interface for backends that support GPU-to-GPU buffer copies.
// This avoids roundtripping through CPU memory.
type BufferCopier interface {
	CopyBuffer(src tensor.DevicePtr, srcOffset int, dst tensor.DevicePtr, dstOffset int, size int)
}

// QuantizedMatMul is an optional interface for backends that support quantized matrix operations.
// Backends that don't implement this will fall back to dequantized (F32) operations.
type QuantizedMatMul interface {
	// MatMulQ4_0 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_0 format.
	// B contains raw Q4_0 data (18 bytes per 32 elements: 2 byte f16 scale + 16 bytes nibbles).
	MatMulQ4_0(a, b, out tensor.DevicePtr, m, n, k int)
}

// Backend represents a compute backend that can execute tensor operations.
// All compute operations use DevicePtr for device-agnostic memory access.
// This interface is implemented by CPU, Metal, and CUDA backends.
type Backend interface {
	// Device returns the device associated with this backend.
	Device() tensor.Device

	// Memory Management

	// Alloc allocates memory on the device and returns a pointer.
	// Size is in bytes. Returns nil DevicePtr on failure.
	Alloc(bytes int) tensor.DevicePtr

	// Free releases memory previously allocated with Alloc.
	Free(ptr tensor.DevicePtr)

	// ToDevice copies data from host memory to device memory.
	// dst must be a valid device pointer, src is host memory.
	ToDevice(dst tensor.DevicePtr, src []byte)

	// ToHost copies data from device memory to host memory.
	// dst is host memory, src must be a valid device pointer.
	ToHost(dst []byte, src tensor.DevicePtr)

	// Sync waits for all pending operations to complete.
	Sync()

	// Compute Kernels
	// All pointers are device pointers. Caller is responsible for ensuring
	// correct sizes and alignments.

	// MatMul performs C = A @ B where A is [M,K], B is [K,N], C is [M,N].
	MatMul(a, b, out tensor.DevicePtr, m, n, k int)

	// MatMulTransposed performs C = A @ B^T where A is [M,K], B is [N,K], C is [M,N].
	MatMulTransposed(a, b, out tensor.DevicePtr, m, n, k int)

	// RMSNorm performs RMS normalization.
	// x: [rows, cols], weight: [cols], out: [rows, cols]
	RMSNorm(x, weight, out tensor.DevicePtr, rows, cols int, eps float32)

	// RoPE applies Rotary Position Embeddings in-place to Q and K.
	// q: [seqLen, numHeads, headDim], k: [seqLen, numKVHeads, headDim] (can be nil)
	RoPE(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos int, theta float32)

	// SiLU applies the SiLU activation function element-wise.
	// x: [n], out: [n]
	SiLU(x, out tensor.DevicePtr, n int)

	// SiLUMul performs fused silu(gate) * up operation.
	// gate: [n], up: [n], out: [n]
	SiLUMul(gate, up, out tensor.DevicePtr, n int)

	// Softmax applies softmax row-wise.
	// x: [rows, cols], out: [rows, cols]
	Softmax(x, out tensor.DevicePtr, rows, cols int)

	// Add performs element-wise addition: out = a + b
	Add(a, b, out tensor.DevicePtr, n int)

	// Mul performs element-wise multiplication: out = a * b
	Mul(a, b, out tensor.DevicePtr, n int)

	// Embedding performs embedding lookup.
	// ids: [numTokens] int32 on device, table: [vocabSize, dim], out: [numTokens, dim]
	Embedding(ids tensor.DevicePtr, numTokens int, table, out tensor.DevicePtr, vocabSize, dim int)

	// SDPA performs Scaled Dot-Product Attention for decode (single query).
	// Q: [numQHeads, headDim] - single query token
	// K: [kvLen, numKVHeads, headDim] - key cache
	// V: [kvLen, numKVHeads, headDim] - value cache
	// out: [numQHeads, headDim]
	SDPA(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32)

	// SDPAPrefill performs SDPA for prefill with causal masking.
	// Q: [seqLen, numQHeads, headDim]
	// K: [seqLen, numKVHeads, headDim]
	// V: [seqLen, numKVHeads, headDim]
	// out: [seqLen, numQHeads, headDim]
	SDPAPrefill(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32)
}

// DeviceTensor represents a tensor allocated on a specific device.
// This bundles a DevicePtr with shape information.
type DeviceTensor struct {
	Ptr   tensor.DevicePtr
	Shape []int
	DType tensor.DType
}

// NumElements returns the total number of elements in the tensor.
func (t DeviceTensor) NumElements() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Bytes returns the size in bytes.
func (t DeviceTensor) Bytes() int {
	return t.NumElements() * t.DType.SizeBytes()
}
