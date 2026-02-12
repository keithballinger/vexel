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
	// CopyBufferBatched copies data from one GPU buffer to another, integrating with command batching.
	// When batching is active, this avoids creating a separate command buffer and sync overhead.
	CopyBufferBatched(src tensor.DevicePtr, srcOffset int, dst tensor.DevicePtr, dstOffset int, size int)
}

// QuantizedMatMul is an optional interface for backends that support quantized matrix operations.
// Backends that don't implement this will fall back to dequantized (F32) operations.
type QuantizedMatMul interface {
	// MatMulQ4_0 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_0 format.
	// B contains raw Q4_0 data (18 bytes per 32 elements: 2 byte f16 scale + 16 bytes nibbles).
	MatMulQ4_0(a, b, out tensor.DevicePtr, m, n, k int)

	// MatMulQ4_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_K format.
	// B contains raw Q4_K data (144 bytes per 256 elements: 4 byte header + 12 scales + 128 qs).
	// Currently only supports M=1 (matvec for decode).
	MatMulQ4_K(a, b, out tensor.DevicePtr, m, n, k int)

	// MatMulQ6_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q6_K format.
	// B contains raw Q6_K data (210 bytes per 256 elements). Used for lm_head.
	// Currently only supports M=1 (matvec for decode).
	MatMulQ6_K(a, b, out tensor.DevicePtr, m, n, k int)

	// MatMulQ5_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q5_K format.
	// B contains raw Q5_K data (176 bytes per 256 elements).
	// Currently only supports M=1 (matvec for decode).
	MatMulQ5_K(a, b, out tensor.DevicePtr, m, n, k int)
}

// FP16Ops is an optional interface for backends that support FP16 (half-precision) operations.
// FP16 provides 2x memory bandwidth savings for memory-bound operations.
type FP16Ops interface {
	// ConvertF32ToF16 converts FP32 data to FP16.
	// in: [n] in FP32, out: [n] in FP16 (buffer must be n*2 bytes)
	ConvertF32ToF16(in, out tensor.DevicePtr, n int)

	// ConvertF16ToF32 converts FP16 data to FP32.
	// in: [n] in FP16, out: [n] in FP32 (buffer must be n*4 bytes)
	ConvertF16ToF32(in, out tensor.DevicePtr, n int)

	// SDPAF16 performs SDPA with FP16 KV cache.
	// Q: [numQHeads, headDim] in FP16
	// K/V: [numKVHeads, kvLen, headDim] in FP16 (head-major layout)
	// out: [numQHeads, headDim] in FP16
	// kvHeadStride: stride between KV heads in elements (maxSeqLen * headDim)
	SDPAF16(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int)

	// SDPAPrefillF16 performs prefill SDPA with FP16 activations (Flash Attention 2).
	// Q, K, V, out: [seqLen, numHeads, headDim] in FP16
	SDPAPrefillF16(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32)

	// RoPEF16 applies Rotary Position Embeddings in-place to FP16 Q and K.
	// Computation done in FP32 for numerical stability, I/O in FP16.
	// q: [seqLen, numHeads, headDim] in FP16, k: [seqLen, numKVHeads, headDim] in FP16
	// ropeDim: number of dimensions to rotate (0 = full headDim)
	// ropeNeox: true = NEOX-style (split pairs: i, i+dim/2), false = LLaMA-style (interleaved: 2i, 2i+1)
	RoPEF16(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float32, ropeNeox bool)

	// ScatterKVF16 transposes KV data from [newTokens, numKVHeads, headDim] to [numKVHeads, maxSeqLen, headDim].
	// This efficiently populates the head-major KV cache layout in a single kernel dispatch.
	ScatterKVF16(src, dst tensor.DevicePtr, newTokens, numKVHeads, headDim, maxSeqLen, seqPos int)

	// ScatterKVF32ToF16 transposes and converts KV data from F32 to F16.
	// src: [newTokens, numKVHeads, headDim] in F32
	// dst: [numKVHeads, maxSeqLen, headDim] in F16
	ScatterKVF32ToF16(src, dst tensor.DevicePtr, newTokens, numKVHeads, headDim, maxSeqLen, seqPos int)
}

// KVScatter is an optional interface for backends that support efficient KV cache population.
type KVScatter interface {
	// ScatterKV transposes KV data from [newTokens, numKVHeads, headDim] to [numKVHeads, maxSeqLen, headDim].
	ScatterKV(src, dst tensor.DevicePtr, newTokens, numKVHeads, headDim, maxSeqLen, seqPos int)
}

// FusedOps is an optional interface for backends that support fused kernel operations.
// Fused kernels combine multiple operations to reduce memory bandwidth.
type FusedOps interface {
	// AddRMSNorm performs fused residual addition + RMSNorm.
	// x = x + residual (in-place), then out = RMSNorm(x, weight)
	// x, residual: [rows, cols], weight: [cols], out: [rows, cols]
	AddRMSNorm(x, residual, weight, out tensor.DevicePtr, rows, cols int, eps float32)

	// MatMulQ4_0_FusedRMSNorm performs RMSNorm on x, then Q4_0 MatVec.
	// x: [1, K] (or [M, K] in future), normWeight: [K]
	// wMat: [N, K] Q4_0, out: [1, N]
	// Computes: out = (RMSNorm(x, normWeight)) @ wMat^T
	MatMulQ4_0_FusedRMSNorm(x, normWeight, wMat, out tensor.DevicePtr, m, n, k int, eps float32)

	// MatMulQ4_0_FusedRMSNormF16 performs RMSNorm on x, then Q4_0 MatVec with FP16 output.
	// This eliminates FP32->FP16 conversion for QKV projections when using FP16 KV cache.
	// x: [1, K] in FP32, normWeight: [K], wMat: [N, K] Q4_0, out: [1, N] in FP16
	MatMulQ4_0_FusedRMSNormF16(x, normWeight, wMat, out tensor.DevicePtr, m, n, k int, eps float32)
}

// Q8_0Ops is an optional interface for backends that support Q8_0 quantized KV cache.
// Q8_0 format: 34 bytes per 32 elements (2-byte f16 scale + 32 int8 values).
// Provides 4x memory savings vs FP32 with minimal accuracy loss.
type Q8_0Ops interface {
	// QuantizeF32ToQ8_0 quantizes FP32 data to Q8_0 format.
	// in: [n] in FP32 (n must be multiple of 32)
	// out: [n/32 * 34] bytes in Q8_0 format
	QuantizeF32ToQ8_0(in, out tensor.DevicePtr, n int)

	// DequantizeQ8_0ToF32 dequantizes Q8_0 data to FP32.
	// in: [n/32 * 34] bytes in Q8_0 format
	// out: [n] in FP32
	DequantizeQ8_0ToF32(in, out tensor.DevicePtr, n int)

	// SDPAQ8_0 performs SDPA with Q8_0 KV cache.
	// Q: [numQHeads, headDim] in FP32
	// K/V: [kvLen, numKVHeads, headDim] in Q8_0 format
	// out: [numQHeads, headDim] in FP32
	SDPAQ8_0(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32)
}

// ArgmaxOps is an optional interface for backends that support GPU-side argmax.
// This avoids transferring large logits arrays to CPU for greedy sampling.
type ArgmaxOps interface {
	// Argmax returns the index of the maximum value in the input tensor.
	// input: [n] in FP32, returns index of maximum element
	Argmax(input tensor.DevicePtr, n int) int
}

// LayerNormOps is an optional interface for backends that support LayerNorm.
// Required for Phi, GPT-2, and other architectures that use LayerNorm instead of RMSNorm.
type LayerNormOps interface {
	// LayerNorm performs Layer Normalization with mean subtraction.
	// x: [rows, cols], weight: [cols], bias: [cols], out: [rows, cols]
	// Computes: out = (x - mean) / sqrt(var + eps) * weight + bias
	LayerNorm(x, weight, bias, out tensor.DevicePtr, rows, cols int, eps float32)
}

// GELUOps is an optional interface for backends that support GELU activation.
// Required for Phi, GPT-2, BERT, and other architectures that use GELU instead of SiLU.
type GELUOps interface {
	// GELU applies the Gaussian Error Linear Unit activation function.
	// Uses fast approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
	// x: [n], out: [n]
	GELU(x, out tensor.DevicePtr, n int)
}

// BiasOps is an optional interface for backends that support bias addition.
// Required for architectures with bias terms in linear projections (Phi, GPT-2).
type BiasOps interface {
	// AddBias performs row-wise bias addition: out[i] = x[i] + bias[i % cols]
	// x: [rows, cols], bias: [cols], out: [rows, cols]
	AddBias(x, bias, out tensor.DevicePtr, rows, cols int)
}

// TrainingOps is an optional interface for backends that support neural network training.
// These operations enable GPU-accelerated gradient computation and weight updates.
type TrainingOps interface {
	// ReLUInplace applies ReLU activation in-place: x = max(0, x)
	ReLUInplace(x tensor.DevicePtr, n int)

	// ReLUBackward applies ReLU gradient mask: dx *= (x > 0)
	// x: forward activations, dx: gradient (modified in place)
	ReLUBackward(x, dx tensor.DevicePtr, n int)

	// BatchedOuterProduct computes out[i,j] += sum_b(a[b,i] * b[b,j])
	// a: [batch, M], b: [batch, N], out: [M, N]
	// Used for computing weight gradients in backpropagation.
	BatchedOuterProduct(a, b, out tensor.DevicePtr, batch, M, N int)

	// SGDUpdate applies SGD weight update with weight decay: w = w*(1-lr*wd) - lr*grad
	SGDUpdate(w, grad tensor.DevicePtr, lr, weightDecay float32, n int)

	// Zero fills a buffer with zeros.
	Zero(x tensor.DevicePtr, n int)
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
	// ropeDim: number of dimensions to rotate (0 = full headDim for LLaMA-style,
	//          otherwise partial rotation for Phi-2 where only first ropeDim are rotated)
	// ropeNeox: true = NEOX-style (split pairs: i, i+dim/2), false = LLaMA-style (interleaved: 2i, 2i+1)
	RoPE(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float32, ropeNeox bool)

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
	// K: [numKVHeads, maxSeqLen, headDim] - key cache (head-major layout)
	// V: [numKVHeads, maxSeqLen, headDim] - value cache (head-major layout)
	// out: [numQHeads, headDim]
	// kvHeadStride: stride between KV heads (maxSeqLen * headDim)
	SDPA(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int)

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

// PagedKVOps is an optional interface for backends that support paged KV cache.
type PagedKVOps interface {
	// ReshapePagedKV copies and reshapes data into a paged KV cache.
	// src: [numTokens, numKVHeads, headDim] (source K or V data)
	// dstBase: Base pointer to the block pool
	// pageTable: [numTokens] int32 (physical block indices)
	// blockOffsets: [numTokens] int32 (token indices within blocks)
	// numTokens, numKVHeads, headDim: dimensions
	// blockSize: tokens per block
	// isValue: true if writing V, false for K
	ReshapePagedKV(src, dstBase, pageTable, blockOffsets tensor.DevicePtr, numTokens, numKVHeads, headDim, blockSize int, isValue bool)
}
