//go:build metal && darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders
#include "metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// Ensure Backend implements the interface
var _ backend.Backend = (*Backend)(nil)

// bufferPool manages a pool of reusable Metal buffers by size.
type bufferPool struct {
	// Map from buffer size to list of available buffers
	available map[int][]unsafe.Pointer
	// Track all buffers currently in use (for ResetPool)
	inUse []unsafe.Pointer
	// Stats
	hits   int
	misses int
}

func newBufferPool() *bufferPool {
	return &bufferPool{
		available: make(map[int][]unsafe.Pointer),
	}
}

// Backend implements GPU acceleration using Apple Metal.
type Backend struct {
	device   unsafe.Pointer
	queue    unsafe.Pointer
	library  unsafe.Pointer
	deviceID int

	// Buffer pool for temporary allocations
	pool *bufferPool

	// Cached pipeline states
	matmulPipeline               unsafe.Pointer
	matvecPipeline               unsafe.Pointer
	matvecQ4Pipeline             unsafe.Pointer
	matvecQ4MultiOutputPipeline  unsafe.Pointer
	softmaxPipeline          unsafe.Pointer
	rmsnormPipeline          unsafe.Pointer
	ropePipeline             unsafe.Pointer
	ropeGQAPipeline          unsafe.Pointer
	siluPipeline             unsafe.Pointer
	siluMulPipeline          unsafe.Pointer
	addPipeline              unsafe.Pointer
	mulPipeline              unsafe.Pointer
	sdpaDecodePipeline        unsafe.Pointer
	sdpaFlashDecodePipeline   unsafe.Pointer
	sdpaPrefillPipeline       unsafe.Pointer
	flashAttention2Pipeline   unsafe.Pointer
	matmulQ4BatchedPipeline   unsafe.Pointer
	matmulQ4SimdgroupPipeline unsafe.Pointer
	matvecQ6KPipeline         unsafe.Pointer
	matvecQ4KPipeline         unsafe.Pointer

	// FP16 (Half-Precision) pipelines
	addF16Pipeline      unsafe.Pointer
	mulF16Pipeline      unsafe.Pointer
	siluF16Pipeline     unsafe.Pointer
	siluMulF16Pipeline  unsafe.Pointer
	rmsnormF16Pipeline   unsafe.Pointer
	matvecQ4F16Pipeline  unsafe.Pointer
	sdpaDecodeF16Pipeline unsafe.Pointer
}

// NewBackend creates a new Metal backend.
func NewBackend(deviceID int) (*Backend, error) {
	b := &Backend{
		deviceID: deviceID,
		pool:     newBufferPool(),
	}

	// Initialize Metal device
	b.device = C.metal_create_device()
	if b.device == nil {
		return nil, fmt.Errorf("no Metal device available")
	}

	b.queue = C.metal_create_command_queue(b.device)
	if b.queue == nil {
		return nil, fmt.Errorf("failed to create command queue")
	}

	// Compile shader library
	b.library = C.metal_compile_library(b.device, nil)
	if b.library == nil {
		return nil, fmt.Errorf("failed to compile Metal shaders")
	}

	// Create pipeline states
	b.matmulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_transposed_f32"))
	b.matvecPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_transposed_f32"))
	b.matvecQ4Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_transposed_f32"))
	b.matvecQ4MultiOutputPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_multi_output_f32"))
	b.softmaxPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("softmax_f32"))
	b.rmsnormPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rmsnorm_f32"))
	b.ropePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_f32"))
	b.ropeGQAPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_gqa_f32"))
	b.siluPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_f32"))
	b.siluMulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_mul_f32"))
	b.addPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_f32"))
	b.mulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("mul_f32"))
	b.sdpaDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_gqa_f32"))
	b.sdpaFlashDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_flash_decode_f32"))
	b.sdpaPrefillPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_prefill_f32"))
	b.flashAttention2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("flash_attention_2_f32"))
	b.matmulQ4BatchedPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_q4_0_batched_f32"))
	b.matmulQ4SimdgroupPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_q4_0_simdgroup_f32"))
	b.matvecQ6KPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q6k_multi_output_f32"))
	b.matvecQ4KPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4k_multi_output_f32"))

	// FP16 pipelines
	b.addF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_f16"))
	b.mulF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("mul_f16"))
	b.siluF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_f16"))
	b.siluMulF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_mul_f16"))
	b.rmsnormF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rmsnorm_f16"))
	b.matvecQ4F16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_f16"))
	b.sdpaDecodeF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_decode_f16"))

	return b, nil
}

// Close releases all Metal resources.
func (b *Backend) Close() {
	if b.library != nil {
		C.metal_release(b.library)
	}
	if b.queue != nil {
		C.metal_release(b.queue)
	}
	if b.device != nil {
		C.metal_release(b.device)
	}
}

// DeviceName returns the Metal device name.
func (b *Backend) DeviceName() string {
	name := C.metal_device_name(b.device)
	return C.GoString(name)
}

// Device returns the tensor device descriptor.
func (b *Backend) Device() tensor.Device {
	return tensor.NewDevice(tensor.Metal, b.deviceID)
}

// =============================================================================
// Memory Management
// =============================================================================

// Alloc allocates a Metal buffer and returns a DevicePtr.
// Uses buffer pooling to reuse buffers of the same size.
func (b *Backend) Alloc(bytes int) tensor.DevicePtr {
	// Check if we have a buffer of this exact size in the pool
	if buffers, ok := b.pool.available[bytes]; ok && len(buffers) > 0 {
		// Pop from available pool
		buf := buffers[len(buffers)-1]
		b.pool.available[bytes] = buffers[:len(buffers)-1]
		b.pool.inUse = append(b.pool.inUse, buf)
		return tensor.NewDevicePtr(tensor.Metal, uintptr(buf))
	}

	// No pooled buffer available, allocate new one
	buf := C.metal_alloc_buffer(b.device, C.size_t(bytes))
	if buf == nil {
		return tensor.DevicePtr{}
	}
	b.pool.inUse = append(b.pool.inUse, buf)
	return tensor.NewDevicePtr(tensor.Metal, uintptr(buf))
}

// AllocPermanent allocates a buffer that won't be recycled (for weights).
func (b *Backend) AllocPermanent(bytes int) tensor.DevicePtr {
	buf := C.metal_alloc_buffer(b.device, C.size_t(bytes))
	if buf == nil {
		return tensor.DevicePtr{}
	}
	return tensor.NewDevicePtr(tensor.Metal, uintptr(buf))
}

// Free releases a Metal buffer.
func (b *Backend) Free(ptr tensor.DevicePtr) {
	if !ptr.IsNil() {
		C.metal_release(unsafe.Pointer(ptr.Addr()))
	}
}

// ResetPool returns all in-use buffers to the pool for reuse.
// Call this at the start of each forward pass.
func (b *Backend) ResetPool() {
	for _, buf := range b.pool.inUse {
		// Get buffer size to categorize it
		size := int(C.metal_buffer_size(buf))
		b.pool.available[size] = append(b.pool.available[size], buf)
	}
	b.pool.inUse = b.pool.inUse[:0]
}

// ToDevice copies data from host to device.
func (b *Backend) ToDevice(dst tensor.DevicePtr, src []byte) {
	if dst.IsNil() || len(src) == 0 {
		return
	}
	C.metal_copy_to_buffer(unsafe.Pointer(dst.Addr()), unsafe.Pointer(&src[0]), C.size_t(len(src)))
}

// ToHost copies data from device to host.
func (b *Backend) ToHost(dst []byte, src tensor.DevicePtr) {
	if src.IsNil() || len(dst) == 0 {
		return
	}
	C.metal_copy_from_buffer(unsafe.Pointer(&dst[0]), unsafe.Pointer(src.Addr()), C.size_t(len(dst)))
}

// Sync waits for all pending GPU operations to complete.
func (b *Backend) Sync() {
	C.metal_sync(b.queue)
}

// BeginBatch starts a batch of operations that will share a single command buffer.
// This reduces commit overhead by batching multiple kernel dispatches together.
// Call EndBatch when done to commit all operations.
func (b *Backend) BeginBatch() {
	C.metal_begin_batch(b.queue)
}

// EndBatch commits all batched operations.
func (b *Backend) EndBatch() {
	C.metal_end_batch()
}

// CopyBuffer copies data from one GPU buffer to another (GPU-to-GPU).
func (b *Backend) CopyBuffer(src tensor.DevicePtr, srcOffset int, dst tensor.DevicePtr, dstOffset int, size int) {
	C.metal_copy_buffer(b.queue, unsafe.Pointer(src.Addr()), C.size_t(srcOffset),
		unsafe.Pointer(dst.Addr()), C.size_t(dstOffset), C.size_t(size))
}

// =============================================================================
// Compute Kernels
// =============================================================================

// MatMul performs C = A @ B where A is [M,K], B is [K,N], C is [M,N].
func (b *Backend) MatMul(a, bMat, out tensor.DevicePtr, m, n, k int) {
	C.metal_matmul_f32(b.queue, b.matmulPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(m), C.int(n), C.int(k))
}

// MatMulTransposed performs C = A @ B^T where A is [M,K], B is [N,K], C is [M,N].
func (b *Backend) MatMulTransposed(a, bMat, out tensor.DevicePtr, m, n, k int) {
	if m == 1 && b.matvecPipeline != nil {
		// Use optimized matrix-vector kernel for single-row case
		C.metal_matvec_transposed_f32(b.queue, b.matvecPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	} else {
		C.metal_matmul_f32(b.queue, b.matmulPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(m), C.int(n), C.int(k))
	}
}

// MatMulQ4_0 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_0 format.
// B contains raw Q4_0 data (18 bytes per 32 elements).
func (b *Backend) MatMulQ4_0(a, bMat, out tensor.DevicePtr, m, n, k int) {
	if b.matvecQ4MultiOutputPipeline == nil {
		panic("MatMulQ4_0 called but no matvecQ4MultiOutputPipeline available")
	}
	if m == 1 {
		// Single row - use multi-output matvec for better thread utilization
		// Each threadgroup computes 8 outputs, each simdgroup handles one output
		C.metal_matvec_q4_0_multi_output_f32(b.queue, b.matvecQ4MultiOutputPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	} else if m >= 8 && b.matmulQ4SimdgroupPipeline != nil {
		// Use simdgroup_matrix kernel for larger batches (prefill)
		// simdgroup_matrix hardware units provide better compute throughput
		C.metal_matmul_q4_0_simdgroup_f32(b.queue, b.matmulQ4SimdgroupPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(m), C.int(n), C.int(k))
	} else {
		// Small batch (2-7 rows) - use simple batched kernel
		C.metal_matmul_q4_0_batched_f32(b.queue, b.matmulQ4BatchedPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(m), C.int(n), C.int(k))
	}
}

// MatMulQ6_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q6_K format.
// B contains raw Q6_K data (210 bytes per 256 elements).
// Only supports M=1 (matvec) for now - used for lm_head during decode.
func (b *Backend) MatMulQ6_K(a, bMat, out tensor.DevicePtr, m, n, k int) {
	if b.matvecQ6KPipeline == nil {
		panic("MatMulQ6_K called but no matvecQ6KPipeline available")
	}
	if m != 1 {
		panic("MatMulQ6_K only supports M=1 (matvec) for now")
	}
	C.metal_matvec_q6k_multi_output_f32(b.queue, b.matvecQ6KPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(n), C.int(k))
}

// MatMulQ4_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_K format.
// B contains raw Q4_K data (144 bytes per 256 elements).
// Only supports M=1 (matvec) for now.
func (b *Backend) MatMulQ4_K(a, bMat, out tensor.DevicePtr, m, n, k int) {
	if b.matvecQ4KPipeline == nil {
		panic("MatMulQ4_K called but no matvecQ4KPipeline available")
	}
	if m != 1 {
		panic("MatMulQ4_K only supports M=1 (matvec) for now")
	}
	C.metal_matvec_q4k_multi_output_f32(b.queue, b.matvecQ4KPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(n), C.int(k))
}

// RMSNorm performs RMS normalization.
func (b *Backend) RMSNorm(x, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	C.metal_rmsnorm_f32(b.queue, b.rmsnormPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(weight.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols), C.float(eps))
}

// RoPE applies rotary position encoding.
func (b *Backend) RoPE(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos int, theta float32) {
	// Use GQA-aware RoPE kernel which handles Q and K head counts separately
	C.metal_rope_gqa_f32(b.queue, b.ropeGQAPipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		C.int(seqLen), C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.int(startPos), C.float(theta))
}

// Softmax applies softmax row-wise.
func (b *Backend) Softmax(x, out tensor.DevicePtr, rows, cols int) {
	C.metal_softmax_f32(b.queue, b.softmaxPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols))
}

// SiLU applies the SiLU activation function.
func (b *Backend) SiLU(x, out tensor.DevicePtr, n int) {
	C.metal_silu_f32(b.queue, b.siluPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// SiLUMul performs fused silu(gate) * up operation.
func (b *Backend) SiLUMul(gate, up, out tensor.DevicePtr, n int) {
	C.metal_silu_mul_f32(b.queue, b.siluMulPipeline,
		unsafe.Pointer(gate.Addr()), unsafe.Pointer(up.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// Add performs element-wise addition.
func (b *Backend) Add(a, bIn, out tensor.DevicePtr, n int) {
	C.metal_add_f32(b.queue, b.addPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// Mul performs element-wise multiplication.
func (b *Backend) Mul(a, bIn, out tensor.DevicePtr, n int) {
	C.metal_mul_f32(b.queue, b.mulPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// Embedding performs embedding lookup.
// ids should be int32 values in an MTLBuffer on device
func (b *Backend) Embedding(ids tensor.DevicePtr, numTokens int, table, out tensor.DevicePtr, vocabSize, dim int) {
	C.metal_embedding_f32(b.queue,
		unsafe.Pointer(ids.Addr()), unsafe.Pointer(table.Addr()), unsafe.Pointer(out.Addr()),
		C.int(numTokens), C.int(vocabSize), C.int(dim))
}

// SDPA performs scaled dot-product attention for decode (single query token).
// Uses Flash Decoding for longer sequences, naive for short ones.
func (b *Backend) SDPA(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32) {
	// Use Flash Decoding for longer KV lengths where parallelism helps
	// For short sequences, the overhead of threadgroup sync isn't worth it
	useFlash := b.sdpaFlashDecodePipeline != nil && kvLen >= 16

	if useFlash {
		C.metal_sdpa_flash_decode_f32(b.queue, b.sdpaFlashDecodePipeline,
			unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
			unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
			C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale))
	} else {
		// Use naive SDPA for short sequences
		C.metal_sdpa_decode_f32(b.queue, b.sdpaDecodePipeline,
			unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
			unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
			C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale))
	}
}

// SDPAPrefill performs SDPA for prefill with causal masking.
// Uses Flash Attention 2 for longer sequences where K/V tiling provides benefit.
func (b *Backend) SDPAPrefill(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	// Use Flash Attention 2 for longer sequences (>=256 tokens) where tiling helps
	// FA2 tiles K/V in shared memory and exploits GQA to share loads across Q heads
	if b.flashAttention2Pipeline != nil && seqLen >= 256 {
		C.metal_flash_attention_2_f32(b.queue, b.flashAttention2Pipeline,
			unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
			unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
			C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale))
		return
	}
	C.metal_sdpa_prefill_f32(b.queue, b.sdpaPrefillPipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}

// FlashAttention2 performs optimized SDPA with K/V tiling in shared memory.
// Uses larger tiles (64 K positions) and exploits GQA to share K/V loads.
func (b *Backend) FlashAttention2(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	C.metal_flash_attention_2_f32(b.queue, b.flashAttention2Pipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}

// =============================================================================
// FP16 (Half-Precision) Operations
// These provide 2x memory bandwidth for memory-bound operations.
// =============================================================================

// AddF16 performs element-wise addition on FP16 data.
func (b *Backend) AddF16(a, bIn, out tensor.DevicePtr, n int) {
	C.metal_add_f16(b.queue, b.addF16Pipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// MulF16 performs element-wise multiplication on FP16 data.
func (b *Backend) MulF16(a, bIn, out tensor.DevicePtr, n int) {
	C.metal_mul_f16(b.queue, b.mulF16Pipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// SiLUF16 applies the SiLU activation function on FP16 data.
func (b *Backend) SiLUF16(x, out tensor.DevicePtr, n int) {
	C.metal_silu_f16(b.queue, b.siluF16Pipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// SiLUMulF16 performs fused silu(gate) * up operation on FP16 data.
func (b *Backend) SiLUMulF16(gate, up, out tensor.DevicePtr, n int) {
	C.metal_silu_mul_f16(b.queue, b.siluMulF16Pipeline,
		unsafe.Pointer(gate.Addr()), unsafe.Pointer(up.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// RMSNormF16 performs RMS normalization with FP16 input/output.
// x: [rows, cols] in FP16, weight: [cols] in FP32, out: [rows, cols] in FP16
func (b *Backend) RMSNormF16(x, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	C.metal_rmsnorm_f16(b.queue, b.rmsnormF16Pipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(weight.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols), C.float(eps))
}

// MatMulQ4_0_F16 performs C = A @ B^T where A is FP16, B is Q4_0, C is FP16.
// A: [1, K] in FP16, B: [N, K] in Q4_0 format, C: [1, N] in FP16.
// This provides 2x activation bandwidth savings while maintaining Q4_0 weight compression.
func (b *Backend) MatMulQ4_0_F16(a, bMat, out tensor.DevicePtr, n, k int) {
	C.metal_matvec_q4_0_f16(b.queue, b.matvecQ4F16Pipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(n), C.int(k))
}

// SDPAF16 performs scaled dot-product attention with FP16 KV cache.
// Q: [numQHeads, headDim], K/V: [kvLen, numKVHeads, headDim], out: [numQHeads, headDim]
// All tensors in FP16. Provides 2x KV cache bandwidth savings.
func (b *Backend) SDPAF16(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32) {
	C.metal_sdpa_decode_f16(b.queue, b.sdpaDecodeF16Pipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}
