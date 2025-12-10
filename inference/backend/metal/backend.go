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
	matmulPipeline           unsafe.Pointer
	matvecPipeline           unsafe.Pointer
	matvecQ4Pipeline         unsafe.Pointer
	softmaxPipeline          unsafe.Pointer
	rmsnormPipeline          unsafe.Pointer
	ropePipeline             unsafe.Pointer
	ropeGQAPipeline          unsafe.Pointer
	siluPipeline             unsafe.Pointer
	addPipeline              unsafe.Pointer
	mulPipeline              unsafe.Pointer
	sdpaDecodePipeline       unsafe.Pointer
	sdpaFlashDecodePipeline  unsafe.Pointer
	sdpaPrefillPipeline      unsafe.Pointer
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
	b.softmaxPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("softmax_f32"))
	b.rmsnormPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rmsnorm_f32"))
	b.ropePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_f32"))
	b.ropeGQAPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_gqa_f32"))
	b.siluPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_f32"))
	b.addPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_f32"))
	b.mulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("mul_f32"))
	b.sdpaDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_gqa_f32"))
	b.sdpaFlashDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_flash_decode_f32"))
	b.sdpaPrefillPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_prefill_f32"))

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
	if b.matvecQ4Pipeline == nil {
		panic("MatMulQ4_0 called but no matvecQ4Pipeline available")
	}
	if m == 1 {
		// Single row - use direct matvec
		C.metal_matvec_q4_0_transposed_f32(b.queue, b.matvecQ4Pipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	} else {
		// Multiple rows - use batched version with buffer offsets
		C.metal_matmul_q4_0_batched_f32(b.queue, b.matvecQ4Pipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(m), C.int(n), C.int(k))
	}
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
func (b *Backend) SDPAPrefill(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	C.metal_sdpa_prefill_f32(b.queue, b.sdpaPrefillPipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}
