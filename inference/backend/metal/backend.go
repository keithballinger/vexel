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

	"vexel/inference/tensor"
)

// Backend implements GPU acceleration using Apple Metal.
type Backend struct {
	device   unsafe.Pointer
	queue    unsafe.Pointer
	library  unsafe.Pointer
	deviceID int

	// Cached pipeline states
	matmulPipeline      unsafe.Pointer
	softmaxPipeline     unsafe.Pointer
	rmsnormPipeline     unsafe.Pointer
	ropePipeline        unsafe.Pointer
	ropeGQAPipeline     unsafe.Pointer
	siluPipeline        unsafe.Pointer
	addPipeline         unsafe.Pointer
	mulPipeline         unsafe.Pointer
	sdpaDecodePipeline  unsafe.Pointer
	sdpaPrefillPipeline unsafe.Pointer
}

// NewBackend creates a new Metal backend.
func NewBackend(deviceID int) (*Backend, error) {
	b := &Backend{deviceID: deviceID}

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
	b.softmaxPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("softmax_f32"))
	b.rmsnormPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rmsnorm_f32"))
	b.ropePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_f32"))
	b.ropeGQAPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_gqa_f32"))
	b.siluPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_f32"))
	b.addPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_f32"))
	b.mulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("mul_f32"))
	b.sdpaDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_gqa_f32"))
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

// CreateStream creates a command queue (Metal's equivalent of a stream).
func (b *Backend) CreateStream() (interface{}, error) {
	queue := C.metal_create_command_queue(b.device)
	if queue == nil {
		return nil, fmt.Errorf("failed to create command queue")
	}
	return queue, nil
}

// AllocBuffer allocates a Metal buffer.
func (b *Backend) AllocBuffer(size int) unsafe.Pointer {
	return C.metal_alloc_buffer(b.device, C.size_t(size))
}

// FreeBuffer releases a Metal buffer.
func (b *Backend) FreeBuffer(buf unsafe.Pointer) {
	C.metal_release(buf)
}

// CopyToDevice copies data from host to device.
func (b *Backend) CopyToDevice(dst unsafe.Pointer, src []float32) {
	C.metal_copy_to_buffer(dst, unsafe.Pointer(&src[0]), C.size_t(len(src)*4))
}

// CopyFromDevice copies data from device to host.
func (b *Backend) CopyFromDevice(dst []float32, src unsafe.Pointer) {
	C.metal_copy_from_buffer(unsafe.Pointer(&dst[0]), src, C.size_t(len(dst)*4))
}

// Sync waits for all pending GPU operations to complete.
func (b *Backend) Sync() {
	C.metal_sync(b.queue)
}

// MatMul performs C = A @ B^T
func (b *Backend) MatMul(a, bMat, c unsafe.Pointer, M, N, K int) {
	C.metal_matmul_f32(b.queue, b.matmulPipeline, a, bMat, c, C.int(M), C.int(N), C.int(K))
}

// RMSNorm performs RMS normalization.
func (b *Backend) RMSNorm(x, weight, out unsafe.Pointer, batchSize, dim int, eps float32) {
	C.metal_rmsnorm_f32(b.queue, b.rmsnormPipeline, x, weight, out,
		C.int(batchSize), C.int(dim), C.float(eps))
}

// RoPE applies rotary position encoding.
func (b *Backend) RoPE(q, k unsafe.Pointer, batchSize, seqLen, numHeads, headDim, startPos int, theta float32) {
	C.metal_rope_f32(b.queue, b.ropePipeline, q, k,
		C.int(batchSize), C.int(seqLen), C.int(numHeads), C.int(headDim),
		C.int(startPos), C.float(theta))
}

// Softmax applies softmax along the last dimension.
func (b *Backend) Softmax(x, out unsafe.Pointer, batchSize, dim int) {
	C.metal_softmax_f32(b.queue, b.softmaxPipeline, x, out, C.int(batchSize), C.int(dim))
}

// SiLU applies the SiLU activation function.
func (b *Backend) SiLU(x, out unsafe.Pointer, n int) {
	C.metal_silu_f32(b.queue, b.siluPipeline, x, out, C.int(n))
}

// Add performs element-wise addition.
func (b *Backend) Add(a, bIn, out unsafe.Pointer, n int) {
	C.metal_add_f32(b.queue, b.addPipeline, a, bIn, out, C.int(n))
}

// Mul performs element-wise multiplication.
func (b *Backend) Mul(a, bIn, out unsafe.Pointer, n int) {
	C.metal_mul_f32(b.queue, b.mulPipeline, a, bIn, out, C.int(n))
}

// Embedding performs embedding lookup.
func (b *Backend) Embedding(tokens []int, table, out unsafe.Pointer, vocabSize, dim int) {
	// Convert tokens to C int array
	cTokens := make([]C.int, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.int(t)
	}
	C.metal_embedding_f32(b.queue, (*C.int)(&cTokens[0]), table, out,
		C.int(len(tokens)), C.int(vocabSize), C.int(dim))
}

// SDPADecode performs scaled dot-product attention for decode (single query token).
// Q: [numQHeads, headDim] - single query token
// K: [kvLen, numKVHeads, headDim] - key cache
// V: [kvLen, numKVHeads, headDim] - value cache
// out: [numQHeads, headDim] - output
func (b *Backend) SDPADecode(q, k, v, out unsafe.Pointer,
	kvLen, numQHeads, numKVHeads, headDim int, scale float32) {
	C.metal_sdpa_decode_f32(b.queue, b.sdpaDecodePipeline, q, k, v, out,
		C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}

// SDPAPrefill performs scaled dot-product attention for prefill with causal masking.
// Q: [seqLen, numQHeads, headDim]
// K: [seqLen, numKVHeads, headDim]
// V: [seqLen, numKVHeads, headDim]
// out: [seqLen, numQHeads, headDim]
func (b *Backend) SDPAPrefill(q, k, v, out unsafe.Pointer,
	seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	C.metal_sdpa_prefill_f32(b.queue, b.sdpaPrefillPipeline, q, k, v, out,
		C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}

// ScaledDotProductAttention is deprecated, use SDPADecode or SDPAPrefill instead.
func (b *Backend) ScaledDotProductAttention(q, k, v, out unsafe.Pointer,
	batchSize, numHeads, seqLen, headDim int, scale float32, causal bool) {
	// Legacy interface - does nothing
}
