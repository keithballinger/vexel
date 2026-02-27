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
	"math"
	"os"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// Ensure Backend implements the interface
var _ backend.Backend = (*Backend)(nil)
var _ backend.ScaledRoPEOps = (*Backend)(nil)

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

	// Dispatch profiler for kernel counting and allocation tracking
	profiler *DispatchProfiler

	// Scratch allocator for sub-allocating from a single MTLBuffer
	scratch *ScratchAllocator

	// Cached pipeline states
	matmulPipeline              unsafe.Pointer // For matmul_transposed_f32 (C = A @ B^T)
	matmulNonTransposedPipeline unsafe.Pointer // For matmul_f32 (C = A @ B)
	matvecPipeline              unsafe.Pointer
	matvecQ4Pipeline            unsafe.Pointer
	matvecQ4MultiOutputPipeline unsafe.Pointer
	matvecQ4NR2Pipeline         unsafe.Pointer
	matvecQ4NR4Pipeline         unsafe.Pointer
	matvecQ4CollabPipeline      unsafe.Pointer
	matvecQ4OptimizedPipeline   unsafe.Pointer
	matvecQ4FusedRMSNormPipeline    unsafe.Pointer
	matvecQ4FusedRMSNormF16Pipeline unsafe.Pointer // FP16 output version
	matvecQ4FusedMLPPipeline        unsafe.Pointer
	softmaxPipeline                 unsafe.Pointer
	rmsnormPipeline                 unsafe.Pointer
	layernormPipeline               unsafe.Pointer
	geluPipeline                    unsafe.Pointer
	geluMulPipeline                 unsafe.Pointer
	sdpaSoftCapDecodePipeline       unsafe.Pointer // SDPA decode with logit soft-capping (Gemma 2)
	sdpaSoftCapPrefillPipeline      unsafe.Pointer // SDPA prefill with logit soft-capping (Gemma 2)
	addBiasPipeline                 unsafe.Pointer
	addRMSNormPipeline              unsafe.Pointer
	ropePipeline                    unsafe.Pointer
	ropeGQAPipeline                 unsafe.Pointer
	ropeGQAScaledPipeline           unsafe.Pointer // Learned RoPE frequencies (Gemma 2)
	ropeGQAF16Pipeline              unsafe.Pointer // FP16 version
	siluPipeline                unsafe.Pointer
	siluMulPipeline             unsafe.Pointer
	addPipeline                 unsafe.Pointer
	mulPipeline                 unsafe.Pointer
	argmaxPipeline              unsafe.Pointer
	sdpaDecodePipeline          unsafe.Pointer
	sdpaFlashDecodePipeline     unsafe.Pointer
	sdpaPrefillPipeline         unsafe.Pointer
	flashAttention2Pipeline     unsafe.Pointer
	flashAttention2F16Pipeline  unsafe.Pointer
	matmulQ4BatchedPipeline     unsafe.Pointer
	matmulQ4SimdgroupPipeline   unsafe.Pointer
	matvecQ6KPipeline           unsafe.Pointer
			matvecQ6KNR2Pipeline        unsafe.Pointer // Optimized Q6_K with nr0=2
			matvecQ4KPipeline           unsafe.Pointer
			matvecQ4KNR2Pipeline        unsafe.Pointer
			matvecQ5KPipeline           unsafe.Pointer
		
			matvecQ5KNR2Pipeline        unsafe.Pointer
			matmulQ4KBatchedPipeline    unsafe.Pointer
		
	// FP16 (Half-Precision) pipelines
	addF16Pipeline          unsafe.Pointer
	mulF16Pipeline          unsafe.Pointer
	siluF16Pipeline         unsafe.Pointer
	siluMulF16Pipeline      unsafe.Pointer
	rmsnormF16Pipeline      unsafe.Pointer
	matvecQ4F16Pipeline         unsafe.Pointer
	sdpaDecodeF16Pipeline       unsafe.Pointer
	sdpaDecodeF16VecPipeline    unsafe.Pointer // Vectorized version
	sdpaDecodeF16HD64Pipeline   unsafe.Pointer // Specialized for headDim=64
	sdpaDecodeF16HD64SimdPipeline unsafe.Pointer // SIMD version
	convertF32ToF16Pipeline     unsafe.Pointer
	convertF16ToF32Pipeline     unsafe.Pointer
	scatterKVF16Pipeline        unsafe.Pointer
	scatterKVF32Pipeline        unsafe.Pointer
	scatterKVF32ToF16Pipeline   unsafe.Pointer

	// Q8_0 Quantization pipelines (for KV cache)
	quantizeF32ToQ8_0Pipeline   unsafe.Pointer
	dequantizeQ8_0ToF32Pipeline unsafe.Pointer
	sdpaDecodeQ8_0Pipeline      unsafe.Pointer

	// Q8_0 Matmul pipelines (for weight matrices)
	matvecQ8_0NR2Pipeline       unsafe.Pointer
	matmulQ8_0BatchedPipeline   unsafe.Pointer

	// BF16 Matmul pipelines (for weight matrices in BFloat16 format)
	matvecBF16NR2Pipeline     unsafe.Pointer
	matmulBF16BatchedPipeline unsafe.Pointer

	// Training pipelines (for Medusa heads)
	reluInplacePipeline         unsafe.Pointer
	reluBackwardPipeline        unsafe.Pointer
	batchedOuterProductPipeline unsafe.Pointer
	sgdUpdatePipeline           unsafe.Pointer
	zeroPipeline                unsafe.Pointer

	// Utility pipelines
	memcpyComputePipeline      unsafe.Pointer // Compute-based memory copy (avoids blit encoder)
	reshapePagedKVPipeline     unsafe.Pointer
	sdpaPagedDecodePipeline    unsafe.Pointer
}

var (
	fa2MinSeqLen     int
	fa2MinSeqLenOnce sync.Once
)

func flashAttentionMinSeqLen() int {
	fa2MinSeqLenOnce.Do(func() {
		// Default threshold: 16 to engage FA2 earlier (benefits long tiles on M-series GPUs)
		fa2MinSeqLen = 16
		if v := os.Getenv("VEXEL_FA2_MIN_SEQ"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				// Clamp to a practical minimum of 8 to avoid tiny-sequence overhead
				if n < 8 {
					n = 8
				}
				fa2MinSeqLen = n
			}
		}
	})
	return fa2MinSeqLen
}

// NewBackend creates a new Metal backend.
func NewBackend(deviceID int) (*Backend, error) {
	b := &Backend{
		deviceID: deviceID,
		pool:     newBufferPool(),
		profiler: NewDispatchProfiler(),
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
	b.matmulNonTransposedPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_f32"))
	b.matvecPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_transposed_f32"))
	b.matvecQ4Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_transposed_f32"))
	b.matvecQ4MultiOutputPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_multi_output_f32"))
	b.matvecQ4NR2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_nr2_f32"))
	b.matvecQ4NR4Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_nr4_f32"))
	b.matvecQ4CollabPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_collab_f32"))
	b.matvecQ4OptimizedPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_optimized_f32"))
	b.matvecQ4FusedRMSNormPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_fused_rmsnorm_f32"))
	b.matvecQ4FusedRMSNormF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_fused_rmsnorm_f16_out"))
	b.matvecQ4FusedMLPPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_fused_mlp_f32"))
	b.softmaxPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("softmax_f32"))
	b.rmsnormPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rmsnorm_f32"))
	b.layernormPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("layernorm_f32"))
	b.geluPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("gelu_f32"))
	b.geluMulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("gelu_mul_f32"))
	b.sdpaSoftCapDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_softcap_decode_f32"))
	b.sdpaSoftCapPrefillPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_softcap_prefill_f32"))
	b.addBiasPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_bias_f32"))
	b.addRMSNormPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_rmsnorm_f32"))
	b.ropePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_f32"))
	b.ropeGQAPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_gqa_f32"))
	b.ropeGQAScaledPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_gqa_scaled_f32"))
	b.ropeGQAF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_gqa_f16"))
	b.siluPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_f32"))
	b.siluMulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_mul_f32"))
	b.addPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_f32"))
	b.mulPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("mul_f32"))
	b.argmaxPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("argmax_f32"))
	b.sdpaDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_gqa_f32"))
	b.sdpaFlashDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_flash_decode_f32"))
	b.sdpaPrefillPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_prefill_f32"))
	b.flashAttention2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("flash_attention_2_f32"))
	b.flashAttention2F16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("flash_attention_2_f16"))
	b.matmulQ4BatchedPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_q4_0_batched_f32"))
	b.matmulQ4SimdgroupPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_q4_0_simdgroup_f32"))
	b.matvecQ6KPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q6k_multi_output_f32"))
	b.matvecQ6KNR2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q6k_nr2_f32"))
	b.matvecQ4KPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4k_multi_output_f32"))
	b.matvecQ4KNR2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4k_nr2_f32"))
	b.matvecQ5KPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q5k_multi_output_f32"))
	b.matvecQ5KNR2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q5k_nr2_f32"))
	b.matmulQ4KBatchedPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_q4k_batched_f32"))

	// Q8_0 matmul pipelines
	b.matvecQ8_0NR2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q8_0_nr2_f32"))
	b.matmulQ8_0BatchedPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_q8_0_batched_f32"))

	// BF16 matmul pipelines
	b.matvecBF16NR2Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_bf16_nr2_f32"))
	b.matmulBF16BatchedPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matmul_bf16_batched_f32"))

	// FP16 pipelines
	b.addF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("add_f16"))
	b.mulF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("mul_f16"))
	b.siluF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_f16"))
	b.siluMulF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_mul_f16"))
	b.rmsnormF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rmsnorm_f16"))
	b.matvecQ4F16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("matvec_q4_0_f16"))
	b.sdpaDecodeF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_decode_f16"))
	b.sdpaDecodeF16VecPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_decode_f16_vec"))
	b.sdpaDecodeF16HD64Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_decode_f16_hd64"))
	b.sdpaDecodeF16HD64SimdPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_decode_f16_hd64_simd"))
	b.convertF32ToF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("convert_f32_to_f16"))
	b.convertF16ToF32Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("convert_f16_to_f32"))
	b.scatterKVF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("scatter_kv_f16"))
	b.scatterKVF32Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("scatter_kv_f32"))
	b.scatterKVF32ToF16Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("scatter_kv_f32_to_f16"))

	// Q8_0 quantization pipelines
	b.quantizeF32ToQ8_0Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("quantize_f32_to_q8_0"))
	b.dequantizeQ8_0ToF32Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("dequantize_q8_0_to_f32"))
	b.sdpaDecodeQ8_0Pipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_decode_q8_0"))

	// Training pipelines
	b.reluInplacePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("relu_inplace_f32"))
	b.reluBackwardPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("relu_backward_f32"))
	b.batchedOuterProductPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("batched_outer_product_f32"))
	b.sgdUpdatePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sgd_update_f32"))
	b.zeroPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("zero_f32"))

	// Utility pipelines
	b.memcpyComputePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("memcpy_compute"))
	b.reshapePagedKVPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("reshape_paged_kv_f32"))
	b.sdpaPagedDecodePipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_paged_decode_f32"))

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

// DispatchProfiler returns the backend's dispatch profiler for kernel counting.
func (b *Backend) DispatchProfiler() *DispatchProfiler {
	return b.profiler
}

// InitScratch creates a scratch allocator with the given capacity.
// Once initialized, ScratchAlloc returns sub-regions of a single MTLBuffer.
func (b *Backend) InitScratch(capacity int) error {
	sa, err := NewScratchAllocator(b, capacity)
	if err != nil {
		return err
	}
	b.scratch = sa
	return nil
}

// ScratchAllocator returns the backend's scratch allocator (may be nil).
func (b *Backend) ScratchAllocator() *ScratchAllocator {
	return b.scratch
}

// ScratchAlloc allocates from the scratch buffer. Falls back to pool Alloc if
// scratch is not initialized or has insufficient space.
func (b *Backend) ScratchAlloc(bytes int) tensor.DevicePtr {
	if b.scratch != nil {
		ptr := b.scratch.Alloc(bytes)
		if !ptr.IsNil() {
			return ptr
		}
	}
	// Fallback to pool-based allocation
	return b.Alloc(bytes)
}

// ScratchReset resets the scratch allocator's bump pointer to 0.
// Call this at the start of each forward pass alongside ResetPool.
func (b *Backend) ScratchReset() {
	if b.scratch != nil {
		b.scratch.Reset()
	}
}

// =============================================================================
// Memory Management
// =============================================================================

// Alloc allocates a Metal buffer and returns a DevicePtr.
// Uses buffer pooling to reuse buffers of the same size.
func (b *Backend) Alloc(bytes int) tensor.DevicePtr {
	var start time.Time
	profiling := b.profiler.IsEnabled()
	if profiling {
		start = time.Now()
	}

	// Check if we have a buffer of this exact size in the pool
	if buffers, ok := b.pool.available[bytes]; ok && len(buffers) > 0 {
		// Pop from available pool
		buf := buffers[len(buffers)-1]
		b.pool.available[bytes] = buffers[:len(buffers)-1]
		b.pool.inUse = append(b.pool.inUse, buf)
		if profiling {
			b.profiler.RecordAlloc(bytes, true, time.Since(start))
		}
		return tensor.NewDevicePtr(tensor.Metal, uintptr(buf))
	}

	// No pooled buffer available, allocate new one
	buf := C.metal_alloc_buffer(b.device, C.size_t(bytes))
	if buf == nil {
		return tensor.DevicePtr{}
	}
	b.pool.inUse = append(b.pool.inUse, buf)
	if profiling {
		b.profiler.RecordAlloc(bytes, false, time.Since(start))
	}
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

// MemoryBarrier inserts a buffer-scope memory barrier in the current batch encoder.
// This ensures all prior buffer writes are visible to subsequent reads. Required when
// multiple dispatches share the same MTLBuffer (scratch allocator). No-op outside batch mode.
func (b *Backend) MemoryBarrier() {
	C.metal_memory_barrier()
}

// CopyBuffer copies data from one GPU buffer to another (GPU-to-GPU).
func (b *Backend) CopyBuffer(src tensor.DevicePtr, srcOffset int, dst tensor.DevicePtr, dstOffset int, size int) {
	C.metal_copy_buffer(b.queue, unsafe.Pointer(src.Addr()), C.size_t(srcOffset),
		unsafe.Pointer(dst.Addr()), C.size_t(dstOffset), C.size_t(size))
}

// CopyBufferBatched copies data integrating with command batching.
// VEXEL_USE_COMPUTE_COPY=1 uses compute-based copy (experimental).
// Default is blit-based copy which requires mid-layer sync for correct output.
func (b *Backend) CopyBufferBatched(src tensor.DevicePtr, srcOffset int, dst tensor.DevicePtr, dstOffset int, size int) {
	useComputeCopy := os.Getenv("VEXEL_USE_COMPUTE_COPY") == "1"
	if useComputeCopy {
		C.metal_copy_buffer_compute(b.queue, b.memcpyComputePipeline,
			unsafe.Pointer(src.Addr()), C.size_t(srcOffset),
			unsafe.Pointer(dst.Addr()), C.size_t(dstOffset), C.size_t(size))
	} else {
		C.metal_copy_buffer_batched(b.queue, unsafe.Pointer(src.Addr()), C.size_t(srcOffset),
			unsafe.Pointer(dst.Addr()), C.size_t(dstOffset), C.size_t(size))
	}
}

// GPUProfileStats holds GPU profiling statistics.
type GPUProfileStats struct {
	TotalTimeNs uint64
	BatchCount  uint64
	KernelCount uint64
	SyncTimeNs  uint64
}

// GetGPUProfile returns GPU profiling statistics (enable with VEXEL_GPU_PROFILE=1).
func GetGPUProfile() GPUProfileStats {
	var totalTime, batchCount, kernelCount, syncTime C.uint64_t
	C.metal_get_gpu_profile(&totalTime, &batchCount, &kernelCount, &syncTime)
	return GPUProfileStats{
		TotalTimeNs: uint64(totalTime),
		BatchCount:  uint64(batchCount),
		KernelCount: uint64(kernelCount),
		SyncTimeNs:  uint64(syncTime),
	}
}

// ResetGPUProfile clears GPU profiling statistics.
func ResetGPUProfile() {
	C.metal_reset_gpu_profile()
}

// PrintGPUProfile prints GPU profiling statistics to stderr.
func PrintGPUProfile() {
	stats := GetGPUProfile()
	if stats.BatchCount == 0 {
		return
	}
	gpuMs := float64(stats.TotalTimeNs) / 1e6
	syncMs := float64(stats.SyncTimeNs) / 1e6
	avgBatchUs := float64(stats.TotalTimeNs) / float64(stats.BatchCount) / 1e3
	fmt.Fprintf(os.Stderr, "[GPU PROFILE] GPU: %.2f ms, Sync: %.2f ms, Batches: %d, Avg: %.2f µs/batch\n",
		gpuMs, syncMs, stats.BatchCount, avgBatchUs)
}

// =============================================================================
// Compute Kernels
// =============================================================================

// MatMul performs C = A @ B where A is [M,K], B is [K,N], C is [M,N].
func (b *Backend) MatMul(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMul")
	C.metal_matmul_f32(b.queue, b.matmulNonTransposedPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(m), C.int(n), C.int(k))
}

// MatMulTransposed performs C = A @ B^T where A is [M,K], B is [N,K], C is [M,N].
func (b *Backend) MatMulTransposed(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMulTransposed")
	if m == 1 && b.matvecPipeline != nil {
		// Use optimized matrix-vector kernel for single-row case
		C.metal_matvec_transposed_f32(b.queue, b.matvecPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	} else {
		// Use transposed matmul kernel
		C.metal_matmul_transposed_f32(b.queue, b.matmulPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(m), C.int(n), C.int(k))
	}
}

// MatMulQ4_0 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_0 format.
// B contains raw Q4_0 data (18 bytes per 32 elements).
func (b *Backend) MatMulQ4_0(a, bMat, out tensor.DevicePtr, m, n, k int) {
	// Auto-detect scratch-allocated DevicePtrs (non-zero offset) and dispatch
	// to offset-aware C functions. Pool-allocated ptrs have offset=0 so this
	// is a no-op for the traditional path.
	if a.Offset() != 0 || out.Offset() != 0 {
		b.MatMulQ4_0Offset(a, bMat, out, m, n, k)
		return
	}
	b.profiler.RecordDispatch("MatMulQ4_0")
	if b.matvecQ4NR2Pipeline == nil {
		panic("MatMulQ4_0 called but no matvecQ4NR2Pipeline available")
	}
	if m == 1 {
		// For partial blocks (K not multiple of 32), fall back to the scalar-safe single-output kernel.
		if k%32 != 0 && b.matvecQ4Pipeline != nil {
			C.metal_matvec_q4_0_transposed_f32(b.queue, b.matvecQ4Pipeline,
				unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
				C.int(n), C.int(k))
			return
		}
		// Single row (decode) - use multi-output matvec (8 outputs per threadgroup)
		// Benchmarked: multi_output (8 out/TG) is faster than NR2 (16 out/TG) for Q4_0
		// because Q4_0's 18-byte block layout causes cache pressure when 2 rows share a simdgroup.
		C.metal_matvec_q4_0_multi_output_f32(b.queue, b.matvecQ4MultiOutputPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	} else if m >= 8 && b.matmulQ4SimdgroupPipeline != nil {
		// Use simdgroup_matrix kernel for prefill (M>=8)
		// 32×64 tiled kernel with 8 simdgroups, optimized for M=32/64/128
		C.metal_matmul_q4_0_simdgroup_f32(b.queue, b.matmulQ4SimdgroupPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(m), C.int(n), C.int(k))
	} else {
		// Fallback: simple batched kernel (one threadgroup per output element)
		C.metal_matmul_q4_0_batched_f32(b.queue, b.matmulQ4BatchedPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(m), C.int(n), C.int(k))
	}
}

// MatVecQ4_0MultiOutput executes the 8-output-per-threadgroup kernel explicitly.
// Primarily used for validation/testing of the multi-output implementation.
func (b *Backend) MatVecQ4_0MultiOutput(a, bMat, out tensor.DevicePtr, n, k int) {
	b.profiler.RecordDispatch("MatMulQ4_0")
	switch {
	case b.matvecQ4MultiOutputPipeline != nil:
		C.metal_matvec_q4_0_multi_output_f32(b.queue, b.matvecQ4MultiOutputPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	case b.matvecQ4NR2Pipeline != nil:
		// Fallback to NR2 path if multi-output pipeline is unavailable.
		C.metal_matvec_q4_0_nr2_f32(b.queue, b.matvecQ4NR2Pipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	case b.matvecQ4Pipeline != nil:
		// Last resort: single-output matvec.
		C.metal_matvec_q4_0_transposed_f32(b.queue, b.matvecQ4Pipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	default:
		panic("MatVecQ4_0MultiOutput called but no Q4 matvec pipeline available")
	}
}

// MatMulQ6_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q6_K format.
// B contains raw Q6_K data (210 bytes per 256 elements).
// Currently only supports M=1 (matvec) natively, uses loop for M>1.
var q6kNR2DebugOnce sync.Once

func (b *Backend) MatMulQ6_K(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMulQ6_K")
	if m == 1 {
		// Use optimized NR2 kernel (2 outputs per simdgroup) if available
		if b.matvecQ6KNR2Pipeline != nil {
			q6kNR2DebugOnce.Do(func() {
				fmt.Println("[DEBUG] Using Q6_K NR2 kernel for LM head")
			})
			C.metal_matvec_q6k_nr2_f32(b.queue, b.matvecQ6KNR2Pipeline,
				unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
				C.int(n), C.int(k))
			return
		}
		// Fall back to original kernel
		if b.matvecQ6KPipeline == nil {
			panic("MatMulQ6_K called but no Q6_K pipeline available")
		}
		C.metal_matvec_q6k_multi_output_f32(b.queue, b.matvecQ6KPipeline,
			unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
			C.int(n), C.int(k))
	} else {
		// Looped matvec for prefill
		stateRowBytes := uintptr(k * 4)
		outRowBytes := uintptr(n * 4)
		for i := 0; i < m; i++ {
			rowA := tensor.DevicePtrOffset(a, uintptr(i)*stateRowBytes)
			rowOut := tensor.DevicePtrOffset(out, uintptr(i)*outRowBytes)

			if b.matvecQ6KNR2Pipeline != nil {
				C.metal_matvec_q6k_nr2_f32(b.queue, b.matvecQ6KNR2Pipeline,
					unsafe.Pointer(rowA.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(rowOut.Addr()),
					C.int(n), C.int(k))
			} else {
				C.metal_matvec_q6k_multi_output_f32(b.queue, b.matvecQ6KPipeline,
					unsafe.Pointer(rowA.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(rowOut.Addr()),
					C.int(n), C.int(k))
			}
		}
	}
}

// MatMulQ4_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_K format.
// B contains raw Q4_K data (144 bytes per 256 elements).
func (b *Backend) MatMulQ4_K(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMulQ4_K")
	if m == 1 {
		// Decode: use optimized matvec kernel
		if b.matvecQ4KNR2Pipeline != nil {
			C.metal_matvec_q4k_nr2_f32(b.queue, b.matvecQ4KNR2Pipeline,
				unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
				unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
				unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
				C.int(n), C.int(k))
			return
		}
		if b.matvecQ4KPipeline == nil {
			panic("MatMulQ4_K called but no matvecQ4KPipeline available")
		}
		C.metal_matvec_q4k_multi_output_f32(b.queue, b.matvecQ4KPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(n), C.int(k))
	} else {
		// Prefill: use batched kernel
		if b.matmulQ4KBatchedPipeline == nil {
			panic("MatMulQ4_K called with M>1 but no matmulQ4KBatchedPipeline available")
		}
		C.metal_matmul_q4k_batched_f32(b.queue, b.matmulQ4KBatchedPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(m), C.int(n), C.int(k))
	}
}

// MatMulQ5_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q5_K format.
// B contains raw Q5_K data (176 bytes per 256 elements).
// Currently only supports M=1 (matvec) natively, uses loop for M>1.
func (b *Backend) MatMulQ5_K(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMulQ5_K")
	if m == 1 {
		// if b.matvecQ5KNR2Pipeline != nil {
		// 	offsets := []C.uint64_t{
		// 		C.uint64_t(a.Offset()),
		// 		C.uint64_t(bMat.Offset()),
		// 		C.uint64_t(out.Offset()),
		// 	}
		// 	C.metal_matvec_q5k_nr2_f32_v4(
		// 		b.queue,
		// 		b.matvecQ5KNR2Pipeline,
		// 		unsafe.Pointer(a.Addr()),
		// 		unsafe.Pointer(bMat.Addr()),
		// 		unsafe.Pointer(out.Addr()),
		// 		C.int(n),
		// 		C.int(k),
		// 		unsafe.Pointer(&offsets[0]))
		// 	return
		// }
		if b.matvecQ5KPipeline == nil {
			panic("MatMulQ5_K called but no matvecQ5KPipeline available")
		}
		C.metal_matvec_q5k_multi_output_f32(b.queue, b.matvecQ5KPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(n), C.int(k))
	} else {
		// Looped matvec for prefill
		stateRowBytes := uintptr(k * 4)
		outRowBytes := uintptr(n * 4)
		for i := 0; i < m; i++ {
			rowA := tensor.DevicePtrOffset(a, uintptr(i)*stateRowBytes)
			rowOut := tensor.DevicePtrOffset(out, uintptr(i)*outRowBytes)

			C.metal_matvec_q5k_multi_output_f32(b.queue, b.matvecQ5KPipeline,
				unsafe.Pointer(rowA.Addr()), C.uint64_t(rowA.Offset()),
				unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
				unsafe.Pointer(rowOut.Addr()), C.uint64_t(rowOut.Offset()),
				C.int(n), C.int(k))
		}
	}
}

// MatMulQ8_0 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q8_0 format.
// B contains raw Q8_0 data (34 bytes per 32 elements: 2 byte f16 scale + 32 int8 values).
// Track 4: Quantization Expansion, Phase 2 Task 2.
func (b *Backend) MatMulQ8_0(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMulQ8_0")
	if m == 1 {
		if b.matvecQ8_0NR2Pipeline == nil {
			panic("MatMulQ8_0 called but no matvecQ8_0NR2Pipeline available")
		}
		C.metal_matvec_q8_0_nr2_f32(b.queue, b.matvecQ8_0NR2Pipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(n), C.int(k))
	} else {
		if b.matmulQ8_0BatchedPipeline == nil {
			panic("MatMulQ8_0 batched called but no matmulQ8_0BatchedPipeline available")
		}
		C.metal_matmul_q8_0_batched_f32(b.queue, b.matmulQ8_0BatchedPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(m), C.int(n), C.int(k))
	}
}

// MatMulBF16 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in BF16 format.
// B contains raw BF16 data (2 bytes per element). Kernel converts BF16→F32 on the fly.
// Track 4: Quantization Expansion, Phase 2 Task 3.
func (b *Backend) MatMulBF16(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMulBF16")
	if m == 1 {
		if b.matvecBF16NR2Pipeline == nil {
			panic("MatMulBF16 called but no matvecBF16NR2Pipeline available")
		}
		C.metal_matvec_bf16_nr2_f32(b.queue, b.matvecBF16NR2Pipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(n), C.int(k))
	} else {
		if b.matmulBF16BatchedPipeline == nil {
			panic("MatMulBF16 batched called but no matmulBF16BatchedPipeline available")
		}
		C.metal_matmul_bf16_batched_f32(b.queue, b.matmulBF16BatchedPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()), C.uint64_t(bMat.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(m), C.int(n), C.int(k))
	}
}

// MatMulQ4_0_FusedRMSNorm performs fused RMSNorm(x) + Q4_0 MatVec.
// Only supports M=1 (decode) currently.
func (b *Backend) MatMulQ4_0_FusedRMSNorm(x, normWeight, wMat, out tensor.DevicePtr, m, n, k int, eps float32) {
	b.profiler.RecordDispatch("FusedRMSNorm+MatMul")
	if m != 1 {
		panic("MatMulQ4_0_FusedRMSNorm only supports M=1")
	}
	if b.matvecQ4FusedRMSNormPipeline == nil {
		panic("MatMulQ4_0_FusedRMSNorm called but pipeline unavailable")
	}
	if out.Offset() != 0 {
		C.metal_matvec_q4_0_fused_rmsnorm_f32_offset(b.queue, b.matvecQ4FusedRMSNormPipeline,
			unsafe.Pointer(x.Addr()), unsafe.Pointer(normWeight.Addr()),
			unsafe.Pointer(wMat.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(n), C.int(k), C.float(eps))
		return
	}
	C.metal_matvec_q4_0_fused_rmsnorm_f32(b.queue, b.matvecQ4FusedRMSNormPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(normWeight.Addr()),
		unsafe.Pointer(wMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(n), C.int(k), C.float(eps))
}

// MatMulQ4_0_FusedRMSNormF16 performs fused RMSNorm + Q4_0 matmul with FP16 output.
// Eliminates FP32->FP16 conversion after QKV projections.
func (b *Backend) MatMulQ4_0_FusedRMSNormF16(x, normWeight, wMat, out tensor.DevicePtr, m, n, k int, eps float32) {
	b.profiler.RecordDispatch("FusedRMSNorm+MatMul")
	if m != 1 {
		panic("MatMulQ4_0_FusedRMSNormF16 only supports M=1")
	}
	if b.matvecQ4FusedRMSNormF16Pipeline == nil {
		panic("MatMulQ4_0_FusedRMSNormF16 called but pipeline unavailable")
	}
	C.metal_matvec_q4_0_fused_rmsnorm_f16_out(b.queue, b.matvecQ4FusedRMSNormF16Pipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(normWeight.Addr()),
		unsafe.Pointer(wMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(n), C.int(k), C.float(eps))
}

// MatMulQ4_0_FusedMLP performs fused MLP: SiLU(x @ W1) * (x @ W3).
// Only supports M=1 (decode) currently.
func (b *Backend) MatMulQ4_0_FusedMLP(x, w1, w3, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("FusedMLP")
	if m != 1 {
		panic("MatMulQ4_0_FusedMLP only supports M=1")
	}
	if b.matvecQ4FusedMLPPipeline == nil {
		panic("MatMulQ4_0_FusedMLP called but pipeline unavailable")
	}
	if x.Offset() != 0 || out.Offset() != 0 {
		C.metal_matvec_q4_0_fused_mlp_f32_offset(b.queue, b.matvecQ4FusedMLPPipeline,
			unsafe.Pointer(x.Addr()), C.uint64_t(x.Offset()),
			unsafe.Pointer(w1.Addr()), unsafe.Pointer(w3.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(n), C.int(k))
		return
	}
	C.metal_matvec_q4_0_fused_mlp_f32(b.queue, b.matvecQ4FusedMLPPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(w1.Addr()),
		unsafe.Pointer(w3.Addr()), unsafe.Pointer(out.Addr()),
		C.int(n), C.int(k))
}

// RMSNorm performs RMS normalization.
func (b *Backend) RMSNorm(x, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	if x.Offset() != 0 || out.Offset() != 0 {
		b.RMSNormOffset(x, weight, out, rows, cols, eps)
		return
	}
	b.profiler.RecordDispatch("RMSNorm")
	C.metal_rmsnorm_f32(b.queue, b.rmsnormPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(weight.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols), C.float(eps))
}

// LayerNorm performs Layer normalization (for Phi-2 and similar architectures).
// out = (x - mean) / sqrt(var + eps) * weight + bias
func (b *Backend) LayerNorm(x, weight, bias, out tensor.DevicePtr, rows, cols int, eps float32) {
	b.profiler.RecordDispatch("LayerNorm")
	C.metal_layernorm_f32(b.queue, b.layernormPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(weight.Addr()),
		unsafe.Pointer(bias.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols), C.float(eps))
}

// GELU applies the GELU activation function.
// out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func (b *Backend) GELU(x, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("GELU")
	C.metal_gelu_f32(b.queue, b.geluPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// GELUMul performs fused gelu(gate) * up operation (GeGLU for Gemma).
func (b *Backend) GELUMul(gate, up, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("GELUMul")
	C.metal_gelu_mul_f32(b.queue, b.geluMulPipeline,
		unsafe.Pointer(gate.Addr()), unsafe.Pointer(up.Addr()),
		unsafe.Pointer(out.Addr()), C.int(n))
}

// AddBias adds bias to each row: out[row, col] = x[row, col] + bias[col]
func (b *Backend) AddBias(x, bias, out tensor.DevicePtr, rows, cols int) {
	b.profiler.RecordDispatch("AddBias")
	C.metal_add_bias_f32(b.queue, b.addBiasPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(bias.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols))
}

// AddRMSNorm performs fused residual addition + RMSNorm.
// x = x + residual (in-place), then out = RMSNorm(x, weight)
// This saves one memory round-trip compared to separate Add + RMSNorm.
func (b *Backend) AddRMSNorm(x, residual, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	b.profiler.RecordDispatch("AddRMSNorm")
	if residual.Offset() != 0 || out.Offset() != 0 {
		C.metal_add_rmsnorm_f32_offset(b.queue, b.addRMSNormPipeline,
			unsafe.Pointer(x.Addr()),
			unsafe.Pointer(residual.Addr()), C.uint64_t(residual.Offset()),
			unsafe.Pointer(weight.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(rows), C.int(cols), C.float(eps))
		return
	}
	C.metal_add_rmsnorm_f32(b.queue, b.addRMSNormPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(residual.Addr()),
		unsafe.Pointer(weight.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols), C.float(eps))
}

// RoPE applies rotary position encoding.
// ropeDim: dimensions to rotate (0 = full headDim). For partial RoPE like Phi-2.
// ropeNeox: true = NEOX-style (split pairs), false = LLaMA-style (interleaved pairs)
func (b *Backend) RoPE(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float32, ropeNeox bool) {
	if q.Offset() != 0 || k.Offset() != 0 {
		b.RoPEOffset(q, k, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim, theta, ropeNeox)
		return
	}
	b.profiler.RecordDispatch("RoPE")
	// If ropeDim is 0 or equals headDim, rotate all dimensions
	effectiveRopeDim := ropeDim
	if effectiveRopeDim == 0 {
		effectiveRopeDim = headDim
	}
	// Use GQA-aware RoPE kernel which handles Q and K head counts separately
	neoxFlag := 0
	if ropeNeox {
		neoxFlag = 1
	}
	C.metal_rope_gqa_f32(b.queue, b.ropeGQAPipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		C.int(seqLen), C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.int(startPos), C.int(effectiveRopeDim), C.float(theta), C.int(neoxFlag))
}

// RoPEF16 applies rotary position encoding with FP16 inputs/outputs.
// Computation done in FP32 for numerical stability, I/O in FP16.
// ropeDim: dimensions to rotate (0 = full headDim). For partial RoPE like Phi-2.
// ropeNeox: true = NEOX-style (split pairs), false = LLaMA-style (interleaved pairs)
func (b *Backend) RoPEF16(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float32, ropeNeox bool) {
	b.profiler.RecordDispatch("RoPE")
	if b.ropeGQAF16Pipeline == nil {
		panic("RoPEF16 called but pipeline unavailable")
	}
	effectiveRopeDim := ropeDim
	if effectiveRopeDim == 0 {
		effectiveRopeDim = headDim
	}
	neoxFlag := 0
	if ropeNeox {
		neoxFlag = 1
	}
	C.metal_rope_gqa_f16(b.queue, b.ropeGQAF16Pipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		C.int(seqLen), C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.int(startPos), C.int(effectiveRopeDim), C.float(theta), C.int(neoxFlag))
}

// RoPEWithFreqs applies rotary position encoding using pre-computed per-dimension
// inverse frequencies from a device buffer. Used for learned RoPE scaling (Gemma 2).
// freqs: [headDim/2] float32 pre-computed inverse frequencies on device.
func (b *Backend) RoPEWithFreqs(q, k, freqs tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos int, ropeNeox bool) {
	b.profiler.RecordDispatch("RoPEScaled")
	if b.ropeGQAScaledPipeline == nil {
		panic("RoPEWithFreqs called but pipeline unavailable")
	}
	neoxFlag := 0
	if ropeNeox {
		neoxFlag = 1
	}
	C.metal_rope_gqa_scaled_f32(b.queue, b.ropeGQAScaledPipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()), unsafe.Pointer(freqs.Addr()),
		C.int(seqLen), C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.int(startPos), C.int(neoxFlag))
}

// Softmax applies softmax row-wise.
func (b *Backend) Softmax(x, out tensor.DevicePtr, rows, cols int) {
	b.profiler.RecordDispatch("Softmax")
	C.metal_softmax_f32(b.queue, b.softmaxPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols))
}

// SiLU applies the SiLU activation function.
func (b *Backend) SiLU(x, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("SiLU")
	C.metal_silu_f32(b.queue, b.siluPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// SiLUMul performs fused silu(gate) * up operation.
func (b *Backend) SiLUMul(gate, up, out tensor.DevicePtr, n int) {
	if gate.Offset() != 0 || up.Offset() != 0 || out.Offset() != 0 {
		b.SiLUMulOffset(gate, up, out, n)
		return
	}
	b.profiler.RecordDispatch("SiLUMul")
	C.metal_silu_mul_f32(b.queue, b.siluMulPipeline,
		unsafe.Pointer(gate.Addr()), unsafe.Pointer(up.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// Add performs element-wise addition.
func (b *Backend) Add(a, bIn, out tensor.DevicePtr, n int) {
	if a.Offset() != 0 || bIn.Offset() != 0 || out.Offset() != 0 {
		b.AddOffset(a, bIn, out, n)
		return
	}
	b.profiler.RecordDispatch("Add")
	C.metal_add_f32(b.queue, b.addPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// AddOffset performs element-wise addition using offset-aware dispatch.
// This allows operating on sub-regions of a shared MTLBuffer (scratch allocation).
func (b *Backend) AddOffset(a, bIn, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Add")
	C.metal_add_f32_offset(b.queue, b.addPipeline,
		unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
		unsafe.Pointer(bIn.Addr()), C.uint64_t(bIn.Offset()),
		unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()), C.int(n))
}

// RMSNormOffset performs RMS normalization using offset-aware dispatch.
func (b *Backend) RMSNormOffset(x, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	b.profiler.RecordDispatch("RMSNorm")
	C.metal_rmsnorm_f32_offset(b.queue, b.rmsnormPipeline,
		unsafe.Pointer(x.Addr()), C.uint64_t(x.Offset()),
		unsafe.Pointer(weight.Addr()), C.uint64_t(weight.Offset()),
		unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
		C.int(rows), C.int(cols), C.float(eps))
}

// SiLUMulOffset performs fused silu(gate) * up using offset-aware dispatch.
func (b *Backend) SiLUMulOffset(gate, up, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("SiLUMul")
	C.metal_silu_mul_f32_offset(b.queue, b.siluMulPipeline,
		unsafe.Pointer(gate.Addr()), C.uint64_t(gate.Offset()),
		unsafe.Pointer(up.Addr()), C.uint64_t(up.Offset()),
		unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()), C.int(n))
}

// MatMulQ4_0Offset performs Q4_0 quantized matmul with offset-aware dispatch.
// A and out use offsets (scratch-allocated), bMat (weights) always at offset 0.
func (b *Backend) MatMulQ4_0Offset(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.profiler.RecordDispatch("MatMulQ4_0")
	if m == 1 {
		if k%32 != 0 && b.matvecQ4Pipeline != nil {
			C.metal_matvec_q4_0_transposed_f32_offset(b.queue, b.matvecQ4Pipeline,
				unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
				unsafe.Pointer(bMat.Addr()),
				unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
				C.int(n), C.int(k))
			return
		}
		// multi_output: 8 outputs per threadgroup — faster than NR2 for Q4_0 (see above)
		C.metal_matvec_q4_0_multi_output_f32_offset(b.queue, b.matvecQ4MultiOutputPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(n), C.int(k))
	} else if m >= 8 && b.matmulQ4SimdgroupPipeline != nil {
		C.metal_matmul_q4_0_simdgroup_f32_offset(b.queue, b.matmulQ4SimdgroupPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(m), C.int(n), C.int(k))
	} else {
		C.metal_matmul_q4_0_batched_f32_offset(b.queue, b.matmulQ4BatchedPipeline,
			unsafe.Pointer(a.Addr()), C.uint64_t(a.Offset()),
			unsafe.Pointer(bMat.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(m), C.int(n), C.int(k))
	}
}

// RoPEOffset applies rotary position encoding with offset-aware dispatch.
// Q and K use offsets (scratch-allocated activations).
func (b *Backend) RoPEOffset(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float32, ropeNeox bool) {
	b.profiler.RecordDispatch("RoPE")
	effectiveRopeDim := ropeDim
	if effectiveRopeDim == 0 {
		effectiveRopeDim = headDim
	}
	neoxFlag := 0
	if ropeNeox {
		neoxFlag = 1
	}
	C.metal_rope_gqa_f32_offset(b.queue, b.ropeGQAPipeline,
		unsafe.Pointer(q.Addr()), C.uint64_t(q.Offset()),
		unsafe.Pointer(k.Addr()), C.uint64_t(k.Offset()),
		C.int(seqLen), C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.int(startPos), C.int(effectiveRopeDim), C.float(theta), C.int(neoxFlag))
}

// SDPAOffset performs SDPA with offset-aware dispatch.
// Q and out use offsets (scratch-allocated), K/V (KV cache) at offset 0.
func (b *Backend) SDPAOffset(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int) {
	b.profiler.RecordDispatch("SDPA")
	useFlash := b.sdpaFlashDecodePipeline != nil && kvLen >= 16

	if useFlash {
		C.metal_sdpa_flash_decode_f32_offset(b.queue, b.sdpaFlashDecodePipeline,
			unsafe.Pointer(q.Addr()), C.uint64_t(q.Offset()),
			unsafe.Pointer(k.Addr()), unsafe.Pointer(v.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale), C.int(kvHeadStride))
	} else {
		C.metal_sdpa_decode_f32_offset(b.queue, b.sdpaDecodePipeline,
			unsafe.Pointer(q.Addr()), C.uint64_t(q.Offset()),
			unsafe.Pointer(k.Addr()), unsafe.Pointer(v.Addr()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()),
			C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale), C.int(kvHeadStride))
	}
}

// Mul performs element-wise multiplication.
func (b *Backend) Mul(a, bIn, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Mul")
	C.metal_mul_f32(b.queue, b.mulPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// Argmax returns the index of the maximum value in the input tensor.
// This runs entirely on GPU, avoiding the 128KB logits transfer for greedy sampling.
func (b *Backend) Argmax(input tensor.DevicePtr, n int) int {
	b.profiler.RecordDispatch("Argmax")
	// Allocate a small buffer for the result (4 bytes for int32)
	resultBuf := C.metal_alloc_buffer(b.device, 4)
	defer C.metal_release(resultBuf)

	// Run argmax kernel
	C.metal_argmax_f32(b.queue, b.argmaxPipeline,
		unsafe.Pointer(input.Addr()), resultBuf, C.int(n))

	// Sync and read result
	C.metal_sync(b.queue)
	var result int32
	C.metal_copy_from_buffer(unsafe.Pointer(&result), resultBuf, 4)

	return int(result)
}

// =============================================================================
// Training Operations for Medusa Heads
// =============================================================================

// ReLUInplace performs in-place ReLU: x = max(0, x)
func (b *Backend) ReLUInplace(x tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("ReLU")
	C.metal_relu_inplace_f32(b.queue, b.reluInplacePipeline,
		unsafe.Pointer(x.Addr()), C.int(n))
}

// ReLUBackward performs ReLU backward: dx = dx * (x > 0)
// x is the pre-ReLU input, dx is the gradient (modified in-place)
func (b *Backend) ReLUBackward(x, dx tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("ReLU")
	C.metal_relu_backward_f32(b.queue, b.reluBackwardPipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(dx.Addr()), C.int(n))
}

// SiLUInplace performs in-place SiLU: x = x * sigmoid(x)
// Falls back to CPU implementation since Metal shader for SiLU is pending.
func (b *Backend) SiLUInplace(x tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("SiLU")
	// CPU fallback: read from GPU, apply SiLU, write back
	buf := make([]byte, n*4)
	b.ToHost(buf, x)
	b.Sync()
	data := bytesToFloat32Slice(buf)
	for i := 0; i < n; i++ {
		v := data[i]
		data[i] = v / (1.0 + float32(math.Exp(float64(-v))))
	}
	b.ToDevice(x, float32SliceToBytes(data))
}

// SiLUBackward applies SiLU gradient: dx *= sigmoid(x) * (1 + x*(1-sigmoid(x)))
// Falls back to CPU implementation since Metal shader for SiLU backward is pending.
func (b *Backend) SiLUBackward(x, dx tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("SiLUBackward")
	// CPU fallback: read from GPU, compute gradient, write back
	xBuf := make([]byte, n*4)
	dxBuf := make([]byte, n*4)
	b.ToHost(xBuf, x)
	b.ToHost(dxBuf, dx)
	b.Sync()
	xData := bytesToFloat32Slice(xBuf)
	dxData := bytesToFloat32Slice(dxBuf)
	for i := 0; i < n; i++ {
		sig := float32(1.0 / (1.0 + math.Exp(float64(-xData[i]))))
		dxData[i] *= sig * (1.0 + xData[i]*(1.0-sig))
	}
	b.ToDevice(dx, float32SliceToBytes(dxData))
}

// BatchedOuterProduct computes out[i,j] += sum_b(a[b,i] * bIn[b,j])
// a: [batch, M], bIn: [batch, N], out: [M, N]
// Used for computing weight gradients dFC1 and dFC2
func (b *Backend) BatchedOuterProduct(a, bIn, out tensor.DevicePtr, batch, M, N int) {
	b.profiler.RecordDispatch("BatchedOuterProduct")
	C.metal_batched_outer_product_f32(b.queue, b.batchedOuterProductPipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()),
		C.int(batch), C.int(M), C.int(N))
}

// SGDUpdate performs w = w*(1-lr*wd) - lr*grad (SGD with weight decay)
func (b *Backend) SGDUpdate(w, grad tensor.DevicePtr, lr, weightDecay float32, n int) {
	b.profiler.RecordDispatch("SGDUpdate")
	C.metal_sgd_update_f32(b.queue, b.sgdUpdatePipeline,
		unsafe.Pointer(w.Addr()), unsafe.Pointer(grad.Addr()), C.float(lr), C.float(weightDecay), C.int(n))
}

// Zero sets all elements to zero
func (b *Backend) Zero(x tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Zero")
	C.metal_zero_f32(b.queue, b.zeroPipeline,
		unsafe.Pointer(x.Addr()), C.int(n))
}

// Embedding performs embedding lookup.
// ids should be int32 values in an MTLBuffer on device
func (b *Backend) Embedding(ids tensor.DevicePtr, numTokens int, table, out tensor.DevicePtr, vocabSize, dim int) {
	b.profiler.RecordDispatch("Embedding")
	C.metal_embedding_f32(b.queue,
		unsafe.Pointer(ids.Addr()), unsafe.Pointer(table.Addr()), unsafe.Pointer(out.Addr()),
		C.int(numTokens), C.int(vocabSize), C.int(dim))
}

// SDPA performs scaled dot-product attention for decode (single query token).
// Uses Flash Decoding for longer sequences, naive for short ones.
func (b *Backend) SDPA(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int) {
	if q.Offset() != 0 || out.Offset() != 0 {
		b.SDPAOffset(q, k, v, out, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
		return
	}
	b.profiler.RecordDispatch("SDPA")
	// Use Flash Decoding for longer KV lengths where parallelism helps
	// For short sequences, the overhead of threadgroup sync isn't worth it
	useFlash := b.sdpaFlashDecodePipeline != nil && kvLen >= 16

	if useFlash {
		C.metal_sdpa_flash_decode_f32(b.queue, b.sdpaFlashDecodePipeline,
			unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
			unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
			C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale), C.int(kvHeadStride))
	} else {
		// Use naive SDPA for short sequences
		C.metal_sdpa_decode_f32(b.queue, b.sdpaDecodePipeline,
			unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
			unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
			C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale), C.int(kvHeadStride))
	}
}

// SDPAPrefill performs SDPA for prefill with causal masking.
// Uses Flash Attention 2 for longer sequences where K/V tiling provides benefit.
func (b *Backend) SDPAPrefill(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	b.profiler.RecordDispatch("SDPAPrefill")
	// Use Flash Attention 2 for sequences >= 32 tokens
	// FA2 uses two-pass tiling (find max, then accumulate) to reduce register pressure
	// and caches Q in registers while streaming K/V tiles from shared memory
	if b.flashAttention2Pipeline != nil && seqLen >= flashAttentionMinSeqLen() {
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

// SDPAPrefillStandard calls the standard SDPA kernel directly, bypassing FA2 threshold.
// Used for benchmarking to compare FA2 vs standard at same sequence lengths.
func (b *Backend) SDPAPrefillStandard(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	b.profiler.RecordDispatch("SDPAPrefill")
	C.metal_sdpa_prefill_f32(b.queue, b.sdpaPrefillPipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}

// SDPASoftCap performs SDPA with logit soft-capping applied before softmax.
// Used by Gemma 2: scores = cap * tanh(scores / cap) before softmax.
// When softcap=0, behaves identically to regular SDPA.
func (b *Backend) SDPASoftCap(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale, softcap float32, kvHeadStride int) {
	b.profiler.RecordDispatch("SDPASoftCap")
	C.metal_sdpa_softcap_decode_f32(b.queue, b.sdpaSoftCapDecodePipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale), C.int(kvHeadStride), C.float(softcap))
}

// SDPAPrefillSoftCap performs prefill SDPA with logit soft-capping and causal masking.
// Used by Gemma 2 during the prefill phase.
func (b *Backend) SDPAPrefillSoftCap(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale, softcap float32) {
	b.profiler.RecordDispatch("SDPAPrefillSoftCap")
	C.metal_sdpa_softcap_prefill_f32(b.queue, b.sdpaSoftCapPrefillPipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale), C.float(softcap))
}

// FlashAttention2 performs optimized SDPA with K/V tiling in shared memory.
// Uses larger tiles (64 K positions) and exploits GQA to share K/V loads.
func (b *Backend) FlashAttention2(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	b.profiler.RecordDispatch("FlashAttention2")
	C.metal_flash_attention_2_f32(b.queue, b.flashAttention2Pipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(seqLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}

// SDPAPrefillF16 performs prefill SDPA with FP16 inputs/outputs (Flash Attention 2).
// Q, K, V, and out are expected to be in FP16 format.
// Provides 2x memory bandwidth savings for activation loading.
func (b *Backend) SDPAPrefillF16(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	b.profiler.RecordDispatch("SDPAPrefill")
	C.metal_flash_attention_2_f16(b.queue, b.flashAttention2F16Pipeline,
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
	b.profiler.RecordDispatch("Add")
	C.metal_add_f16(b.queue, b.addF16Pipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// MulF16 performs element-wise multiplication on FP16 data.
func (b *Backend) MulF16(a, bIn, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Mul")
	C.metal_mul_f16(b.queue, b.mulF16Pipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bIn.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// SiLUF16 applies the SiLU activation function on FP16 data.
func (b *Backend) SiLUF16(x, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("SiLU")
	C.metal_silu_f16(b.queue, b.siluF16Pipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// SiLUMulF16 performs fused silu(gate) * up operation on FP16 data.
func (b *Backend) SiLUMulF16(gate, up, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("SiLUMul")
	C.metal_silu_mul_f16(b.queue, b.siluMulF16Pipeline,
		unsafe.Pointer(gate.Addr()), unsafe.Pointer(up.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// RMSNormF16 performs RMS normalization with FP16 input/output.
// x: [rows, cols] in FP16, weight: [cols] in FP32, out: [rows, cols] in FP16
func (b *Backend) RMSNormF16(x, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	b.profiler.RecordDispatch("RMSNorm")
	C.metal_rmsnorm_f16(b.queue, b.rmsnormF16Pipeline,
		unsafe.Pointer(x.Addr()), unsafe.Pointer(weight.Addr()), unsafe.Pointer(out.Addr()),
		C.int(rows), C.int(cols), C.float(eps))
}

// MatMulQ4_0_F16 performs C = A @ B^T where A is FP16, B is Q4_0, C is FP16.
// A: [1, K] in FP16, B: [N, K] in Q4_0 format, C: [1, N] in FP16.
// This provides 2x activation bandwidth savings while maintaining Q4_0 weight compression.
func (b *Backend) MatMulQ4_0_F16(a, bMat, out tensor.DevicePtr, n, k int) {
	b.profiler.RecordDispatch("MatMulQ4_0")
	C.metal_matvec_q4_0_f16(b.queue, b.matvecQ4F16Pipeline,
		unsafe.Pointer(a.Addr()), unsafe.Pointer(bMat.Addr()), unsafe.Pointer(out.Addr()),
		C.int(n), C.int(k))
}

// SDPAF16 performs scaled dot-product attention with FP16 KV cache.
// Q: [numQHeads, headDim], K/V: [numKVHeads, kvLen, headDim] (head-major), out: [numQHeads, headDim]
// All tensors in FP16. Provides 2x KV cache bandwidth savings.
// kvHeadStride: stride between KV heads in elements (maxSeqLen * headDim).
func (b *Backend) SDPAF16(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int) {
	b.profiler.RecordDispatch("SDPA")
	// Use vectorized kernel if available and headDim is multiple of 4
	if b.sdpaDecodeF16VecPipeline != nil && headDim%4 == 0 {
		C.metal_sdpa_decode_f16(b.queue, b.sdpaDecodeF16VecPipeline,
			unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
			unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
			C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
			C.float(scale), C.int(kvHeadStride))
		return
	}

	C.metal_sdpa_decode_f16(b.queue, b.sdpaDecodeF16Pipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale), C.int(kvHeadStride))
}

// ConvertF32ToF16 converts FP32 data to FP16.
// in: [n] in FP32, out: [n] in FP16 (out buffer must be n*2 bytes)
// Auto-detects scratch-allocated buffers and uses offset-aware dispatch.
func (b *Backend) ConvertF32ToF16(in, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Convert")
	if in.Offset() != 0 || out.Offset() != 0 {
		C.metal_convert_f32_to_f16_offset(b.queue, b.convertF32ToF16Pipeline,
			unsafe.Pointer(in.Addr()), C.uint64_t(in.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()), C.int(n))
		return
	}
	C.metal_convert_f32_to_f16(b.queue, b.convertF32ToF16Pipeline,
		unsafe.Pointer(in.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// ConvertF16ToF32 converts FP16 data to FP32.
// in: [n] in FP16, out: [n] in FP32 (out buffer must be n*4 bytes)
// Auto-detects scratch-allocated buffers and uses offset-aware dispatch.
func (b *Backend) ConvertF16ToF32(in, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Convert")
	if in.Offset() != 0 || out.Offset() != 0 {
		C.metal_convert_f16_to_f32_offset(b.queue, b.convertF16ToF32Pipeline,
			unsafe.Pointer(in.Addr()), C.uint64_t(in.Offset()),
			unsafe.Pointer(out.Addr()), C.uint64_t(out.Offset()), C.int(n))
		return
	}
	C.metal_convert_f16_to_f32(b.queue, b.convertF16ToF32Pipeline,
		unsafe.Pointer(in.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// ScatterKVF16 transposes KV data from [newTokens, numKVHeads, headDim] to [numKVHeads, maxSeqLen, headDim].
// This is used to efficiently populate the head-major KV cache layout in a single kernel dispatch.
func (b *Backend) ScatterKVF16(src, dst tensor.DevicePtr, newTokens, numKVHeads, headDim, maxSeqLen, seqPos int) {
	b.profiler.RecordDispatch("ScatterKV")
	C.metal_scatter_kv_f16(b.queue, b.scatterKVF16Pipeline,
		unsafe.Pointer(src.Addr()), unsafe.Pointer(dst.Addr()),
		C.int(newTokens), C.int(numKVHeads), C.int(headDim), C.int(maxSeqLen), C.int(seqPos))
}

func (b *Backend) ScatterKVF32ToF16(src, dst tensor.DevicePtr, newTokens, numKVHeads, headDim, maxSeqLen, seqPos int) {
	b.profiler.RecordDispatch("ScatterKV")
	if b.scatterKVF32ToF16Pipeline == nil {
		panic("ScatterKVF32ToF16 called but pipeline unavailable")
	}
	C.metal_scatter_kv_f32_to_f16(b.queue, b.scatterKVF32ToF16Pipeline,
		unsafe.Pointer(src.Addr()), unsafe.Pointer(dst.Addr()),
		C.int(newTokens), C.int(numKVHeads), C.int(headDim), C.int(maxSeqLen), C.int(seqPos))
}


func (b *Backend) ScatterKV(src, dst tensor.DevicePtr, newTokens, numKVHeads, headDim, maxSeqLen, seqPos int) {
	b.profiler.RecordDispatch("ScatterKV")
	if src.Offset() != 0 {
		C.metal_scatter_kv_f32_offset(b.queue, b.scatterKVF32Pipeline,
			unsafe.Pointer(src.Addr()), C.uint64_t(src.Offset()),
			unsafe.Pointer(dst.Addr()),
			C.int(newTokens), C.int(numKVHeads), C.int(headDim),
			C.int(maxSeqLen), C.int(seqPos))
		return
	}
	C.metal_scatter_kv_f32(b.queue, b.scatterKVF32Pipeline,
		unsafe.Pointer(src.Addr()), unsafe.Pointer(dst.Addr()),
		C.int(newTokens), C.int(numKVHeads), C.int(headDim),
		C.int(maxSeqLen), C.int(seqPos))
}


// =============================================================================
// Q8_0 Quantization Operations
// Q8_0 format: 34 bytes per 32 elements (2-byte f16 scale + 32 int8 values)
// Provides 4x memory savings vs FP32 with minimal accuracy loss.
// =============================================================================

// QuantizeF32ToQ8_0 quantizes FP32 data to Q8_0 format.
// in: [n] in FP32 (n must be multiple of 32)
// out: [n/32 * 34] bytes in Q8_0 format
func (b *Backend) QuantizeF32ToQ8_0(in, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Quantize")
	C.metal_quantize_f32_to_q8_0(b.queue, b.quantizeF32ToQ8_0Pipeline,
		unsafe.Pointer(in.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// DequantizeQ8_0ToF32 dequantizes Q8_0 data to FP32.
// in: [n/32 * 34] bytes in Q8_0 format
// out: [n] in FP32
func (b *Backend) DequantizeQ8_0ToF32(in, out tensor.DevicePtr, n int) {
	b.profiler.RecordDispatch("Dequantize")
	C.metal_dequantize_q8_0_to_f32(b.queue, b.dequantizeQ8_0ToF32Pipeline,
		unsafe.Pointer(in.Addr()), unsafe.Pointer(out.Addr()), C.int(n))
}

// SDPAQ8_0 performs scaled dot-product attention with Q8_0 KV cache.
// Q: [numQHeads, headDim] in FP32
// K/V: [kvLen, numKVHeads, headDim] in Q8_0 format
// out: [numQHeads, headDim] in FP32
// Provides 4x KV cache memory savings.
func (b *Backend) SDPAQ8_0(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32) {
	b.profiler.RecordDispatch("SDPA")
	C.metal_sdpa_decode_q8_0(b.queue, b.sdpaDecodeQ8_0Pipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(k.Addr()),
		unsafe.Pointer(v.Addr()), unsafe.Pointer(out.Addr()),
		C.int(kvLen), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale))
}

// ReshapePagedKV copies and reshapes data into a paged KV cache.
func (b *Backend) ReshapePagedKV(src, dstBase, pageTable, blockOffsets tensor.DevicePtr, numTokens, numKVHeads, headDim, blockSize int, isValue bool) {
	b.profiler.RecordDispatch("ReshapePagedKV")
	if b.reshapePagedKVPipeline == nil {
		panic("ReshapePagedKV called but pipeline unavailable")
	}
	
	valFlag := 0
	if isValue {
		valFlag = 1
	}

	C.metal_reshape_paged_kv_f32(b.queue, b.reshapePagedKVPipeline,
		unsafe.Pointer(src.Addr()), unsafe.Pointer(dstBase.Addr()),
		unsafe.Pointer(pageTable.Addr()), unsafe.Pointer(blockOffsets.Addr()),
		C.int(numTokens), C.int(numKVHeads), C.int(headDim), C.int(blockSize), C.int(valFlag))
}

// SDPAPagedDecode performs scaled dot-product attention reading K/V from a paged block pool.
// Q: [numQHeads, headDim], kvPool: base of block pool, blockTable: [numBlocks] int32.
func (b *Backend) SDPAPagedDecode(q, kvPool, blockTable, out tensor.DevicePtr, numBlocks, blockSize, numQHeads, numKVHeads, headDim int, scale float32, tokensInLastBlock int) {
	b.profiler.RecordDispatch("SDPAPagedDecode")
	if b.sdpaPagedDecodePipeline == nil {
		panic("SDPAPagedDecode called but pipeline unavailable")
	}

	C.metal_sdpa_paged_decode_f32(b.queue, b.sdpaPagedDecodePipeline,
		unsafe.Pointer(q.Addr()), unsafe.Pointer(kvPool.Addr()),
		unsafe.Pointer(blockTable.Addr()), unsafe.Pointer(out.Addr()),
		C.int(numBlocks), C.int(blockSize),
		C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		C.float(scale), C.int(tokensInLastBlock))
}

// bytesToFloat32Slice reinterprets bytes as a float32 slice.
func bytesToFloat32Slice(b []byte) []float32 {
	n := len(b) / 4
	f := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		f[i] = math.Float32frombits(bits)
	}
	return f
}

// float32SliceToBytes converts a float32 slice to bytes.
func float32SliceToBytes(f []float32) []byte {
	b := make([]byte, len(f)*4)
	for i, v := range f {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}
