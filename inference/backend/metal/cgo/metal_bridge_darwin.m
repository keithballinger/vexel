// metal_bridge.m - Objective-C Metal implementation
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_bridge.h"

// Embedded Metal shader source
static NSString* metalShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Matrix multiplication: C = A @ B^T
// A: [M, K], B: [N, K], C: [M, N]
kernel void matmul_transposed_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= (uint)N || gid.y >= (uint)M) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[gid.x * K + k];
    }
    C[gid.y * N + gid.x] = sum;
}

// RMS Normalization
kernel void rmsnorm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    int row = gid / dim;
    int col = gid % dim;

    // Compute sum of squares for this row
    threadgroup float shared[256];

    float val = x[gid];
    float sq = val * val;

    // Reduce within threadgroup
    shared[tid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < (uint)dim) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared[0] / float(dim) + eps);
    out[gid] = val * rms * weight[col];
}

// RoPE - Rotary Position Encoding
kernel void rope_f32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant int& seqLen [[buffer(2)]],
    constant int& numHeads [[buffer(3)]],
    constant int& headDim [[buffer(4)]],
    constant int& startPos [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int head = gid.y;
    int pos = gid.x;

    if (pos >= seqLen || head >= numHeads) return;

    int absPos = startPos + pos;

    for (int i = 0; i < headDim / 2; i++) {
        float freq = 1.0f / pow(theta, float(2 * i) / float(headDim));
        float angle = float(absPos) * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        int idx = (pos * numHeads + head) * headDim + i * 2;

        // Apply to Q
        float q0 = q[idx];
        float q1 = q[idx + 1];
        q[idx] = q0 * cos_val - q1 * sin_val;
        q[idx + 1] = q0 * sin_val + q1 * cos_val;

        // Apply to K
        float k0 = k[idx];
        float k1 = k[idx + 1];
        k[idx] = k0 * cos_val - k1 * sin_val;
        k[idx + 1] = k0 * sin_val + k1 * cos_val;
    }
}

// Softmax along last dimension
kernel void softmax_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int row = gid;
    int base = row * dim;

    // Find max
    float maxVal = x[base];
    for (int i = 1; i < dim; i++) {
        maxVal = max(maxVal, x[base + i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float e = exp(x[base + i] - maxVal);
        out[base + i] = e;
        sum += e;
    }

    // Normalize
    for (int i = 0; i < dim; i++) {
        out[base + i] /= sum;
    }
}

// SiLU activation: x * sigmoid(x)
kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float val = x[gid];
    out[gid] = val / (1.0f + exp(-val));
}

// Element-wise add
kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

// Element-wise multiply
kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] * b[gid];
}

// Embedding lookup
kernel void embedding_f32(
    device const int* tokens [[buffer(0)]],
    device const float* table [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int tokenIdx = gid.y;
    int elemIdx = gid.x;

    if (elemIdx >= dim) return;

    int token = tokens[tokenIdx];
    out[tokenIdx * dim + elemIdx] = table[token * dim + elemIdx];
}

// Scaled dot-product attention
kernel void sdpa_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& seqLen [[buffer(4)]],
    constant int& headDim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& causal [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int queryPos = gid.y;
    int dim = gid.x;

    if (queryPos >= seqLen || dim >= headDim) return;

    // Compute attention scores for this query position
    float scores[512]; // Max sequence length support
    float maxScore = -INFINITY;

    int maxK = causal ? (queryPos + 1) : seqLen;
    for (int k = 0; k < maxK; k++) {
        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += Q[queryPos * headDim + d] * K[k * headDim + d];
        }
        score *= scale;
        scores[k] = score;
        maxScore = max(maxScore, score);
    }

    // Softmax
    float sum = 0.0f;
    for (int k = 0; k < maxK; k++) {
        scores[k] = exp(scores[k] - maxScore);
        sum += scores[k];
    }
    for (int k = 0; k < maxK; k++) {
        scores[k] /= sum;
    }

    // Compute weighted sum of values
    float result = 0.0f;
    for (int k = 0; k < maxK; k++) {
        result += scores[k] * V[k * headDim + dim];
    }
    out[queryPos * headDim + dim] = result;
}
)";

// Device and queue management
void* metal_create_device(void) {
    return (__bridge_retained void*)MTLCreateSystemDefaultDevice();
}

void* metal_create_command_queue(void* device) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    return (__bridge_retained void*)[dev newCommandQueue];
}

const char* metal_device_name(void* device) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    return [[dev name] UTF8String];
}

void metal_release(void* obj) {
    if (obj) {
        CFRelease(obj);
    }
}

// Buffer management
void* metal_alloc_buffer(void* device, size_t size) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buffer = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buffer;
}

void metal_copy_to_buffer(void* buffer, const void* src, size_t size) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    memcpy([buf contents], src, size);
}

void metal_copy_from_buffer(void* dst, void* buffer, size_t size) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    memcpy(dst, [buf contents], size);
}

// Shader compilation
void* metal_compile_library(void* device, const char* source) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    NSError* error = nil;

    NSString* src = source ? [NSString stringWithUTF8String:source] : metalShaderSource;

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;

    id<MTLLibrary> library = [dev newLibraryWithSource:src options:options error:&error];
    if (error) {
        NSLog(@"Metal compile error: %@", error);
        return nil;
    }
    return (__bridge_retained void*)library;
}

void* metal_create_pipeline(void* device, void* library, const char* functionName) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;

    NSString* name = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [lib newFunctionWithName:name];
    if (!function) {
        NSLog(@"Function not found: %@", name);
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [dev newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        NSLog(@"Pipeline error: %@", error);
        return nil;
    }

    return (__bridge_retained void*)pipeline;
}

// Synchronization
void metal_sync(void* commandQueue) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}

// Helper to dispatch a compute kernel
static void dispatch_kernel(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                           NSArray* buffers, NSArray* constants, MTLSize gridSize) {
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    // Set buffers
    for (NSUInteger i = 0; i < buffers.count; i++) {
        id<MTLBuffer> buf = buffers[i];
        [encoder setBuffer:buf offset:0 atIndex:i];
    }

    // Set constants (starting after buffers)
    NSUInteger constIdx = buffers.count;
    for (NSUInteger i = 0; i < constants.count; i++) {
        NSData* data = constants[i];
        [encoder setBytes:[data bytes] length:[data length] atIndex:constIdx + i];
    }

    MTLSize threadgroupSize = MTLSizeMake(
        MIN(pipeline.maxTotalThreadsPerThreadgroup, gridSize.width),
        1, 1
    );
    MTLSize threadgroups = MTLSizeMake(
        (gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width,
        gridSize.height,
        gridSize.depth
    );

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}

// Kernel implementations
void metal_matmul_transposed_f32(void* queuePtr, void* pipelinePtr,
                                  void* A, void* B, void* C,
                                  int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)A,
        (__bridge id<MTLBuffer>)B,
        (__bridge id<MTLBuffer>)C
    ];
    NSArray* constants = @[
        [NSData dataWithBytes:&M length:sizeof(M)],
        [NSData dataWithBytes:&N length:sizeof(N)],
        [NSData dataWithBytes:&K length:sizeof(K)]
    ];

    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(N, M, 1));
}

void metal_matmul_f32(void* queue, void* pipeline,
                      void* A, void* B, void* C,
                      int M, int N, int K) {
    // For now, use transposed version
    metal_matmul_transposed_f32(queue, pipeline, A, B, C, M, N, K);
}

void metal_rmsnorm_f32(void* queuePtr, void* pipelinePtr,
                       void* x, void* weight, void* out,
                       int batchSize, int dim, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)x,
        (__bridge id<MTLBuffer>)weight,
        (__bridge id<MTLBuffer>)out
    ];
    NSArray* constants = @[
        [NSData dataWithBytes:&dim length:sizeof(dim)],
        [NSData dataWithBytes:&eps length:sizeof(eps)]
    ];

    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(batchSize * dim, 1, 1));
}

void metal_rope_f32(void* queuePtr, void* pipelinePtr,
                    void* q, void* k,
                    int batchSize, int seqLen, int numHeads, int headDim,
                    int startPos, float theta) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)q,
        (__bridge id<MTLBuffer>)k
    ];
    NSArray* constants = @[
        [NSData dataWithBytes:&seqLen length:sizeof(seqLen)],
        [NSData dataWithBytes:&numHeads length:sizeof(numHeads)],
        [NSData dataWithBytes:&headDim length:sizeof(headDim)],
        [NSData dataWithBytes:&startPos length:sizeof(startPos)],
        [NSData dataWithBytes:&theta length:sizeof(theta)]
    ];

    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(seqLen, numHeads, 1));
}

void metal_softmax_f32(void* queuePtr, void* pipelinePtr,
                       void* x, void* out,
                       int batchSize, int dim) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)x,
        (__bridge id<MTLBuffer>)out
    ];
    NSArray* constants = @[
        [NSData dataWithBytes:&dim length:sizeof(dim)]
    ];

    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(batchSize, 1, 1));
}

void metal_silu_f32(void* queuePtr, void* pipelinePtr,
                    void* x, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)x,
        (__bridge id<MTLBuffer>)out
    ];

    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_add_f32(void* queuePtr, void* pipelinePtr,
                   void* a, void* b, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)a,
        (__bridge id<MTLBuffer>)b,
        (__bridge id<MTLBuffer>)out
    ];

    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_mul_f32(void* queuePtr, void* pipelinePtr,
                   void* a, void* b, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)a,
        (__bridge id<MTLBuffer>)b,
        (__bridge id<MTLBuffer>)out
    ];

    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_embedding_f32(void* queuePtr,
                         const int* tokens, void* table, void* out,
                         int numTokens, int vocabSize, int dim) {
    // For embedding, we use MPS or a simple copy
    // This is a simple CPU fallback for now
    id<MTLBuffer> tableBuf = (__bridge id<MTLBuffer>)table;
    id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)out;

    float* tablePtr = (float*)[tableBuf contents];
    float* outPtr = (float*)[outBuf contents];

    for (int t = 0; t < numTokens; t++) {
        int token = tokens[t];
        memcpy(outPtr + t * dim, tablePtr + token * dim, dim * sizeof(float));
    }
}

void metal_scaled_dot_product_attention(void* queuePtr,
                                        void* Q, void* K, void* V, void* out,
                                        int batchSize, int numHeads, int seqLen, int headDim,
                                        float scale, int causal) {
    // Use Metal Performance Shaders for optimal attention
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLDevice> device = [queue device];

    // For now, fall back to the kernel-based implementation
    // A full MPS implementation would use MPSGraph or custom optimized kernels

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];

    // TODO: Implement efficient attention using MPS
    // For now, this is a placeholder that processes sequentially

    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}
