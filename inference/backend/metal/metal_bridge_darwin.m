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

// RoPE - Rotary Position Encoding (paired-element layout)
// Data layout: [seqLen, numHeads, headDim]
// Rotation pairs: (dim[j], dim[j + headDim/2]) for j in 0..headDim/2
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
    int pos = gid.x;
    int head = gid.y;

    if (pos >= seqLen || head >= numHeads) return;

    int absPos = startPos + pos;
    int halfDim = headDim / 2;

    // Base offset for this position and head
    // Layout: [seqLen, numHeads, headDim]
    int offset = (pos * numHeads + head) * headDim;

    for (int j = 0; j < halfDim; j++) {
        // Frequency for this dimension pair
        float freq = 1.0f / pow(theta, float(2 * j) / float(headDim));
        float angle = float(absPos) * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        // Apply rotation to Q using paired elements (j and j+halfDim)
        float q0 = q[offset + j];
        float q1 = q[offset + j + halfDim];
        q[offset + j] = q0 * cos_val - q1 * sin_val;
        q[offset + j + halfDim] = q0 * sin_val + q1 * cos_val;

        // Apply to K
        float k0 = k[offset + j];
        float k1 = k[offset + j + halfDim];
        k[offset + j] = k0 * cos_val - k1 * sin_val;
        k[offset + j + halfDim] = k0 * sin_val + k1 * cos_val;
    }
}

// RoPE for GQA - separate Q and K head counts
// Q layout: [seqLen, numQHeads, headDim]
// K layout: [seqLen, numKVHeads, headDim]
kernel void rope_gqa_f32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant int& seqLen [[buffer(2)]],
    constant int& numQHeads [[buffer(3)]],
    constant int& numKVHeads [[buffer(4)]],
    constant int& headDim [[buffer(5)]],
    constant int& startPos [[buffer(6)]],
    constant float& theta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int pos = gid.x;
    int head = gid.y;

    if (pos >= seqLen) return;

    int absPos = startPos + pos;
    int halfDim = headDim / 2;

    // Process Q heads
    if (head < numQHeads) {
        int qOffset = (pos * numQHeads + head) * headDim;
        for (int j = 0; j < halfDim; j++) {
            float freq = 1.0f / pow(theta, float(2 * j) / float(headDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            float q0 = q[qOffset + j];
            float q1 = q[qOffset + j + halfDim];
            q[qOffset + j] = q0 * cos_val - q1 * sin_val;
            q[qOffset + j + halfDim] = q0 * sin_val + q1 * cos_val;
        }
    }

    // Process K heads (fewer due to GQA)
    if (head < numKVHeads) {
        int kOffset = (pos * numKVHeads + head) * headDim;
        for (int j = 0; j < halfDim; j++) {
            float freq = 1.0f / pow(theta, float(2 * j) / float(headDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            float k0 = k[kOffset + j];
            float k1 = k[kOffset + j + halfDim];
            k[kOffset + j] = k0 * cos_val - k1 * sin_val;
            k[kOffset + j + halfDim] = k0 * sin_val + k1 * cos_val;
        }
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

// Scaled dot-product attention for GQA decode
// Q: [numQHeads, headDim] - single query token
// K: [kvLen, numKVHeads, headDim] - key cache
// V: [kvLen, numKVHeads, headDim] - value cache
// out: [numQHeads, headDim] - output
kernel void sdpa_gqa_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    int qHead = gid;
    if (qHead >= numQHeads) return;

    // GQA: map Q head to KV head
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;

    // Q head pointer
    int qOffset = qHead * headDim;

    // Compute attention scores: Q dot K for each KV position
    // Use thread-local storage (limited to 2048 positions)
    float scores[2048];
    float maxScore = -INFINITY;

    for (int pos = 0; pos < kvLen && pos < 2048; pos++) {
        // K at position pos for this KV head
        // K layout: [kvLen, numKVHeads, headDim]
        int kOffset = pos * numKVHeads * headDim + kvHead * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        scores[pos] = dot * scale;
        maxScore = max(maxScore, scores[pos]);
    }

    // Softmax
    float sumExp = 0.0f;
    int effectiveLen = min(kvLen, 2048);
    for (int pos = 0; pos < effectiveLen; pos++) {
        scores[pos] = exp(scores[pos] - maxScore);
        sumExp += scores[pos];
    }
    float invSum = 1.0f / sumExp;
    for (int pos = 0; pos < effectiveLen; pos++) {
        scores[pos] *= invSum;
    }

    // Compute weighted sum of V
    int outOffset = qHead * headDim;
    for (int d = 0; d < headDim; d++) {
        float sum = 0.0f;
        for (int pos = 0; pos < effectiveLen; pos++) {
            // V layout: [kvLen, numKVHeads, headDim]
            int vOffset = pos * numKVHeads * headDim + kvHead * headDim;
            sum += scores[pos] * V[vOffset + d];
        }
        out[outOffset + d] = sum;
    }
}

// Batched SDPA for prefill with causal masking
// Q: [seqLen, numQHeads, headDim]
// K: [seqLen, numKVHeads, headDim]
// V: [seqLen, numKVHeads, headDim]
// out: [seqLen, numQHeads, headDim]
kernel void sdpa_prefill_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& seqLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int qPos = gid.x;
    int qHead = gid.y;

    if (qPos >= seqLen || qHead >= numQHeads) return;

    // GQA mapping
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;

    // Q pointer for this position and head
    // Q layout: [seqLen, numQHeads, headDim]
    int qOffset = qPos * numQHeads * headDim + qHead * headDim;

    // Causal attention: only attend to positions <= qPos
    int maxKLen = qPos + 1;

    // Compute attention scores
    float scores[2048];
    float maxScore = -INFINITY;

    for (int kPos = 0; kPos < maxKLen && kPos < 2048; kPos++) {
        // K layout: [seqLen, numKVHeads, headDim]
        int kOffset = kPos * numKVHeads * headDim + kvHead * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        scores[kPos] = dot * scale;
        maxScore = max(maxScore, scores[kPos]);
    }

    // Softmax
    float sumExp = 0.0f;
    for (int kPos = 0; kPos < maxKLen; kPos++) {
        scores[kPos] = exp(scores[kPos] - maxScore);
        sumExp += scores[kPos];
    }
    float invSum = 1.0f / sumExp;
    for (int kPos = 0; kPos < maxKLen; kPos++) {
        scores[kPos] *= invSum;
    }

    // Compute weighted sum of V
    // out layout: [seqLen, numQHeads, headDim]
    int outOffset = qPos * numQHeads * headDim + qHead * headDim;
    for (int d = 0; d < headDim; d++) {
        float sum = 0.0f;
        for (int kPos = 0; kPos < maxKLen; kPos++) {
            // V layout: [seqLen, numKVHeads, headDim]
            int vOffset = kPos * numKVHeads * headDim + kvHead * headDim;
            sum += scores[kPos] * V[vOffset + d];
        }
        out[outOffset + d] = sum;
    }
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
                         void* tokensPtr, void* tablePtr, void* outPtr,
                         int numTokens, int vocabSize, int dim) {
    // For embedding, we use a simple CPU copy (each buffer is standalone)
    id<MTLBuffer> tokensBuf = (__bridge id<MTLBuffer>)tokensPtr;
    id<MTLBuffer> tableBuf = (__bridge id<MTLBuffer>)tablePtr;
    id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)outPtr;

    int32_t* tokensData = (int32_t*)[tokensBuf contents];
    float* tableData = (float*)[tableBuf contents];
    float* outData = (float*)[outBuf contents];

    for (int t = 0; t < numTokens; t++) {
        int32_t token = tokensData[t];
        if (token >= 0 && token < vocabSize) {
            memcpy(outData + t * dim, tableData + token * dim, dim * sizeof(float));
        }
    }
}

void metal_sdpa_decode_f32(void* queuePtr, void* pipelinePtr,
                           void* Q, void* K, void* V, void* out,
                           int kvLen, int numQHeads, int numKVHeads, int headDim,
                           float scale) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)K offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)V offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&kvLen length:sizeof(kvLen) atIndex:4];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:5];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:6];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:7];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:8];

    // Dispatch one thread per Q head
    MTLSize gridSize = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)numQHeads), 1, 1);
    MTLSize threadgroups = MTLSizeMake((numQHeads + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}

void metal_sdpa_prefill_f32(void* queuePtr, void* pipelinePtr,
                            void* Q, void* K, void* V, void* out,
                            int seqLen, int numQHeads, int numKVHeads, int headDim,
                            float scale) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)K offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)V offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:4];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:5];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:6];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:7];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:8];

    // Dispatch [seqLen, numQHeads] grid
    MTLSize threadgroupSize = MTLSizeMake(MIN(16, (NSUInteger)seqLen), MIN(16, (NSUInteger)numQHeads), 1);
    MTLSize threadgroups = MTLSizeMake(
        (seqLen + threadgroupSize.width - 1) / threadgroupSize.width,
        (numQHeads + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}

// Legacy interface (kept for compatibility)
void metal_scaled_dot_product_attention(void* queuePtr,
                                        void* Q, void* K, void* V, void* out,
                                        int batchSize, int numHeads, int seqLen, int headDim,
                                        float scale, int causal) {
    // Deprecated - use metal_sdpa_decode_f32 or metal_sdpa_prefill_f32 instead
}
