// metal_bridge.m - Objective-C Metal implementation
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_bridge.h"

// Embedded Metal shader source
static NSString* metalShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Matrix multiplication: C = A @ B^T with SIMD optimization
// A: [M, K], B: [N, K], C: [M, N]
// Each thread computes one element of C
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

    // Use float4 for SIMD - process 4 elements at a time
    float sum = 0.0f;
    int k = 0;

    // Process 4 elements at a time using SIMD
    device const float4* A4 = (device const float4*)(A + gid.y * K);
    device const float4* B4 = (device const float4*)(B + gid.x * K);
    int k4 = K / 4;

    for (int i = 0; i < k4; i++) {
        float4 a = A4[i];
        float4 b = B4[i];
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    k = k4 * 4;

    // Handle remaining elements
    for (; k < K; k++) {
        sum += A[gid.y * K + k] * B[gid.x * K + k];
    }

    C[gid.y * N + gid.x] = sum;
}

// Matrix-vector multiplication: C = A @ B^T for M=1 case
// A: [1, K], B: [N, K], C: [1, N]
// Uses threadgroup reduction for parallel dot product
// Each threadgroup computes one output element
// Threads within threadgroup split the K dimension
constant int MV_THREADGROUP_SIZE = 256;

kernel void matvec_transposed_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup computes one output element C[gid]
    if (gid >= (uint)N) return;

    // Split K across threads in threadgroup
    float sum = 0.0f;
    device const float* b_row = B + gid * K;

    // Each thread handles K/threadgroup_size elements with stride
    // Use float4 for better memory coalescing
    int k4 = K / 4;
    for (int i = tid; i < k4; i += MV_THREADGROUP_SIZE) {
        int base = i * 4;
        float4 a = float4(A[base], A[base+1], A[base+2], A[base+3]);
        float4 b = float4(b_row[base], b_row[base+1], b_row[base+2], b_row[base+3]);
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    // Handle remainder
    for (int k = k4 * 4 + tid; k < K; k += MV_THREADGROUP_SIZE) {
        sum += A[k] * b_row[k];
    }

    // Warp-level reduction using simdgroup operations
    sum = simd_sum(sum);

    // Store warp results to shared memory
    if (simd_lane == 0) {
        shared[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first warp
    if (simd_group == 0) {
        int num_warps = (MV_THREADGROUP_SIZE + 31) / 32;
        float warp_sum = (simd_lane < (uint)num_warps) ? shared[simd_lane] : 0.0f;
        warp_sum = simd_sum(warp_sum);
        if (tid == 0) {
            C[gid] = warp_sum;
        }
    }
}

// Q4_0 Matrix-vector multiplication: C = A @ B^T where B is Q4_0 quantized
// A: [1, K] in float32, B: [N, K] in Q4_0 format, C: [1, N] in float32
// Q4_0 format: 18 bytes per 32 elements (2 byte f16 scale + 16 bytes of nibbles)
// Each thread computes one output element (simd parallel dot product)
constant int Q4_MV_THREADGROUP_SIZE = 256;
constant int Q4_BLOCK_SIZE = 32;
constant int Q4_BYTES_PER_BLOCK = 18;

// Helper to convert f16 to f32 using software (avoids alignment issues)
inline float q4_f16_to_f32(ushort h) {
    uint sign = (h >> 15) & 0x1;
    uint exp = (h >> 10) & 0x1f;
    uint mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float f = mant * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return sign ? -f : f;
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }

    float f = (1.0f + mant * (1.0f / 1024.0f)) * pow(2.0f, float(exp) - 15.0f);
    return sign ? -f : f;
}

kernel void matvec_q4_0_transposed_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_0 format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup computes one output element C[gid]
    if (gid >= (uint)N) return;

    float sum = 0.0f;

    // Calculate Q4_0 row offset for row gid
    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    device const uchar* b_row = B + gid * numBlocks * Q4_BYTES_PER_BLOCK;

    // Each thread handles some blocks
    for (int block = tid; block < numBlocks; block += Q4_MV_THREADGROUP_SIZE) {
        device const uchar* blockPtr = b_row + block * Q4_BYTES_PER_BLOCK;

        // Read f16 scale
        ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
        float scale = q4_f16_to_f32(scale_u16);

        int base_k = block * Q4_BLOCK_SIZE;

        // Process 16 bytes = 32 nibbles
        for (int i = 0; i < 16 && base_k + i < K; i++) {
            uchar byte_val = blockPtr[2 + i];

            // Low nibble -> position i
            int k0 = base_k + i;
            int q0 = byte_val & 0x0F;
            sum += A[k0] * scale * float(q0 - 8);

            // High nibble -> position i + 16
            int k1 = base_k + i + 16;
            if (k1 < K) {
                int q1 = (byte_val >> 4) & 0x0F;
                sum += A[k1] * scale * float(q1 - 8);
            }
        }
    }

    // Warp-level reduction
    sum = simd_sum(sum);

    // Store warp results to shared memory
    if (simd_lane == 0) {
        shared[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first warp
    if (simd_group == 0) {
        int num_warps = (Q4_MV_THREADGROUP_SIZE + 31) / 32;
        float warp_sum = (simd_lane < (uint)num_warps) ? shared[simd_lane] : 0.0f;
        warp_sum = simd_sum(warp_sum);
        if (tid == 0) {
            C[gid] = warp_sum;
        }
    }
}

// RMSNorm with threadgroup parallelism
// Each threadgroup processes one row using parallel reduction
constant int RMSNORM_THREADGROUP_SIZE = 256;

kernel void rmsnorm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base = row * dim;

    // Phase 1: Each thread computes partial sum of squares
    float sumSq = 0.0f;
    for (int i = tid; i < dim; i += RMSNORM_THREADGROUP_SIZE) {
        float val = x[base + i];
        sumSq += val * val;
    }

    // Warp-level reduction
    sumSq = simd_sum(sumSq);

    // Store warp results to shared memory
    if (simd_lane == 0) {
        shared[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first warp
    float totalSumSq = 0.0f;
    if (simd_group == 0) {
        int num_warps = (RMSNORM_THREADGROUP_SIZE + 31) / 32;
        float warp_sum = (simd_lane < (uint)num_warps) ? shared[simd_lane] : 0.0f;
        totalSumSq = simd_sum(warp_sum);
    }

    // Broadcast RMS to all threads via shared memory
    if (tid == 0) {
        shared[0] = rsqrt(totalSumSq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = shared[0];

    // Phase 2: Each thread normalizes its portion of elements
    for (int i = tid; i < dim; i += RMSNORM_THREADGROUP_SIZE) {
        out[base + i] = x[base + i] * rms * weight[i];
    }
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
// Uses two-pass algorithm to avoid large local arrays
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

    int qOffset = qHead * headDim;
    int outOffset = qHead * headDim;

    // First pass: find max score for numerical stability
    float maxScore = -INFINITY;
    for (int pos = 0; pos < kvLen; pos++) {
        int kOffset = pos * numKVHeads * headDim + kvHead * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        maxScore = max(maxScore, dot * scale);
    }

    // Initialize output to zero
    for (int d = 0; d < headDim; d++) {
        out[outOffset + d] = 0.0f;
    }

    // Second pass: compute softmax and weighted sum
    float sumExp = 0.0f;
    for (int pos = 0; pos < kvLen; pos++) {
        int kOffset = pos * numKVHeads * headDim + kvHead * headDim;
        int vOffset = pos * numKVHeads * headDim + kvHead * headDim;

        // Recompute score
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        float weight = exp(dot * scale - maxScore);
        sumExp += weight;

        // Accumulate weighted V
        for (int d = 0; d < headDim; d++) {
            out[outOffset + d] += weight * V[vOffset + d];
        }
    }

    // Normalize by sum of exp
    float invSum = 1.0f / sumExp;
    for (int d = 0; d < headDim; d++) {
        out[outOffset + d] *= invSum;
    }
}

// Flash Decoding - parallelized SDPA for decode phase
// Two-phase approach:
// Phase 1: Threads cooperatively compute all scores, find global max and sum
// Phase 2: Each thread handles specific output dimensions, accumulating weighted V
// Each threadgroup handles one Q head
constant int FLASH_DECODE_THREADS = 256;

kernel void sdpa_flash_decode_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int qHead = gid;
    if (qHead >= numQHeads) return;

    // GQA: map Q head to KV head
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;

    int qOffset = qHead * headDim;

    // Shared memory layout:
    // [0..kvLen-1]: attention weights (after softmax)
    // [kvLen..kvLen+7]: warp max/sum values
    threadgroup float* weights = shared;
    threadgroup float* warpVals = shared + kvLen;

    // Phase 1a: Each thread computes scores for its KV positions
    float localMax = -INFINITY;
    for (int pos = tid; pos < kvLen; pos += FLASH_DECODE_THREADS) {
        int kOffset = pos * numKVHeads * headDim + kvHead * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        float score = dot * scale;
        weights[pos] = score;
        localMax = max(localMax, score);
    }

    // Reduce max across threadgroup
    localMax = simd_max(localMax);
    if (simd_lane == 0) {
        warpVals[simd_group] = localMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8) {
        localMax = warpVals[tid];
        localMax = simd_max(localMax);
        if (tid == 0) warpVals[0] = localMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gMax = warpVals[0];

    // Phase 1b: Compute exp(score - max) and local sum
    float localSum = 0.0f;
    for (int pos = tid; pos < kvLen; pos += FLASH_DECODE_THREADS) {
        float expScore = exp(weights[pos] - gMax);
        weights[pos] = expScore;
        localSum += expScore;
    }

    // Reduce sum
    localSum = simd_sum(localSum);
    if (simd_lane == 0) {
        warpVals[simd_group] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8) {
        localSum = warpVals[tid];
        localSum = simd_sum(localSum);
        if (tid == 0) warpVals[0] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = 1.0f / warpVals[0];

    // Normalize weights in place
    for (int pos = tid; pos < kvLen; pos += FLASH_DECODE_THREADS) {
        weights[pos] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Each thread handles specific output dimensions
    // Threads are assigned to output dimensions (strided)
    int outOffset = qHead * headDim;
    for (int d = tid; d < headDim; d += FLASH_DECODE_THREADS) {
        float sum = 0.0f;
        for (int pos = 0; pos < kvLen; pos++) {
            int vOffset = pos * numKVHeads * headDim + kvHead * headDim;
            sum += weights[pos] * V[vOffset + d];
        }
        out[outOffset + d] = sum;
    }
}

// Batched SDPA for prefill with causal masking
// Uses two-pass algorithm to avoid large local arrays
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

    // First pass: find max score for numerical stability
    float maxScore = -INFINITY;
    for (int kPos = 0; kPos < maxKLen; kPos++) {
        // K layout: [seqLen, numKVHeads, headDim]
        int kOffset = kPos * numKVHeads * headDim + kvHead * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        maxScore = max(maxScore, dot * scale);
    }

    // out layout: [seqLen, numQHeads, headDim]
    int outOffset = qPos * numQHeads * headDim + qHead * headDim;

    // Initialize output to zero
    for (int d = 0; d < headDim; d++) {
        out[outOffset + d] = 0.0f;
    }

    // Second pass: compute softmax and weighted sum
    float sumExp = 0.0f;
    for (int kPos = 0; kPos < maxKLen; kPos++) {
        // Recompute K offset and score
        int kOffset = kPos * numKVHeads * headDim + kvHead * headDim;
        int vOffset = kPos * numKVHeads * headDim + kvHead * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        float weight = exp(dot * scale - maxScore);
        sumExp += weight;

        // Accumulate weighted V
        for (int d = 0; d < headDim; d++) {
            out[outOffset + d] += weight * V[vOffset + d];
        }
    }

    // Normalize by sum of exp
    float invSum = 1.0f / sumExp;
    for (int d = 0; d < headDim; d++) {
        out[outOffset + d] *= invSum;
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

size_t metal_buffer_size(void* buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return [buf length];
}

void metal_copy_to_buffer(void* buffer, const void* src, size_t size) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    memcpy([buf contents], src, size);
}

void metal_copy_from_buffer(void* dst, void* buffer, size_t size) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    memcpy(dst, [buf contents], size);
}

// GPU-to-GPU buffer copy using blit encoder
void metal_copy_buffer(void* queue, void* srcBuffer, size_t srcOffset,
                       void* dstBuffer, size_t dstOffset, size_t size) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcBuffer;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuffer;

    id<MTLCommandBuffer> commandBuffer = [q commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];

    [blit copyFromBuffer:src sourceOffset:srcOffset toBuffer:dst destinationOffset:dstOffset size:size];
    [blit endEncoding];

    [commandBuffer commit];
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

// Helper to dispatch a compute kernel (asynchronous - call metal_sync to wait)
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
    // Don't wait here - let commands queue up, sync at end of forward pass
}

// Custom kernel for C = A @ B^T with SIMD
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

// Optimized matrix-vector multiply: C = A @ B^T for M=1 case
// Uses threadgroup reduction for better parallelism
void metal_matvec_transposed_f32(void* queuePtr, void* pipelinePtr,
                                  void* A, void* B, void* C,
                                  int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    // Each threadgroup computes one output element
    // Threads within threadgroup cooperate on the dot product
    int threadgroupSize = 256;  // Must match MV_THREADGROUP_SIZE in shader
    int numWarps = (threadgroupSize + 31) / 32;
    int sharedMemSize = numWarps * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(N, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [cmdBuffer commit];
}

// Q4_0 quantized matrix-vector multiply: C = A @ B^T where B is Q4_0
void metal_matvec_q4_0_transposed_f32(void* queuePtr, void* pipelinePtr,
                                       void* A, void* B, void* C,
                                       int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    int threadgroupSize = 256;  // Must match Q4_MV_THREADGROUP_SIZE in shader
    int numWarps = (threadgroupSize + 31) / 32;
    int sharedMemSize = numWarps * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(N, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [cmdBuffer commit];
}

// Batched Q4_0 matmul: dispatches matvec kernel M times with buffer offsets
void metal_matmul_q4_0_batched_f32(void* queuePtr, void* pipelinePtr,
                                    void* A, void* B, void* C,
                                    int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int threadgroupSize = 256;
    int numWarps = (threadgroupSize + 31) / 32;
    int sharedMemSize = numWarps * sizeof(float);

    size_t aRowBytes = K * sizeof(float);
    size_t cRowBytes = N * sizeof(float);

    // Use a single command buffer with multiple dispatches for efficiency
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];  // B is same for all rows
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(N, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    for (int row = 0; row < M; row++) {
        [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:row * aRowBytes atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:row * cRowBytes atIndex:2];
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    }

    [encoder endEncoding];
    [cmdBuffer commit];
}

void metal_rmsnorm_f32(void* queuePtr, void* pipelinePtr,
                       void* x, void* weight, void* out,
                       int batchSize, int dim, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int threadgroupSize = 256;
    int numWarps = (threadgroupSize + 31) / 32;
    int sharedMemSize = numWarps * sizeof(float);

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
    [encoder setBytes:&dim length:sizeof(dim) atIndex:3];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:4];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per row
    MTLSize threadgroups = MTLSizeMake(batchSize, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    [encoder endEncoding];
    [cmdBuffer commit];
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

void metal_rope_gqa_f32(void* queuePtr, void* pipelinePtr,
                        void* q, void* k,
                        int seqLen, int numQHeads, int numKVHeads, int headDim,
                        int startPos, float theta) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)q,
        (__bridge id<MTLBuffer>)k
    ];
    NSArray* constants = @[
        [NSData dataWithBytes:&seqLen length:sizeof(seqLen)],
        [NSData dataWithBytes:&numQHeads length:sizeof(numQHeads)],
        [NSData dataWithBytes:&numKVHeads length:sizeof(numKVHeads)],
        [NSData dataWithBytes:&headDim length:sizeof(headDim)],
        [NSData dataWithBytes:&startPos length:sizeof(startPos)],
        [NSData dataWithBytes:&theta length:sizeof(theta)]
    ];

    // Dispatch with max(numQHeads, numKVHeads) threads per position
    int maxHeads = numQHeads > numKVHeads ? numQHeads : numKVHeads;
    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(seqLen, maxHeads, 1));
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

// Flash Decoding - parallelized SDPA for decode phase
// Uses threadgroups with shared memory for scores and accumulation
void metal_sdpa_flash_decode_f32(void* queuePtr, void* pipelinePtr,
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

    // Shared memory layout:
    // - weights[kvLen]: attention weights
    // - warpVals[8]: for max/sum reduction
    int threadgroupSize = 256;  // Must match FLASH_DECODE_THREADS
    int sharedMemSize = (kvLen + 8) * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per Q head
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
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
