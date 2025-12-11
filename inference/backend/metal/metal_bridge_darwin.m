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
//
// OPTIMIZED VERSION v3:
// - Uses 2 simdgroups (64 threads) per threadgroup for better occupancy
// Q4_0 constants and helpers
constant int Q4_BLOCK_SIZE = 32;
constant int Q4_BYTES_PER_BLOCK = 18;
constant int Q4_MV_THREADGROUP_SIZE = 256;
constant int Q4_MV_OUTPUTS_PER_TG = 8;  // Multi-output: 8 outputs per threadgroup

// Helper function to convert fp16 to fp32 (for Q4_0 scale)
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

// Q4_0 matvec: each threadgroup handles one output element
// Grid: N threadgroups of 256 threads
// OPTIMIZED: Uses float4 vectorized loads and processes 4 bytes at a time
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

        // Check if we can do full block (most common case)
        if (base_k + 32 <= K) {
            // Process 16 bytes = 32 nibbles using float4 loads for A
            // Load A values as float4 for first 16 elements (low nibbles)
            device const float4* a_vec = (device const float4*)(A + base_k);
            float4 a0 = a_vec[0];  // A[base_k + 0..3]
            float4 a1 = a_vec[1];  // A[base_k + 4..7]
            float4 a2 = a_vec[2];  // A[base_k + 8..11]
            float4 a3 = a_vec[3];  // A[base_k + 12..15]

            // Load A values for second 16 elements (high nibbles)
            device const float4* a_vec_hi = (device const float4*)(A + base_k + 16);
            float4 a4 = a_vec_hi[0];  // A[base_k + 16..19]
            float4 a5 = a_vec_hi[1];  // A[base_k + 20..23]
            float4 a6 = a_vec_hi[2];  // A[base_k + 24..27]
            float4 a7 = a_vec_hi[3];  // A[base_k + 28..31]

            // Process bytes 0-3 (elements 0-3 low, 16-19 high)
            uchar b0 = blockPtr[2];
            uchar b1 = blockPtr[3];
            uchar b2 = blockPtr[4];
            uchar b3 = blockPtr[5];

            float4 q_lo_0 = float4(b0 & 0xF, b1 & 0xF, b2 & 0xF, b3 & 0xF) - 8.0f;
            float4 q_hi_0 = float4(b0 >> 4, b1 >> 4, b2 >> 4, b3 >> 4) - 8.0f;
            sum += scale * dot(a0, q_lo_0);
            sum += scale * dot(a4, q_hi_0);

            // Process bytes 4-7 (elements 4-7 low, 20-23 high)
            uchar b4 = blockPtr[6];
            uchar b5 = blockPtr[7];
            uchar b6 = blockPtr[8];
            uchar b7 = blockPtr[9];

            float4 q_lo_1 = float4(b4 & 0xF, b5 & 0xF, b6 & 0xF, b7 & 0xF) - 8.0f;
            float4 q_hi_1 = float4(b4 >> 4, b5 >> 4, b6 >> 4, b7 >> 4) - 8.0f;
            sum += scale * dot(a1, q_lo_1);
            sum += scale * dot(a5, q_hi_1);

            // Process bytes 8-11 (elements 8-11 low, 24-27 high)
            uchar b8 = blockPtr[10];
            uchar b9 = blockPtr[11];
            uchar b10 = blockPtr[12];
            uchar b11 = blockPtr[13];

            float4 q_lo_2 = float4(b8 & 0xF, b9 & 0xF, b10 & 0xF, b11 & 0xF) - 8.0f;
            float4 q_hi_2 = float4(b8 >> 4, b9 >> 4, b10 >> 4, b11 >> 4) - 8.0f;
            sum += scale * dot(a2, q_lo_2);
            sum += scale * dot(a6, q_hi_2);

            // Process bytes 12-15 (elements 12-15 low, 28-31 high)
            uchar b12 = blockPtr[14];
            uchar b13 = blockPtr[15];
            uchar b14 = blockPtr[16];
            uchar b15 = blockPtr[17];

            float4 q_lo_3 = float4(b12 & 0xF, b13 & 0xF, b14 & 0xF, b15 & 0xF) - 8.0f;
            float4 q_hi_3 = float4(b12 >> 4, b13 >> 4, b14 >> 4, b15 >> 4) - 8.0f;
            sum += scale * dot(a3, q_lo_3);
            sum += scale * dot(a7, q_hi_3);
        } else {
            // Partial block - use scalar fallback
            for (int i = 0; i < 16 && base_k + i < K; i++) {
                uchar byte_val = blockPtr[2 + i];
                int k0 = base_k + i;
                int q0 = byte_val & 0x0F;
                sum += A[k0] * scale * float(q0 - 8);

                int k1 = base_k + i + 16;
                if (k1 < K) {
                    int q1 = (byte_val >> 4) & 0x0F;
                    sum += A[k1] * scale * float(q1 - 8);
                }
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

// Q4_0 matvec MULTI-OUTPUT: compute 8 outputs per threadgroup
// Each simdgroup (32 threads) handles one output
// 8 simdgroups = 8 outputs, each thread processes K/32 blocks
// Grid: ceil(N/8) threadgroups of 256 threads
kernel void matvec_q4_0_multi_output_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_0 format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles 8 outputs: [gid*8 .. gid*8+7]
    // Each simdgroup handles one output
    int output_idx = gid * Q4_MV_OUTPUTS_PER_TG + simd_group;
    if (output_idx >= N) return;

    float sum = 0.0f;

    // Q4_0 row layout
    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q4_BYTES_PER_BLOCK;

    // Each thread in simdgroup handles blocks with stride 32
    // For K=2048 (64 blocks), each thread handles 2 blocks
    for (int block = simd_lane; block < numBlocks; block += 32) {
        device const uchar* blockPtr = b_row + block * Q4_BYTES_PER_BLOCK;

        // Read f16 scale
        ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
        float scale = q4_f16_to_f32(scale_u16);

        int base_k = block * Q4_BLOCK_SIZE;

        if (base_k + 32 <= K) {
            // Full block - vectorized processing
            device const float4* a_vec = (device const float4*)(A + base_k);
            float4 a0 = a_vec[0], a1 = a_vec[1], a2 = a_vec[2], a3 = a_vec[3];

            device const float4* a_vec_hi = (device const float4*)(A + base_k + 16);
            float4 a4 = a_vec_hi[0], a5 = a_vec_hi[1], a6 = a_vec_hi[2], a7 = a_vec_hi[3];

            // Process bytes 0-3
            uchar b0 = blockPtr[2], b1 = blockPtr[3], b2 = blockPtr[4], b3 = blockPtr[5];
            sum += scale * dot(a0, float4(b0 & 0xF, b1 & 0xF, b2 & 0xF, b3 & 0xF) - 8.0f);
            sum += scale * dot(a4, float4(b0 >> 4, b1 >> 4, b2 >> 4, b3 >> 4) - 8.0f);

            // Process bytes 4-7
            uchar b4 = blockPtr[6], b5 = blockPtr[7], b6 = blockPtr[8], b7 = blockPtr[9];
            sum += scale * dot(a1, float4(b4 & 0xF, b5 & 0xF, b6 & 0xF, b7 & 0xF) - 8.0f);
            sum += scale * dot(a5, float4(b4 >> 4, b5 >> 4, b6 >> 4, b7 >> 4) - 8.0f);

            // Process bytes 8-11
            uchar b8 = blockPtr[10], b9 = blockPtr[11], b10 = blockPtr[12], b11 = blockPtr[13];
            sum += scale * dot(a2, float4(b8 & 0xF, b9 & 0xF, b10 & 0xF, b11 & 0xF) - 8.0f);
            sum += scale * dot(a6, float4(b8 >> 4, b9 >> 4, b10 >> 4, b11 >> 4) - 8.0f);

            // Process bytes 12-15
            uchar b12 = blockPtr[14], b13 = blockPtr[15], b14 = blockPtr[16], b15 = blockPtr[17];
            sum += scale * dot(a3, float4(b12 & 0xF, b13 & 0xF, b14 & 0xF, b15 & 0xF) - 8.0f);
            sum += scale * dot(a7, float4(b12 >> 4, b13 >> 4, b14 >> 4, b15 >> 4) - 8.0f);
        } else {
            // Partial block - scalar fallback
            for (int i = 0; i < 16 && base_k + i < K; i++) {
                uchar byte_val = blockPtr[2 + i];
                int k0 = base_k + i;
                sum += A[k0] * scale * float((byte_val & 0x0F) - 8);
                int k1 = base_k + i + 16;
                if (k1 < K) {
                    sum += A[k1] * scale * float(((byte_val >> 4) & 0x0F) - 8);
                }
            }
        }
    }

    // Simdgroup reduction - only 32 threads, single simd_sum needed
    sum = simd_sum(sum);

    // Lane 0 of each simdgroup writes its output
    if (simd_lane == 0) {
        C[output_idx] = sum;
    }
}

// Q4_0 matmul: C[m,n] = A[m,k] @ B[n,k]^T
// Each threadgroup handles ONE output element
// Grid: (N, M) threadgroups of 256 threads
// OPTIMIZED: Uses float4 vectorized loads and dot products

kernel void matmul_q4_0_batched_f32(
    device const float* A [[buffer(0)]],           // [M, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_0 format
    device float* C [[buffer(2)]],                 // [M, N] output
    constant int& M [[buffer(3)]],                 // Number of input rows
    constant int& N [[buffer(4)]],                 // Number of output columns
    constant int& K [[buffer(5)]],                 // Inner dimension
    threadgroup float* shared [[threadgroup(0)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // gid.x = output column (n), gid.y = input row (m)
    int n = gid.x;
    int m = gid.y;

    if (n >= N || m >= M) return;

    float sum = 0.0f;

    // A row pointer: A[m, :]
    device const float* a_row = A + m * K;

    // B row pointer in Q4_0 format: B[n, :]
    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    device const uchar* b_row = B + n * numBlocks * Q4_BYTES_PER_BLOCK;

    // Each thread handles some blocks
    for (int block = tid; block < numBlocks; block += Q4_MV_THREADGROUP_SIZE) {
        device const uchar* blockPtr = b_row + block * Q4_BYTES_PER_BLOCK;

        // Read f16 scale
        ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
        float scale = q4_f16_to_f32(scale_u16);

        int base_k = block * Q4_BLOCK_SIZE;

        // Check if we can do full block (most common case)
        if (base_k + 32 <= K) {
            // Process 16 bytes = 32 nibbles using float4 loads for A
            device const float4* a_vec = (device const float4*)(a_row + base_k);
            float4 a0 = a_vec[0];
            float4 a1 = a_vec[1];
            float4 a2 = a_vec[2];
            float4 a3 = a_vec[3];

            device const float4* a_vec_hi = (device const float4*)(a_row + base_k + 16);
            float4 a4 = a_vec_hi[0];
            float4 a5 = a_vec_hi[1];
            float4 a6 = a_vec_hi[2];
            float4 a7 = a_vec_hi[3];

            // Process bytes 0-3
            uchar b0 = blockPtr[2];
            uchar b1 = blockPtr[3];
            uchar b2 = blockPtr[4];
            uchar b3 = blockPtr[5];

            float4 q_lo_0 = float4(b0 & 0xF, b1 & 0xF, b2 & 0xF, b3 & 0xF) - 8.0f;
            float4 q_hi_0 = float4(b0 >> 4, b1 >> 4, b2 >> 4, b3 >> 4) - 8.0f;
            sum += scale * dot(a0, q_lo_0);
            sum += scale * dot(a4, q_hi_0);

            // Process bytes 4-7
            uchar b4 = blockPtr[6];
            uchar b5 = blockPtr[7];
            uchar b6 = blockPtr[8];
            uchar b7 = blockPtr[9];

            float4 q_lo_1 = float4(b4 & 0xF, b5 & 0xF, b6 & 0xF, b7 & 0xF) - 8.0f;
            float4 q_hi_1 = float4(b4 >> 4, b5 >> 4, b6 >> 4, b7 >> 4) - 8.0f;
            sum += scale * dot(a1, q_lo_1);
            sum += scale * dot(a5, q_hi_1);

            // Process bytes 8-11
            uchar b8 = blockPtr[10];
            uchar b9 = blockPtr[11];
            uchar b10 = blockPtr[12];
            uchar b11 = blockPtr[13];

            float4 q_lo_2 = float4(b8 & 0xF, b9 & 0xF, b10 & 0xF, b11 & 0xF) - 8.0f;
            float4 q_hi_2 = float4(b8 >> 4, b9 >> 4, b10 >> 4, b11 >> 4) - 8.0f;
            sum += scale * dot(a2, q_lo_2);
            sum += scale * dot(a6, q_hi_2);

            // Process bytes 12-15
            uchar b12 = blockPtr[14];
            uchar b13 = blockPtr[15];
            uchar b14 = blockPtr[16];
            uchar b15 = blockPtr[17];

            float4 q_lo_3 = float4(b12 & 0xF, b13 & 0xF, b14 & 0xF, b15 & 0xF) - 8.0f;
            float4 q_hi_3 = float4(b12 >> 4, b13 >> 4, b14 >> 4, b15 >> 4) - 8.0f;
            sum += scale * dot(a3, q_lo_3);
            sum += scale * dot(a7, q_hi_3);
        } else {
            // Partial block - use scalar fallback
            for (int i = 0; i < 16 && base_k + i < K; i++) {
                uchar byte_val = blockPtr[2 + i];
                int k0 = base_k + i;
                int q0 = byte_val & 0x0F;
                sum += a_row[k0] * scale * float(q0 - 8);

                int k1 = base_k + i + 16;
                if (k1 < K) {
                    int q1 = (byte_val >> 4) & 0x0F;
                    sum += a_row[k1] * scale * float(q1 - 8);
                }
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
            // Output: C[m, n]
            C[m * N + n] = warp_sum;
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

    // Interleaved layout: pairs are (0,1), (2,3), (4,5), ...
    // This matches llama.cpp's implementation for Llama-family models.
    for (int j = 0; j < halfDim; j++) {
        int idx = j * 2;
        // Frequency for this dimension pair
        float freq = 1.0f / pow(theta, float(2 * j) / float(headDim));
        float angle = float(absPos) * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        // Apply rotation to Q using interleaved pairs (2j and 2j+1)
        float q0 = q[offset + idx];
        float q1 = q[offset + idx + 1];
        q[offset + idx] = q0 * cos_val - q1 * sin_val;
        q[offset + idx + 1] = q0 * sin_val + q1 * cos_val;

        // Apply to K
        float k0 = k[offset + idx];
        float k1 = k[offset + idx + 1];
        k[offset + idx] = k0 * cos_val - k1 * sin_val;
        k[offset + idx + 1] = k0 * sin_val + k1 * cos_val;
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

    // Process Q heads - interleaved layout: pairs are (0,1), (2,3), (4,5), ...
    if (head < numQHeads) {
        int qOffset = (pos * numQHeads + head) * headDim;
        for (int j = 0; j < halfDim; j++) {
            int idx = j * 2;
            float freq = 1.0f / pow(theta, float(2 * j) / float(headDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            float q0 = q[qOffset + idx];
            float q1 = q[qOffset + idx + 1];
            q[qOffset + idx] = q0 * cos_val - q1 * sin_val;
            q[qOffset + idx + 1] = q0 * sin_val + q1 * cos_val;
        }
    }

    // Process K heads (fewer due to GQA) - interleaved layout
    if (head < numKVHeads) {
        int kOffset = (pos * numKVHeads + head) * headDim;
        for (int j = 0; j < halfDim; j++) {
            int idx = j * 2;
            float freq = 1.0f / pow(theta, float(2 * j) / float(headDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            float k0 = k[kOffset + idx];
            float k1 = k[kOffset + idx + 1];
            k[kOffset + idx] = k0 * cos_val - k1 * sin_val;
            k[kOffset + idx + 1] = k0 * sin_val + k1 * cos_val;
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

// Fused SiLU+Mul: out = silu(gate) * up
// Combines activation and multiplication into single kernel
kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = gate[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = silu_g * up[gid];
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

// Tiled SDPA for prefill with causal masking (Flash Attention style)
// Uses online softmax to compute attention in a single pass with tiling
// Each threadgroup handles one (qPos, qHead) pair with threads parallelizing over headDim
// Q: [seqLen, numQHeads, headDim]
// K: [seqLen, numKVHeads, headDim]
// V: [seqLen, numKVHeads, headDim]
// out: [seqLen, numQHeads, headDim]
constant int PREFILL_THREADS = 64;  // Threads per threadgroup
constant int PREFILL_TILE_K = 16;   // K positions per tile

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
    threadgroup float* shared [[threadgroup(0)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int qPos = gid.x;
    int qHead = gid.y;

    if (qPos >= seqLen || qHead >= numQHeads) return;

    // GQA mapping
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;

    // Q layout: [seqLen, numQHeads, headDim]
    int qOffset = qPos * numQHeads * headDim + qHead * headDim;

    // Causal attention: only attend to positions <= qPos
    int maxKLen = qPos + 1;

    // Shared memory layout:
    // [0..PREFILL_TILE_K-1]: attention scores for current tile
    // [PREFILL_TILE_K..PREFILL_TILE_K+7]: warp scratch for reduction
    threadgroup float* scores = shared;
    threadgroup float* warpScratch = shared + PREFILL_TILE_K;

    // Online softmax state
    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    // Output accumulator (per-thread for dimensions it handles)
    // Each thread handles headDim/PREFILL_THREADS dimensions
    float acc[4] = {0.0f};  // Assuming headDim <= 256, max 4 dims per thread with 64 threads

    // Process K in tiles
    for (int tileStart = 0; tileStart < maxKLen; tileStart += PREFILL_TILE_K) {
        int tileEnd = min(tileStart + PREFILL_TILE_K, maxKLen);
        int tileSize = tileEnd - tileStart;

        // Phase 1: Compute Q·K scores for this tile (parallel over K positions)
        // Each thread computes one score
        float localScore = -INFINITY;
        if ((int)tid < tileSize) {
            int kPos = tileStart + tid;
            int kOffset = kPos * numKVHeads * headDim + kvHead * headDim;

            float dot = 0.0f;
            for (int d = 0; d < headDim; d++) {
                dot += Q[qOffset + d] * K[kOffset + d];
            }
            localScore = dot * scale;
        }

        // Store score to shared memory
        if ((int)tid < tileSize) {
            scores[tid] = localScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: Find tile max (parallel reduction)
        float tileMax = -INFINITY;
        for (int i = tid; i < tileSize; i += PREFILL_THREADS) {
            tileMax = max(tileMax, scores[i]);
        }
        tileMax = simd_max(tileMax);
        if (simd_lane == 0) {
            warpScratch[simd_group] = tileMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 2) {
            tileMax = max(warpScratch[0], warpScratch[1]);
            warpScratch[0] = tileMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tileMax = warpScratch[0];

        // Phase 3: Online softmax update
        float newMax = max(runningMax, tileMax);
        float rescale = exp(runningMax - newMax);

        // Rescale existing accumulator
        for (int i = 0; i < 4; i++) {
            acc[i] *= rescale;
        }
        runningSum *= rescale;

        // Compute exp(score - newMax) and sum for this tile
        float tileSum = 0.0f;
        for (int i = tid; i < tileSize; i += PREFILL_THREADS) {
            float expScore = exp(scores[i] - newMax);
            scores[i] = expScore;  // Store for V accumulation
            tileSum += expScore;
        }

        // Reduce tile sum
        tileSum = simd_sum(tileSum);
        if (simd_lane == 0) {
            warpScratch[simd_group] = tileSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 2) {
            tileSum = warpScratch[0] + warpScratch[1];
            warpScratch[0] = tileSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tileSum = warpScratch[0];

        runningSum += tileSum;
        runningMax = newMax;

        // Phase 4: Accumulate weighted V
        // Each thread handles specific dimensions
        for (int kIdx = 0; kIdx < tileSize; kIdx++) {
            float weight = scores[kIdx];
            int kPos = tileStart + kIdx;
            int vOffset = kPos * numKVHeads * headDim + kvHead * headDim;

            // Each thread accumulates for its assigned dimensions
            for (int i = 0; i < 4 && (int)(tid + i * PREFILL_THREADS) < headDim; i++) {
                int d = tid + i * PREFILL_THREADS;
                acc[i] += weight * V[vOffset + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output (each thread writes its dimensions)
    int outOffset = qPos * numQHeads * headDim + qHead * headDim;
    float invSum = 1.0f / runningSum;

    for (int i = 0; i < 4 && (int)(tid + i * PREFILL_THREADS) < headDim; i++) {
        int d = tid + i * PREFILL_THREADS;
        out[outOffset + d] = acc[i] * invSum;
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
    [commandBuffer waitUntilCompleted];
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

// =============================================================================
// Command Buffer Batching
// =============================================================================
// Global batch state (thread-local would be better for multi-threading)
static id<MTLCommandQueue> g_batchQueue = nil;
static id<MTLCommandBuffer> g_batchCmdBuffer = nil;
static id<MTLComputeCommandEncoder> g_batchEncoder = nil;

// Begin a batch of operations - all subsequent kernel dispatches use the same command buffer
void metal_begin_batch(void* queuePtr) {
    if (g_batchEncoder != nil) {
        // Already in a batch - this shouldn't happen, but handle gracefully
        return;
    }
    g_batchQueue = (__bridge id<MTLCommandQueue>)queuePtr;
    g_batchCmdBuffer = [g_batchQueue commandBuffer];
    g_batchEncoder = [g_batchCmdBuffer computeCommandEncoder];
}

// End the batch and commit all operations
void metal_end_batch(void) {
    if (g_batchEncoder == nil) {
        return;
    }
    [g_batchEncoder endEncoding];
    [g_batchCmdBuffer commit];
    g_batchEncoder = nil;
    g_batchCmdBuffer = nil;
    g_batchQueue = nil;
}

// Check if we're in batch mode
static inline bool is_batch_mode(void) {
    return g_batchEncoder != nil;
}

// Helper: get encoder and whether caller should commit
// If batching, returns the batch encoder and *commit=false
// Otherwise creates a new command buffer/encoder and *commit=true
static id<MTLComputeCommandEncoder> get_encoder(id<MTLCommandQueue> queue, id<MTLCommandBuffer>* cmdBufOut, bool* shouldCommit) {
    if (is_batch_mode()) {
        *cmdBufOut = nil;
        *shouldCommit = false;
        return g_batchEncoder;
    }
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    *cmdBufOut = cmdBuf;
    *shouldCommit = true;
    return [cmdBuf computeCommandEncoder];
}

// Helper: finish encoding and possibly commit
static inline void finish_encode(id<MTLComputeCommandEncoder> encoder, id<MTLCommandBuffer> cmdBuf, bool shouldCommit) {
    if (shouldCommit) {
        [encoder endEncoding];
        [cmdBuf commit];
    }
    // In batch mode, don't end encoding - let metal_end_batch do it
}

// Helper to dispatch a compute kernel (supports batching)
static void dispatch_kernel(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                           NSArray* buffers, NSArray* constants, MTLSize gridSize) {
    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

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
    finish_encode(encoder, cmdBuffer, shouldCommit);
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
// Each threadgroup handles one output element with 256 threads
void metal_matvec_q4_0_transposed_f32(void* queuePtr, void* pipelinePtr,
                                       void* A, void* B, void* C,
                                       int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    // One threadgroup per output element, 256 threads per threadgroup
    int threadgroupSize = 256;
    int numWarps = (threadgroupSize + 31) / 32;
    int sharedMemSize = numWarps * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(N, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 multi-output matvec: 8 outputs per threadgroup using simdgroups
// Grid: ceil(N/8) threadgroups of 256 threads
void metal_matvec_q4_0_multi_output_f32(void* queuePtr, void* pipelinePtr,
                                         void* A, void* B, void* C,
                                         int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    // 8 outputs per threadgroup, 256 threads (8 simdgroups of 32)
    int outputsPerTG = 8;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 batched matmul: one threadgroup per output element
// Grid: (N, M) threadgroups of 256 threads
void metal_matmul_q4_0_batched_f32(void* queuePtr, void* pipelinePtr,
                                    void* A, void* B, void* C,
                                    int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int threadgroupSize = 256;
    int numWarps = (threadgroupSize + 31) / 32;
    int sharedMemSize = numWarps * sizeof(float);

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // 2D grid: (N, M) threadgroups - one per output element
    MTLSize threadgroups = MTLSizeMake(N, M, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
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

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

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

    finish_encode(encoder, cmdBuffer, shouldCommit);
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

// Fused SiLU+Mul: out = silu(gate) * up
void metal_silu_mul_f32(void* queuePtr, void* pipelinePtr,
                        void* gate, void* up, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)gate,
        (__bridge id<MTLBuffer>)up,
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

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

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
    MTLSize threadgroupSize = MTLSizeMake(MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)numQHeads), 1, 1);
    MTLSize threadgroups = MTLSizeMake((numQHeads + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Flash Decoding - parallelized SDPA for decode phase
// Uses threadgroups with shared memory for scores and accumulation
void metal_sdpa_flash_decode_f32(void* queuePtr, void* pipelinePtr,
                                  void* Q, void* K, void* V, void* out,
                                  int kvLen, int numQHeads, int numKVHeads, int headDim,
                                  float scale) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

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
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_sdpa_prefill_f32(void* queuePtr, void* pipelinePtr,
                            void* Q, void* K, void* V, void* out,
                            int seqLen, int numQHeads, int numKVHeads, int headDim,
                            float scale) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

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

    // Tiled Flash Attention: each threadgroup handles one (qPos, qHead) pair
    // PREFILL_THREADS = 64 threads per threadgroup
    // Shared memory: PREFILL_TILE_K (16) + warp scratch (8) = 24 floats
    int PREFILL_THREADS = 64;
    int PREFILL_TILE_K = 16;
    int sharedMemSize = (PREFILL_TILE_K + 8) * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // Dispatch grid: (seqLen, numQHeads) threadgroups
    MTLSize threadgroups = MTLSizeMake(seqLen, numQHeads, 1);
    MTLSize threadsPerGroup = MTLSizeMake(PREFILL_THREADS, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Legacy interface (kept for compatibility)
void metal_scaled_dot_product_attention(void* queuePtr,
                                        void* Q, void* K, void* V, void* out,
                                        int batchSize, int numHeads, int seqLen, int headDim,
                                        float scale, int causal) {
    // Deprecated - use metal_sdpa_decode_f32 or metal_sdpa_prefill_f32 instead
}
