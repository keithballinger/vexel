// metal_bridge.m - Objective-C Metal implementation
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <mach/mach_time.h>
#include "metal_bridge.h"

// Embedded Metal shader source
static NSString* metalShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Matrix multiplication: C = A @ B (non-transposed)
// A: [M, K], B: [K, N], C: [M, N]
// Each thread computes one element of C
kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int row = gid.y;
    int col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A[row, k] * B[k, col]
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

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

        // Scalar path for correctness across partial blocks
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
// OPTIMIZED: Uses as_type<half> for fast fp16 conversion
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

        // Read f16 scale using fast hardware conversion
        float scale = float(as_type<half>(*((device const ushort*)blockPtr)));

        int base_k = block * Q4_BLOCK_SIZE;

        if (base_k + 32 <= K) {
            // Full block: vectorized path using float4 loads
            device const float4* a_vec = (device const float4*)(A + base_k);
            float4 a0 = a_vec[0], a1 = a_vec[1], a2 = a_vec[2], a3 = a_vec[3];

            device const float4* a_vec_hi = (device const float4*)(A + base_k + 16);
            float4 a4 = a_vec_hi[0], a5 = a_vec_hi[1], a6 = a_vec_hi[2], a7 = a_vec_hi[3];

            // Bytes 0-3
            uchar b0 = blockPtr[2], b1 = blockPtr[3], b2 = blockPtr[4], b3 = blockPtr[5];
            sum += scale * dot(a0, float4(b0 & 0xF, b1 & 0xF, b2 & 0xF, b3 & 0xF) - 8.0f);
            sum += scale * dot(a4, float4(b0 >> 4, b1 >> 4, b2 >> 4, b3 >> 4) - 8.0f);

            // Bytes 4-7
            uchar b4 = blockPtr[6], b5 = blockPtr[7], b6 = blockPtr[8], b7 = blockPtr[9];
            sum += scale * dot(a1, float4(b4 & 0xF, b5 & 0xF, b6 & 0xF, b7 & 0xF) - 8.0f);
            sum += scale * dot(a5, float4(b4 >> 4, b5 >> 4, b6 >> 4, b7 >> 4) - 8.0f);

            // Bytes 8-11
            uchar b8 = blockPtr[10], b9 = blockPtr[11], b10 = blockPtr[12], b11 = blockPtr[13];
            sum += scale * dot(a2, float4(b8 & 0xF, b9 & 0xF, b10 & 0xF, b11 & 0xF) - 8.0f);
            sum += scale * dot(a6, float4(b8 >> 4, b9 >> 4, b10 >> 4, b11 >> 4) - 8.0f);

            // Bytes 12-15
            uchar b12 = blockPtr[14], b13 = blockPtr[15], b14 = blockPtr[16], b15 = blockPtr[17];
            sum += scale * dot(a3, float4(b12 & 0xF, b13 & 0xF, b14 & 0xF, b15 & 0xF) - 8.0f);
            sum += scale * dot(a7, float4(b12 >> 4, b13 >> 4, b14 >> 4, b15 >> 4) - 8.0f);
        } else {
            // Partial block: scalar fallback
            for (int i = 0; i < 16 && base_k + i < K; i++) {
                uchar byte_val = blockPtr[2 + i];
                int k0 = base_k + i;
                int k1 = base_k + i + 16;

                sum += A[k0] * scale * float((byte_val & 0x0F) - 8);
                if (k1 < K) {
                    sum += A[k1] * scale * float((byte_val >> 4) - 8);
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

// =============================================================================
// Q4_0 NR2 MATVEC: 2 outputs per simdgroup (llama.cpp NR0 technique)
// =============================================================================
// Each simdgroup computes 2 outputs instead of 1, amortizing activation loads.
// No shared memory barrier - each thread loads activations into registers.
// Grid: ceil(N/16) threadgroups of 256 threads (8 simdgroups * 2 outputs = 16)
constant int Q4_NR2_OUTPUTS_PER_SG = 2;
constant int Q4_NR2_OUTPUTS_PER_TG = 16;  // 8 simdgroups * 2 outputs

kernel void matvec_q4_0_nr2_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_0 format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each simdgroup handles 2 consecutive outputs
    int base_output = gid * Q4_NR2_OUTPUTS_PER_TG + simd_group * Q4_NR2_OUTPUTS_PER_SG;
    int out0 = base_output;
    int out1 = base_output + 1;

    // Early exit if both outputs are out of bounds
    if (out0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    // Calculate row pointers
    device const uchar* b_row0 = B + out0 * numBlocks * Q4_BYTES_PER_BLOCK;
    device const uchar* b_row1 = (out1 < N) ? B + out1 * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;

    // Each thread in simdgroup handles blocks with stride 32
    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * Q4_BLOCK_SIZE;
        if (base_k >= K) break;

        bool full_block = (base_k + Q4_BLOCK_SIZE) <= K;

        if (full_block) {
            // Load activations ONCE for both outputs (32 floats per block)
            device const float4* a_vec = (device const float4*)(A + base_k);
            float4 a0 = a_vec[0], a1 = a_vec[1], a2 = a_vec[2], a3 = a_vec[3];
            float4 a4 = a_vec[4], a5 = a_vec[5], a6 = a_vec[6], a7 = a_vec[7];

            // Process output 0 - use uint16 reads for packed nibbles
            {
                device const uchar* blockPtr = b_row0 + block * Q4_BYTES_PER_BLOCK;
                float scale = float(as_type<half>(*((device const ushort*)blockPtr)));

                device const ushort* qs16 = (device const ushort*)(blockPtr + 2);
                ushort w0 = qs16[0], w1 = qs16[1], w2 = qs16[2], w3 = qs16[3];
                ushort w4 = qs16[4], w5 = qs16[5], w6 = qs16[6], w7 = qs16[7];

                sum0 += scale * (
                    dot(a0, float4( w0        & 0xF,  (w0 >> 8) & 0xF,  w1        & 0xF,  (w1 >> 8) & 0xF) - 8.0f) +
                    dot(a4, float4((w0 >> 4)  & 0xF,  (w0 >> 12)& 0xF, (w1 >> 4)  & 0xF,  (w1 >> 12)& 0xF) - 8.0f) +
                    dot(a1, float4( w2        & 0xF,  (w2 >> 8) & 0xF,  w3        & 0xF,  (w3 >> 8) & 0xF) - 8.0f) +
                    dot(a5, float4((w2 >> 4)  & 0xF,  (w2 >> 12)& 0xF, (w3 >> 4)  & 0xF,  (w3 >> 12)& 0xF) - 8.0f) +
                    dot(a2, float4( w4        & 0xF,  (w4 >> 8) & 0xF,  w5        & 0xF,  (w5 >> 8) & 0xF) - 8.0f) +
                    dot(a6, float4((w4 >> 4)  & 0xF,  (w4 >> 12)& 0xF, (w5 >> 4)  & 0xF,  (w5 >> 12)& 0xF) - 8.0f) +
                    dot(a3, float4( w6        & 0xF,  (w6 >> 8) & 0xF,  w7        & 0xF,  (w7 >> 8) & 0xF) - 8.0f) +
                    dot(a7, float4((w6 >> 4)  & 0xF,  (w6 >> 12)& 0xF, (w7 >> 4)  & 0xF,  (w7 >> 12)& 0xF) - 8.0f));
            }

            // Process output 1 (if valid)
            if (b_row1) {
                device const uchar* blockPtr = b_row1 + block * Q4_BYTES_PER_BLOCK;
                float scale = float(as_type<half>(*((device const ushort*)blockPtr)));

                device const ushort* qs16 = (device const ushort*)(blockPtr + 2);
                ushort w0 = qs16[0], w1 = qs16[1], w2 = qs16[2], w3 = qs16[3];
                ushort w4 = qs16[4], w5 = qs16[5], w6 = qs16[6], w7 = qs16[7];

                sum1 += scale * (
                    dot(a0, float4( w0        & 0xF,  (w0 >> 8) & 0xF,  w1        & 0xF,  (w1 >> 8) & 0xF) - 8.0f) +
                    dot(a4, float4((w0 >> 4)  & 0xF,  (w0 >> 12)& 0xF, (w1 >> 4)  & 0xF,  (w1 >> 12)& 0xF) - 8.0f) +
                    dot(a1, float4( w2        & 0xF,  (w2 >> 8) & 0xF,  w3        & 0xF,  (w3 >> 8) & 0xF) - 8.0f) +
                    dot(a5, float4((w2 >> 4)  & 0xF,  (w2 >> 12)& 0xF, (w3 >> 4)  & 0xF,  (w3 >> 12)& 0xF) - 8.0f) +
                    dot(a2, float4( w4        & 0xF,  (w4 >> 8) & 0xF,  w5        & 0xF,  (w5 >> 8) & 0xF) - 8.0f) +
                    dot(a6, float4((w4 >> 4)  & 0xF,  (w4 >> 12)& 0xF, (w5 >> 4)  & 0xF,  (w5 >> 12)& 0xF) - 8.0f) +
                    dot(a3, float4( w6        & 0xF,  (w6 >> 8) & 0xF,  w7        & 0xF,  (w7 >> 8) & 0xF) - 8.0f) +
                    dot(a7, float4((w6 >> 4)  & 0xF,  (w6 >> 12)& 0xF, (w7 >> 4)  & 0xF,  (w7 >> 12)& 0xF) - 8.0f));
            }
        } else {
            // Partial block fallback to avoid out-of-bounds reads on A
            device const uchar* blockPtr0 = b_row0 + block * Q4_BYTES_PER_BLOCK;
            float scale0 = float(as_type<half>(*((device const ushort*)blockPtr0)));

            for (int i = 0; i < 16 && (base_k + i) < K; i++) {
                uchar byte_val = blockPtr0[2 + i];
                int k0 = base_k + i;
                int k1 = base_k + i + 16;

                sum0 += A[k0] * scale0 * float((byte_val & 0x0F) - 8);
                if (k1 < K) {
                    sum0 += A[k1] * scale0 * float(((byte_val >> 4) & 0x0F) - 8);
                }
            }

            if (b_row1) {
                device const uchar* blockPtr1 = b_row1 + block * Q4_BYTES_PER_BLOCK;
                float scale1 = float(as_type<half>(*((device const ushort*)blockPtr1)));

                for (int i = 0; i < 16 && (base_k + i) < K; i++) {
                    uchar byte_val = blockPtr1[2 + i];
                    int k0 = base_k + i;
                    int k1 = base_k + i + 16;

                    sum1 += A[k0] * scale1 * float((byte_val & 0x0F) - 8);
                    if (k1 < K) {
                        sum1 += A[k1] * scale1 * float(((byte_val >> 4) & 0x0F) - 8);
                    }
                }
            }
        }
    }

    // Simdgroup reduction
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);

    // Lane 0 writes outputs
    if (simd_lane == 0) {
        C[out0] = sum0;
        if (out1 < N) C[out1] = sum1;
    }
}

// =============================================================================
// Q4_0 NR4 MATVEC: 4 outputs per simdgroup (llama.cpp NR0=4 technique)
// =============================================================================
// Each simdgroup computes 4 outputs, maximizing activation reuse.
// Grid: ceil(N/32) threadgroups of 256 threads (8 simdgroups * 4 outputs = 32)
constant int Q4_NR4_OUTPUTS_PER_SG = 4;
constant int Q4_NR4_OUTPUTS_PER_TG = 32;

kernel void matvec_q4_0_nr4_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base_output = gid * Q4_NR4_OUTPUTS_PER_TG + simd_group * Q4_NR4_OUTPUTS_PER_SG;
    if (base_output >= N) return;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    // Calculate row pointers for 4 outputs
    device const uchar* b_row0 = B + base_output * numBlocks * Q4_BYTES_PER_BLOCK;
    device const uchar* b_row1 = (base_output + 1 < N) ? B + (base_output + 1) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    device const uchar* b_row2 = (base_output + 2 < N) ? B + (base_output + 2) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    device const uchar* b_row3 = (base_output + 3 < N) ? B + (base_output + 3) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;

    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * Q4_BLOCK_SIZE;
        if (base_k >= K) break;

        // Load activations ONCE for all 4 outputs
        device const float4* a_vec = (device const float4*)(A + base_k);
        float4 a0 = a_vec[0], a1 = a_vec[1], a2 = a_vec[2], a3 = a_vec[3];
        float4 a4 = a_vec[4], a5 = a_vec[5], a6 = a_vec[6], a7 = a_vec[7];

        // Macro to process one output row
        #define PROCESS_Q4_ROW(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* bp = row_ptr + block * Q4_BYTES_PER_BLOCK; \
            float s = float(as_type<half>(*((device const ushort*)bp))); \
            uchar b0 = bp[2], b1 = bp[3], b2 = bp[4], b3 = bp[5]; \
            uchar b4 = bp[6], b5 = bp[7], b6 = bp[8], b7 = bp[9]; \
            uchar b8 = bp[10], b9 = bp[11], b10 = bp[12], b11 = bp[13]; \
            uchar b12 = bp[14], b13 = bp[15], b14 = bp[16], b15 = bp[17]; \
            sum_var += s * (dot(a0, float4(b0 & 0xF, b1 & 0xF, b2 & 0xF, b3 & 0xF) - 8.0f) + \
                           dot(a4, float4(b0 >> 4, b1 >> 4, b2 >> 4, b3 >> 4) - 8.0f) + \
                           dot(a1, float4(b4 & 0xF, b5 & 0xF, b6 & 0xF, b7 & 0xF) - 8.0f) + \
                           dot(a5, float4(b4 >> 4, b5 >> 4, b6 >> 4, b7 >> 4) - 8.0f) + \
                           dot(a2, float4(b8 & 0xF, b9 & 0xF, b10 & 0xF, b11 & 0xF) - 8.0f) + \
                           dot(a6, float4(b8 >> 4, b9 >> 4, b10 >> 4, b11 >> 4) - 8.0f) + \
                           dot(a3, float4(b12 & 0xF, b13 & 0xF, b14 & 0xF, b15 & 0xF) - 8.0f) + \
                           dot(a7, float4(b12 >> 4, b13 >> 4, b14 >> 4, b15 >> 4) - 8.0f)); \
        }

        PROCESS_Q4_ROW(b_row0, sum0);
        PROCESS_Q4_ROW(b_row1, sum1);
        PROCESS_Q4_ROW(b_row2, sum2);
        PROCESS_Q4_ROW(b_row3, sum3);
        #undef PROCESS_Q4_ROW
    }

    // Simdgroup reductions
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    // Lane 0 writes outputs
    if (simd_lane == 0) {
        C[base_output] = sum0;
        if (base_output + 1 < N) C[base_output + 1] = sum1;
        if (base_output + 2 < N) C[base_output + 2] = sum2;
        if (base_output + 3 < N) C[base_output + 3] = sum3;
    }
}

// =============================================================================
// Q4_0 COLLABORATIVE MATVEC (llama.cpp-style thread collaboration)
// =============================================================================
// Key insight from llama.cpp: 2 threads collaborate per block, each handling 16 elements
// This improves register pressure distribution and memory coalescing.
//
// Thread organization:
// - 32 threads in simdgroup split into 16 pairs
// - Each pair collaborates on one block (2 threads per block)
// - Thread offset: (simd_lane % 2) * 16 gives offset 0 or 16 within block
// - Block index: simd_lane / 2 gives which of 16 blocks this thread pair handles
// - Block stride: 16 (process 16 blocks per iteration across 32 threads)
//
// Grid: ceil(N/32) threadgroups of 256 threads (8 simdgroups * 4 outputs = 32)
constant int Q4_COLLAB_OUTPUTS_PER_SG = 4;
constant int Q4_COLLAB_OUTPUTS_PER_TG = 32;

kernel void matvec_q4_0_collab_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base_output = gid * Q4_COLLAB_OUTPUTS_PER_TG + simd_group * Q4_COLLAB_OUTPUTS_PER_SG;
    if (base_output >= N) return;

    // Thread collaboration: 2 threads per block
    // ix: which of 16 blocks this thread pair handles (0-15)
    // use_high: whether to use high nibbles (elements 16-31) or low nibbles (elements 0-15)
    short ix = simd_lane / 2;  // Block index (0-15)
    bool use_high = (simd_lane % 2) == 1;  // Thread 0 = low nibbles, Thread 1 = high nibbles

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    // Calculate row pointers
    device const uchar* b_row0 = B + base_output * numBlocks * Q4_BYTES_PER_BLOCK;
    device const uchar* b_row1 = (base_output + 1 < N) ? B + (base_output + 1) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    device const uchar* b_row2 = (base_output + 2 < N) ? B + (base_output + 2) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    device const uchar* b_row3 = (base_output + 3 < N) ? B + (base_output + 3) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;

    // Process 16 blocks per iteration (stride 16 instead of 32)
    for (int block = ix; block < numBlocks; block += 16) {
        int base_k = block * Q4_BLOCK_SIZE;
        // For high nibbles (elements 16-31), offset activation read by 16
        int act_offset = use_high ? 16 : 0;
        if (base_k + act_offset >= K) break;

        // Load 16 activation values (this thread's half of the block)
        device const float4* a_vec = (device const float4*)(A + base_k + act_offset);
        float4 a0 = a_vec[0], a1 = a_vec[1], a2 = a_vec[2], a3 = a_vec[3];

        // Macro to process 16 elements for one output row
        // Q4_0 layout: byte[i] low nibble = element[i], high nibble = element[i+16]
        // For low nibbles (use_high=false): elements 0-15 from all 16 bytes
        // For high nibbles (use_high=true): elements 16-31 from all 16 bytes
        #define PROCESS_HALF_BLOCK(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* bp = row_ptr + block * Q4_BYTES_PER_BLOCK; \
            float scale = float(as_type<half>(*((device const ushort*)bp))); \
            device const uchar* qs = bp + 2; \
            float4 q0, q1, q2, q3; \
            if (use_high) { \
                /* Elements 16-19, 20-23, 24-27, 28-31 from high nibbles */ \
                q0 = float4(qs[0] >> 4, qs[1] >> 4, qs[2] >> 4, qs[3] >> 4) - 8.0f; \
                q1 = float4(qs[4] >> 4, qs[5] >> 4, qs[6] >> 4, qs[7] >> 4) - 8.0f; \
                q2 = float4(qs[8] >> 4, qs[9] >> 4, qs[10] >> 4, qs[11] >> 4) - 8.0f; \
                q3 = float4(qs[12] >> 4, qs[13] >> 4, qs[14] >> 4, qs[15] >> 4) - 8.0f; \
            } else { \
                /* Elements 0-3, 4-7, 8-11, 12-15 from low nibbles */ \
                q0 = float4(qs[0] & 0xF, qs[1] & 0xF, qs[2] & 0xF, qs[3] & 0xF) - 8.0f; \
                q1 = float4(qs[4] & 0xF, qs[5] & 0xF, qs[6] & 0xF, qs[7] & 0xF) - 8.0f; \
                q2 = float4(qs[8] & 0xF, qs[9] & 0xF, qs[10] & 0xF, qs[11] & 0xF) - 8.0f; \
                q3 = float4(qs[12] & 0xF, qs[13] & 0xF, qs[14] & 0xF, qs[15] & 0xF) - 8.0f; \
            } \
            sum_var += scale * (dot(a0, q0) + dot(a1, q1) + dot(a2, q2) + dot(a3, q3)); \
        }

        PROCESS_HALF_BLOCK(b_row0, sum0);
        PROCESS_HALF_BLOCK(b_row1, sum1);
        PROCESS_HALF_BLOCK(b_row2, sum2);
        PROCESS_HALF_BLOCK(b_row3, sum3);
        #undef PROCESS_HALF_BLOCK
    }

    // Simdgroup reductions (sum across all 32 threads)
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    // Lane 0 writes outputs
    if (simd_lane == 0) {
        C[base_output] = sum0;
        if (base_output + 1 < N) C[base_output + 1] = sum1;
        if (base_output + 2 < N) C[base_output + 2] = sum2;
        if (base_output + 3 < N) C[base_output + 3] = sum3;
    }
}

// =============================================================================
// Q4_0 OPTIMIZED MATVEC WITH SHARED MEMORY (llama.cpp-inspired)
// =============================================================================
// Key optimizations over multi_output:
// 1. Cache input activations in shared memory - load once, use for all outputs
// 2. Process 4 outputs per simdgroup using NR0 technique
// 3. Use uint16 reads for packed quantized values
// 4. Use as_type<half> for fast fp16->fp32 conversion
//
// Grid: ceil(N/32) threadgroups of 256 threads (8 simdgroups)
// Each simdgroup computes 4 outputs (32 total per threadgroup)
constant int Q4_OPT_OUTPUTS_PER_SG = 4;
constant int Q4_OPT_OUTPUTS_PER_TG = 32;  // 8 simdgroups * 4 outputs

kernel void matvec_q4_0_optimized_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_0 format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    threadgroup float* shared_A [[threadgroup(0)]], // [K] shared activations
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperatively load activations into shared memory
    // 256 threads load K floats (typically K=2048, so 8 floats per thread)
    for (int i = tid; i < K; i += 256) {
        shared_A[i] = A[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup handles 4 consecutive outputs
    int base_output = gid * Q4_OPT_OUTPUTS_PER_TG + simd_group * Q4_OPT_OUTPUTS_PER_SG;

    // Initialize accumulators for 4 outputs
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    // Calculate row pointers for each output
    device const uchar* b_row0 = (base_output + 0 < N) ? B + (base_output + 0) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    device const uchar* b_row1 = (base_output + 1 < N) ? B + (base_output + 1) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    device const uchar* b_row2 = (base_output + 2 < N) ? B + (base_output + 2) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    device const uchar* b_row3 = (base_output + 3 < N) ? B + (base_output + 3) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;

    // Each thread in simdgroup processes blocks with stride 32
    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * Q4_BLOCK_SIZE;
        if (base_k >= K) break;

        // Load activations from shared memory (32 values per block)
        // Using float4 loads for efficiency
        threadgroup const float4* a_vec = (threadgroup const float4*)(shared_A + base_k);
        float4 a0 = a_vec[0], a1 = a_vec[1], a2 = a_vec[2], a3 = a_vec[3];
        float4 a4 = a_vec[4], a5 = a_vec[5], a6 = a_vec[6], a7 = a_vec[7];

        // Process each output row
        #define PROCESS_ROW(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* blockPtr = row_ptr + block * Q4_BYTES_PER_BLOCK; \
            float scale = as_type<half>(*((device const ushort*)blockPtr)); \
            device const uchar* qs = blockPtr + 2; \
            float4 q_lo_0 = float4(qs[0] & 0xF, qs[1] & 0xF, qs[2] & 0xF, qs[3] & 0xF) - 8.0f; \
            float4 q_hi_0 = float4(qs[0] >> 4, qs[1] >> 4, qs[2] >> 4, qs[3] >> 4) - 8.0f; \
            float4 q_lo_1 = float4(qs[4] & 0xF, qs[5] & 0xF, qs[6] & 0xF, qs[7] & 0xF) - 8.0f; \
            float4 q_hi_1 = float4(qs[4] >> 4, qs[5] >> 4, qs[6] >> 4, qs[7] >> 4) - 8.0f; \
            float4 q_lo_2 = float4(qs[8] & 0xF, qs[9] & 0xF, qs[10] & 0xF, qs[11] & 0xF) - 8.0f; \
            float4 q_hi_2 = float4(qs[8] >> 4, qs[9] >> 4, qs[10] >> 4, qs[11] >> 4) - 8.0f; \
            float4 q_lo_3 = float4(qs[12] & 0xF, qs[13] & 0xF, qs[14] & 0xF, qs[15] & 0xF) - 8.0f; \
            float4 q_hi_3 = float4(qs[12] >> 4, qs[13] >> 4, qs[14] >> 4, qs[15] >> 4) - 8.0f; \
            sum_var += scale * (dot(a0, q_lo_0) + dot(a4, q_hi_0) + \
                                dot(a1, q_lo_1) + dot(a5, q_hi_1) + \
                                dot(a2, q_lo_2) + dot(a6, q_hi_2) + \
                                dot(a3, q_lo_3) + dot(a7, q_hi_3)); \
        }

        PROCESS_ROW(b_row0, sum0);
        PROCESS_ROW(b_row1, sum1);
        PROCESS_ROW(b_row2, sum2);
        PROCESS_ROW(b_row3, sum3);
        #undef PROCESS_ROW
    }

    // Simdgroup reduction
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    // Lane 0 writes outputs
    if (simd_lane == 0) {
        if (base_output + 0 < N) C[base_output + 0] = sum0;
        if (base_output + 1 < N) C[base_output + 1] = sum1;
        if (base_output + 2 < N) C[base_output + 2] = sum2;
        if (base_output + 3 < N) C[base_output + 3] = sum3;
    }
}

// =============================================================================
// Q4_0 TILED MATMUL WITH SIMDGROUP_MATRIX (HALF-PRECISION)
// =============================================================================
// C[M,N] = A[M,K] @ B[N,K]^T where B is Q4_0 quantized
// Uses simdgroup_matrix operations for 8x8 tiled computation
// Each threadgroup computes a TILE_M x TILE_N output tile
// Threads cooperatively load A and dequant B tiles into half-precision
// threadgroup memory. simdgroup_half8x8 inputs with float accumulators
// gives 2x MAC throughput on Apple Silicon AMX units.
//
// Tile sizes: TILE_M=32, TILE_N=64, TILE_K=64 (2 Q4_0 blocks per K-tile)
// Threadgroup: 256 threads = 8 simdgroups
// 8 simdgroups in 2×4 layout, each computes a 16×16 output tile = 32×64 total
// TILE_K=64 halves barrier count: 64 barriers for K=4096 vs 128 with TILE_K=32
// Block-based B dequant: threads 0-127 each handle one Q4_0 block (32 values)
// with one scale load per block (32x fewer scale reads).
// Shared memory: 4096 (A, half) + 8192 (B, half) = 12288 bytes total

constant int SMM_TILE_M = 32;
constant int SMM_TILE_N = 64;
constant int SMM_TILE_K = 64;

kernel void matmul_q4_0_simdgroup_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    threadgroup half* shared_A [[threadgroup(0)]],
    threadgroup half* shared_B [[threadgroup(1)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int tile_m = tg_pos.y * SMM_TILE_M;
    int tile_n = tg_pos.x * SMM_TILE_N;
    if (tile_m >= M || tile_n >= N) return;

    int sg_row = (simd_group / 4) * 16;
    int sg_col = (simd_group % 4) * 16;

    simdgroup_float8x8 acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int numKTiles = (K + SMM_TILE_K - 1) / SMM_TILE_K;

    for (int k_tile = 0; k_tile < numKTiles; k_tile++) {
        int k_base = k_tile * SMM_TILE_K;

        // === Cooperative A loading: all 256 threads, float → half ===
        for (int i = tid; i < SMM_TILE_M * SMM_TILE_K; i += 256) {
            int local_m = i / SMM_TILE_K;
            int local_k = i % SMM_TILE_K;
            int global_m = tile_m + local_m;
            int global_k = k_base + local_k;
            half val = (global_m < M && global_k < K) ? (half)A[global_m * K + global_k] : (half)0;
            shared_A[local_m * SMM_TILE_K + local_k] = val;
        }

        // === Block-based B dequant: each thread handles one Q4_0 block ===
        // 128 blocks (64 N-cols × 2 K-blocks), threads 0-127 each get one block
        // One scale load per 32 values (vs per value in flat approach)
        if (tid < SMM_TILE_N * 2) {
            int block_id = tid;
            int local_n = block_id >> 1;          // 0-63 (N column)
            int block_in_tile = block_id & 1;     // 0-1 (which Q4_0 block)
            int k_offset = block_in_tile * 32;
            int global_n = tile_n + local_n;
            int q4_block_idx = k_tile * 2 + block_in_tile;

            if (global_n < N && k_base + k_offset < K) {
                device const uchar* blockPtr = B + global_n * numBlocks * Q4_BYTES_PER_BLOCK
                                                + q4_block_idx * Q4_BYTES_PER_BLOCK;
                ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
                half scale = as_type<half>(scale_u16);

                // Process all 32 nibbles with one scale load
                for (int j = 0; j < 16; j++) {
                    uchar byte_val = blockPtr[2 + j];
                    half lo = scale * (half)((byte_val & 0xF) - 8);
                    half hi = scale * (half)(((byte_val >> 4) & 0xF) - 8);
                    shared_B[(k_offset + j) * SMM_TILE_N + local_n] = lo;
                    shared_B[(k_offset + j + 16) * SMM_TILE_N + local_n] = hi;
                }
            } else {
                for (int j = 0; j < 32; j++) {
                    shared_B[(k_offset + j) * SMM_TILE_N + local_n] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === simdgroup_multiply_accumulate: half8x8 inputs → float accumulation ===
        for (int k = 0; k < SMM_TILE_K; k += 8) {
            simdgroup_half8x8 matA0, matA1, matB0, matB1;

            simdgroup_load(matA0, shared_A + (sg_row + 0) * SMM_TILE_K + k, SMM_TILE_K);
            simdgroup_load(matA1, shared_A + (sg_row + 8) * SMM_TILE_K + k, SMM_TILE_K);
            simdgroup_load(matB0, shared_B + k * SMM_TILE_N + (sg_col + 0), SMM_TILE_N);
            simdgroup_load(matB1, shared_B + k * SMM_TILE_N + (sg_col + 8), SMM_TILE_N);

            simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
            simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
            simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
            simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int out_m = tile_m + sg_row;
    int out_n = tile_n + sg_col;

    if (out_m < M && out_n < N)
        simdgroup_store(acc00, C + (out_m + 0) * N + (out_n + 0), N);
    if (out_m < M && out_n + 8 < N)
        simdgroup_store(acc01, C + (out_m + 0) * N + (out_n + 8), N);
    if (out_m + 8 < M && out_n < N)
        simdgroup_store(acc10, C + (out_m + 8) * N + (out_n + 0), N);
    if (out_m + 8 < M && out_n + 8 < N)
        simdgroup_store(acc11, C + (out_m + 8) * N + (out_n + 8), N);
}

// =============================================================================
// ORIGINAL Q4_0 BATCHED MATMUL (FALLBACK)
// =============================================================================
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

// =============================================================================
// Q6_K MATVEC FOR LM_HEAD
// =============================================================================
// Q6_K format (256 elements per block, 210 bytes):
//   - ql[128]: lower 4 bits of 6-bit quants
//   - qh[64]: upper 2 bits of 6-bit quants
//   - scales[16]: 8-bit signed scales for 16-element super-blocks
//   - d[2]: f16 global scale (at offset 208)
//
// Each simdgroup handles one output row
// Each thread in simdgroup processes elements with stride
// Grid: ceil(N/8) threadgroups of 256 threads (8 simdgroups)

constant int Q6K_BLOCK_SIZE = 256;
constant int Q6K_BYTES_PER_BLOCK = 210;
constant int Q6K_OUTPUTS_PER_TG = 8;

kernel void matvec_q6k_multi_output_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q6_K format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements (vocab_size)
    constant int& K [[buffer(4)]],                 // Inner dimension (hidden_size)
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each simdgroup handles one output
    int output_idx = gid * Q6K_OUTPUTS_PER_TG + simd_group;
    if (output_idx >= N) return;

    float sum = 0.0f;

    // Q6_K row layout
    int numBlocks = (K + Q6K_BLOCK_SIZE - 1) / Q6K_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q6K_BYTES_PER_BLOCK;

    // Each thread processes elements with stride 32 within each block
    // For K=2048, numBlocks=8, total 2048 elements, each thread handles 64 elements
    for (int block = 0; block < numBlocks; block++) {
        device const uchar* blockPtr = b_row + block * Q6K_BYTES_PER_BLOCK;

        // Block offsets
        device const uchar* ql = blockPtr;           // 128 bytes
        device const uchar* qh = blockPtr + 128;     // 64 bytes
        device const char* scales = (device const char*)(blockPtr + 192);  // 16 signed bytes
        ushort d_u16 = ((ushort)blockPtr[209] << 8) | blockPtr[208];
        float d = q4_f16_to_f32(d_u16);  // reuse helper

        int base_k = block * Q6K_BLOCK_SIZE;

        // Each thread handles elements at simd_lane, simd_lane+32, simd_lane+64, ...
        // Process 8 elements per thread (256/32 = 8)
        for (int elem_offset = simd_lane; elem_offset < 256 && base_k + elem_offset < K; elem_offset += 32) {
            int k_idx = base_k + elem_offset;
            float a_val = A[k_idx];

            // Determine which half of block (n: 0=first 128, 1=second 128)
            int n = elem_offset / 128;
            int elem_in_half = elem_offset - n * 128;

            // Within half, determine super-block (0-3 in each 32-element chunk)
            // elem_in_half: 0-31 → chunk 0, 32-63 → chunk 1, 64-95 → chunk 2, 96-127 → chunk 3
            int chunk = elem_in_half / 32;
            int l = elem_in_half % 32;  // position within chunk (0-31)

            // Scale index: based on super-block pattern
            // scales layout: sc0,sc1 for l<16, sc0,sc1 for l>=16, then sc2,sc3, sc4,sc5, sc6,sc7
            int is = l / 16;  // 0 for l<16, 1 for l>=16

            // Offset into scales array
            int scOff = n * 8;  // 0 for first half, 8 for second half

            // Get scale based on chunk
            float sc;
            if (chunk == 0) sc = float(scales[scOff + is + 0]);
            else if (chunk == 1) sc = float(scales[scOff + is + 2]);
            else if (chunk == 2) sc = float(scales[scOff + is + 4]);
            else sc = float(scales[scOff + is + 6]);

            // Dequantize the 6-bit value
            // ql layout: ql[0..63] for first half, ql[64..127] for second half
            // qh layout: qh[0..31] for first half, qh[32..63] for second half
            int qlOff = n * 64;
            int qhOff = n * 32;

            int ql_idx, qh_idx, qh_shift;
            uchar ql_byte, qh_byte;
            int q;

            // Map chunk and l to ql/qh indices
            // chunk 0: ql[l], qh[l], shift 0
            // chunk 1: ql[l+32], qh[l], shift 2
            // chunk 2: ql[l] high nibble, qh[l], shift 4
            // chunk 3: ql[l+32] high nibble, qh[l], shift 6
            if (chunk == 0) {
                ql_byte = ql[qlOff + l];
                qh_byte = qh[qhOff + l];
                q = ((ql_byte & 0xF) | ((qh_byte >> 0) & 3) << 4) - 32;
            } else if (chunk == 1) {
                ql_byte = ql[qlOff + l + 32];
                qh_byte = qh[qhOff + l];
                q = ((ql_byte & 0xF) | ((qh_byte >> 2) & 3) << 4) - 32;
            } else if (chunk == 2) {
                ql_byte = ql[qlOff + l];
                qh_byte = qh[qhOff + l];
                q = ((ql_byte >> 4) | ((qh_byte >> 4) & 3) << 4) - 32;
            } else {  // chunk == 3
                ql_byte = ql[qlOff + l + 32];
                qh_byte = qh[qhOff + l];
                q = ((ql_byte >> 4) | ((qh_byte >> 6) & 3) << 4) - 32;
            }

            // Accumulate: d * scale * q * a
            sum += a_val * d * sc * float(q);
        }
    }

    // Simdgroup reduction
    sum = simd_sum(sum);

    // Lane 0 writes output
    if (simd_lane == 0) {
        C[output_idx] = sum;
    }
}

// =============================================================================
// Q6_K NR2 MATVEC - 2 outputs per simdgroup (llama.cpp style)
// =============================================================================
// Optimized for LM head: large vocab (32k), small hidden (2k)
// Key optimizations:
//   - nr0=2: Each simdgroup produces 2 outputs (2x activation reuse)
//   - Thread collaboration: 2 threads per block (ix = lane%2)
//   - Vectorized activations: yl[16] array for efficient loads
// Grid: ceil(N/16) threadgroups of 256 threads (8 simdgroups × 2 outputs)

constant int Q6K_NR2_OUTPUTS_PER_SG = 2;
constant int Q6K_NR2_OUTPUTS_PER_TG = 16;  // 8 simdgroups × 2 outputs

kernel void matvec_q6k_nr2_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q6_K format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements (vocab_size)
    constant int& K [[buffer(4)]],                 // Inner dimension (hidden_size)
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Bit masks for extracting 6-bit values from Q6_K
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    // Number of Q6_K blocks per row
    int numBlocks = (K + Q6K_BLOCK_SIZE - 1) / Q6K_BLOCK_SIZE;

    // First output row for this simdgroup (2 outputs per simdgroup)
    int first_row = (gid * Q6K_NR2_OUTPUTS_PER_TG) + simd_group * Q6K_NR2_OUTPUTS_PER_SG;

    // Early exit if both outputs are out of bounds
    if (first_row >= N) return;

    // Accumulators for 2 outputs
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    // Vectorized activation cache
    float yl[16];

    // Thread collaboration: 2 threads per block
    // tid: 0-15 (32 lanes / 2)
    // ix: 0 or 1 (which half of block pair)
    short tid = simd_lane / 2;
    short ix = simd_lane % 2;

    // Mapping within Q6_K block (256 elements):
    // ip: 0=first 128 elements, 1=second 128 elements
    // il: position within half (0-7, each handles 4 elements)
    // l0: starting element within group of 32
    short ip = tid / 8;          // 0 or 1
    short il = tid % 8;          // 0-7
    short l0 = 4 * il;           // 0, 4, 8, ... 28
    short is = 8 * ip + l0 / 16; // scale index

    // Offsets into block data
    short y_offset = 128 * ip + l0;
    short q_offset_l = 64 * ip + l0;
    short q_offset_h = 32 * ip + l0;

    // Row pointers
    device const uchar* row0 = B + first_row * numBlocks * Q6K_BYTES_PER_BLOCK;
    device const uchar* row1 = (first_row + 1 < N) ? B + (first_row + 1) * numBlocks * Q6K_BYTES_PER_BLOCK : nullptr;

    // Process blocks with stride 2 (thread collaboration)
    for (int i = ix; i < numBlocks; i += 2) {
        // Block base pointer for row 0
        device const uchar* blockPtr0 = row0 + i * Q6K_BYTES_PER_BLOCK;
        device const uchar* ql0 = blockPtr0;
        device const uchar* qh0 = blockPtr0 + 128;
        device const char* sc0 = (device const char*)(blockPtr0 + 192);
        float d0 = float(as_type<half>(*((device const ushort*)(blockPtr0 + 208))));

        // Load vectorized activations for this block
        int base_k = i * Q6K_BLOCK_SIZE + y_offset;
        if (base_k + 96 + 4 <= K) {
            // Full load
            for (short l = 0; l < 4; ++l) {
                yl[4*l + 0] = A[base_k + l + 0];
                yl[4*l + 1] = A[base_k + l + 32];
                yl[4*l + 2] = A[base_k + l + 64];
                yl[4*l + 3] = A[base_k + l + 96];
            }
        } else {
            // Partial load with bounds checking
            for (short l = 0; l < 4; ++l) {
                int k0 = base_k + l;
                int k1 = base_k + l + 32;
                int k2 = base_k + l + 64;
                int k3 = base_k + l + 96;
                yl[4*l + 0] = (k0 < K) ? A[k0] : 0.0f;
                yl[4*l + 1] = (k1 < K) ? A[k1] : 0.0f;
                yl[4*l + 2] = (k2 < K) ? A[k2] : 0.0f;
                yl[4*l + 3] = (k3 < K) ? A[k3] : 0.0f;
            }
        }

        // Process row 0
        {
            device const uchar* q1 = ql0 + q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh_ptr = qh0 + q_offset_h;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                uchar qhv = qh_ptr[l];
                sums[0] += yl[4*l + 0] * float((char)((q1[l] & 0xF) | ((qhv & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * float((char)((q2[l] & 0xF) | ((qhv & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * float((char)((q1[l] >> 4)  | ((qhv & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * float((char)((q2[l] >> 4)  | ((qhv & kmask4) >> 2)) - 32);
            }
            sumf0 += d0 * (sums[0] * float(sc0[is + 0]) + sums[1] * float(sc0[is + 2]) +
                           sums[2] * float(sc0[is + 4]) + sums[3] * float(sc0[is + 6]));
        }

        // Process row 1 (if valid)
        if (row1) {
            device const uchar* blockPtr1 = row1 + i * Q6K_BYTES_PER_BLOCK;
            device const uchar* ql1 = blockPtr1;
            device const uchar* qh1 = blockPtr1 + 128;
            device const char* sc1 = (device const char*)(blockPtr1 + 192);
            float d1 = float(as_type<half>(*((device const ushort*)(blockPtr1 + 208))));

            device const uchar* q1 = ql1 + q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh_ptr = qh1 + q_offset_h;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                uchar qhv = qh_ptr[l];
                sums[0] += yl[4*l + 0] * float((char)((q1[l] & 0xF) | ((qhv & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * float((char)((q2[l] & 0xF) | ((qhv & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * float((char)((q1[l] >> 4)  | ((qhv & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * float((char)((q2[l] >> 4)  | ((qhv & kmask4) >> 2)) - 32);
            }
            sumf1 += d1 * (sums[0] * float(sc1[is + 0]) + sums[1] * float(sc1[is + 2]) +
                           sums[2] * float(sc1[is + 4]) + sums[3] * float(sc1[is + 6]));
        }
    }

    // Simdgroup reduction for both outputs
    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);

    // Lane 0 writes outputs
    if (simd_lane == 0) {
        C[first_row] = sumf0;
        if (first_row + 1 < N) {
            C[first_row + 1] = sumf1;
        }
    }
}

// =============================================================================
// Q4_K MATVEC FOR ATTENTION PROJECTIONS
// =============================================================================
// Q4_K format (256 elements per block, 144 bytes):
//   - d (2 bytes): fp16 super-block scale for quantized scales
//   - dmin (2 bytes): fp16 super-block scale for quantized mins
//   - scales[12] (12 bytes): 6-bit packed scales/mins (8 scales + 8 mins)
//   - qs[128] (128 bytes): 4-bit quantized values
//
// Each simdgroup handles one output row
// Grid: ceil(N/8) threadgroups of 256 threads (8 simdgroups)

constant int Q4K_BLOCK_SIZE = 256;
constant int Q4K_BYTES_PER_BLOCK = 144;
constant int Q4K_OUTPUTS_PER_TG = 8;
constant int Q4K_NR2_OUTPUTS_PER_TG = 16;

constant int Q5K_BLOCK_SIZE = 256;
constant int Q5K_BYTES_PER_BLOCK = 176;
constant int Q5K_OUTPUTS_PER_TG = 8;
constant int Q5K_NR2_OUTPUTS_PER_TG = 16;

kernel void matvec_q4k_multi_output_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Sub-block strided Q4_K matvec: each lane independently processes 32-element
    // sub-blocks with stride-32, matching Q4_0 multi_output's block-striding pattern.
    //
    // Q4_K has 8 sub-blocks of 32 elements per 256-element block. For K=4096:
    // 16 blocks × 8 sub-blocks = 128 total → 128/32 = 4 sub-blocks per lane.
    // Each sub-block: 8 float4 loads + 8 dot products (same density as Q4_0).
    //
    // Key optimizations vs previous cooperative approach:
    // 1. Hardware fp16 via as_type<half> (not software q4_f16_to_f32 with pow())
    // 2. Independent lane processing (no serial block loop, better ILP)
    // 3. Selective scale/min extraction (1 scale + 1 min per sub-block, not all 8)
    // 4. uint vector loads for qs bytes (1 load per 4 bytes vs 4 individual loads)
    int output_idx = gid * Q4K_OUTPUTS_PER_TG + simd_group;
    if (output_idx >= N) return;

    float sum = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q4K_BYTES_PER_BLOCK;
    int totalSubBlocks = numBlocks << 3;  // numBlocks * 8

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;           // which Q4_K block
        int j = sb & 7;                    // sub-block index within block (0..7)

        device const uchar* blockPtr = b_row + block_idx * Q4K_BYTES_PER_BLOCK;

        // Hardware fp16→fp32 conversion (single cycle vs software pow())
        float d = float(as_type<half>(*((device const ushort*)blockPtr)));
        float dmin = float(as_type<half>(*((device const ushort*)(blockPtr + 2))));

        // Extract only this sub-block's scale and min from packed 6-bit header
        device const uchar* sd = blockPtr + 4;
        int j_lo = j & 3;  // 0..3 for either half
        uchar sc, mn;
        if (j < 4) {
            sc = sd[j_lo] & 0x3F;
            mn = sd[j_lo + 4] & 0x3F;
        } else {
            sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4);
            mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4);
        }

        float dsc = d * float(sc);
        float dmn = dmin * float(mn);

        // Sub-blocks 0-3 use low nibbles, 4-7 use high nibbles of same qs bytes.
        // Both halves share qs_base offset; 'shift' selects the nibble.
        int qs_base = j_lo << 5;              // (j & 3) * 32
        int shift = (j >> 2) << 2;            // 0 for j<4, 4 for j>=4
        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);  // j * 32

        if (elem_start + 32 <= K) {
            // Full sub-block: 32 elements = 8 × float4, matching Q4_0 density
            device const float4* a_vec = (device const float4*)(A + elem_start);
            device const uint* qs32 = (device const uint*)(blockPtr + 16 + qs_base);

            // 8 dot products: each uint holds 4 qs bytes, extract 4 nibbles
            for (int i = 0; i < 8; i++) {
                float4 a = a_vec[i];
                uint w = qs32[i];
                float4 q = float4((w >> shift) & 0xF, (w >> (shift + 8)) & 0xF,
                                  (w >> (shift + 16)) & 0xF, (w >> (shift + 24)) & 0xF);
                sum += dot(a, dsc * q - dmn);
            }
        } else if (elem_start < K) {
            // Partial sub-block: scalar fallback for boundary
            device const uchar* qs = blockPtr + 16 + qs_base;
            for (int i = 0; i < 32 && elem_start + i < K; i++) {
                float q_val = float((qs[i] >> shift) & 0xF);
                sum += A[elem_start + i] * (dsc * q_val - dmn);
            }
        }
    }

    // Simdgroup reduction
    sum = simd_sum(sum);

    // Lane 0 writes output
    if (simd_lane == 0) {
        C[output_idx] = sum;
    }
}

kernel void matvec_q4k_nr2_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each simdgroup handles TWO outputs
    int row0 = gid * Q4K_NR2_OUTPUTS_PER_TG + simd_group * 2;
    int row1 = row0 + 1;
    if (row0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    device const uchar* b_row0 = B + row0 * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + row1 * numBlocks * Q4K_BYTES_PER_BLOCK;

    for (int block = 0; block < numBlocks; block++) {
        device const uchar* p0 = b_row0 + block * Q4K_BYTES_PER_BLOCK;
        device const uchar* p1 = (row1 < N) ? b_row1 + block * Q4K_BYTES_PER_BLOCK : p0;

        // Parse block headers (hardware fp16 conversion)
        float d0 = float(as_type<half>(*((device const ushort*)p0)));
        float dm0 = float(as_type<half>(*((device const ushort*)(p0 + 2))));
        float d1 = (row1 < N) ? float(as_type<half>(*((device const ushort*)p1))) : 0.0f;
        float dm1 = (row1 < N) ? float(as_type<half>(*((device const ushort*)(p1 + 2)))) : 0.0f;

        device const uchar* sd0 = p0 + 4;
        device const uchar* sd1 = p1 + 4;

        // Unpack scales and mins (6-bit each, packed into 12 bytes)
        uchar s0[8], m0[8], s1[8], m1[8];
        s0[0] = sd0[0] & 0x3F; s0[1] = sd0[1] & 0x3F; s0[2] = sd0[2] & 0x3F; s0[3] = sd0[3] & 0x3F;
        m0[0] = sd0[4] & 0x3F; m0[1] = sd0[5] & 0x3F; m0[2] = sd0[6] & 0x3F; m0[3] = sd0[7] & 0x3F;
        s0[4] = (sd0[8] & 0x0F) | ((sd0[0] >> 6) << 4); s0[5] = (sd0[9] & 0x0F) | ((sd0[1] >> 6) << 4);
        s0[6] = (sd0[10] & 0x0F) | ((sd0[2] >> 6) << 4); s0[7] = (sd0[11] & 0x0F) | ((sd0[3] >> 6) << 4);
        m0[4] = (sd0[8] >> 4) | ((sd0[4] >> 6) << 4); m0[5] = (sd0[9] >> 4) | ((sd0[5] >> 6) << 4);
        m0[6] = (sd0[10] >> 4) | ((sd0[6] >> 6) << 4); m0[7] = (sd0[11] >> 4) | ((sd0[7] >> 6) << 4);

        if (row1 < N) {
            s1[0] = sd1[0] & 0x3F; s1[1] = sd1[1] & 0x3F; s1[2] = sd1[2] & 0x3F; s1[3] = sd1[3] & 0x3F;
            m1[0] = sd1[4] & 0x3F; m1[1] = sd1[5] & 0x3F; m1[2] = sd1[6] & 0x3F; m1[3] = sd1[7] & 0x3F;
            s1[4] = (sd1[8] & 0x0F) | ((sd1[0] >> 6) << 4); s1[5] = (sd1[9] & 0x0F) | ((sd1[1] >> 6) << 4);
            s1[6] = (sd1[10] & 0x0F) | ((sd1[2] >> 6) << 4); s1[7] = (sd1[11] & 0x0F) | ((sd1[3] >> 6) << 4);
            m1[4] = (sd1[8] >> 4) | ((sd1[4] >> 6) << 4); m1[5] = (sd1[9] >> 4) | ((sd1[5] >> 6) << 4);
            m1[6] = (sd1[10] >> 4) | ((sd1[6] >> 6) << 4); m1[7] = (sd1[11] >> 4) | ((sd1[7] >> 6) << 4);
        }

        device const uchar* qs0 = p0 + 16;
        device const uchar* qs1 = p1 + 16;

        int base_k = block * Q4K_BLOCK_SIZE;

        for (int i = simd_lane; i < 256 && base_k + i < K; i += 32) {
            float a = A[base_k + i];
            int is = i / 32;
            int iqs = (i / 64) * 32 + (i % 32);
            int nib = (i / 32) % 2;

            int q0 = (qs0[iqs] >> (nib ? 4 : 0)) & 0xF;
            sum0 += a * (d0 * float(s0[is]) * float(q0) - dm0 * float(m0[is]));

            if (row1 < N) {
                int q1 = (qs1[iqs] >> (nib ? 4 : 0)) & 0xF;
                sum1 += a * (d1 * float(s1[is]) * float(q1) - dm1 * float(m1[is]));
            }
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        C[row0] = sum0;
        if (row1 < N) C[row1] = sum1;
    }
}

kernel void matvec_q5k_multi_output_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q5_K format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each simdgroup handles one output
    int output_idx = gid * Q5K_OUTPUTS_PER_TG + simd_group;
    if (output_idx >= N) return;

    float sum = 0.0f;

    // Q5_K row layout: d (2), dmin (2), scales (12), qs (128), qh (32) = 176
    int numBlocks = (K + Q5K_BLOCK_SIZE - 1) / Q5K_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q5K_BYTES_PER_BLOCK;

    for (int block = 0; block < numBlocks; block++) {
        device const uchar* blockPtr = b_row + block * Q5K_BYTES_PER_BLOCK;

        // Parse block header
        ushort d_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
        ushort dmin_u16 = ((ushort)blockPtr[3] << 8) | blockPtr[2];
        float d = q4_f16_to_f32(d_u16);
        float dmin = q4_f16_to_f32(dmin_u16);

        device const uchar* scalesData = blockPtr + 4;
        device const uchar* qh = blockPtr + 16;  // 5th bit (32 bytes)
        device const uchar* qs = blockPtr + 48;  // 4-bit quantized values (128 bytes)

        // Unpack scales and mins (same logic as Q4_K)
        uchar scales[8];
        uchar mins[8];
        scales[0] = scalesData[0] & 0x3F;
        scales[1] = scalesData[1] & 0x3F;
        scales[2] = scalesData[2] & 0x3F;
        scales[3] = scalesData[3] & 0x3F;
        mins[0] = scalesData[4] & 0x3F;
        mins[1] = scalesData[5] & 0x3F;
        mins[2] = scalesData[6] & 0x3F;
        mins[3] = scalesData[7] & 0x3F;
        scales[4] = (scalesData[8] & 0x0F) | ((scalesData[0] >> 6) << 4);
        scales[5] = (scalesData[9] & 0x0F) | ((scalesData[1] >> 6) << 4);
        scales[6] = (scalesData[10] & 0x0F) | ((scalesData[2] >> 6) << 4);
        scales[7] = (scalesData[11] & 0x0F) | ((scalesData[3] >> 6) << 4);
        mins[4] = (scalesData[8] >> 4) | ((scalesData[4] >> 6) << 4);
        mins[5] = (scalesData[9] >> 4) | ((scalesData[5] >> 6) << 4);
        mins[6] = (scalesData[10] >> 4) | ((scalesData[6] >> 6) << 4);
        mins[7] = (scalesData[11] >> 4) | ((scalesData[7] >> 6) << 4);

        int base_k = block * Q5K_BLOCK_SIZE;

        for (int elem_offset = simd_lane; elem_offset < 256 && base_k + elem_offset < K; elem_offset += 32) {
            int k_idx = base_k + elem_offset;
            float a_val = A[k_idx];

            int scale_idx = elem_offset / 32;
            float sc = float(scales[scale_idx]);
            float m = float(mins[scale_idx]);

            int qs_byte_idx = (elem_offset / 64) * 32 + (elem_offset % 32);
            int nibble_is_high = (elem_offset / 32) % 2;
            uchar qs_byte = qs[qs_byte_idx];

            int q;
            if (nibble_is_high == 0) {
                q = qs_byte & 0x0F;
            } else {
                q = (qs_byte >> 4) & 0x0F;
            }

            // High bit from qh (32 bytes = 256 bits)
            // qh[m] bit 'is' corresponds to element is*32 + m
            int qh_bit_idx = elem_offset / 32;
            int qh_byte_idx = elem_offset % 32;
            int high_bit = (qh[qh_byte_idx] >> qh_bit_idx) & 1;
            q |= (high_bit << 4);

            float dequant = d * sc * float(q) - dmin * m;
            sum += a_val * dequant;
        }
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        C[output_idx] = sum;
    }
}

kernel void matvec_q5k_nr2_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q5_K format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each simdgroup handles TWO outputs
    int row0 = gid * Q5K_NR2_OUTPUTS_PER_TG + simd_group * 2;
    int row1 = row0 + 1;
    if (row0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int numBlocks = (K + Q5K_BLOCK_SIZE - 1) / Q5K_BLOCK_SIZE;
    device const uchar* b_row0 = B + row0 * numBlocks * Q5K_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + row1 * numBlocks * Q5K_BYTES_PER_BLOCK;

    for (int block = 0; block < numBlocks; block++) {
        device const uchar* p0 = b_row0 + block * Q5K_BYTES_PER_BLOCK;
        device const uchar* p1 = (row1 < N) ? b_row1 + block * Q5K_BYTES_PER_BLOCK : p0;

        float d0 = q4_f16_to_f32(((ushort)p0[1] << 8) | p0[0]);
        float dm0 = q4_f16_to_f32(((ushort)p0[3] << 8) | p0[2]);
        float d1 = (row1 < N) ? q4_f16_to_f32(((ushort)p1[1] << 8) | p1[0]) : 0.0f;
        float dm1 = (row1 < N) ? q4_f16_to_f32(((ushort)p1[3] << 8) | p1[2]) : 0.0f;

        device const uchar* sd0 = p0 + 4;
        device const uchar* sd1 = p1 + 4;
        
        uchar s0[8], m0[8], s1[8], m1[8];
        s0[0] = sd0[0] & 0x3F; s0[1] = sd0[1] & 0x3F; s0[2] = sd0[2] & 0x3F; s0[3] = sd0[3] & 0x3F;
        m0[0] = sd0[4] & 0x3F; m0[1] = sd0[5] & 0x3F; m0[2] = sd0[6] & 0x3F; m0[3] = sd0[7] & 0x3F;
        s0[4] = (sd0[8] & 0x0F) | ((sd0[0] >> 6) << 4); s0[5] = (sd0[9] & 0x0F) | ((sd0[1] >> 6) << 4);
        s0[6] = (sd0[10] & 0x0F) | ((sd0[2] >> 6) << 4); s0[7] = (sd0[11] & 0x0F) | ((sd0[3] >> 6) << 4);
        m0[4] = (sd0[8] >> 4) | ((sd0[4] >> 6) << 4); m0[5] = (sd0[9] >> 4) | ((sd0[5] >> 6) << 4);
        m0[6] = (sd0[10] >> 4) | ((sd0[6] >> 6) << 4); m0[7] = (sd0[11] >> 4) | ((sd0[7] >> 6) << 4);

        if (row1 < N) {
            s1[0] = sd1[0] & 0x3F; s1[1] = sd1[1] & 0x3F; s1[2] = sd1[2] & 0x3F; s1[3] = sd1[3] & 0x3F;
            m1[0] = sd1[4] & 0x3F; m1[1] = sd1[5] & 0x3F; m1[2] = sd1[6] & 0x3F; m1[3] = sd1[7] & 0x3F;
            s1[4] = (sd1[8] & 0x0F) | ((sd1[0] >> 6) << 4); s1[5] = (sd1[9] & 0x0F) | ((sd1[1] >> 6) << 4);
            s1[6] = (sd1[10] & 0x0F) | ((sd1[2] >> 6) << 4); s1[7] = (sd1[11] & 0x0F) | ((sd1[3] >> 6) << 4);
            m1[4] = (sd1[8] >> 4) | ((sd1[4] >> 6) << 4); m1[5] = (sd1[9] >> 4) | ((sd1[5] >> 6) << 4);
            m1[6] = (sd1[10] >> 4) | ((sd1[6] >> 6) << 4); m1[7] = (sd1[11] >> 4) | ((sd1[7] >> 6) << 4);
        }

        device const uchar* qh0 = p0 + 16;
        device const uchar* qs0 = p0 + 48;
        device const uchar* qh1 = p1 + 16;
        device const uchar* qs1 = p1 + 48;

        int base_k = block * Q5K_BLOCK_SIZE;

        for (int i = simd_lane; i < 256 && base_k + i < K; i += 32) {
            float a = A[base_k + i];
            int is = i / 32;
            int iqs = (i / 64) * 32 + (i % 32);
            int nib = (i / 32) % 2;

            int q0 = (qs0[iqs] >> (nib ? 4 : 0)) & 0xF;
            q0 |= ((qh0[i % 32] >> is) & 1) << 4;
            sum0 += a * (d0 * float(s0[is]) * float(q0) - dm0 * float(m0[is]));

            if (row1 < N) {
                int q1 = (qs1[iqs] >> (nib ? 4 : 0)) & 0xF;
                q1 |= ((qh1[i % 32] >> is) & 1) << 4;
                sum1 += a * (d1 * float(s1[is]) * float(q1) - dm1 * float(m1[is]));
            }
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        C[row0] = sum0;
        if (row1 < N) C[row1] = sum1;
    }
}

// =============================================================================
// Q4_K BATCHED MATMUL (FOR M > 1, PREFILL) — Optimized Track 4
// =============================================================================
// C[m,n] = A[m,k] @ B[n,k]^T where B is Q4_K quantized
// NR2-style: 8 simdgroups × 2 N outputs = 16 outputs per threadgroup per M row.
// Grid: (ceil(N/16), M) threadgroups of 256 threads.

// Batched Q4_K matmul: [M,K] × [N,K]^T → [M,N] using NR2 pattern.
// Each threadgroup handles one M row, each simdgroup handles 2 N outputs.
// Grid: (ceil(N/16), M) — 16 outputs per threadgroup (8 simdgroups × 2).
// Track 4: Quantization Expansion, Phase 1 Task 2.
kernel void matmul_q4k_batched_f32(
    device const float* A [[buffer(0)]],           // [M, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [M, N] output
    constant int& M [[buffer(3)]],                 // Number of input rows
    constant int& N [[buffer(4)]],                 // Number of output columns
    constant int& K [[buffer(5)]],                 // Inner dimension
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // gid.x = N tile index, gid.y = input row (m)
    int m = gid.y;
    if (m >= M) return;

    // Each simdgroup handles 2 N rows (NR2 pattern)
    int row0 = gid.x * Q4K_NR2_OUTPUTS_PER_TG + simd_group * 2;
    int row1 = row0 + 1;
    if (row0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    // A row pointer: A[m, :]
    device const float* a_row = A + m * K;

    // B row pointers in Q4_K format
    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    device const uchar* b_row0 = B + row0 * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + row1 * numBlocks * Q4K_BYTES_PER_BLOCK;

    for (int block = 0; block < numBlocks; block++) {
        device const uchar* p0 = b_row0 + block * Q4K_BYTES_PER_BLOCK;
        device const uchar* p1 = (row1 < N) ? b_row1 + block * Q4K_BYTES_PER_BLOCK : p0;

        // Parse block headers
        float d0 = q4_f16_to_f32(((ushort)p0[1] << 8) | p0[0]);
        float dm0 = q4_f16_to_f32(((ushort)p0[3] << 8) | p0[2]);
        float d1 = (row1 < N) ? q4_f16_to_f32(((ushort)p1[1] << 8) | p1[0]) : 0.0f;
        float dm1 = (row1 < N) ? q4_f16_to_f32(((ushort)p1[3] << 8) | p1[2]) : 0.0f;

        device const uchar* sd0 = p0 + 4;
        device const uchar* sd1 = p1 + 4;

        // Unpack 6-bit scales and mins for row0
        uchar s0[8], mn0[8];
        s0[0] = sd0[0] & 0x3F; s0[1] = sd0[1] & 0x3F; s0[2] = sd0[2] & 0x3F; s0[3] = sd0[3] & 0x3F;
        mn0[0] = sd0[4] & 0x3F; mn0[1] = sd0[5] & 0x3F; mn0[2] = sd0[6] & 0x3F; mn0[3] = sd0[7] & 0x3F;
        s0[4] = (sd0[8] & 0x0F) | ((sd0[0] >> 6) << 4); s0[5] = (sd0[9] & 0x0F) | ((sd0[1] >> 6) << 4);
        s0[6] = (sd0[10] & 0x0F) | ((sd0[2] >> 6) << 4); s0[7] = (sd0[11] & 0x0F) | ((sd0[3] >> 6) << 4);
        mn0[4] = (sd0[8] >> 4) | ((sd0[4] >> 6) << 4); mn0[5] = (sd0[9] >> 4) | ((sd0[5] >> 6) << 4);
        mn0[6] = (sd0[10] >> 4) | ((sd0[6] >> 6) << 4); mn0[7] = (sd0[11] >> 4) | ((sd0[7] >> 6) << 4);

        // Unpack 6-bit scales and mins for row1
        uchar s1[8], mn1[8];
        if (row1 < N) {
            s1[0] = sd1[0] & 0x3F; s1[1] = sd1[1] & 0x3F; s1[2] = sd1[2] & 0x3F; s1[3] = sd1[3] & 0x3F;
            mn1[0] = sd1[4] & 0x3F; mn1[1] = sd1[5] & 0x3F; mn1[2] = sd1[6] & 0x3F; mn1[3] = sd1[7] & 0x3F;
            s1[4] = (sd1[8] & 0x0F) | ((sd1[0] >> 6) << 4); s1[5] = (sd1[9] & 0x0F) | ((sd1[1] >> 6) << 4);
            s1[6] = (sd1[10] & 0x0F) | ((sd1[2] >> 6) << 4); s1[7] = (sd1[11] & 0x0F) | ((sd1[3] >> 6) << 4);
            mn1[4] = (sd1[8] >> 4) | ((sd1[4] >> 6) << 4); mn1[5] = (sd1[9] >> 4) | ((sd1[5] >> 6) << 4);
            mn1[6] = (sd1[10] >> 4) | ((sd1[6] >> 6) << 4); mn1[7] = (sd1[11] >> 4) | ((sd1[7] >> 6) << 4);
        }

        device const uchar* qs0 = p0 + 16;
        device const uchar* qs1 = p1 + 16;

        int base_k = block * Q4K_BLOCK_SIZE;

        // Each lane processes elements stride-32 through the 256-element super-block
        for (int i = simd_lane; i < 256 && base_k + i < K; i += 32) {
            float a = a_row[base_k + i];
            int is = i / 32;           // sub-block index (0-7)
            int iqs = (i / 64) * 32 + (i % 32);  // byte index in qs
            int nib = (i / 32) % 2;    // nibble selector

            int q0 = (qs0[iqs] >> (nib ? 4 : 0)) & 0xF;
            sum0 += a * (d0 * float(s0[is]) * float(q0) - dm0 * float(mn0[is]));

            if (row1 < N) {
                int q1 = (qs1[iqs] >> (nib ? 4 : 0)) & 0xF;
                sum1 += a * (d1 * float(s1[is]) * float(q1) - dm1 * float(mn1[is]));
            }
        }
    }

    // Simdgroup reduction and store
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        C[m * N + row0] = sum0;
        if (row1 < N) C[m * N + row1] = sum1;
    }
}

// =============================================================================
// Q4_K SIMDGROUP TILED MATMUL (PREFILL)
// =============================================================================
// Uses simdgroup_matrix hardware 8×8 matrix multiply for Q4_K prefill.
// Same tile layout as Q4_0 simdgroup: TILE_M=32, TILE_N=64, TILE_K=32.
// TILE_K=32 aligns with Q4_K sub-block size (8 sub-blocks × 32 = 256 per block).
// Each k_tile processes exactly one Q4_K sub-block, so j is constant per tile.
// 8 simdgroups in 2×4 layout, each computing 16×16 output tile = 32×64 total.

kernel void matmul_q4k_simdgroup_f32(
    device const float* A [[buffer(0)]],           // [M, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [M, N] output
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    threadgroup half* shared_A [[threadgroup(0)]],   // [TILE_M, TILE_K] half
    threadgroup half* shared_B [[threadgroup(1)]],   // [TILE_K, TILE_N] half (B transposed!)
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int tile_m = tg_pos.y * SMM_TILE_M;
    int tile_n = tg_pos.x * SMM_TILE_N;
    if (tile_m >= M || tile_n >= N) return;

    // Simdgroup layout: 2×4 grid of 16×16 tiles = 32×64 output
    int sg_row = (simd_group / 4) * 16;
    int sg_col = (simd_group % 4) * 16;

    // Result accumulators (float for precision, half inputs for speed)
    simdgroup_float8x8 acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    int numQ4KBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    int numKTiles = (K + SMM_TILE_K - 1) / SMM_TILE_K;

    for (int k_tile = 0; k_tile < numKTiles; k_tile++) {
        int k_base = k_tile * SMM_TILE_K;

        // === Cooperative load of A tile [TILE_M, TILE_K] as half ===
        for (int i = tid; i < SMM_TILE_M * SMM_TILE_K; i += 256) {
            int local_m = i / SMM_TILE_K;
            int local_k = i % SMM_TILE_K;
            int global_m = tile_m + local_m;
            int global_k = k_base + local_k;

            half val = 0.0h;
            if (global_m < M && global_k < K) {
                val = half(A[global_m * K + global_k]);
            }
            shared_A[local_m * SMM_TILE_K + local_k] = val;
        }

        // === Block-based B dequant (Q4_K): threads 0-127 each handle one sub-block ===
        // TILE_K=64 spans 2 Q4_K sub-blocks (32 elements each).
        // 128 sub-blocks total: TILE_N=64 rows × 2 sub-blocks per row.
        // Both sub-blocks are always in the same Q4_K super-block.
        if (tid < SMM_TILE_N * 2) {
            int block_id = tid;
            int local_n = block_id >> 1;       // 0..63 — which N-row
            int sub_in_tile = block_id & 1;    // 0 or 1 — first or second sub-block
            int global_n = tile_n + local_n;

            if (global_n < N) {
                int j_first = (k_base / 32) & 7;
                int j = j_first + sub_in_tile;
                int k_offset = sub_in_tile * 32;

                int q4k_block_idx = k_base / Q4K_BLOCK_SIZE;
                device const uchar* blockPtr = B + global_n * numQ4KBlocks * Q4K_BYTES_PER_BLOCK
                                                + q4k_block_idx * Q4K_BYTES_PER_BLOCK;

                // Super-block header: d and dmin (safe byte-wise load)
                half d_h = as_type<half>((ushort)((ushort)blockPtr[1] << 8 | blockPtr[0]));
                half dmin_h = as_type<half>((ushort)((ushort)blockPtr[3] << 8 | blockPtr[2]));

                // Extract scale and min for sub-block j
                device const uchar* sd = blockPtr + 4;
                int j_lo = j & 3;
                half sc_h, mn_h;
                if (j < 4) {
                    sc_h = half(sd[j_lo] & 0x3F);
                    mn_h = half(sd[j_lo + 4] & 0x3F);
                } else {
                    sc_h = half((sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4));
                    mn_h = half((sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4));
                }

                half d_sc = d_h * sc_h;
                half dmin_mn = dmin_h * mn_h;

                // Dequantize 32 nibbles from this sub-block
                // Q4_K interleaved layout: groups of 64 elements share 32 qs bytes
                // Even sub-blocks (j%2==0) use low nibble, odd use high nibble
                int qs_base = (j >> 1) << 5;    // (j/2)*32: byte offset in qs
                int shift = (j & 1) << 2;       // 0 for even (low nib), 4 for odd (high nib)
                device const uchar* qs = blockPtr + 16 + qs_base;

                for (int i = 0; i < 32; i++) {
                    int q = (qs[i] >> shift) & 0xF;
                    half val = d_sc * half(q) - dmin_mn;
                    shared_B[(k_offset + i) * SMM_TILE_N + local_n] = val;
                }
            } else {
                // Out of bounds: zero fill
                int k_offset = (tid & 1) * 32;
                for (int i = 0; i < 32; i++) {
                    shared_B[(k_offset + i) * SMM_TILE_N + local_n] = 0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Compute using simdgroup_matrix: half inputs, float accumulators ===
        for (int k = 0; k < SMM_TILE_K; k += 8) {
            simdgroup_half8x8 matA0, matA1, matB0, matB1;

            simdgroup_load(matA0, shared_A + (sg_row + 0) * SMM_TILE_K + k, SMM_TILE_K);
            simdgroup_load(matA1, shared_A + (sg_row + 8) * SMM_TILE_K + k, SMM_TILE_K);

            simdgroup_load(matB0, shared_B + k * SMM_TILE_N + (sg_col + 0), SMM_TILE_N);
            simdgroup_load(matB1, shared_B + k * SMM_TILE_N + (sg_col + 8), SMM_TILE_N);

            simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
            simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
            simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
            simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store results ===
    int out_m = tile_m + sg_row;
    int out_n = tile_n + sg_col;

    if (out_m < M && out_n < N)
        simdgroup_store(acc00, C + (out_m + 0) * N + (out_n + 0), N);
    if (out_m < M && out_n + 8 < N)
        simdgroup_store(acc01, C + (out_m + 0) * N + (out_n + 8), N);
    if (out_m + 8 < M && out_n < N)
        simdgroup_store(acc10, C + (out_m + 8) * N + (out_n + 0), N);
    if (out_m + 8 < M && out_n + 8 < N)
        simdgroup_store(acc11, C + (out_m + 8) * N + (out_n + 8), N);
}

// =============================================================================
// Q8_0 MATMUL KERNELS
// =============================================================================
// Q8_0 format: 32 elements per block, 34 bytes (2 byte f16 scale + 32 int8 values).
// Track 4: Quantization Expansion, Phase 2 Task 2.

constant int Q8_0_BLOCK_SIZE = 32;
constant int Q8_0_BYTES_PER_BLOCK = 34;
constant int Q8_0_NR2_OUTPUTS_PER_TG = 16; // 8 simdgroups × 2 outputs

// Q8_0 NR2 matvec: [1,K] × [N,K]^T → [1,N]
// Each simdgroup handles 2 N outputs, lanes stride through blocks.
kernel void matvec_q8_0_nr2_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int row0 = gid * Q8_0_NR2_OUTPUTS_PER_TG + simd_group * 2;
    int row1 = row0 + 1;
    if (row0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int numBlocks = (K + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    device const uchar* b_row0 = B + row0 * numBlocks * Q8_0_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + row1 * numBlocks * Q8_0_BYTES_PER_BLOCK;

    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * Q8_0_BLOCK_SIZE;

        // Load A values for this block (reuse across both rows)
        float a_vals[32];
        int elems = min(32, K - base_k);
        for (int i = 0; i < elems; i++) {
            a_vals[i] = A[base_k + i];
        }

        // Row 0
        {
            device const uchar* p = b_row0 + block * Q8_0_BYTES_PER_BLOCK;
            float d = q4_f16_to_f32(((ushort)p[1] << 8) | p[0]);
            float s = 0.0f;
            for (int i = 0; i < elems; i++) {
                s += a_vals[i] * float((char)p[2 + i]);
            }
            sum0 += d * s;
        }

        // Row 1
        if (row1 < N) {
            device const uchar* p = b_row1 + block * Q8_0_BYTES_PER_BLOCK;
            float d = q4_f16_to_f32(((ushort)p[1] << 8) | p[0]);
            float s = 0.0f;
            for (int i = 0; i < elems; i++) {
                s += a_vals[i] * float((char)p[2 + i]);
            }
            sum1 += d * s;
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        C[row0] = sum0;
        if (row1 < N) C[row1] = sum1;
    }
}

// Q8_0 batched matmul: [M,K] × [N,K]^T → [M,N] using NR2 pattern.
// Grid: (ceil(N/16), M). Each TG handles one M row, 16 N outputs.
kernel void matmul_q8_0_batched_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int m = gid.y;
    if (m >= M) return;

    int row0 = gid.x * Q8_0_NR2_OUTPUTS_PER_TG + simd_group * 2;
    int row1 = row0 + 1;
    if (row0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    device const float* a_row = A + m * K;
    int numBlocks = (K + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    device const uchar* b_row0 = B + row0 * numBlocks * Q8_0_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + row1 * numBlocks * Q8_0_BYTES_PER_BLOCK;

    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * Q8_0_BLOCK_SIZE;

        float a_vals[32];
        int elems = min(32, K - base_k);
        for (int i = 0; i < elems; i++) {
            a_vals[i] = a_row[base_k + i];
        }

        {
            device const uchar* p = b_row0 + block * Q8_0_BYTES_PER_BLOCK;
            float d = q4_f16_to_f32(((ushort)p[1] << 8) | p[0]);
            float s = 0.0f;
            for (int i = 0; i < elems; i++) {
                s += a_vals[i] * float((char)p[2 + i]);
            }
            sum0 += d * s;
        }

        if (row1 < N) {
            device const uchar* p = b_row1 + block * Q8_0_BYTES_PER_BLOCK;
            float d = q4_f16_to_f32(((ushort)p[1] << 8) | p[0]);
            float s = 0.0f;
            for (int i = 0; i < elems; i++) {
                s += a_vals[i] * float((char)p[2 + i]);
            }
            sum1 += d * s;
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        C[m * N + row0] = sum0;
        if (row1 < N) C[m * N + row1] = sum1;
    }
}

// =============================================================================
// BF16 MatMul Kernels
// BF16 format: 2 bytes per element (upper 16 bits of float32)
// Conversion: float = as_type<float>(uint32(bf16_bits) << 16)
// =============================================================================

constant int BF16_NR2_OUTPUTS_PER_TG = 16; // 8 simdgroups × 2 outputs each

// Helper: convert BF16 bits to float32
inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// BF16 NR2 matvec: C = A @ B^T where A is [1,K] F32, B is [N,K] BF16, C is [1,N] F32.
// Each simdgroup handles 2 output rows, 8 simdgroups per threadgroup = 16 outputs.
kernel void matvec_bf16_nr2_f32(
    device const float* A [[buffer(0)]],
    device const ushort* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int row0 = gid * BF16_NR2_OUTPUTS_PER_TG + simd_group * 2;
    int row1 = row0 + 1;
    if (row0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    device const ushort* b_row0 = B + row0 * K;
    device const ushort* b_row1 = B + row1 * K;

    // Each lane processes elements at stride 32
    for (int i = simd_lane; i < K; i += 32) {
        float a_val = A[i];
        sum0 += a_val * bf16_to_f32(b_row0[i]);
        if (row1 < N) {
            sum1 += a_val * bf16_to_f32(b_row1[i]);
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        C[row0] = sum0;
        if (row1 < N) C[row1] = sum1;
    }
}

// BF16 batched matmul: C = A @ B^T where A is [M,K] F32, B is [N,K] BF16, C is [M,N] F32.
// 2D grid: (nTiles, M) where nTiles = ceil(N / BF16_NR2_OUTPUTS_PER_TG).
kernel void matmul_bf16_batched_f32(
    device const float* A [[buffer(0)]],
    device const ushort* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int m = gid.y;
    if (m >= M) return;

    int row0 = gid.x * BF16_NR2_OUTPUTS_PER_TG + simd_group * 2;
    int row1 = row0 + 1;
    if (row0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    device const float* a_row = A + m * K;
    device const ushort* b_row0 = B + row0 * K;
    device const ushort* b_row1 = B + row1 * K;

    for (int i = simd_lane; i < K; i += 32) {
        float a_val = a_row[i];
        sum0 += a_val * bf16_to_f32(b_row0[i]);
        if (row1 < N) {
            sum1 += a_val * bf16_to_f32(b_row1[i]);
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        C[m * N + row0] = sum0;
        if (row1 < N) C[m * N + row1] = sum1;
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

// =============================================================================
// LAYERNORM KERNEL (for Phi-2 and other non-LLaMA architectures)
// =============================================================================
// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
// Unlike RMSNorm, this computes both mean and variance.

constant int LAYERNORM_THREADGROUP_SIZE = 256;

kernel void layernorm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],  // Need 2 * num_warps floats
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base = row * dim;

    // Phase 1: Each thread computes partial sum and sum of squares
    float sum = 0.0f;
    float sumSq = 0.0f;
    for (int i = tid; i < dim; i += LAYERNORM_THREADGROUP_SIZE) {
        float val = x[base + i];
        sum += val;
        sumSq += val * val;
    }

    // Warp-level reduction for both sum and sumSq
    sum = simd_sum(sum);
    sumSq = simd_sum(sumSq);

    // Store warp results to shared memory (interleaved: sum, sumSq)
    if (simd_lane == 0) {
        shared[simd_group * 2] = sum;
        shared[simd_group * 2 + 1] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first warp
    float totalSum = 0.0f;
    float totalSumSq = 0.0f;
    if (simd_group == 0) {
        int num_warps = (LAYERNORM_THREADGROUP_SIZE + 31) / 32;
        float warp_sum = (simd_lane < (uint)num_warps) ? shared[simd_lane * 2] : 0.0f;
        float warp_sumSq = (simd_lane < (uint)num_warps) ? shared[simd_lane * 2 + 1] : 0.0f;
        totalSum = simd_sum(warp_sum);
        totalSumSq = simd_sum(warp_sumSq);
    }

    // Compute mean and inverse std, broadcast via shared memory
    if (tid == 0) {
        float mean = totalSum / float(dim);
        float var = (totalSumSq / float(dim)) - (mean * mean);
        shared[0] = mean;
        shared[1] = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shared[0];
    float invStd = shared[1];

    // Phase 2: Each thread normalizes its portion: (x - mean) * invStd * weight + bias
    for (int i = tid; i < dim; i += LAYERNORM_THREADGROUP_SIZE) {
        float normalized = (x[base + i] - mean) * invStd;
        out[base + i] = normalized * weight[i] + bias[i];
    }
}

// =============================================================================
// GELU ACTIVATION KERNEL (for Phi-2 and other architectures)
// =============================================================================
// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

kernel void gelu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // NOTE: No bounds check needed - dispatch_kernel handles exact thread count
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float val = x[gid];

    // Clamp very large values to avoid numerical issues
    // For |x| > 10, GELU(x) ≈ x (for positive) or 0 (for negative)
    if (val > 10.0f) {
        out[gid] = val;
        return;
    }
    if (val < -10.0f) {
        out[gid] = 0.0f;
        return;
    }

    // Fast GELU approximation for normal range
    float x3 = val * val * val;
    float tanh_arg = 0.7978845608f * (val + 0.044715f * x3);  // sqrt(2/pi) ≈ 0.7978845608
    float tanh_val = tanh(tanh_arg);
    out[gid] = 0.5f * val * (1.0f + tanh_val);
}

// GELU with FP16 input/output
kernel void gelu_f16(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float val = float(x[gid]);
    float x3 = val * val * val;
    float tanh_arg = 0.7978845608f * (val + 0.044715f * x3);
    float tanh_val = tanh(tanh_arg);
    out[gid] = half(0.5f * val * (1.0f + tanh_val));
}

// Fused GELU-gated multiply for GeGLU activation (Gemma).
// out[i] = GELU(gate[i]) * up[i]
kernel void gelu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float val = gate[gid];
    float gelu_val;
    if (val > 10.0f) {
        gelu_val = val;
    } else if (val < -10.0f) {
        gelu_val = 0.0f;
    } else {
        float x3 = val * val * val;
        float tanh_arg = 0.7978845608f * (val + 0.044715f * x3);
        gelu_val = 0.5f * val * (1.0f + tanh(tanh_arg));
    }
    out[gid] = gelu_val * up[gid];
}

// =============================================================================
// ADDBIAS KERNEL (for architectures with bias terms)
// =============================================================================
// Adds bias to each row: out[row, col] = x[row, col] + bias[col]

kernel void add_bias_f32(
    device const float* x [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int col = gid % cols;
    out[gid] = x[gid] + bias[col];
}

// =============================================================================
// FUSED RMSNORM + Q4_0 MATVEC KERNEL
// =============================================================================
// Combines RMSNorm(x) and Q4_0 MatVec(x_norm, W) into one kernel.
// Avoids writing x_norm to global memory and reading it back.
//
// Steps:
// 1. Load x into shared memory (cooperative)
// 2. Compute sum of squares of x (cooperative reduction)
// 3. Compute RMS = rsqrt(mean + eps)
// 4. Perform MatVec using (x_shared[i] * rms * weight[i]) as activation
//
// Grid: ceil(N/32) threadgroups of 256 threads (standard optimized matvec grid)
// Shared memory: K floats for x, plus overhead for reduction

kernel void matvec_q4_0_fused_rmsnorm_f32(
    device const float* x [[buffer(0)]],           // [K] input activations
    device const float* normWeight [[buffer(1)]],  // [K] RMSNorm weights
    device const uchar* W [[buffer(2)]],           // [N, K] Q4_0 weights
    device float* out [[buffer(3)]],               // [N] output logits
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_x [[threadgroup(0)]], // [K] shared activations
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & Sum Squares
    // 256 threads load K elements (e.g. 2048 / 256 = 8 elements per thread)
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i];
        shared_x[i] = val; // Store raw x
        localSumSq += val * val;
    }

    // Warp-level reduction of sum-squares
    localSumSq = simd_sum(localSumSq);

    // Threadgroup reduction (via shared memory scratch at end of buffer?)
    // Or just use first few floats of shared_x if we are careful? No, need x.
    // We need a small scratch area. Let's assume shared_x is size K.
    // We can use a dedicated variable for reduction or reuse a known safe spot?
    // Better: allocate extra shared memory in dispatch.
    // Let's assume shared memory size passed is (K + 8) * 4 bytes.
    // Scratch at offset K.
    threadgroup float* scratch = shared_x + K;

    if (simd_lane == 0) {
        scratch[simd_group] = localSumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first warp
    float totalSumSq = 0.0f;
    if (simd_group == 0) {
        float val = (simd_lane < 8) ? scratch[simd_lane] : 0.0f;
        totalSumSq = simd_sum(val);
    }
    
    // Broadcast RMS to all threads (using scratch[0])
    if (tid == 0) {
        scratch[0] = rsqrt(totalSumSq / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = scratch[0];

    // Phase 2: MatVec using normalized x
    // Similar to matvec_q4_0_optimized_f32, but we apply norm on the fly
    // Each simdgroup handles 4 outputs (32 total per threadgroup)
    
    int base_output = gid * 32 + simd_group * 4;
    
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int numBlocks = (K + 32 - 1) / 32;

    // Row pointers
    device const uchar* row0 = (base_output + 0 < N) ? W + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* row1 = (base_output + 1 < N) ? W + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* row2 = (base_output + 2 < N) ? W + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* row3 = (base_output + 3 < N) ? W + (base_output + 3) * numBlocks * 18 : nullptr;

    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * 32;
        if (base_k >= K) break;

        // Load 32 activations from shared memory
        // Apply RMSNorm: x_norm = x_raw * rms * normWeight
        // We need to load normWeight from global (vectorized)
        device const float4* w_ptr = (device const float4*)(normWeight + base_k);
        float4 w0 = w_ptr[0];
        float4 w1 = w_ptr[1];
        float4 w2 = w_ptr[2];
        float4 w3 = w_ptr[3];
        float4 w4 = w_ptr[4];
        float4 w5 = w_ptr[5];
        float4 w6 = w_ptr[6];
        float4 w7 = w_ptr[7];

        threadgroup const float4* x_ptr = (threadgroup const float4*)(shared_x + base_k);
        float4 x0 = x_ptr[0];
        float4 x1 = x_ptr[1];
        float4 x2 = x_ptr[2];
        float4 x3 = x_ptr[3];
        float4 x4 = x_ptr[4];
        float4 x5 = x_ptr[5];
        float4 x6 = x_ptr[6];
        float4 x7 = x_ptr[7];

        // Normalize activations
        float4 a0 = x0 * rms * w0;
        float4 a1 = x1 * rms * w1;
        float4 a2 = x2 * rms * w2;
        float4 a3 = x3 * rms * w3;
        float4 a4 = x4 * rms * w4;
        float4 a5 = x5 * rms * w5;
        float4 a6 = x6 * rms * w6;
        float4 a7 = x7 * rms * w7;

        // Re-pack into 8 float4s for dot product (0,4,1,5,2,6,3,7 order used in optimized kernel)
        // Actually, optimized kernel loads 0,1,2,3 then 4,5,6,7. Let's match usage.
        // The macro uses a0..a3 (low) and a4..a7 (high).
        
        #define PROCESS_ROW(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* blockPtr = row_ptr + block * 18; \
            float scale = as_type<half>(*((device const ushort*)blockPtr)); \
            device const uchar* qs = blockPtr + 2; \
            float4 q_lo_0 = float4(qs[0] & 0xF, qs[1] & 0xF, qs[2] & 0xF, qs[3] & 0xF) - 8.0f; \
            float4 q_hi_0 = float4(qs[0] >> 4, qs[1] >> 4, qs[2] >> 4, qs[3] >> 4) - 8.0f; \
            float4 q_lo_1 = float4(qs[4] & 0xF, qs[5] & 0xF, qs[6] & 0xF, qs[7] & 0xF) - 8.0f; \
            float4 q_hi_1 = float4(qs[4] >> 4, qs[5] >> 4, qs[6] >> 4, qs[7] >> 4) - 8.0f; \
            float4 q_lo_2 = float4(qs[8] & 0xF, qs[9] & 0xF, qs[10] & 0xF, qs[11] & 0xF) - 8.0f; \
            float4 q_hi_2 = float4(qs[8] >> 4, qs[9] >> 4, qs[10] >> 4, qs[11] >> 4) - 8.0f; \
            float4 q_lo_3 = float4(qs[12] & 0xF, qs[13] & 0xF, qs[14] & 0xF, qs[15] & 0xF) - 8.0f; \
            float4 q_hi_3 = float4(qs[12] >> 4, qs[13] >> 4, qs[14] >> 4, qs[15] >> 4) - 8.0f; \
            sum_var += scale * (dot(a0, q_lo_0) + dot(a4, q_hi_0) + \
                                dot(a1, q_lo_1) + dot(a5, q_hi_1) + \
                                dot(a2, q_lo_2) + dot(a6, q_hi_2) + \
                                dot(a3, q_lo_3) + dot(a7, q_hi_3)); \
        }

        PROCESS_ROW(row0, sum0);
        PROCESS_ROW(row1, sum1);
        PROCESS_ROW(row2, sum2);
        PROCESS_ROW(row3, sum3);
        #undef PROCESS_ROW
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    if (simd_lane == 0) {
        if (base_output + 0 < N) out[base_output + 0] = sum0;
        if (base_output + 1 < N) out[base_output + 1] = sum1;
        if (base_output + 2 < N) out[base_output + 2] = sum2;
        if (base_output + 3 < N) out[base_output + 3] = sum3;
    }
}

// Fused RMSNorm + Q4_0 MatVec with FP16 OUTPUT
// Same as above but outputs half-precision for FP16 attention path
// Eliminates FP32->FP16 conversion after QKV projections
kernel void matvec_q4_0_fused_rmsnorm_f16_out(
    device const float* x [[buffer(0)]],           // [K] input activations (FP32)
    device const float* normWeight [[buffer(1)]],  // [K] RMSNorm weights
    device const uchar* W [[buffer(2)]],           // [N, K] Q4_0 weights
    device half* out [[buffer(3)]],                // [N] output (FP16!)
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_x [[threadgroup(0)]], // [K] shared activations
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & Sum Squares (identical to F32 version)
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i];
        shared_x[i] = val;
        localSumSq += val * val;
    }

    localSumSq = simd_sum(localSumSq);

    threadgroup float* scratch = shared_x + K;
    if (simd_lane == 0) {
        scratch[simd_group] = localSumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float totalSumSq = 0.0f;
    if (simd_group == 0) {
        float val = (simd_lane < 8) ? scratch[simd_lane] : 0.0f;
        totalSumSq = simd_sum(val);
    }

    if (tid == 0) {
        scratch[0] = rsqrt(totalSumSq / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = scratch[0];

    // Phase 2: MatVec (identical to F32 version)
    int base_output = gid * 32 + simd_group * 4;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int numBlocks = (K + 32 - 1) / 32;

    device const uchar* row0 = (base_output + 0 < N) ? W + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* row1 = (base_output + 1 < N) ? W + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* row2 = (base_output + 2 < N) ? W + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* row3 = (base_output + 3 < N) ? W + (base_output + 3) * numBlocks * 18 : nullptr;

    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * 32;
        if (base_k >= K) break;

        device const float4* w_ptr = (device const float4*)(normWeight + base_k);
        float4 w0 = w_ptr[0], w1 = w_ptr[1], w2 = w_ptr[2], w3 = w_ptr[3];
        float4 w4 = w_ptr[4], w5 = w_ptr[5], w6 = w_ptr[6], w7 = w_ptr[7];

        threadgroup const float4* x_ptr = (threadgroup const float4*)(shared_x + base_k);
        float4 x0 = x_ptr[0], x1 = x_ptr[1], x2 = x_ptr[2], x3 = x_ptr[3];
        float4 x4 = x_ptr[4], x5 = x_ptr[5], x6 = x_ptr[6], x7 = x_ptr[7];

        float4 a0 = x0 * rms * w0, a1 = x1 * rms * w1, a2 = x2 * rms * w2, a3 = x3 * rms * w3;
        float4 a4 = x4 * rms * w4, a5 = x5 * rms * w5, a6 = x6 * rms * w6, a7 = x7 * rms * w7;

        #define PROCESS_ROW_F16(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* blockPtr = row_ptr + block * 18; \
            float scale = as_type<half>(*((device const ushort*)blockPtr)); \
            device const uchar* qs = blockPtr + 2; \
            float4 q_lo_0 = float4(qs[0] & 0xF, qs[1] & 0xF, qs[2] & 0xF, qs[3] & 0xF) - 8.0f; \
            float4 q_hi_0 = float4(qs[0] >> 4, qs[1] >> 4, qs[2] >> 4, qs[3] >> 4) - 8.0f; \
            float4 q_lo_1 = float4(qs[4] & 0xF, qs[5] & 0xF, qs[6] & 0xF, qs[7] & 0xF) - 8.0f; \
            float4 q_hi_1 = float4(qs[4] >> 4, qs[5] >> 4, qs[6] >> 4, qs[7] >> 4) - 8.0f; \
            float4 q_lo_2 = float4(qs[8] & 0xF, qs[9] & 0xF, qs[10] & 0xF, qs[11] & 0xF) - 8.0f; \
            float4 q_hi_2 = float4(qs[8] >> 4, qs[9] >> 4, qs[10] >> 4, qs[11] >> 4) - 8.0f; \
            float4 q_lo_3 = float4(qs[12] & 0xF, qs[13] & 0xF, qs[14] & 0xF, qs[15] & 0xF) - 8.0f; \
            float4 q_hi_3 = float4(qs[12] >> 4, qs[13] >> 4, qs[14] >> 4, qs[15] >> 4) - 8.0f; \
            sum_var += scale * (dot(a0, q_lo_0) + dot(a4, q_hi_0) + \
                                dot(a1, q_lo_1) + dot(a5, q_hi_1) + \
                                dot(a2, q_lo_2) + dot(a6, q_hi_2) + \
                                dot(a3, q_lo_3) + dot(a7, q_hi_3)); \
        }

        PROCESS_ROW_F16(row0, sum0);
        PROCESS_ROW_F16(row1, sum1);
        PROCESS_ROW_F16(row2, sum2);
        PROCESS_ROW_F16(row3, sum3);
        #undef PROCESS_ROW_F16
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    // Write output as FP16 (the only difference from F32 version!)
    if (simd_lane == 0) {
        if (base_output + 0 < N) out[base_output + 0] = half(sum0);
        if (base_output + 1 < N) out[base_output + 1] = half(sum1);
        if (base_output + 2 < N) out[base_output + 2] = half(sum2);
        if (base_output + 3 < N) out[base_output + 3] = half(sum3);
    }
}

// Fused MLP Kernel: Gate(x) * Up(x) -> SiLU(x @ W1) * (x @ W3)
// Weights are Q4_0. Only for decode (seqLen=1).
// This runs TWO matvecs in parallel (interleaved) and fuses the SiLU/Mul.
// W1: [Intermediate, Hidden], W3: [Intermediate, Hidden]
// Output: [Intermediate]
kernel void matvec_q4_0_fused_mlp_f32(
    device const float* x [[buffer(0)]],      // [Hidden] input
    device const uchar* W1 [[buffer(1)]],     // [Inter, Hidden] Q4_0 Gate
    device const uchar* W3 [[buffer(2)]],     // [Inter, Hidden] Q4_0 Up
    device float* out [[buffer(3)]],          // [Inter] output
    constant int& N [[buffer(4)]],            // Intermediate size
    constant int& K [[buffer(5)]],            // Hidden size
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles 32 output elements (rows of W1/W3)
    // 4 outputs per simdgroup (8 simdgroups = 32 outputs)
    // Similar logic to matvec_q4_0_optimized_f32 but computing TWO values per row
    
    int base_output = gid * 32 + simd_group * 4;
    
    // Accumulators for W1 (Gate) and W3 (Up)
    float sumGate0 = 0.0f, sumGate1 = 0.0f, sumGate2 = 0.0f, sumGate3 = 0.0f;
    float sumUp0 = 0.0f, sumUp1 = 0.0f, sumUp2 = 0.0f, sumUp3 = 0.0f;
    
    int numBlocks = (K + 32 - 1) / 32; // Q4_0 blocks (32 elements)
    
    // W1 rows
    device const uchar* w1_row0 = (base_output + 0 < N) ? W1 + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* w1_row1 = (base_output + 1 < N) ? W1 + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* w1_row2 = (base_output + 2 < N) ? W1 + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* w1_row3 = (base_output + 3 < N) ? W1 + (base_output + 3) * numBlocks * 18 : nullptr;

    // W3 rows
    device const uchar* w3_row0 = (base_output + 0 < N) ? W3 + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* w3_row1 = (base_output + 1 < N) ? W3 + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* w3_row2 = (base_output + 2 < N) ? W3 + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* w3_row3 = (base_output + 3 < N) ? W3 + (base_output + 3) * numBlocks * 18 : nullptr;

    // Iterate over K (hidden dim)
    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * 32;
        if (base_k >= K) break; // Should be guarded by block count anyway

        // Load 32 activations (FP32)
        device const float4* x_ptr = (device const float4*)(x + base_k);
        float4 x0 = x_ptr[0], x1 = x_ptr[1], x2 = x_ptr[2], x3 = x_ptr[3];
        float4 x4 = x_ptr[4], x5 = x_ptr[5], x6 = x_ptr[6], x7 = x_ptr[7];

        // Process W1 (Gate)
        #define PROCESS_ROW(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* blockPtr = row_ptr + block * 18; \
            float scale = as_type<half>(*((device const ushort*)blockPtr)); \
            device const uchar* qs = blockPtr + 2; \
            /* Dequantize 32 nibbles */ \
            float4 q_lo_0 = float4(qs[0] & 0xF, qs[1] & 0xF, qs[2] & 0xF, qs[3] & 0xF) - 8.0f; \
            float4 q_hi_0 = float4(qs[0] >> 4, qs[1] >> 4, qs[2] >> 4, qs[3] >> 4) - 8.0f; \
            float4 q_lo_1 = float4(qs[4] & 0xF, qs[5] & 0xF, qs[6] & 0xF, qs[7] & 0xF) - 8.0f; \
            float4 q_hi_1 = float4(qs[4] >> 4, qs[5] >> 4, qs[6] >> 4, qs[7] >> 4) - 8.0f; \
            float4 q_lo_2 = float4(qs[8] & 0xF, qs[9] & 0xF, qs[10] & 0xF, qs[11] & 0xF) - 8.0f; \
            float4 q_hi_2 = float4(qs[8] >> 4, qs[9] >> 4, qs[10] >> 4, qs[11] >> 4) - 8.0f; \
            float4 q_lo_3 = float4(qs[12] & 0xF, qs[13] & 0xF, qs[14] & 0xF, qs[15] & 0xF) - 8.0f; \
            float4 q_hi_3 = float4(qs[12] >> 4, qs[13] >> 4, qs[14] >> 4, qs[15] >> 4) - 8.0f; \
            sum_var += scale * (dot(x0, q_lo_0) + dot(x4, q_hi_0) + \
                                dot(x1, q_lo_1) + dot(x5, q_hi_1) + \
                                dot(x2, q_lo_2) + dot(x6, q_hi_2) + \
                                dot(x3, q_lo_3) + dot(x7, q_hi_3)); \
        }

        PROCESS_ROW(w1_row0, sumGate0);
        PROCESS_ROW(w1_row1, sumGate1);
        PROCESS_ROW(w1_row2, sumGate2);
        PROCESS_ROW(w1_row3, sumGate3);

        // Process W3 (Up)
        PROCESS_ROW(w3_row0, sumUp0);
        PROCESS_ROW(w3_row1, sumUp1);
        PROCESS_ROW(w3_row2, sumUp2);
        PROCESS_ROW(w3_row3, sumUp3);
        
        #undef PROCESS_ROW
    }

    // Reduce
    sumGate0 = simd_sum(sumGate0); sumUp0 = simd_sum(sumUp0);
    sumGate1 = simd_sum(sumGate1); sumUp1 = simd_sum(sumUp1);
    sumGate2 = simd_sum(sumGate2); sumUp2 = simd_sum(sumUp2);
    sumGate3 = simd_sum(sumGate3); sumUp3 = simd_sum(sumUp3);

    // Apply SiLU and Multiply: out = (x * sigmoid(x)) * y
    // SiLU(g) * u
    if (simd_lane == 0) {
        if (base_output + 0 < N) {
            float sigmoid = 1.0f / (1.0f + exp(-sumGate0));
            out[base_output + 0] = (sumGate0 * sigmoid) * sumUp0;
        }
        if (base_output + 1 < N) {
            float sigmoid = 1.0f / (1.0f + exp(-sumGate1));
            out[base_output + 1] = (sumGate1 * sigmoid) * sumUp1;
        }
        if (base_output + 2 < N) {
            float sigmoid = 1.0f / (1.0f + exp(-sumGate2));
            out[base_output + 2] = (sumGate2 * sigmoid) * sumUp2;
        }
        if (base_output + 3 < N) {
            float sigmoid = 1.0f / (1.0f + exp(-sumGate3));
            out[base_output + 3] = (sumGate3 * sigmoid) * sumUp3;
        }
    }
}

// =============================================================================
// FUSED ADD + RMSNORM KERNEL
// =============================================================================
// Fuses residual addition with RMSNorm to save one memory round-trip.
// x = x + residual (in-place), then out = RMSNorm(x)
// This is the pattern after attention output projection and FFN output projection.

kernel void add_rmsnorm_f32(
    device float* x [[buffer(0)]],              // [rows, dim] - updated in-place
    device const float* residual [[buffer(1)]], // [rows, dim] - residual to add
    device const float* weight [[buffer(2)]],   // [dim] - RMSNorm weight
    device float* out [[buffer(3)]],            // [rows, dim] - normalized output
    constant int& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base = row * dim;

    // Phase 1: Add residual and compute sum of squares simultaneously
    float sumSq = 0.0f;
    for (int i = tid; i < dim; i += RMSNORM_THREADGROUP_SIZE) {
        float val = x[base + i] + residual[base + i];
        x[base + i] = val;  // Store updated x in-place
        sumSq += val * val;
    }

    // Simd reduction
    sumSq = simd_sum(sumSq);

    // Store simdgroup results to shared memory
    if (simd_lane == 0) {
        shared[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first simdgroup
    float totalSumSq = 0.0f;
    if (simd_group == 0) {
        int num_simd = (RMSNORM_THREADGROUP_SIZE + 31) / 32;
        float simd_val = (simd_lane < (uint)num_simd) ? shared[simd_lane] : 0.0f;
        totalSumSq = simd_sum(simd_val);
    }

    // Broadcast RMS to all threads
    if (tid == 0) {
        shared[0] = rsqrt(totalSumSq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = shared[0];

    // Phase 2: Apply RMSNorm
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
// ropeNeox: 0 = LLaMA-style (interleaved pairs (0,1), (2,3)...),
//           1 = NEOX-style (split pairs (i, i+ropeDim/2))
kernel void rope_gqa_f32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant int& seqLen [[buffer(2)]],
    constant int& numQHeads [[buffer(3)]],
    constant int& numKVHeads [[buffer(4)]],
    constant int& headDim [[buffer(5)]],
    constant int& startPos [[buffer(6)]],
    constant int& ropeDim [[buffer(7)]],  // Dimensions to rotate (can be < headDim for partial RoPE)
    constant float& theta [[buffer(8)]],
    constant int& ropeNeox [[buffer(9)]],  // 0 = LLaMA-style, 1 = NEOX-style
    uint2 gid [[thread_position_in_grid]]
) {
    int pos = gid.x;
    int head = gid.y;

    if (pos >= seqLen) return;

    int absPos = startPos + pos;
    // For partial RoPE (like Phi-2), only rotate first ropeDim dimensions
    int halfRopeDim = ropeDim / 2;

    // Process Q heads
    if (head < numQHeads) {
        int qOffset = (pos * numQHeads + head) * headDim;
        for (int j = 0; j < halfRopeDim; j++) {
            // Use ropeDim for frequency calculation to match expected RoPE behavior
            float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            int idx0, idx1;
            if (ropeNeox != 0) {
                // NEOX-style: pairs are (i, i + ropeDim/2)
                idx0 = j;
                idx1 = j + halfRopeDim;
            } else {
                // LLaMA-style (interleaved): pairs are (2j, 2j+1)
                idx0 = j * 2;
                idx1 = j * 2 + 1;
            }

            float q0 = q[qOffset + idx0];
            float q1 = q[qOffset + idx1];
            q[qOffset + idx0] = q0 * cos_val - q1 * sin_val;
            q[qOffset + idx1] = q0 * sin_val + q1 * cos_val;
        }
    }

    // Process K heads (fewer due to GQA)
    if (head < numKVHeads) {
        int kOffset = (pos * numKVHeads + head) * headDim;
        for (int j = 0; j < halfRopeDim; j++) {
            float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            int idx0, idx1;
            if (ropeNeox != 0) {
                // NEOX-style: pairs are (i, i + ropeDim/2)
                idx0 = j;
                idx1 = j + halfRopeDim;
            } else {
                // LLaMA-style (interleaved): pairs are (2j, 2j+1)
                idx0 = j * 2;
                idx1 = j * 2 + 1;
            }

            float k0 = k[kOffset + idx0];
            float k1 = k[kOffset + idx1];
            k[kOffset + idx0] = k0 * cos_val - k1 * sin_val;
            k[kOffset + idx1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// RoPE for GQA with pre-computed per-dimension inverse frequencies.
// Used for learned RoPE scaling (Gemma 2). Reads frequencies from a buffer
// instead of computing from theta. freqs has [headDim/2] float32 values.
kernel void rope_gqa_scaled_f32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    const device float* freqs [[buffer(2)]],  // Pre-computed inverse frequencies [headDim/2]
    constant int& seqLen [[buffer(3)]],
    constant int& numQHeads [[buffer(4)]],
    constant int& numKVHeads [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& startPos [[buffer(7)]],
    constant int& ropeNeox [[buffer(8)]],
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
            float freq = freqs[j];
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            int idx0, idx1;
            if (ropeNeox != 0) {
                idx0 = j;
                idx1 = j + halfDim;
            } else {
                idx0 = j * 2;
                idx1 = j * 2 + 1;
            }

            float q0 = q[qOffset + idx0];
            float q1 = q[qOffset + idx1];
            q[qOffset + idx0] = q0 * cos_val - q1 * sin_val;
            q[qOffset + idx1] = q0 * sin_val + q1 * cos_val;
        }
    }

    // Process K heads (fewer due to GQA)
    if (head < numKVHeads) {
        int kOffset = (pos * numKVHeads + head) * headDim;
        for (int j = 0; j < halfDim; j++) {
            float freq = freqs[j];
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            int idx0, idx1;
            if (ropeNeox != 0) {
                idx0 = j;
                idx1 = j + halfDim;
            } else {
                idx0 = j * 2;
                idx1 = j * 2 + 1;
            }

            float k0 = k[kOffset + idx0];
            float k1 = k[kOffset + idx1];
            k[kOffset + idx0] = k0 * cos_val - k1 * sin_val;
            k[kOffset + idx1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// RoPE for GQA with FP16 inputs/outputs
// Computation done in FP32 for numerical stability, I/O in FP16
// Eliminates FP32->FP16 conversions in attention path
// ropeDim: dimensions to rotate (can be < headDim for partial RoPE like Phi-2)
// ropeNeox: 0 = LLaMA-style (interleaved pairs (0,1), (2,3)...),
//           1 = NEOX-style (split pairs (i, i+ropeDim/2))
kernel void rope_gqa_f16(
    device half* q [[buffer(0)]],
    device half* k [[buffer(1)]],
    constant int& seqLen [[buffer(2)]],
    constant int& numQHeads [[buffer(3)]],
    constant int& numKVHeads [[buffer(4)]],
    constant int& headDim [[buffer(5)]],
    constant int& startPos [[buffer(6)]],
    constant int& ropeDim [[buffer(7)]],  // Dimensions to rotate (can be < headDim for partial RoPE)
    constant float& theta [[buffer(8)]],
    constant int& ropeNeox [[buffer(9)]],  // 0 = LLaMA-style, 1 = NEOX-style
    uint2 gid [[thread_position_in_grid]]
) {
    int pos = gid.x;
    int head = gid.y;

    if (pos >= seqLen) return;

    int absPos = startPos + pos;
    // For partial RoPE (like Phi-2), only rotate first ropeDim dimensions
    int halfRopeDim = ropeDim / 2;

    // Process Q heads
    if (head < numQHeads) {
        int qOffset = (pos * numQHeads + head) * headDim;
        for (int j = 0; j < halfRopeDim; j++) {
            // Use ropeDim for frequency calculation to match expected RoPE behavior
            float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            int idx0, idx1;
            if (ropeNeox != 0) {
                // NEOX-style: pairs are (i, i + ropeDim/2)
                idx0 = j;
                idx1 = j + halfRopeDim;
            } else {
                // LLaMA-style (interleaved): pairs are (2j, 2j+1)
                idx0 = j * 2;
                idx1 = j * 2 + 1;
            }

            // Read as FP16, compute in FP32
            float q0 = float(q[qOffset + idx0]);
            float q1 = float(q[qOffset + idx1]);
            // Write back as FP16
            q[qOffset + idx0] = half(q0 * cos_val - q1 * sin_val);
            q[qOffset + idx1] = half(q0 * sin_val + q1 * cos_val);
        }
    }

    // Process K heads (fewer due to GQA)
    if (head < numKVHeads) {
        int kOffset = (pos * numKVHeads + head) * headDim;
        for (int j = 0; j < halfRopeDim; j++) {
            float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
            float angle = float(absPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            int idx0, idx1;
            if (ropeNeox != 0) {
                // NEOX-style: pairs are (i, i + ropeDim/2)
                idx0 = j;
                idx1 = j + halfRopeDim;
            } else {
                // LLaMA-style (interleaved): pairs are (2j, 2j+1)
                idx0 = j * 2;
                idx1 = j * 2 + 1;
            }

            // Read as FP16, compute in FP32
            float k0 = float(k[kOffset + idx0]);
            float k1 = float(k[kOffset + idx1]);
            // Write back as FP16
            k[kOffset + idx0] = half(k0 * cos_val - k1 * sin_val);
            k[kOffset + idx1] = half(k0 * sin_val + k1 * cos_val);
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

// =============================================================================
// Training Kernels for Medusa Head Training
// =============================================================================

// LeakyReLU in-place: x = x > 0 ? x : alpha * x
// Using alpha=0.01 to prevent dead neurons during training
constant float LEAKY_RELU_ALPHA = 0.01f;

kernel void relu_inplace_f32(
    device float* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float val = x[gid];
    x[gid] = val > 0.0f ? val : LEAKY_RELU_ALPHA * val;
}

// LeakyReLU backward: dx = dx * (x > 0 ? 1 : alpha)
kernel void relu_backward_f32(
    device const float* x [[buffer(0)]],      // forward input (pre-activation)
    device float* dx [[buffer(1)]],           // gradient (modified in-place)
    uint gid [[thread_position_in_grid]]
) {
    dx[gid] = (x[gid] > 0.0f) ? dx[gid] : LEAKY_RELU_ALPHA * dx[gid];
}

// Batched outer product for gradient accumulation
// C[i,j] += sum_b(A[b,i] * B[b,j]) for all b in batch
// A: [batch, M], B: [batch, N], C: [M, N]
// Used for computing dFC1 and dFC2 gradients
kernel void batched_outer_product_f32(
    device const float* A [[buffer(0)]],      // [batch, M]
    device const float* B [[buffer(1)]],      // [batch, N]
    device float* C [[buffer(2)]],            // [M, N]
    constant int& batch [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int i = gid.y;  // M dimension
    int j = gid.x;  // N dimension
    if (i >= M || j >= N) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum += A[b * M + i] * B[b * N + j];
    }
    C[i * N + j] += sum;
}

/// SGD weight update with weight decay: w = w * (1 - lr * wd) - lr * grad
// This is L2 regularization which prevents weights from growing unbounded
kernel void sgd_update_f32(
    device float* w [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    constant float& lr [[buffer(2)]],
    constant float& weightDecay [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float decay = 1.0f - lr * weightDecay;
    w[gid] = w[gid] * decay - lr * grad[gid];
}

// Zero out a buffer (for initializing gradients)
kernel void zero_f32(
    device float* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    x[gid] = 0.0f;
}

// =============================================================================
// FP16 (Half-Precision) Kernels
// These provide 2x memory bandwidth for memory-bound operations.
// Accumulation is done in FP32 for numerical stability where needed.
// =============================================================================

// Element-wise add for FP16
kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

// Element-wise multiply for FP16
kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] * b[gid];
}

// SiLU activation for FP16
// Uses half4 for better vectorization (process 4 elements at a time)
kernel void silu_f16(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    half val = x[gid];
    // Compute SiLU in FP32 for numerical stability, then convert back
    float f = float(val);
    float silu = f / (1.0f + exp(-f));
    out[gid] = half(silu);
}

// Fused SiLU+Mul for FP16
kernel void silu_mul_f16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = float(gate[gid]);
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = half(silu_g * float(up[gid]));
}

// FP32 to FP16 conversion kernel
// Converts float32 array to float16
kernel void convert_f32_to_f16(
    device const float* in [[buffer(0)]],
    device half* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = half(in[gid]);
}

// FP16 to FP32 conversion kernel
// Converts float16 array to float32
kernel void convert_f16_to_f32(
    device const half* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = float(in[gid]);
}

// =============================================================================
// KV Cache Scatter Kernel
// Transposes from [newTokens, numKVHeads, headDim] to [numKVHeads, maxSeqLen, headDim]
// =============================================================================

// Scatter KV data for FP16 cache
// Input layout: [newTokens, numKVHeads, headDim] - contiguous per-token KV
// Output layout: [numKVHeads, maxSeqLen, headDim] - contiguous per-head across sequence
// One thread per element, scatters to correct head position
kernel void scatter_kv_f16(
    device const half* src [[buffer(0)]],  // Input: [newTokens, numKVHeads, headDim]
    device half* dst [[buffer(1)]],        // Output: [numKVHeads, maxSeqLen, headDim]
    constant int& newTokens [[buffer(2)]],
    constant int& numKVHeads [[buffer(3)]],
    constant int& headDim [[buffer(4)]],
    constant int& maxSeqLen [[buffer(5)]],
    constant int& seqPos [[buffer(6)]],    // Starting position in sequence
    uint gid [[thread_position_in_grid]]
) {
    int totalElements = newTokens * numKVHeads * headDim;
    if (gid >= (uint)totalElements) return;

    // Decode source indices
    // gid = t * numKVHeads * headDim + h * headDim + d
    int srcHeadStride = numKVHeads * headDim;
    int t = gid / srcHeadStride;
    int remainder = gid % srcHeadStride;
    int h = remainder / headDim;
    int d = remainder % headDim;

    // Compute destination offset
    // dst[h][seqPos + t][d] = dst[h * maxSeqLen * headDim + (seqPos + t) * headDim + d]
    int dstOffset = h * maxSeqLen * headDim + (seqPos + t) * headDim + d;

    dst[dstOffset] = src[gid];
}

kernel void scatter_kv_f32(
    device const float* src [[buffer(0)]], // Input: [newTokens, numKVHeads, headDim]
    device float* dst [[buffer(1)]],       // Output: [numKVHeads, maxSeqLen, headDim]
    constant int& newTokens [[buffer(2)]],
    constant int& numKVHeads [[buffer(3)]],
    constant int& headDim [[buffer(4)]],
    constant int& maxSeqLen [[buffer(5)]],
    constant int& seqPos [[buffer(6)]],    // Starting position in sequence
    uint gid [[thread_position_in_grid]]
) {
    int totalElements = newTokens * numKVHeads * headDim;
    if (gid >= (uint)totalElements) return;

    // Decode source indices
    int srcHeadStride = numKVHeads * headDim;
    int t = gid / srcHeadStride;
    int remainder = gid % srcHeadStride;
    int h = remainder / headDim;
    int d = remainder % headDim;

    // Compute destination offset
    int dstOffset = h * maxSeqLen * headDim + (seqPos + t) * headDim + d;

    dst[dstOffset] = src[gid];
}

kernel void scatter_kv_f32_to_f16(
    device const float* src [[buffer(0)]], // Input: [newTokens, numKVHeads, headDim] in F32
    device half* dst [[buffer(1)]],        // Output: [numKVHeads, maxSeqLen, headDim] in F16
    constant int& newTokens [[buffer(2)]],
    constant int& numKVHeads [[buffer(3)]],
    constant int& headDim [[buffer(4)]],
    constant int& maxSeqLen [[buffer(5)]],
    constant int& seqPos [[buffer(6)]],    // Starting position in sequence
    uint gid [[thread_position_in_grid]]
) {
    int totalElements = newTokens * numKVHeads * headDim;
    if (gid >= (uint)totalElements) return;

    // Decode source indices
    int srcHeadStride = numKVHeads * headDim;
    int t = gid / srcHeadStride;
    int remainder = gid % srcHeadStride;
    int h = remainder / headDim;
    int d = remainder % headDim;

    // Compute destination offset
    int dstOffset = h * maxSeqLen * headDim + (seqPos + t) * headDim + d;

    dst[dstOffset] = half(src[gid]);
}

// Reshape KV cache for paged attention
// Copies data from src [numTokens, numKVHeads, headDim] to paged blocks.
// blockIndices: [numTokens] int32, physical block index for each token
// blockOffsets: [numTokens] int32, token index within the block for each token
// dstBase: Base pointer to the block pool
// blockSize: number of tokens per block
// Layout of one block: K [blockSize, numKVHeads, headDim], then V [blockSize, numKVHeads, headDim]
// Total block size in elements: 2 * blockSize * numKVHeads * headDim
kernel void reshape_paged_kv_f32(
    device const float* src [[buffer(0)]],      // Input: [newTokens, numKVHeads, headDim] (K then V? No, usually interleaved or separate. Let's assume src is K then V concatenated? No, usually src is [numTokens, 2, numKVHeads, headDim] or we launch twice? Let's assume src contains K only for now, and we call it twice for V?)
                                                // Actually, standard is to pass K and V src pointers separately? Or src has K and V?
                                                // Let's look at `AppendKV`: it takes kPtr and vPtr.
                                                // So we should have `reshape_paged_kv_f32` take `src` and write to `dstBase` which is EITHER K or V part of the block?
                                                // Block layout: K data... V data...
                                                // If we call this for K, we write to K part. If for V, we write to V part.
                                                // So we need a `dstOffsetBytes` or similar?
                                                // Let's assume `src` is [newTokens, numKVHeads, headDim].
                                                // `dstBase` is the start of the block pool.
                                                // We need to know if we are writing K or V.
                                                // Let's add `isV` flag or `dstBlockOffset`?
                                                // Better: `dstBase` is adjusted by caller to point to start of K or V section in the pool?
                                                // No, blocks are interleaved. Block 0 [K...V...], Block 1 [K...V...]
                                                // So `dstBase` is start of pool.
                                                // We need `isValue` boolean?
    device float* dstBase [[buffer(1)]],        // Base of block pool
    device const int* blockIndices [[buffer(2)]], // [newTokens]
    device const int* blockOffsets [[buffer(3)]], // [newTokens]
    constant int& newTokens [[buffer(4)]],
    constant int& numKVHeads [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& blockSize [[buffer(7)]],
    constant int& isValue [[buffer(8)]],        // 0 for K, 1 for V
    uint gid [[thread_position_in_grid]]
) {
    // Total threads: newTokens * numKVHeads * headDim
    int totalElements = newTokens * numKVHeads * headDim;
    if (gid >= (uint)totalElements) return;

    // Decode source indices
    int srcHeadStride = numKVHeads * headDim;
    int t = gid / srcHeadStride;
    int remainder = gid % srcHeadStride;
    int h = remainder / headDim;
    int d = remainder % headDim;

    // Get block info for token t
    int blockIdx = blockIndices[t];
    int tokenInBlock = blockOffsets[t];

    // Block layout:
    // K part: [blockSize, numKVHeads, headDim]
    // V part: [blockSize, numKVHeads, headDim]
    // Elements per part: blockSize * numKVHeads * headDim
    int elementsPerBlockPart = blockSize * numKVHeads * headDim;
    int elementsPerBlock = 2 * elementsPerBlockPart;

    // Calculate destination offset in elements
    // Block base
    long dstIdx = (long)blockIdx * elementsPerBlock;
    
    // Add V offset if needed
    if (isValue) {
        dstIdx += elementsPerBlockPart;
    }

    // Offset within part: token_idx * (heads * dim) + head * dim + dim_idx
    int offsetInPart = tokenInBlock * srcHeadStride + h * headDim + d;
    dstIdx += offsetInPart;

    dstBase[dstIdx] = src[gid];
}

// =============================================================================
// Q8_0 Quantization for KV Cache
// Q8_0 format: 34 bytes per 32 elements (2-byte f16 scale + 32 int8 values)
// Provides 4x memory reduction vs FP32, 2x vs FP16
// =============================================================================

constant int Q8_BLOCK_SIZE = 32;
constant int Q8_BYTES_PER_BLOCK = 34;

// Quantize FP32 to Q8_0
// in: [n] float32, out: [n/32 * 34] bytes in Q8_0 format
// Each threadgroup handles one block of 32 elements
kernel void quantize_f32_to_q8_0(
    device const float* in [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Each threadgroup handles one Q8_0 block (32 elements)
    int blockIdx = gid;
    int numBlocks = (n + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    if (blockIdx >= numBlocks) return;

    int baseIdx = blockIdx * Q8_BLOCK_SIZE;
    device uchar* blockOut = out + blockIdx * Q8_BYTES_PER_BLOCK;

    // Thread 0 finds max abs and writes scale
    if (tid == 0) {
        float maxAbs = 0.0f;
        for (int i = 0; i < Q8_BLOCK_SIZE && (baseIdx + i) < n; i++) {
            float val = in[baseIdx + i];
            float absVal = abs(val);
            if (absVal > maxAbs) maxAbs = absVal;
        }

        // Scale: max_abs / 127
        float scale = maxAbs / 127.0f;

        // Write scale as f16 (first 2 bytes)
        half scaleHalf = half(scale);
        device half* scalePtr = (device half*)blockOut;
        *scalePtr = scaleHalf;

        // Quantize values
        float invScale = (scale > 0.0f) ? (127.0f / maxAbs) : 0.0f;
        for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
            float val = (baseIdx + i < n) ? in[baseIdx + i] : 0.0f;
            int8_t q = int8_t(round(val * invScale));
            // Cast int8_t to uchar via uint8_t (bit-preserving cast)
            blockOut[2 + i] = uchar(uint8_t(q));
        }
    }
}

// Dequantize Q8_0 to FP32
// in: Q8_0 format, out: [n] float32
kernel void dequantize_q8_0_to_f32(
    device const uchar* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)n) return;

    int blockIdx = gid / Q8_BLOCK_SIZE;
    int inBlockIdx = gid % Q8_BLOCK_SIZE;

    device const uchar* block = in + blockIdx * Q8_BYTES_PER_BLOCK;

    // Read scale (f16)
    half scaleHalf = *((device const half*)block);
    float scale = float(scaleHalf);

    // Read quantized value (int8)
    int8_t q = *((device const int8_t*)(block + 2 + inBlockIdx));

    out[gid] = float(q) * scale;
}

// SDPA decode with Q8_0 KV cache
// Q: [numQHeads, headDim] in FP32
// K/V: [kvLen, numKVHeads, headDim] in Q8_0 format
// out: [numQHeads, headDim] in FP32
// Each threadgroup handles one Q head
constant int SDPA_Q8_THREADS = 256;

// Helper to dequantize a single element from Q8_0 block
inline float dequant_q8_0_elem(device const uchar* block, int elemInBlock) {
    half scaleHalf = *((device const half*)block);
    int8_t q = *((device const int8_t*)(block + 2 + elemInBlock));
    return float(q) * float(scaleHalf);
}

kernel void sdpa_decode_q8_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K [[buffer(1)]],
    device const uchar* V [[buffer(2)]],
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

    // Calculate Q8_0 layout for KV cache
    // Each KV position has numKVHeads * headDim elements, quantized in blocks of 32
    int kvPosElements = numKVHeads * headDim;
    int blocksPerPos = (kvPosElements + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    int bytesPerPos = blocksPerPos * Q8_BYTES_PER_BLOCK;

    // This head's element range within KV position
    int kvHeadStart = kvHead * headDim;

    // Shared memory layout: [weights, warpVals]
    threadgroup float* weights = shared;
    threadgroup float* warpVals = shared + kvLen;

    // Phase 1a: Compute Q·K scores with Q8_0 dequantization
    float localMax = -INFINITY;
    for (int pos = tid; pos < kvLen; pos += SDPA_Q8_THREADS) {
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            int elemIdx = kvHeadStart + d;
            int blockIdx = elemIdx / Q8_BLOCK_SIZE;
            int inBlockIdx = elemIdx % Q8_BLOCK_SIZE;
            device const uchar* kBlock = K + pos * bytesPerPos + blockIdx * Q8_BYTES_PER_BLOCK;
            float kVal = dequant_q8_0_elem(kBlock, inBlockIdx);
            dot += Q[qOffset + d] * kVal;
        }
        float score = dot * scale;
        weights[pos] = score;
        localMax = max(localMax, score);
    }

    // Reduce max across threadgroup
    localMax = simd_max(localMax);
    if (simd_lane == 0) warpVals[simd_group] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8) {
        localMax = warpVals[tid];
        localMax = simd_max(localMax);
        if (tid == 0) warpVals[0] = localMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gMax = warpVals[0];

    // Phase 1b: Compute exp(score - max) and sum
    float localSum = 0.0f;
    for (int pos = tid; pos < kvLen; pos += SDPA_Q8_THREADS) {
        float expScore = exp(weights[pos] - gMax);
        weights[pos] = expScore;
        localSum += expScore;
    }

    // Reduce sum
    localSum = simd_sum(localSum);
    if (simd_lane == 0) warpVals[simd_group] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8) {
        localSum = warpVals[tid];
        localSum = simd_sum(localSum);
        if (tid == 0) warpVals[0] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = 1.0f / warpVals[0];

    // Normalize weights
    for (int pos = tid; pos < kvLen; pos += SDPA_Q8_THREADS) {
        weights[pos] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute weighted sum of V with Q8_0 dequantization
    int outOffset = qHead * headDim;
    for (int d = tid; d < headDim; d += SDPA_Q8_THREADS) {
        float sum = 0.0f;
        int elemIdx = kvHeadStart + d;
        int blockIdx = elemIdx / Q8_BLOCK_SIZE;
        int inBlockIdx = elemIdx % Q8_BLOCK_SIZE;

        for (int pos = 0; pos < kvLen; pos++) {
            device const uchar* vBlock = V + pos * bytesPerPos + blockIdx * Q8_BYTES_PER_BLOCK;
            float vVal = dequant_q8_0_elem(vBlock, inBlockIdx);
            sum += weights[pos] * vVal;
        }
        out[outOffset + d] = sum;
    }
}

// RMSNorm for FP16 input/output with FP32 accumulation
// x: [rows, dim] in FP16, weight: [dim] in FP32, out: [rows, dim] in FP16
constant int RMSNORM_F16_THREADGROUP_SIZE = 256;

kernel void rmsnorm_f16(
    device const half* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base = row * dim;

    // Phase 1: Each thread computes partial sum of squares in FP32
    float sumSq = 0.0f;
    for (int i = tid; i < dim; i += RMSNORM_F16_THREADGROUP_SIZE) {
        float val = float(x[base + i]);  // Convert to FP32 for accumulation
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
        int num_warps = (RMSNORM_F16_THREADGROUP_SIZE + 31) / 32;
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
    for (int i = tid; i < dim; i += RMSNORM_F16_THREADGROUP_SIZE) {
        float val = float(x[base + i]);
        out[base + i] = half(val * rms * weight[i]);
    }
}

// FP16 SDPA for decode - K/V cache in FP16 for 2x bandwidth savings
// Q: [numQHeads, headDim] in FP16
// K: [kvLen, numKVHeads, headDim] in FP16
// V: [kvLen, numKVHeads, headDim] in FP16
// out: [numQHeads, headDim] in FP16
constant int SDPA_F16_THREADS = 256;

kernel void sdpa_decode_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant int& kvHeadStride [[buffer(9)]],  // Stride between KV heads (maxSeqLen * headDim)
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

    // KV cache layout: [numKVHeads, maxSeqLen, headDim]
    // For head h, position p: offset = h * kvHeadStride + p * headDim
    // This makes position iteration contiguous for each head (good for memory bandwidth)
    int kvHeadBase = kvHead * kvHeadStride;

    // Shared memory layout:
    // [0..kvLen-1]: attention weights
    // [kvLen..kvLen+7]: warp max/sum values
    threadgroup float* weights = shared;
    threadgroup float* warpVals = shared + kvLen;

    // Phase 1a: Compute Q·K scores (convert FP16 to FP32 for computation)
    // K reads are now contiguous per position: K[kvHeadBase + pos * headDim + d]
    float localMax = -INFINITY;
    for (int pos = tid; pos < kvLen; pos += SDPA_F16_THREADS) {
        int kOffset = kvHeadBase + pos * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += float(Q[qOffset + d]) * float(K[kOffset + d]);
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

    // Phase 1b: Compute exp(score - max) and sum
    float localSum = 0.0f;
    for (int pos = tid; pos < kvLen; pos += SDPA_F16_THREADS) {
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

    // Normalize weights
    for (int pos = tid; pos < kvLen; pos += SDPA_F16_THREADS) {
        weights[pos] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute weighted sum of V (output in FP16)
    // V reads are now contiguous per position: V[kvHeadBase + pos * headDim + d]
    int outOffset = qHead * headDim;
    for (int d = tid; d < headDim; d += SDPA_F16_THREADS) {
        float sum = 0.0f;
        for (int pos = 0; pos < kvLen; pos++) {
            int vOffset = kvHeadBase + pos * headDim;
            sum += weights[pos] * float(V[vOffset + d]);
        }
        out[outOffset + d] = half(sum);
    }
}

// =============================================================================
// Specialized SDPA kernel for TinyLlama (headDim=64, numKVHeads=4)
// Uses half4 vectorization and online softmax for maximum efficiency
// =============================================================================

// Vectorized SDPA decode - keeps position parallelization but uses half4 for dot products
// This is an improved version of sdpa_decode_f16 with vectorized loads
// Q is cached in shared memory and K is loaded with half4 for vectorized dot products
kernel void sdpa_decode_f16_vec(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant int& kvHeadStride [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int qHead = gid;
    if (qHead >= numQHeads) return;

    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;
    int kvBase = kvHead * kvHeadStride;

    // Shared memory layout:
    // [0..headDim-1]: q_cache (float4 * numChunks)
    // [headDim..]: weights
    threadgroup float4* q_cache = (threadgroup float4*)shared;
    
    // Ensure weights start after Q cache (aligned to 4 floats/16 bytes)
    int qCacheSize = (headDim + 3) / 4 * 4;
    threadgroup float* weights = shared + qCacheSize;
    threadgroup float* warpVals = weights + kvLen;

    // Load Q once into shared memory using half4 (for headDim=64: 16 chunks)
    device const half4* Q4 = (device const half4*)(Q + qHead * headDim);
    int numChunks = headDim / 4;
    for (int c = tid; c < numChunks; c += 256) {
        q_cache[c] = float4(Q4[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1a: Compute Q·K scores with vectorized dot product
    float localMax = -INFINITY;
    for (int pos = tid; pos < kvLen; pos += 256) {
        device const half4* K4 = (device const half4*)(K + kvBase + pos * headDim);
        float dot = 0.0f;
        for (int c = 0; c < numChunks; c++) {
            float4 k = float4(K4[c]);
            float4 q = q_cache[c];
            dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
        }
        float score = dot * scale;
        weights[pos] = score;
        localMax = max(localMax, score);
    }

    // Reduce max across threadgroup
    localMax = simd_max(localMax);
    if (simd_lane == 0) warpVals[simd_group] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8) {
        localMax = warpVals[tid];
        localMax = simd_max(localMax);
        if (tid == 0) warpVals[0] = localMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gMax = warpVals[0];

    // Phase 1b: exp and sum
    float localSum = 0.0f;
    for (int pos = tid; pos < kvLen; pos += 256) {
        float expScore = exp(weights[pos] - gMax);
        weights[pos] = expScore;
        localSum += expScore;
    }

    localSum = simd_sum(localSum);
    if (simd_lane == 0) warpVals[simd_group] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8) {
        localSum = warpVals[tid];
        localSum = simd_sum(localSum);
        if (tid == 0) warpVals[0] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = 1.0f / warpVals[0];

    // Normalize
    for (int pos = tid; pos < kvLen; pos += 256) {
        weights[pos] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: weighted V sum
    // Each thread handles headDim/256 dimensions (with 256 threads, each handles at most 1 for headDim=64)
    for (int d = tid; d < headDim; d += 256) {
        float sum = 0.0f;
        for (int pos = 0; pos < kvLen; pos++) {
            sum += weights[pos] * float(V[kvBase + pos * headDim + d]);
        }
        out[qHead * headDim + d] = half(sum);
    }
}

// Specialized SDPA decode for headDim=64
// Uses half4 vectorization: 16 chunks of 4 elements
// Online softmax: no intermediate storage of all scores
// Q cached in registers, loaded once per head
kernel void sdpa_decode_f16_hd64(
    device const half* Q [[buffer(0)]],      // [numQHeads, 64] in FP16
    device const half* K [[buffer(1)]],      // [numKVHeads, maxSeqLen, 64] in FP16
    device const half* V [[buffer(2)]],      // [numKVHeads, maxSeqLen, 64] in FP16
    device half* out [[buffer(3)]],          // [numQHeads, 64] in FP16
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant int& kvHeadStride [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // One threadgroup per Q head, 64 threads per threadgroup
    // Each thread handles 1 dimension of the output (64 threads = 64 dims)
    int qHead = gid;
    if (qHead >= numQHeads) return;

    // GQA: map Q head to KV head
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;
    int kvBase = kvHead * kvHeadStride;

    // Load Q into registers using half4 (16 chunks × 4 elements = 64)
    // Each thread loads its portion for the simdgroup reduction
    device const half4* Q4 = (device const half4*)(Q + qHead * 64);
    float4 q_reg[16];
    for (int i = 0; i < 16; i++) {
        q_reg[i] = float4(Q4[i]);
    }

    // Online softmax state: track running max and sum
    float m = -INFINITY;  // running max
    float l = 0.0f;       // running sum of exp(score - m)

    // Accumulator for weighted V sum (float4 for precision)
    float4 acc[16];
    for (int i = 0; i < 16; i++) {
        acc[i] = float4(0.0f);
    }

    // Process KV positions with online softmax
    // Each thread computes the full dot product, then we use simd ops for reduction
    for (int pos = 0; pos < kvLen; pos++) {
        // Load K at this position using half4
        device const half4* K4 = (device const half4*)(K + kvBase + pos * 64);

        // Compute Q·K dot product (vectorized)
        float score = 0.0f;
        for (int i = 0; i < 16; i++) {
            float4 k = float4(K4[i]);
            score += dot(q_reg[i], k);
        }
        score *= scale;

        // Online softmax update
        float m_new = max(m, score);
        float exp_diff = exp(m - m_new);
        float exp_score = exp(score - m_new);

        // Update sum: l_new = l * exp(m - m_new) + exp(score - m_new)
        l = l * exp_diff + exp_score;

        // Update accumulator: acc = acc * exp(m - m_new) + exp(score - m_new) * V
        device const half4* V4 = (device const half4*)(V + kvBase + pos * 64);
        for (int i = 0; i < 16; i++) {
            float4 v = float4(V4[i]);
            acc[i] = acc[i] * exp_diff + exp_score * v;
        }

        m = m_new;
    }

    // Final normalization: out = acc / l
    float inv_l = 1.0f / l;
    device half4* out4 = (device half4*)(out + qHead * 64);
    for (int i = 0; i < 16; i++) {
        out4[i] = half4(acc[i] * inv_l);
    }
}

// Parallelized version: multiple threads cooperate on dot product via simd reduction
// Better utilization for longer sequences
kernel void sdpa_decode_f16_hd64_simd(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant int& kvHeadStride [[buffer(8)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // 32 threads per threadgroup (1 simdgroup)
    // Each thread handles 2 half4 chunks of the 64-dim vectors (2 × 4 = 8 dims)
    // Dot product reduced across simdgroup

    int qHead = gid;
    if (qHead >= numQHeads) return;

    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;
    int kvBase = kvHead * kvHeadStride;

    // Each thread loads 2 half4 chunks (8 elements) of Q
    // Thread 0: chunks 0,1; Thread 1: chunks 2,3; ... Thread 15: chunks 30,31
    // With 32 threads: threads 0-15 each handle 2 chunks, threads 16-31 also handle their chunks
    // Actually for 64 elements with half4: 16 chunks total
    // With 32 threads: each thread handles 16/32 = 0.5 chunks...
    // Better: each thread handles 2 elements, use simd to sum

    // Simpler: each of 32 threads handles 2 float values from Q (64/32 = 2)
    int myDim = tid * 2;  // Each thread owns 2 dimensions

    device const half* Qh = Q + qHead * 64;
    float q0 = float(Qh[myDim]);
    float q1 = float(Qh[myDim + 1]);

    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    // Shared memory for max/sum reduction
    threadgroup float* warpMax = shared;
    threadgroup float* warpSum = shared + 8;

    for (int pos = 0; pos < kvLen; pos++) {
        device const half* Kh = K + kvBase + pos * 64;

        // Each thread computes partial dot for its 2 dimensions
        float partial = q0 * float(Kh[myDim]) + q1 * float(Kh[myDim + 1]);

        // Reduce across simdgroup to get full dot product
        float score = simd_sum(partial) * scale;

        // Online softmax (all threads have same score after reduction)
        float m_new = max(m, score);
        float exp_diff = exp(m - m_new);
        float exp_score = exp(score - m_new);

        l = l * exp_diff + exp_score;

        // Update accumulators
        device const half* Vh = V + kvBase + pos * 64;
        acc0 = acc0 * exp_diff + exp_score * float(Vh[myDim]);
        acc1 = acc1 * exp_diff + exp_score * float(Vh[myDim + 1]);

        m = m_new;
    }

    // Final normalization and write
    float inv_l = 1.0f / l;
    device half* outh = out + qHead * 64;
    outh[myDim] = half(acc0 * inv_l);
    outh[myDim + 1] = half(acc1 * inv_l);
}

// =============================================================================
// Flash Attention SDPA decode for FP16 KV cache
// Split-KV approach: simdgroups process independent KV chunks with online softmax,
// then merge partial results. Shared memory is O(headDim), not O(kvLen).
// Supports any headDim that is a multiple of 32 (64, 128, 256).
// =============================================================================
constant int FLASH_F16_DECODE_THREADS = 256;  // 8 simdgroups × 32 threads
constant int FLASH_F16_DECODE_NUM_SG = 8;

kernel void sdpa_flash_decode_f16(
    device const half* Q [[buffer(0)]],         // [numQHeads, headDim] in FP16
    device const half* K [[buffer(1)]],         // [numKVHeads, maxSeqLen, headDim] in FP16
    device const half* V [[buffer(2)]],         // [numKVHeads, maxSeqLen, headDim] in FP16
    device half* out [[buffer(3)]],             // [numQHeads, headDim] in FP16
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant int& kvHeadStride [[buffer(9)]],   // Stride between KV heads (maxSeqLen * headDim)
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
    int kvBase = kvHead * kvHeadStride;

    // Each thread in a simdgroup handles headDim/32 consecutive elements.
    // For headDim=128: 4 elements/thread, headDim=64: 2, headDim=256: 8.
    int elemsPerThread = headDim / 32;
    int threadBase = simd_lane * elemsPerThread;

    // Load Q into registers (each thread loads its own portion)
    float q_reg[8];  // supports up to headDim=256 (256/32=8)
    for (int e = 0; e < elemsPerThread; e++) {
        q_reg[e] = float(Q[qHead * headDim + threadBase + e]);
    }

    // Per-simdgroup online softmax state
    float m_i = -INFINITY;    // running max
    float l_i = 0.0f;         // running sum of exp(score - m_i)
    float o_i[8];              // weighted V accumulator
    for (int e = 0; e < 8; e++) o_i[e] = 0.0f;

    // Split KV positions across simdgroups for parallel processing
    int chunkSize = (kvLen + FLASH_F16_DECODE_NUM_SG - 1) / FLASH_F16_DECODE_NUM_SG;
    int startPos = simd_group * chunkSize;
    int endPos = min(startPos + chunkSize, kvLen);

    // Process this simdgroup's KV chunk with online softmax
    for (int pos = startPos; pos < endPos; pos++) {
        int kBase = kvBase + pos * headDim;

        // Q·K dot product: each thread computes partial, simd_sum reduces
        float partial = 0.0f;
        for (int e = 0; e < elemsPerThread; e++) {
            partial += q_reg[e] * float(K[kBase + threadBase + e]);
        }
        float score = simd_sum(partial) * scale;
        // score is broadcast to ALL lanes by simd_sum

        // Online softmax update
        float m_new = max(m_i, score);
        float alpha = exp(m_i - m_new);   // rescale factor for old accumulator
        float p = exp(score - m_new);      // attention weight for this position

        l_i = l_i * alpha + p;

        // Rescale old accumulator and add weighted V
        int vBase = kvBase + pos * headDim;
        for (int e = 0; e < elemsPerThread; e++) {
            o_i[e] = o_i[e] * alpha + p * float(V[vBase + threadBase + e]);
        }

        m_i = m_new;
    }

    // ---- Cross-simdgroup merge ----
    // Shared memory layout:
    //   [0 .. NUM_SG-1]:                          per-SG max values
    //   [NUM_SG .. 2*NUM_SG-1]:                   per-SG sum values
    //   [2*NUM_SG .. 2*NUM_SG + NUM_SG*headDim-1]: per-SG output accumulators
    threadgroup float* sg_maxs = shared;
    threadgroup float* sg_sums = shared + FLASH_F16_DECODE_NUM_SG;
    threadgroup float* sg_accs = shared + 2 * FLASH_F16_DECODE_NUM_SG;

    // Write this simdgroup's results to shared memory
    if (simd_lane == 0) {
        sg_maxs[simd_group] = m_i;
        sg_sums[simd_group] = l_i;
    }
    for (int e = 0; e < elemsPerThread; e++) {
        sg_accs[simd_group * headDim + threadBase + e] = o_i[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup 0 merges all partial results and writes final output
    if (simd_group == 0) {
        // Find global max across all simdgroups
        float global_max = sg_maxs[0];
        for (int s = 1; s < FLASH_F16_DECODE_NUM_SG; s++) {
            global_max = max(global_max, sg_maxs[s]);
        }

        // Merge partial results with online softmax correction
        float total_sum = 0.0f;
        float result[8];
        for (int e = 0; e < 8; e++) result[e] = 0.0f;

        for (int s = 0; s < FLASH_F16_DECODE_NUM_SG; s++) {
            // If this simdgroup processed no positions, skip it
            float w = (sg_sums[s] > 0.0f) ? exp(sg_maxs[s] - global_max) : 0.0f;
            total_sum += sg_sums[s] * w;
            for (int e = 0; e < elemsPerThread; e++) {
                result[e] += sg_accs[s * headDim + threadBase + e] * w;
            }
        }

        // Normalize and write output
        float inv_l = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
        int outBase = qHead * headDim;
        for (int e = 0; e < elemsPerThread; e++) {
            out[outBase + threadBase + e] = half(result[e] * inv_l);
        }
    }
}

// Q4_0 Matrix-vector with FP16 input/output
// A: [1, K] in FP16, B: [N, K] in Q4_0 format, C: [1, N] in FP16
// Uses FP32 accumulation for precision, 2x bandwidth savings on activations
kernel void matvec_q4_0_f16(
    device const half* A [[buffer(0)]],            // [1, K] activations in FP16
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_0 format
    device half* C [[buffer(2)]],                  // [1, N] output in FP16
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

    float sum = 0.0f;  // Accumulate in FP32

    // Q4_0 row layout
    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q4_BYTES_PER_BLOCK;

    // Each thread in simdgroup handles blocks with stride 32
    for (int block = simd_lane; block < numBlocks; block += 32) {
        device const uchar* blockPtr = b_row + block * Q4_BYTES_PER_BLOCK;

        // Read f16 scale
        ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
        float scale = q4_f16_to_f32(scale_u16);

        int base_k = block * Q4_BLOCK_SIZE;

        if (base_k + 32 <= K) {
            // Full block - read FP16 and convert to FP32 for computation
            // Using half4 reads for 2x bandwidth improvement
            device const half4* a_vec = (device const half4*)(A + base_k);
            float4 a0 = float4(a_vec[0]), a1 = float4(a_vec[1]);
            float4 a2 = float4(a_vec[2]), a3 = float4(a_vec[3]);

            device const half4* a_vec_hi = (device const half4*)(A + base_k + 16);
            float4 a4 = float4(a_vec_hi[0]), a5 = float4(a_vec_hi[1]);
            float4 a6 = float4(a_vec_hi[2]), a7 = float4(a_vec_hi[3]);

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
                sum += float(A[k0]) * scale * float((byte_val & 0x0F) - 8);
                int k1 = base_k + i + 16;
                if (k1 < K) {
                    sum += float(A[k1]) * scale * float(((byte_val >> 4) & 0x0F) - 8);
                }
            }
        }
    }

    // Simdgroup reduction
    sum = simd_sum(sum);

    // Lane 0 of each simdgroup writes its output in FP16
    if (simd_lane == 0) {
        C[output_idx] = half(sum);
    }
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
    constant int& kvHeadStride [[buffer(9)]],  // Stride between KV heads (maxSeqLen * headDim)
    uint gid [[thread_position_in_grid]]
) {
    int qHead = gid;
    if (qHead >= numQHeads) return;

    // GQA: map Q head to KV head
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;

    int qOffset = qHead * headDim;
    int outOffset = qHead * headDim;

    // KV cache layout: [numKVHeads, maxSeqLen, headDim]
    // For head h, position p: offset = h * kvHeadStride + p * headDim
    int kvHeadBase = kvHead * kvHeadStride;

    // First pass: find max score for numerical stability
    float maxScore = -INFINITY;
    for (int pos = 0; pos < kvLen; pos++) {
        int kOffset = kvHeadBase + pos * headDim;
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
        int kOffset = kvHeadBase + pos * headDim;
        int vOffset = kvHeadBase + pos * headDim;

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

// SDPA decode with logit soft-capping (Gemma 2).
// Applies cap * tanh(score / cap) before softmax.
// softcap: soft-cap value (typically 30.0 for Gemma 2).
kernel void sdpa_softcap_decode_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant int& kvHeadStride [[buffer(9)]],
    constant float& softcap [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    int qHead = gid;
    if (qHead >= numQHeads) return;

    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;

    int qOffset = qHead * headDim;
    int outOffset = qHead * headDim;
    int kvHeadBase = kvHead * kvHeadStride;

    // First pass: find max score (with soft-capping)
    float maxScore = -INFINITY;
    for (int pos = 0; pos < kvLen; pos++) {
        int kOffset = kvHeadBase + pos * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        float score = dot * scale;
        // Apply soft-capping: cap * tanh(score / cap)
        if (softcap > 0.0f) {
            score = softcap * tanh(score / softcap);
        }
        maxScore = max(maxScore, score);
    }

    // Initialize output to zero
    for (int d = 0; d < headDim; d++) {
        out[outOffset + d] = 0.0f;
    }

    // Second pass: compute softmax and weighted sum
    float sumExp = 0.0f;
    for (int pos = 0; pos < kvLen; pos++) {
        int kOffset = kvHeadBase + pos * headDim;
        int vOffset = kvHeadBase + pos * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * K[kOffset + d];
        }
        float score = dot * scale;
        if (softcap > 0.0f) {
            score = softcap * tanh(score / softcap);
        }
        float weight = exp(score - maxScore);
        sumExp += weight;

        for (int d = 0; d < headDim; d++) {
            out[outOffset + d] += weight * V[vOffset + d];
        }
    }

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
    constant int& kvHeadStride [[buffer(9)]],  // Stride between KV heads (maxSeqLen * headDim)
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

    // KV cache layout: [numKVHeads, maxSeqLen, headDim]
    // For head h, position p: offset = h * kvHeadStride + p * headDim
    int kvHeadBase = kvHead * kvHeadStride;

    // Shared memory layout:
    // [0..kvLen-1]: attention weights (after softmax)
    // [kvLen..kvLen+7]: warp max/sum values
    threadgroup float* weights = shared;
    threadgroup float* warpVals = shared + kvLen;

    // Phase 1a: Each thread computes scores for its KV positions
    float localMax = -INFINITY;
    for (int pos = tid; pos < kvLen; pos += FLASH_DECODE_THREADS) {
        int kOffset = kvHeadBase + pos * headDim;
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
            int vOffset = kvHeadBase + pos * headDim;
            sum += weights[pos] * V[vOffset + d];
        }
        out[outOffset + d] = sum;
    }
}

// Paged SDPA decode: reads K/V from a paged block pool via block table indirection.
// Same two-phase algorithm as sdpa_flash_decode_f32, but K/V are in paged blocks
// instead of contiguous buffers.
//
// Block pool layout per physical block:
//   K: [blockSize, numKVHeads, headDim] float32
//   V: [blockSize, numKVHeads, headDim] float32
//   Total: 2 * blockSize * numKVHeads * headDim elements
//
// blockTable: [numBlocks] int32 mapping logical block index → physical block index
constant int PAGED_DECODE_THREADS = 256;

kernel void sdpa_paged_decode_f32(
    device const float* Q [[buffer(0)]],
    device const float* kvPool [[buffer(1)]],
    device const int* blockTable [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& numBlocks [[buffer(4)]],
    constant int& blockSize [[buffer(5)]],
    constant int& numQHeads [[buffer(6)]],
    constant int& numKVHeads [[buffer(7)]],
    constant int& headDim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant int& tokensInLastBlock [[buffer(10)]],
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

    // Block pool geometry
    int elementsPerBlockPart = blockSize * numKVHeads * headDim;
    int elementsPerBlock = 2 * elementsPerBlockPart;

    // Total KV sequence length
    int kvLen = (numBlocks - 1) * blockSize + tokensInLastBlock;

    // Shared memory: weights[kvLen] + warpVals[8]
    threadgroup float* weights = shared;
    threadgroup float* warpVals = shared + kvLen;

    // Phase 1a: Compute attention scores Q·K for all KV positions
    float localMax = -INFINITY;
    for (int pos = (int)tid; pos < kvLen; pos += PAGED_DECODE_THREADS) {
        int logicalBlock = pos / blockSize;
        int tokenInBlock = pos % blockSize;
        int physicalBlock = blockTable[logicalBlock];

        // K offset: physBlock * elementsPerBlock + token * numKVHeads * headDim + kvHead * headDim
        long kBase = (long)physicalBlock * elementsPerBlock +
                     tokenInBlock * numKVHeads * headDim +
                     kvHead * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qOffset + d] * kvPool[kBase + d];
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

    // Phase 1b: Compute exp(score - max) and sum
    float localSum = 0.0f;
    for (int pos = (int)tid; pos < kvLen; pos += PAGED_DECODE_THREADS) {
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

    // Normalize weights
    for (int pos = (int)tid; pos < kvLen; pos += PAGED_DECODE_THREADS) {
        weights[pos] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Accumulate weighted V values
    int outOffset = qHead * headDim;
    for (int d = (int)tid; d < headDim; d += PAGED_DECODE_THREADS) {
        float sum = 0.0f;
        for (int pos = 0; pos < kvLen; pos++) {
            int logicalBlock = pos / blockSize;
            int tokenInBlock = pos % blockSize;
            int physicalBlock = blockTable[logicalBlock];

            // V offset: physBlock * elementsPerBlock + elementsPerBlockPart + token * numKVHeads * headDim + kvHead * headDim
            long vBase = (long)physicalBlock * elementsPerBlock +
                         elementsPerBlockPart +
                         tokenInBlock * numKVHeads * headDim +
                         kvHead * headDim;

            sum += weights[pos] * kvPool[vBase + d];
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

// SDPA prefill with logit soft-capping (Gemma 2).
// Same tiled algorithm as sdpa_prefill_f32, with soft-cap applied to scores.
kernel void sdpa_softcap_prefill_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& seqLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int qPos = gid.x;
    int qHead = gid.y;

    if (qPos >= seqLen || qHead >= numQHeads) return;

    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;

    int qOffset = qPos * numQHeads * headDim + qHead * headDim;
    int maxKLen = qPos + 1;

    threadgroup float* scores = shared;
    threadgroup float* warpScratch = shared + PREFILL_TILE_K;

    float runningMax = -INFINITY;
    float runningSum = 0.0f;
    float acc[4] = {0.0f};

    for (int tileStart = 0; tileStart < maxKLen; tileStart += PREFILL_TILE_K) {
        int tileEnd = min(tileStart + PREFILL_TILE_K, maxKLen);
        int tileSize = tileEnd - tileStart;

        float localScore = -INFINITY;
        if ((int)tid < tileSize) {
            int kPos = tileStart + tid;
            int kOffset = kPos * numKVHeads * headDim + kvHead * headDim;

            float dot = 0.0f;
            for (int d = 0; d < headDim; d++) {
                dot += Q[qOffset + d] * K[kOffset + d];
            }
            localScore = dot * scale;
            // Apply soft-capping
            if (softcap > 0.0f) {
                localScore = softcap * tanh(localScore / softcap);
            }
        }

        if ((int)tid < tileSize) {
            scores[tid] = localScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find tile max
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

        // Online softmax update
        float newMax = max(runningMax, tileMax);
        float rescale = exp(runningMax - newMax);

        for (int i = 0; i < 4; i++) {
            acc[i] *= rescale;
        }
        runningSum *= rescale;

        float tileSum = 0.0f;
        for (int i = tid; i < tileSize; i += PREFILL_THREADS) {
            float expScore = exp(scores[i] - newMax);
            scores[i] = expScore;
            tileSum += expScore;
        }

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

        // Accumulate weighted V
        for (int kIdx = 0; kIdx < tileSize; kIdx++) {
            float weight = scores[kIdx];
            int kPos = tileStart + kIdx;
            int vOffset = kPos * numKVHeads * headDim + kvHead * headDim;

            for (int i = 0; i < 4 && (int)(tid + i * PREFILL_THREADS) < headDim; i++) {
                int d = tid + i * PREFILL_THREADS;
                acc[i] += weight * V[vOffset + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int outOffset = qPos * numQHeads * headDim + qHead * headDim;
    float invSum = 1.0f / runningSum;

    for (int i = 0; i < 4 && (int)(tid + i * PREFILL_THREADS) < headDim; i++) {
        int d = tid + i * PREFILL_THREADS;
        out[outOffset + d] = acc[i] * invSum;
    }
}

// Flash Attention 2 optimized kernel for prefill
// Key optimizations:
// 1. K/V tiles loaded to shared memory, shared across Q heads (GQA)
// 2. Larger tile size (64 K positions vs 16)
// 3. 8 Q heads computed in parallel (one simdgroup each)
// 4. Two-pass per tile: first find max, then compute exp and accumulate V
//    (avoids storing 64 scores per thread - reduces register pressure)
//
// Threadgroup layout:
// - 8 simdgroups (256 threads total)
// - Each simdgroup handles one Q head (8 Q heads share one KV head in GQA)
// - Threads within simdgroup handle different dimensions of headDim
//
// Q: [seqLen, numQHeads, headDim]
// K: [seqLen, numKVHeads, headDim]
// V: [seqLen, numKVHeads, headDim]
// out: [seqLen, numQHeads, headDim]

constant int FA2_TILE_KV = 64;      // K/V positions per tile
constant int FA2_SIMDGROUPS = 8;    // Simdgroups per threadgroup (matches GQA ratio)
constant int FA2_THREADS = FA2_SIMDGROUPS * 32;  // 256 threads total

// Flash Attention 2 - FP32 version
// FIXED: Now handles headDim > 64 by iterating over dimension blocks
kernel void flash_attention_2_f32(
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
    int qPos = gid.x;      // Query position this threadgroup handles
    int kvHead = gid.y;    // KV head this threadgroup handles

    if (qPos >= seqLen || kvHead >= numKVHeads) return;

    // GQA: compute which Q heads map to this KV head
    int headsPerKV = numQHeads / numKVHeads;
    int qHeadBase = kvHead * headsPerKV;

    // This simdgroup handles one Q head
    int qHeadLocal = simd_group;  // 0-7 for 8 Q heads per KV head
    bool active = (qHeadLocal < headsPerKV);
    int qHead = qHeadBase + qHeadLocal;

    // Shared memory layout:
    // [0..FA2_TILE_KV*headDim-1]: K tile
    // [FA2_TILE_KV*headDim..2*FA2_TILE_KV*headDim-1]: V tile
    threadgroup float* Ktile = shared;
    threadgroup float* Vtile = shared + FA2_TILE_KV * headDim;

    // Q offset for this (qPos, qHead)
    int qOffset = qPos * numQHeads * headDim + qHead * headDim;

    // Each thread handles 2 dimensions per 64-dimension block
    // For headDim=64: 1 iteration, for headDim=128: 2 iterations, etc.
    // Maximum supported: 256 dimensions (4 blocks of 64)
    int numDimBlocks = (headDim + 63) / 64;

    // Load Q values for all dimension blocks this thread handles
    float q_vals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Max 4 blocks * 2 dims
    if (active) {
        for (int block = 0; block < numDimBlocks; block++) {
            int d0 = block * 64 + simd_lane * 2;
            int d1 = d0 + 1;
            q_vals[block * 2] = (d0 < headDim) ? Q[qOffset + d0] : 0.0f;
            q_vals[block * 2 + 1] = (d1 < headDim) ? Q[qOffset + d1] : 0.0f;
        }
    }

    // Causal attention: only attend to positions <= qPos
    int maxKLen = qPos + 1;

    // Online softmax state (per Q head)
    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    // Accumulators for output dimensions
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Process K/V in tiles
    // Process K/V in tiles
    for (int tileStart = 0; tileStart < maxKLen; tileStart += FA2_TILE_KV) {
        int tileEnd = min(tileStart + FA2_TILE_KV, maxKLen);
        int tileSize = tileEnd - tileStart;

        // Cooperative load: all threads load K and V tiles to shared memory
        int loadCount = (FA2_TILE_KV * headDim + FA2_THREADS - 1) / FA2_THREADS;
        for (int i = 0; i < loadCount; i++) {
            int idx = tid + i * FA2_THREADS;
            if (idx < FA2_TILE_KV * headDim) {
                int kPos = tileStart + idx / headDim;
                int d = idx % headDim;
                if (kPos < tileEnd) {
                    int kOffset = kPos * numKVHeads * headDim + kvHead * headDim + d;
                    Ktile[idx] = K[kOffset];
                    Vtile[idx] = V[kOffset];
                } else {
                    Ktile[idx] = 0.0f;
                    Vtile[idx] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 1: Compute Q·K scores and find tile max
        float tileMax = -INFINITY;

        for (int k = 0; k < tileSize; k++) {
            // Compute Q·K dot product across all dimension blocks
            float dot = 0.0f;
            for (int block = 0; block < numDimBlocks; block++) {
                int d0 = block * 64 + simd_lane * 2;
                int d1 = d0 + 1;
                if (d0 < headDim) {
                    dot += q_vals[block * 2] * Ktile[k * headDim + d0];
                }
                if (d1 < headDim) {
                    dot += q_vals[block * 2 + 1] * Ktile[k * headDim + d1];
                }
            }
            dot = simd_sum(dot);
            float score = dot * scale;
            tileMax = max(tileMax, score);
        }

        // Online softmax update
        float newMax = max(runningMax, tileMax);
        float rescale = exp(runningMax - newMax);

        // Rescale existing accumulator
        for (int i = 0; i < numDimBlocks * 2; i++) {
            acc[i] *= rescale;
        }
        runningSum *= rescale;

        // Pass 2: Recompute scores, compute exp, accumulate V
        float tileSum = 0.0f;
        for (int k = 0; k < tileSize; k++) {
            // Recompute Q·K dot product
            float dot = 0.0f;
            for (int block = 0; block < numDimBlocks; block++) {
                int d0 = block * 64 + simd_lane * 2;
                int d1 = d0 + 1;
                if (d0 < headDim) {
                    dot += q_vals[block * 2] * Ktile[k * headDim + d0];
                }
                if (d1 < headDim) {
                    dot += q_vals[block * 2 + 1] * Ktile[k * headDim + d1];
                }
            }
            dot = simd_sum(dot);
            float score = dot * scale;

            float expScore = exp(score - newMax);
            tileSum += expScore;

            // Accumulate weighted V for all dimension blocks
            for (int block = 0; block < numDimBlocks; block++) {
                int d0 = block * 64 + simd_lane * 2;
                int d1 = d0 + 1;
                if (d0 < headDim) {
                    acc[block * 2] += expScore * Vtile[k * headDim + d0];
                }
                if (d1 < headDim) {
                    acc[block * 2 + 1] += expScore * Vtile[k * headDim + d1];
                }
            }
        }

        runningSum += tileSum;
        runningMax = newMax;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    int outOffset = qPos * numQHeads * headDim + qHead * headDim;
    float invSum = 1.0f / runningSum;

    if (active) {
        for (int block = 0; block < numDimBlocks; block++) {
            int d0 = block * 64 + simd_lane * 2;
            int d1 = d0 + 1;
            if (d0 < headDim) out[outOffset + d0] = acc[block * 2] * invSum;
            if (d1 < headDim) out[outOffset + d1] = acc[block * 2 + 1] * invSum;
        }
    }
}

// ============================================================================
// Flash Attention 2 v2 — Optimized FP32 prefill kernel
// Fixes 3 critical issues in the original FA2:
//   1. Single-pass online softmax (no Q·K recomputation)
//   2. Tiles Q positions across simdgroups (fixes GQA utilization waste)
//   3. Strided dimension mapping for bank-conflict-free shared memory access
//
// Grid: (ceil(seqLen / tileQ), numKVHeads)
//   tileQ = max(1, 8 / headsPerKV): 8 for headsPerKV=1, 2 for headsPerKV=4
// Each threadgroup handles tileQ Q positions × headsPerKV Q heads = 8 tasks
// All 8 simdgroups are fully utilized.
//
// Shared memory: K tile [TILE_KV, headDim] + V tile [TILE_KV, headDim] in FP32
//   TILE_KV=32, headDim=128 → 2×32×128×4 = 32KB (fits in threadgroup memory)
// ============================================================================
constant int FA2V2_TILE_KV = 32;
constant int FA2V2_NUM_SG = 8;
constant int FA2V2_THREADS = FA2V2_NUM_SG * 32;

kernel void flash_attention_2_v2_f32(
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
    int kvHead = gid.y;
    int headsPerKV = numQHeads / numKVHeads;

    // Tile Q: distribute Q positions across simdgroups
    // For headsPerKV=1: tileQ=8 (each SG = different Q position, same head)
    // For headsPerKV=4: tileQ=2 (each SG = one of 4 heads × 2 positions)
    // For headsPerKV=8: tileQ=1 (each SG = one of 8 heads, same position)
    int tileQ = FA2V2_NUM_SG / headsPerKV;
    if (tileQ < 1) tileQ = 1;
    if (tileQ > FA2V2_NUM_SG) tileQ = FA2V2_NUM_SG;

    int qHeadLocal = (int)simd_group / tileQ;
    int qPosLocal = (int)simd_group % tileQ;

    int qHead = kvHead * headsPerKV + qHeadLocal;
    int qPos = (int)gid.x * tileQ + qPosLocal;

    bool active = (qPos < seqLen) && (qHeadLocal < headsPerKV);

    // Shared memory: K tile then V tile
    threadgroup float* Ktile = shared;
    threadgroup float* Vtile = shared + FA2V2_TILE_KV * headDim;

    // Strided dimension mapping: thread l handles dims l, l+32, l+64, l+96
    // This gives coalesced device reads AND bank-conflict-free shared memory access
    int dimsPerThread = (headDim + 31) / 32;

    // Load Q into registers (reused across all KV tiles)
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active) {
        int qOffset = qPos * numQHeads * headDim + qHead * headDim;
        for (int i = 0; i < dimsPerThread; i++) {
            int d = (int)simd_lane + i * 32;
            if (d < headDim) q_reg[i] = Q[qOffset + d];
        }
    }

    int maxKLen = active ? (qPos + 1) : 0;

    // Online softmax state
    float runningMax = -INFINITY;
    float runningSum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Process KV in tiles
    for (int tileStart = 0; tileStart < seqLen; tileStart += FA2V2_TILE_KV) {
        // Early exit: if all Q positions in this TG are past causal boundary
        int maxQInTG = min((int)gid.x * tileQ + tileQ - 1, seqLen - 1);
        if (tileStart > maxQInTG) break;

        int tileEnd = min(tileStart + FA2V2_TILE_KV, seqLen);
        int tileSize = tileEnd - tileStart;

        // Cooperative K/V tile load (all 256 threads participate)
        int totalElems = FA2V2_TILE_KV * headDim;
        for (int idx = (int)tid; idx < totalElems; idx += FA2V2_THREADS) {
            int kPos = tileStart + idx / headDim;
            int d = idx % headDim;
            if (kPos < tileEnd) {
                int kvOffset = kPos * numKVHeads * headDim + kvHead * headDim + d;
                Ktile[idx] = K[kvOffset];
                Vtile[idx] = V[kvOffset];
            } else {
                Ktile[idx] = 0.0f;
                Vtile[idx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            // Single pass: compute scores + find tile max
            float tileMax = -INFINITY;
            float scores[32]; // FA2V2_TILE_KV = 32

            for (int k = 0; k < tileSize; k++) {
                int kPos = tileStart + k;
                if (kPos > qPos) {
                    scores[k] = -INFINITY;
                    continue;
                }
                // Q·K dot product (strided dims)
                float dot = 0.0f;
                for (int i = 0; i < dimsPerThread; i++) {
                    int d = (int)simd_lane + i * 32;
                    if (d < headDim) {
                        dot += q_reg[i] * Ktile[k * headDim + d];
                    }
                }
                float score = simd_sum(dot) * scale;
                scores[k] = score;
                tileMax = max(tileMax, score);
            }

            // Online softmax: rescale existing accumulators
            float newMax = max(runningMax, tileMax);
            float rescale = (runningMax > -INFINITY) ? exp(runningMax - newMax) : 0.0f;

            for (int i = 0; i < dimsPerThread; i++) {
                acc[i] *= rescale;
            }
            runningSum *= rescale;

            // Accumulate exp(score) * V
            float tileSum = 0.0f;
            for (int k = 0; k < tileSize; k++) {
                int kPos = tileStart + k;
                if (kPos > qPos) continue;

                float expScore = exp(scores[k] - newMax);
                tileSum += expScore;

                for (int i = 0; i < dimsPerThread; i++) {
                    int d = (int)simd_lane + i * 32;
                    if (d < headDim) {
                        acc[i] += expScore * Vtile[k * headDim + d];
                    }
                }
            }

            runningSum += tileSum;
            runningMax = newMax;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output: O = acc / runningSum
    if (active && runningSum > 0.0f) {
        int outOffset = qPos * numQHeads * headDim + qHead * headDim;
        float invSum = 1.0f / runningSum;
        for (int i = 0; i < dimsPerThread; i++) {
            int d = (int)simd_lane + i * 32;
            if (d < headDim) {
                out[outOffset + d] = acc[i] * invSum;
            }
        }
    }
}

// Flash Attention 2 with FP16 activations
// Q, K, V, out are all FP16
// Shared memory K/V tiles are also FP16 (halves memory usage)
// FIXED: Now handles headDim > 64 by iterating over dimension blocks
kernel void flash_attention_2_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant int& seqLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    threadgroup half* shared [[threadgroup(0)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int qPos = gid.x;
    int kvHead = gid.y;

    if (qPos >= seqLen || kvHead >= numKVHeads) return;

    int headsPerKV = numQHeads / numKVHeads;
    int qHeadBase = kvHead * headsPerKV;

    int qHeadLocal = simd_group;
    bool active = (qHeadLocal < headsPerKV);
    int qHead = qHeadBase + qHeadLocal;

    // Shared memory in half precision
    threadgroup half* Ktile = shared;
    threadgroup half* Vtile = shared + FA2_TILE_KV * headDim;

    int qOffset = qPos * numQHeads * headDim + qHead * headDim;

    // Each thread handles 2 dimensions per 64-dimension block
    // For headDim=64: 1 iteration, for headDim=128: 2 iterations, etc.
    // Maximum supported: 256 dimensions (4 blocks of 64)
    int numDimBlocks = (headDim + 63) / 64;

    // Load Q values for all dimension blocks this thread handles
    // Each thread handles dims: lane*2, lane*2+1, 64+lane*2, 64+lane*2+1, ...
    float q_vals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Max 4 blocks * 2 dims
    if (active) {
        for (int block = 0; block < numDimBlocks; block++) {
            int d0 = block * 64 + simd_lane * 2;
            int d1 = d0 + 1;
            q_vals[block * 2] = (d0 < headDim) ? float(Q[qOffset + d0]) : 0.0f;
            q_vals[block * 2 + 1] = (d1 < headDim) ? float(Q[qOffset + d1]) : 0.0f;
        }
    }

    int maxKLen = qPos + 1;

    float runningMax = -INFINITY;
    float runningSum = 0.0f;
    // Accumulators for output dimensions
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int tileStart = 0; tileStart < maxKLen; tileStart += FA2_TILE_KV) {
        int tileEnd = min(tileStart + FA2_TILE_KV, maxKLen);
        int tileSize = tileEnd - tileStart;

        int loadCount = (FA2_TILE_KV * headDim + FA2_THREADS - 1) / FA2_THREADS;
        for (int i = 0; i < loadCount; i++) {
            int idx = tid + i * FA2_THREADS;
            if (idx < FA2_TILE_KV * headDim) {
                int kPos = tileStart + idx / headDim;
                int d = idx % headDim;
                if (kPos < tileEnd) {
                    int kOffset = kPos * numKVHeads * headDim + kvHead * headDim + d;
                    Ktile[idx] = K[kOffset];
                    Vtile[idx] = V[kOffset];
                } else {
                    Ktile[idx] = 0.0h;
                    Vtile[idx] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute tile max - sum dot products across all dimension blocks
        float tileMax = -INFINITY;
        for (int k = 0; k < tileSize; k++) {
            float dot = 0.0f;
            for (int block = 0; block < numDimBlocks; block++) {
                int d0 = block * 64 + simd_lane * 2;
                int d1 = d0 + 1;
                if (d0 < headDim) {
                    dot += q_vals[block * 2] * float(Ktile[k * headDim + d0]);
                }
                if (d1 < headDim) {
                    dot += q_vals[block * 2 + 1] * float(Ktile[k * headDim + d1]);
                }
            }
            dot = simd_sum(dot);
            float score = dot * scale;
            tileMax = max(tileMax, score);
        }

        float newMax = max(runningMax, tileMax);
        float rescale = exp(runningMax - newMax);
        for (int i = 0; i < numDimBlocks * 2; i++) {
            acc[i] *= rescale;
        }
        runningSum *= rescale;

        // Compute softmax-weighted V accumulation
        float tileSum = 0.0f;
        for (int k = 0; k < tileSize; k++) {
            float dot = 0.0f;
            for (int block = 0; block < numDimBlocks; block++) {
                int d0 = block * 64 + simd_lane * 2;
                int d1 = d0 + 1;
                if (d0 < headDim) {
                    dot += q_vals[block * 2] * float(Ktile[k * headDim + d0]);
                }
                if (d1 < headDim) {
                    dot += q_vals[block * 2 + 1] * float(Ktile[k * headDim + d1]);
                }
            }
            dot = simd_sum(dot);
            float score = dot * scale;
            float expScore = exp(score - newMax);
            tileSum += expScore;

            // Accumulate weighted V for all dimension blocks
            for (int block = 0; block < numDimBlocks; block++) {
                int d0 = block * 64 + simd_lane * 2;
                int d1 = d0 + 1;
                if (d0 < headDim) {
                    acc[block * 2] += expScore * float(Vtile[k * headDim + d0]);
                }
                if (d1 < headDim) {
                    acc[block * 2 + 1] += expScore * float(Vtile[k * headDim + d1]);
                }
            }
        }

        runningSum += tileSum;
        runningMax = newMax;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int outOffset = qPos * numQHeads * headDim + qHead * headDim;
    float invSum = 1.0f / runningSum;

    if (active) {
        for (int block = 0; block < numDimBlocks; block++) {
            int d0 = block * 64 + simd_lane * 2;
            int d1 = d0 + 1;
            if (d0 < headDim) out[outOffset + d0] = half(acc[block * 2] * invSum);
            if (d1 < headDim) out[outOffset + d1] = half(acc[block * 2 + 1] * invSum);
        }
    }
}

// =============================================================================
// Compute-based memory copy kernel (avoids blit encoder transition issues)
// =============================================================================
// Uses uchar4 (4-byte) copies for efficiency
kernel void memcpy_compute(
    device const uchar4* src [[buffer(0)]],
    device uchar4* dst [[buffer(1)]],
    constant int& numElements [[buffer(2)]],  // Number of uchar4 elements to copy
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(numElements)) {
        dst[gid] = src[gid];
    }
}

// =============================================================================
// Argmax kernel: find index of maximum value
// =============================================================================
// Single threadgroup reduction for vocab-sized arrays (32K elements)
// Returns the index of the maximum value in the input array
kernel void argmax_f32(
    device const float* input [[buffer(0)]],
    device int* result [[buffer(1)]],
    constant int& N [[buffer(2)]],
    threadgroup float* shared_vals [[threadgroup(0)]],
    threadgroup int* shared_idxs [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each thread finds its local max across strided elements
    float localMax = -INFINITY;
    int localIdx = 0;

    for (int i = tid; i < N; i += tg_size) {
        float val = input[i];
        if (val > localMax) {
            localMax = val;
            localIdx = i;
        }
    }

    // Store to shared memory
    shared_vals[tid] = localMax;
    shared_idxs[tid] = localIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes result
    if (tid == 0) {
        result[0] = shared_idxs[0];
    }
}
)";

// Device and queue management
// Note: Without ARC (-fno-objc-arc), objects from "new*" methods already have +1 retain count.
// Simple casts pass ownership to Go, which calls metal_release() -> CFRelease().
void* metal_create_device(void) {
    return (void*)MTLCreateSystemDefaultDevice();
}

void* metal_create_command_queue(void* device) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    return (void*)[dev newCommandQueue];
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
    return (void*)buffer;
}

size_t metal_buffer_size(void* buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return [buf length];
}

void* metal_buffer_contents(void* buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return [buf contents];
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
    // [commandBuffer waitUntilCompleted]; // Make async to avoid CPU blocking
}

// Global batch state (thread-local would be better for multi-threading)
static id<MTLCommandQueue> g_batchQueue = nil;
static id<MTLCommandBuffer> g_batchCmdBuffer = nil;
static id<MTLComputeCommandEncoder> g_batchEncoder = nil;
static int g_batchRefCount = 0;  // Nested batch reference count

// GPU profiling state
static uint64_t g_gpuTotalTime = 0;
static uint64_t g_gpuBatchCount = 0;
static uint64_t g_gpuKernelCount = 0;
static uint64_t g_gpuSyncTime = 0;
static mach_timebase_info_data_t g_timebaseInfo = {0, 0};

static inline uint64_t mach_to_ns(uint64_t mach_time_val) {
    if (g_timebaseInfo.denom == 0) {
        mach_timebase_info(&g_timebaseInfo);
    }
    return mach_time_val * g_timebaseInfo.numer / g_timebaseInfo.denom;
}

// Forward declaration for batch mode check
static inline bool is_batch_mode(void);

// GPU-to-GPU buffer copy that integrates with command batching.
// If in batch mode, ends compute encoder, does blit on same command buffer,
// then starts a new compute encoder - avoiding separate command buffer overhead.
void metal_copy_buffer_batched(void* queue, void* srcBuffer, size_t srcOffset,
                               void* dstBuffer, size_t dstOffset, size_t size) {
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcBuffer;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuffer;

    if (g_batchEncoder != nil) {
        // In batch mode: end compute encoder, do blit, start new compute encoder
        // All on the same command buffer - no sync needed
        [g_batchEncoder endEncoding];

        id<MTLBlitCommandEncoder> blit = [g_batchCmdBuffer blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:srcOffset toBuffer:dst destinationOffset:dstOffset size:size];
        // Ensure dst buffer is visible to subsequent compute encoder
        [blit synchronizeResource:dst];
        [blit endEncoding];

        // Resume compute encoding on the same command buffer
        g_batchEncoder = [g_batchCmdBuffer computeCommandEncoder];
    } else {
        // Not in batch mode: use standard copy
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLCommandBuffer> commandBuffer = [q commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:srcOffset toBuffer:dst destinationOffset:dstOffset size:size];
        [blit endEncoding];
        [commandBuffer commit];
    }
}

// Shader compilation
void* metal_compile_library(void* device, const char* source) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    NSError* error = nil;

    NSString* src = source ? [NSString stringWithUTF8String:source] : metalShaderSource;

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    // macOS 15+: mathMode replaces fastMathEnabled. Prefer fast mode when available.
    if ([options respondsToSelector:@selector(setMathMode:)]) {
        options.mathMode = MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }

    id<MTLLibrary> library = [dev newLibraryWithSource:src options:options error:&error];
    if (error) {
        NSLog(@"Metal compile error: %@", error);
        return nil;
    }
    return (void*)library;
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

    return (void*)pipeline;
}

// Synchronization
void metal_sync(void* commandQueue) {
    uint64_t start = mach_absolute_time();

    // If we're in batch mode, we need to flush the batch first
    if (is_batch_mode()) {
        // End the current batch (commit and wait)
        [g_batchEncoder endEncoding];
        [g_batchCmdBuffer commit];
        [g_batchCmdBuffer waitUntilCompleted];
        g_gpuBatchCount++;

        // Start a new batch for subsequent operations
        g_batchCmdBuffer = [g_batchQueue commandBuffer];
        g_batchEncoder = [g_batchCmdBuffer computeCommandEncoder];
    } else {
        // Not in batch mode — commit an empty buffer as a FIFO barrier.
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }

    g_gpuSyncTime += mach_to_ns(mach_absolute_time() - start);
}

// =============================================================================
// Command Buffer Batching
// =============================================================================
// Global batch state defined earlier in file (see forward declarations)

// Begin a batch of operations - all subsequent kernel dispatches use the same command buffer.
// Supports nesting: if already in a batch, increments ref count without creating a new CB.
// This enables cross-layer batching where the outer caller wraps multiple layers.
void metal_begin_batch(void* queuePtr) {
    if (g_batchEncoder != nil) {
        // Already in a batch - increment ref count for nesting
        g_batchRefCount++;
        return;
    }
    g_batchQueue = (__bridge id<MTLCommandQueue>)queuePtr;
    g_batchCmdBuffer = [g_batchQueue commandBuffer];
    g_batchEncoder = [g_batchCmdBuffer computeCommandEncoder];
    g_batchRefCount = 1;
}

// End the batch - only commits when ref count reaches 0 (outermost caller).
// Nested EndBatch calls just decrement the ref count.
void metal_end_batch(void) {
    if (g_batchEncoder == nil) {
        return;
    }
    g_batchRefCount--;
    if (g_batchRefCount > 0) {
        // Nested batch - don't commit yet, just insert a barrier for safety
        [g_batchEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return;
    }
    [g_batchEncoder endEncoding];
    [g_batchCmdBuffer commit];
    // Test mode: VEXEL_BATCH_WAIT=1 waits for each batch to complete
    // This serializes GPU work but helps identify sync issues
    static int waitMode = -1;
    if (waitMode == -1) {
        const char* env = getenv("VEXEL_BATCH_WAIT");
        waitMode = (env && strcmp(env, "1") == 0) ? 1 : 0;
    }
    if (waitMode) {
        [g_batchCmdBuffer waitUntilCompleted];
    }
    g_gpuBatchCount++;
    g_batchEncoder = nil;
    g_batchCmdBuffer = nil;
    g_batchQueue = nil;
}

// Insert a buffer-scope memory barrier on the current batch encoder.
// This ensures all buffer writes from preceding dispatches are visible to subsequent reads.
// Required when multiple dispatches share the same MTLBuffer (scratch allocator).
// No-op when not in batch mode (separate command buffers already serialize).
void metal_memory_barrier(void) {
    if (g_batchEncoder != nil) {
        [g_batchEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
}

// GPU Profiling functions
void metal_get_gpu_profile(uint64_t* totalTimeNs, uint64_t* batchCount, uint64_t* kernelCount, uint64_t* syncTimeNs) {
    if (totalTimeNs) *totalTimeNs = g_gpuTotalTime;
    if (batchCount) *batchCount = g_gpuBatchCount;
    if (kernelCount) *kernelCount = g_gpuKernelCount;
    if (syncTimeNs) *syncTimeNs = g_gpuSyncTime;
}

void metal_reset_gpu_profile(void) {
    g_gpuTotalTime = 0;
    g_gpuBatchCount = 0;
    g_gpuKernelCount = 0;
    g_gpuSyncTime = 0;
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

// Helper: finish encoding and possibly commit.
// Always waits for completion — required for correct memory visibility between
// separate command buffers on Apple Silicon. The proper way to avoid per-dispatch
// waiting is command buffer batching (metal_begin_batch/metal_end_batch), which
// encodes multiple operations into a single command buffer with one wait.
static inline void finish_encode(id<MTLComputeCommandEncoder> encoder, id<MTLCommandBuffer> cmdBuf, bool shouldCommit) {
    if (shouldCommit) {
        uint64_t start = mach_absolute_time();
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        g_gpuTotalTime += mach_to_ns(mach_absolute_time() - start);
        g_gpuBatchCount++;
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

void metal_matmul_f32(void* queuePtr, void* pipelinePtr,
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

    // Grid covers output matrix [N, M]
    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(N, M, 1));
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

// Q4_0 NR2 matvec: 16 outputs per threadgroup (2 per simdgroup)
// Grid: ceil(N/16) threadgroups of 256 threads
void metal_matvec_q4_0_nr2_f32(void* queuePtr, void* pipelinePtr,
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

    // 16 outputs per threadgroup (8 simdgroups * 2 outputs each)
    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 NR4 matvec: 32 outputs per threadgroup (4 per simdgroup)
// Grid: ceil(N/32) threadgroups of 256 threads
void metal_matvec_q4_0_nr4_f32(void* queuePtr, void* pipelinePtr,
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

    // 32 outputs per threadgroup (8 simdgroups * 4 outputs each)
    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 collaborative matvec (llama.cpp-style thread collaboration)
// 2 threads per block, stride 16, 32 outputs per threadgroup
void metal_matvec_q4_0_collab_f32(void* queuePtr, void* pipelinePtr,
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

    // 32 outputs per threadgroup (8 simdgroups * 4 outputs each)
    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 optimized matvec with shared memory: 32 outputs per threadgroup
// Grid: ceil(N/32) threadgroups of 256 threads
void metal_matvec_q4_0_optimized_f32(void* queuePtr, void* pipelinePtr,
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

    // Shared memory for activations (K floats, typically 2048*4 = 8KB)
    int sharedMemSize = K * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // 32 outputs per threadgroup (8 simdgroups * 4 outputs each)
    int outputsPerTG = 32;
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

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

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
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matmul_q4k_batched_f32(void* queuePtr, void* pipelinePtr,
                                   void* A, uint64_t aOff,
                                   void* B, uint64_t bOff,
                                   void* C, uint64_t cOff,
                                   int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];

    // NR2 pattern: 16 outputs per threadgroup (8 simdgroups × 2)
    // 2D grid: (N tiles, M) — each TG handles one M row, 16 N outputs
    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int nTiles = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(nTiles, M, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_K simdgroup tiled matmul: uses simdgroup_matrix for prefill (M>=8)
// Same tile layout as Q4_0 simdgroup: TILE_M=32, TILE_N=64, TILE_K=64
// Half-precision shared memory, block-based sub-block dequant.
// Grid: (ceil(N/64), ceil(M/32)) threadgroups of 256 threads
void metal_matmul_q4k_simdgroup_f32(void* queuePtr, void* pipelinePtr,
                                     void* A, uint64_t aOff,
                                     void* B, uint64_t bOff,
                                     void* C, uint64_t cOff,
                                     int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int TILE_M = 32;
    int TILE_N = 64;
    int TILE_K = 64;
    int threadgroupSize = 256;

    // Shared memory: A tile + B tile (both half-precision)
    int sharedMemA = TILE_M * TILE_K * sizeof(short);  // 32×64×2 = 4096 bytes
    int sharedMemB = TILE_N * TILE_K * sizeof(short);  // 64×64×2 = 8192 bytes

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemA atIndex:0];
    [encoder setThreadgroupMemoryLength:sharedMemB atIndex:1];

    // 2D grid: tiles in (N, M) dimensions
    int tilesN = (N + TILE_N - 1) / TILE_N;
    int tilesM = (M + TILE_M - 1) / TILE_M;
    MTLSize threadgroups = MTLSizeMake(tilesN, tilesM, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q8_0_nr2_f32(void* queuePtr, void* pipelinePtr,
                                void* A, uint64_t aOff,
                                void* B, uint64_t bOff,
                                void* C, uint64_t cOff,
                                int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matmul_q8_0_batched_f32(void* queuePtr, void* pipelinePtr,
                                    void* A, uint64_t aOff,
                                    void* B, uint64_t bOff,
                                    void* C, uint64_t cOff,
                                    int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];

    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int nTiles = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(nTiles, M, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_bf16_nr2_f32(void* queuePtr, void* pipelinePtr,
                               void* A, uint64_t aOff,
                               void* B, uint64_t bOff,
                               void* C, uint64_t cOff,
                               int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matmul_bf16_batched_f32(void* queuePtr, void* pipelinePtr,
                                    void* A, uint64_t aOff,
                                    void* B, uint64_t bOff,
                                    void* C, uint64_t cOff,
                                    int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];

    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int nTiles = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(nTiles, M, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matmul_q4_0_simdgroup_f32(void* queuePtr, void* pipelinePtr,
                                      void* A, void* B, void* C,
                                      int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    // Tile sizes from kernel: SMM_TILE_M=32, SMM_TILE_N=64, SMM_TILE_K=64
    int TILE_M = 32;
    int TILE_N = 64;
    int TILE_K = 64;   // 2 Q4_0 blocks per K-tile
    int threadgroupSize = 256;  // 8 simdgroups of 32 threads

    // Shared memory: half-precision A and B tiles with TILE_K=64
    int sharedMemA = TILE_M * TILE_K * 2;  // 32×64×2 = 4096 bytes (half)
    int sharedMemB = TILE_N * TILE_K * 2;  // 64×64×2 = 8192 bytes (half)

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemA atIndex:0];
    [encoder setThreadgroupMemoryLength:sharedMemB atIndex:1];

    // 2D grid: tiles in (N, M) dimensions
    int tilesN = (N + TILE_N - 1) / TILE_N;
    int tilesM = (M + TILE_M - 1) / TILE_M;
    MTLSize threadgroups = MTLSizeMake(tilesN, tilesM, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q6_K multi-output matvec: 8 outputs per threadgroup using simdgroups
// Grid: ceil(N/8) threadgroups of 256 threads
void metal_matvec_q6k_multi_output_f32(void* queuePtr, void* pipelinePtr,
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

// Q6_K NR2 matvec: 16 outputs per threadgroup (2 per simdgroup)
// Grid: ceil(N/16) threadgroups of 256 threads
void metal_matvec_q6k_nr2_f32(void* queuePtr, void* pipelinePtr,
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

    // 16 outputs per threadgroup (8 simdgroups × 2 outputs each)
    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_K multi-output matvec: 8 outputs per threadgroup using simdgroups
// Grid: ceil(N/8) threadgroups of 256 threads
// Q4_K multi-output matvec: 8 outputs per threadgroup using simdgroups
// Grid: ceil(N/8) threadgroups of 256 threads
void metal_matvec_q4k_multi_output_f32(void* queuePtr, void* pipelinePtr,
                                        void* A, uint64_t aOff,
                                        void* B, uint64_t bOff,
                                        void* C, uint64_t cOff,
                                        int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
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
void metal_matvec_q4k_nr2_f32(void* queuePtr, void* pipelinePtr,
                               void* A, uint64_t aOff,
                               void* B, uint64_t bOff,
                               void* C, uint64_t cOff,
                               int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    // 16 outputs per threadgroup (8 simdgroups * 2)
    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q5k_multi_output_f32(void* queuePtr, void* pipelinePtr,
                                        void* A, uint64_t aOff,
                                        void* B, uint64_t bOff,
                                        void* C, uint64_t cOff,
                                        int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
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

void metal_matvec_q5k_nr2_f32_v4(void* queuePtr, void* pipelinePtr,
                                  void* A, void* B, void* C,
                                  int N, int K, void* offsetsPtr) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;
    uint64_t* offsets = (uint64_t*)offsetsPtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:offsets[0] atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:offsets[1] atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:offsets[2] atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    // 16 outputs per threadgroup, 256 threads
    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
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

void metal_layernorm_f32(void* queuePtr, void* pipelinePtr,
                         void* x, void* weight, void* bias, void* out,
                         int batchSize, int dim, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int threadgroupSize = 256;
    int numWarps = (threadgroupSize + 31) / 32;
    // LayerNorm needs 2 * numWarps for sum and sumSq
    int sharedMemSize = numWarps * 2 * sizeof(float);

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&dim length:sizeof(dim) atIndex:4];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per row
    MTLSize threadgroups = MTLSizeMake(batchSize, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_gelu_f32(void* queuePtr, void* pipelinePtr,
                    void* x, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)x,
        (__bridge id<MTLBuffer>)out
    ];

    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_gelu_mul_f32(void* queuePtr, void* pipelinePtr,
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

void metal_add_bias_f32(void* queuePtr, void* pipelinePtr,
                        void* x, void* bias, void* out,
                        int rows, int cols) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int totalElements = rows * cols;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];

    // One thread per element
    int threadgroupSize = 256;
    int numThreadgroups = (totalElements + threadgroupSize - 1) / threadgroupSize;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_add_rmsnorm_f32(void* queuePtr, void* pipelinePtr,
                           void* x, void* residual, void* weight, void* out,
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
    [encoder setBuffer:(__bridge id<MTLBuffer>)residual offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&dim length:sizeof(dim) atIndex:4];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

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
                        int startPos, int ropeDim, float theta, int ropeNeox) {
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
        [NSData dataWithBytes:&ropeDim length:sizeof(ropeDim)],
        [NSData dataWithBytes:&theta length:sizeof(theta)],
        [NSData dataWithBytes:&ropeNeox length:sizeof(ropeNeox)]
    ];

    // Dispatch with max(numQHeads, numKVHeads) threads per position
    int maxHeads = numQHeads > numKVHeads ? numQHeads : numKVHeads;
    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(seqLen, maxHeads, 1));
}

// RoPE with pre-computed per-dimension inverse frequencies (learned RoPE scaling, Gemma 2)
void metal_rope_gqa_scaled_f32(void* queuePtr, void* pipelinePtr,
                                void* q, void* k, void* freqs,
                                int seqLen, int numQHeads, int numKVHeads, int headDim,
                                int startPos, int ropeNeox) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)q,
        (__bridge id<MTLBuffer>)k,
        (__bridge id<MTLBuffer>)freqs
    ];
    NSArray* constants = @[
        [NSData dataWithBytes:&seqLen length:sizeof(seqLen)],
        [NSData dataWithBytes:&numQHeads length:sizeof(numQHeads)],
        [NSData dataWithBytes:&numKVHeads length:sizeof(numKVHeads)],
        [NSData dataWithBytes:&headDim length:sizeof(headDim)],
        [NSData dataWithBytes:&startPos length:sizeof(startPos)],
        [NSData dataWithBytes:&ropeNeox length:sizeof(ropeNeox)]
    ];

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

// Offset-aware variants: operate on sub-regions of shared MTLBuffers.
// These enable scratch-allocated tensors to be used without separate buffer objects.

void metal_add_f32_offset(void* queuePtr, void* pipelinePtr,
                          void* a, uint64_t aOff,
                          void* b, uint64_t bOff,
                          void* out, uint64_t outOff, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)a offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)b offset:bOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:2];

    MTLSize threadgroupSize = MTLSizeMake(
        MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)n + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_rmsnorm_f32_offset(void* queuePtr, void* pipelinePtr,
                              void* x, uint64_t xOff,
                              void* weight, uint64_t weightOff,
                              void* out, uint64_t outOff,
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
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:xOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:weightOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:2];
    [encoder setBytes:&dim length:sizeof(dim) atIndex:3];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:4];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(batchSize, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_silu_mul_f32_offset(void* queuePtr, void* pipelinePtr,
                               void* gate, uint64_t gateOff,
                               void* up, uint64_t upOff,
                               void* out, uint64_t outOff, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)gate offset:gateOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)up offset:upOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:2];

    MTLSize threadgroupSize = MTLSizeMake(
        MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)n + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// --- Offset-aware Q4_0 matmul variants ---
// A and C use offsets (scratch-allocated activations), B (weights) always offset 0.

void metal_matvec_q4_0_multi_output_f32_offset(void* queuePtr, void* pipelinePtr,
                                                void* A, uint64_t aOff,
                                                void* B,
                                                void* C, uint64_t cOff,
                                                int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    int outputsPerTG = 8;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 NR2 matvec offset-aware: 16 outputs per threadgroup (2 per simdgroup)
// A and C use offsets (scratch-allocated), B (weights) always at offset 0.
void metal_matvec_q4_0_nr2_f32_offset(void* queuePtr, void* pipelinePtr,
                                       void* A, uint64_t aOff,
                                       void* B,
                                       void* C, uint64_t cOff,
                                       int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    int outputsPerTG = 16;
    int threadgroupSize = 256;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matmul_q4_0_batched_f32_offset(void* queuePtr, void* pipelinePtr,
                                           void* A, uint64_t aOff,
                                           void* B,
                                           void* C, uint64_t cOff,
                                           int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int threadgroupSize = 256;
    int numWarps = (threadgroupSize + 31) / 32;
    int sharedMemSize = numWarps * sizeof(float);

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(N, M, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matmul_q4_0_simdgroup_f32_offset(void* queuePtr, void* pipelinePtr,
                                             void* A, uint64_t aOff,
                                             void* B,
                                             void* C, uint64_t cOff,
                                             int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int TILE_M = 32;
    int TILE_N = 64;
    int TILE_K = 64;   // 2 Q4_0 blocks per K-tile
    int threadgroupSize = 256;  // 8 simdgroups

    int sharedMemA = TILE_M * TILE_K * 2;  // half precision: 32×64×2 = 4096 bytes
    int sharedMemB = TILE_N * TILE_K * 2;  // half precision: 64×64×2 = 8192 bytes

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemA atIndex:0];
    [encoder setThreadgroupMemoryLength:sharedMemB atIndex:1];

    int tilesN = (N + TILE_N - 1) / TILE_N;
    int tilesM = (M + TILE_M - 1) / TILE_M;
    MTLSize threadgroups = MTLSizeMake(tilesN, tilesM, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q4_0_transposed_f32_offset(void* queuePtr, void* pipelinePtr,
                                              void* A, uint64_t aOff,
                                              void* B,
                                              void* C, uint64_t cOff,
                                              int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:aOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:cOff atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];

    MTLSize threadgroupSize = MTLSizeMake(
        MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)N), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)N + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// --- Offset-aware RoPE ---
// Q and K use offsets (scratch-allocated activations).

void metal_rope_gqa_f32_offset(void* queuePtr, void* pipelinePtr,
                                void* q, uint64_t qOff,
                                void* k, uint64_t kOff,
                                int seqLen, int numQHeads, int numKVHeads, int headDim,
                                int startPos, int ropeDim, float theta, int ropeNeox) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)q offset:qOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)k offset:kOff atIndex:1];
    [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:2];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:3];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:4];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:5];
    [encoder setBytes:&startPos length:sizeof(startPos) atIndex:6];
    [encoder setBytes:&ropeDim length:sizeof(ropeDim) atIndex:7];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:8];
    [encoder setBytes:&ropeNeox length:sizeof(ropeNeox) atIndex:9];

    int maxHeads = numQHeads > numKVHeads ? numQHeads : numKVHeads;
    MTLSize threadgroupSize = MTLSizeMake(
        MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)seqLen), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)seqLen + threadgroupSize.width - 1) / threadgroupSize.width,
        maxHeads, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// --- Offset-aware SDPA ---
// Q and out use offsets (scratch-allocated), K/V (KV cache) at offset 0.

void metal_sdpa_decode_f32_offset(void* queuePtr, void* pipelinePtr,
                                   void* Q, uint64_t qOff,
                                   void* K, void* V,
                                   void* out, uint64_t outOff,
                                   int kvLen, int numQHeads, int numKVHeads, int headDim,
                                   float scale, int kvHeadStride) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:qOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)K offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)V offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:3];
    [encoder setBytes:&kvLen length:sizeof(kvLen) atIndex:4];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:5];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:6];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:7];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:8];
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];

    MTLSize threadgroupSize = MTLSizeMake(MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)numQHeads), 1, 1);
    MTLSize threadgroups = MTLSizeMake((numQHeads + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_sdpa_flash_decode_f32_offset(void* queuePtr, void* pipelinePtr,
                                         void* Q, uint64_t qOff,
                                         void* K, void* V,
                                         void* out, uint64_t outOff,
                                         int kvLen, int numQHeads, int numKVHeads, int headDim,
                                         float scale, int kvHeadStride) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:qOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)K offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)V offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:3];
    [encoder setBytes:&kvLen length:sizeof(kvLen) atIndex:4];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:5];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:6];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:7];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:8];
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];

    int threadgroupSize = 256;
    int sharedMemSize = (kvLen + 8) * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// --- Offset-aware fused kernel variants ---

void metal_matvec_q4_0_fused_rmsnorm_f32_offset(void* queuePtr, void* pipelinePtr,
                                                  void* x, void* normWeight, void* wMat,
                                                  void* out, uint64_t outOff,
                                                  int n, int k, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)wMat offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:3];
    [encoder setBytes:&n length:sizeof(n) atIndex:4];
    [encoder setBytes:&k length:sizeof(k) atIndex:5];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:6];

    int sharedMemSize = (k + 8) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int numThreadgroups = (n + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_add_rmsnorm_f32_offset(void* queuePtr, void* pipelinePtr,
                                   void* x,
                                   void* residual, uint64_t residualOff,
                                   void* weight,
                                   void* out, uint64_t outOff,
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
    [encoder setBuffer:(__bridge id<MTLBuffer>)residual offset:residualOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:3];
    [encoder setBytes:&dim length:sizeof(dim) atIndex:4];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:5];
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(batchSize, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_scatter_kv_f32_offset(void* queuePtr, void* pipelinePtr,
                                  void* src, uint64_t srcOff,
                                  void* dst,
                                  int newTokens, int numKVHeads, int headDim,
                                  int maxSeqLen, int seqPos) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)src offset:srcOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
    [encoder setBytes:&newTokens length:sizeof(newTokens) atIndex:2];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:3];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:4];
    [encoder setBytes:&maxSeqLen length:sizeof(maxSeqLen) atIndex:5];
    [encoder setBytes:&seqPos length:sizeof(seqPos) atIndex:6];

    int totalElements = newTokens * numKVHeads * headDim;
    int threadgroupSize = 256;
    MTLSize gridSize = MTLSizeMake(totalElements, 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q4_0_fused_mlp_f32_offset(void* queuePtr, void* pipelinePtr,
                                             void* x, uint64_t xOff,
                                             void* W1, void* W3,
                                             void* out, uint64_t outOff,
                                             int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:xOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W1 offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W3 offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];

    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
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

// Argmax: find index of maximum value (GPU-side greedy sampling)
void metal_argmax_f32(void* queuePtr, void* pipelinePtr,
                       void* input, void* result, int N) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)result offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(N) atIndex:2];

    // Use 256 threads for reduction
    int numThreads = 256;
    // Shared memory: 256 floats + 256 ints
    int sharedMem = numThreads * (sizeof(float) + sizeof(int));
    [encoder setThreadgroupMemoryLength:numThreads * sizeof(float) atIndex:0];
    [encoder setThreadgroupMemoryLength:numThreads * sizeof(int) atIndex:1];

    MTLSize threadgroups = MTLSizeMake(1, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(numThreads, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Compute-based memory copy (avoids blit encoder transition issues)
void metal_copy_buffer_compute(void* queuePtr, void* pipelinePtr,
                                void* srcBuffer, size_t srcOffset,
                                void* dstBuffer, size_t dstOffset, size_t size) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    // Calculate number of uchar4 elements (4 bytes each)
    // Round up to handle non-aligned sizes
    int numElements = (int)((size + 3) / 4);

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)srcBuffer offset:srcOffset atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstBuffer offset:dstOffset atIndex:1];
    [encoder setBytes:&numElements length:sizeof(numElements) atIndex:2];

    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    MTLSize threadgroups = MTLSizeMake((numElements + threadWidth - 1) / threadWidth, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadWidth, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// =============================================================================
// Training Kernel C Dispatch Functions
// =============================================================================

void metal_relu_inplace_f32(void* queuePtr, void* pipelinePtr, void* x, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[(__bridge id<MTLBuffer>)x];
    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_relu_backward_f32(void* queuePtr, void* pipelinePtr,
                             void* x, void* dx, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)x,
        (__bridge id<MTLBuffer>)dx
    ];
    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_batched_outer_product_f32(void* queuePtr, void* pipelinePtr,
                                     void* A, void* B, void* C,
                                     int batch, int M, int N) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)A,
        (__bridge id<MTLBuffer>)B,
        (__bridge id<MTLBuffer>)C
    ];

    NSArray* params = @[
        [NSData dataWithBytes:&batch length:sizeof(batch)],
        [NSData dataWithBytes:&M length:sizeof(M)],
        [NSData dataWithBytes:&N length:sizeof(N)]
    ];
    // Grid is [N, M] to cover all output elements
    dispatch_kernel(queue, pipeline, buffers, params, MTLSizeMake(N, M, 1));
}

void metal_sgd_update_f32(void* queuePtr, void* pipelinePtr,
                          void* w, void* grad, float lr, float weightDecay, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)w,
        (__bridge id<MTLBuffer>)grad
    ];

    NSArray* constants = @[
        [NSData dataWithBytes:&lr length:sizeof(lr)],
        [NSData dataWithBytes:&weightDecay length:sizeof(weightDecay)]
    ];

    dispatch_kernel(queue, pipeline, buffers, constants, MTLSizeMake(n, 1, 1));
}

void metal_zero_f32(void* queuePtr, void* pipelinePtr, void* x, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[(__bridge id<MTLBuffer>)x];
    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

// =============================================================================
// FP16 (Half-Precision) C Dispatch Functions
// =============================================================================

void metal_add_f16(void* queuePtr, void* pipelinePtr,
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

void metal_mul_f16(void* queuePtr, void* pipelinePtr,
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

void metal_silu_f16(void* queuePtr, void* pipelinePtr,
                    void* x, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)x,
        (__bridge id<MTLBuffer>)out
    ];

    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_silu_mul_f16(void* queuePtr, void* pipelinePtr,
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

void metal_convert_f32_to_f16(void* queuePtr, void* pipelinePtr,
                               void* in, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)in,
        (__bridge id<MTLBuffer>)out
    ];

    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

void metal_convert_f16_to_f32(void* queuePtr, void* pipelinePtr,
                               void* in, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    NSArray* buffers = @[
        (__bridge id<MTLBuffer>)in,
        (__bridge id<MTLBuffer>)out
    ];

    dispatch_kernel(queue, pipeline, buffers, @[], MTLSizeMake(n, 1, 1));
}

// Offset-aware FP32 to FP16 conversion for scratch-allocated buffers.
void metal_convert_f32_to_f16_offset(void* queuePtr, void* pipelinePtr,
                                      void* in, uint64_t inOff,
                                      void* out, uint64_t outOff, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)in offset:inOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:1];

    MTLSize threadgroupSize = MTLSizeMake(
        MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)n + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Offset-aware FP16 to FP32 conversion for scratch-allocated buffers.
void metal_convert_f16_to_f32_offset(void* queuePtr, void* pipelinePtr,
                                      void* in, uint64_t inOff,
                                      void* out, uint64_t outOff, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)in offset:inOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:1];

    MTLSize threadgroupSize = MTLSizeMake(
        MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)n + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// KV Cache scatter: transpose from [newTokens, numKVHeads, headDim] to [numKVHeads, maxSeqLen, headDim]
void metal_scatter_kv_f16(void* queuePtr, void* pipelinePtr,
                           void* src, void* dst,
                           int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
    [encoder setBytes:&newTokens length:sizeof(newTokens) atIndex:2];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:3];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:4];
    [encoder setBytes:&maxSeqLen length:sizeof(maxSeqLen) atIndex:5];
    [encoder setBytes:&seqPos length:sizeof(seqPos) atIndex:6];

    int totalElements = newTokens * numKVHeads * headDim;
    int threadgroupSize = 256;
    MTLSize gridSize = MTLSizeMake(totalElements, 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_scatter_kv_f32(void* queuePtr, void* pipelinePtr,
                           void* src, void* dst,
                           int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
    [encoder setBytes:&newTokens length:sizeof(newTokens) atIndex:2];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:3];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:4];
    [encoder setBytes:&maxSeqLen length:sizeof(maxSeqLen) atIndex:5];
    [encoder setBytes:&seqPos length:sizeof(seqPos) atIndex:6];

    int totalElements = newTokens * numKVHeads * headDim;
    int threadgroupSize = 256;
    MTLSize gridSize = MTLSizeMake(totalElements, 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_scatter_kv_f32_to_f16(void* queuePtr, void* pipelinePtr,
                                  void* src, void* dst,
                                  int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
    [encoder setBytes:&newTokens length:sizeof(newTokens) atIndex:2];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:3];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:4];
    [encoder setBytes:&maxSeqLen length:sizeof(maxSeqLen) atIndex:5];
    [encoder setBytes:&seqPos length:sizeof(seqPos) atIndex:6];

    int totalElements = newTokens * numKVHeads * headDim;
    int threadgroupSize = 256;
    MTLSize gridSize = MTLSizeMake(totalElements, 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_reshape_paged_kv_f32(void* queuePtr, void* pipelinePtr,
                                void* src, void* dstBase,
                                void* pageTable,
                                void* blockOffsets,
                                int numTokens, int numKVHeads, int headDim, int blockSize, int isValue) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstBase offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pageTable offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)blockOffsets offset:0 atIndex:3];
    [encoder setBytes:&numTokens length:sizeof(numTokens) atIndex:4];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:5];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:6];
    [encoder setBytes:&blockSize length:sizeof(blockSize) atIndex:7];
    [encoder setBytes:&isValue length:sizeof(isValue) atIndex:8];

    int totalElements = numTokens * numKVHeads * headDim;
    int threadgroupSize = 256;
    int numThreadgroups = (totalElements + threadgroupSize - 1) / threadgroupSize;
    MTLSize gridSize = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    
    // dispatchThreads is preferred for non-uniform grids
    [encoder dispatchThreads:MTLSizeMake(totalElements, 1, 1) threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_sdpa_paged_decode_f32(void* queuePtr, void* pipelinePtr,
                                  void* Q, void* kvPool, void* blockTable, void* out,
                                  int numBlocks, int blockSize,
                                  int numQHeads, int numKVHeads, int headDim,
                                  float scale, int tokensInLastBlock) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)kvPool offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)blockTable offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&numBlocks length:sizeof(numBlocks) atIndex:4];
    [encoder setBytes:&blockSize length:sizeof(blockSize) atIndex:5];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:6];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:7];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:8];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:9];
    [encoder setBytes:&tokensInLastBlock length:sizeof(tokensInLastBlock) atIndex:10];

    // Shared memory: weights[kvLen] + warpVals[8]
    int kvLen = (numBlocks - 1) * blockSize + tokensInLastBlock;
    int sharedMemSize = (kvLen + 8) * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per Q head
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q8_0 Quantization dispatch functions
void metal_quantize_f32_to_q8_0(void* queuePtr, void* pipelinePtr,
                                 void* in, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)in offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
    [encoder setBytes:&n length:sizeof(n) atIndex:2];

    // One threadgroup per Q8_0 block (32 elements)
    int numBlocks = (n + 31) / 32;
    MTLSize threadgroups = MTLSizeMake(numBlocks, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);  // One thread per block

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_dequantize_q8_0_to_f32(void* queuePtr, void* pipelinePtr,
                                   void* in, void* out, int n) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)in offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
    [encoder setBytes:&n length:sizeof(n) atIndex:2];

    // 256 threads per threadgroup, each thread handles one element
    int threadsPerGroup = 256;
    int numGroups = (n + threadsPerGroup - 1) / threadsPerGroup;
    MTLSize threadgroups = MTLSizeMake(numGroups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_sdpa_decode_q8_0(void* queuePtr, void* pipelinePtr,
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

    // Shared memory: weights[kvLen] + warpVals[8]
    int sharedBytes = (kvLen + 8) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedBytes atIndex:0];

    // One threadgroup per Q head
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q4_0_f16(void* queuePtr, void* pipelinePtr,
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

// Flash Attention SDPA decode for FP16 KV cache
// Uses split-KV across simdgroups with online softmax. Shared memory: O(headDim).
void metal_sdpa_flash_decode_f16(void* queuePtr, void* pipelinePtr,
                                  void* Q, void* K, void* V, void* out,
                                  int kvLen, int numQHeads, int numKVHeads, int headDim,
                                  float scale, int kvHeadStride) {
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
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];

    // Shared memory: NUM_SG max + NUM_SG sum + NUM_SG * headDim acc
    // = 2*8 + 8*headDim = 16 + 8*headDim floats
    int numSG = 8;
    int sharedMemSize = (2 * numSG + numSG * headDim) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per Q head, 256 threads (8 simdgroups)
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_rmsnorm_f16(void* queuePtr, void* pipelinePtr,
                       void* x, void* weight, void* out,
                       int batchSize, int dim, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
    [encoder setBytes:&dim length:sizeof(int) atIndex:3];
    [encoder setBytes:&eps length:sizeof(float) atIndex:4];

    // Each threadgroup processes one row
    int threadgroupSize = 256;
    int sharedMemSize = (threadgroupSize / 32 + 1) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize gridSize = MTLSizeMake(batchSize, 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_sdpa_decode_f16(void* queuePtr, void* pipelinePtr,
                            void* Q, void* K, void* V, void* out,
                            int kvLen, int numQHeads, int numKVHeads, int headDim,
                            float scale, int kvHeadStride) {
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
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];

    // Shared memory: q_cache[headDim] + weights[kvLen] + warpVals[8]
    // For vec kernel: q_cache needs to store float4s, so align to 4 floats
    int threadgroupSize = 256;
    int qCacheSize = (headDim + 3) / 4 * 4;
    int sharedMemSize = (qCacheSize + kvLen + 8) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per Q head
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Specialized SDPA for headDim=64 with vectorization and online softmax
void metal_sdpa_decode_f16_hd64(void* queuePtr, void* pipelinePtr,
                                 void* Q, void* K, void* V, void* out,
                                 int kvLen, int numQHeads, int numKVHeads,
                                 float scale, int kvHeadStride) {
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
    [encoder setBytes:&scale length:sizeof(scale) atIndex:7];
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:8];

    // One threadgroup per Q head, 1 thread per threadgroup (single thread does all work)
    // This kernel is designed for low-latency single-head processing
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(1, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// SIMD version: 32 threads cooperate via simd_sum for dot products
void metal_sdpa_decode_f16_hd64_simd(void* queuePtr, void* pipelinePtr,
                                      void* Q, void* K, void* V, void* out,
                                      int kvLen, int numQHeads, int numKVHeads,
                                      float scale, int kvHeadStride) {
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
    [encoder setBytes:&scale length:sizeof(scale) atIndex:7];
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:8];

    // Shared memory for warp reductions (not currently used but reserved)
    int sharedMemSize = 64 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per Q head, 32 threads (1 simdgroup) per threadgroup
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_flash_attention_2_f16(void* queuePtr, void* pipelinePtr,
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

    // Shared memory: K tile + V tile (both half precision)
    // FA2_TILE_KV = 64
    int FA2_TILE_KV = 64;
    int sharedMemSize = FA2_TILE_KV * headDim * 2 * sizeof(short); // half = 2 bytes
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // 256 threads (FA2_THREADS)
    int threadgroupSize = 256;

    // Grid: (seqLen, numKVHeads)
    MTLSize threadgroups = MTLSizeMake(seqLen, numKVHeads, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
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
                           float scale, int kvHeadStride) {
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
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];

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
                                  float scale, int kvHeadStride) {
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
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];

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

// SDPA decode with logit soft-capping (Gemma 2)
void metal_sdpa_softcap_decode_f32(void* queuePtr, void* pipelinePtr,
                                    void* Q, void* K, void* V, void* out,
                                    int kvLen, int numQHeads, int numKVHeads, int headDim,
                                    float scale, int kvHeadStride, float softcap) {
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
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];
    [encoder setBytes:&softcap length:sizeof(softcap) atIndex:10];

    MTLSize threadgroupSize = MTLSizeMake(MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)numQHeads), 1, 1);
    MTLSize threadgroups = MTLSizeMake((numQHeads + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// SDPA prefill with logit soft-capping (Gemma 2)
void metal_sdpa_softcap_prefill_f32(void* queuePtr, void* pipelinePtr,
                                     void* Q, void* K, void* V, void* out,
                                     int seqLen, int numQHeads, int numKVHeads, int headDim,
                                     float scale, float softcap) {
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
    [encoder setBytes:&softcap length:sizeof(softcap) atIndex:9];

    int PREFILL_THREADS_LC = 64;
    int PREFILL_TILE_K_LC = 16;
    int sharedMemSize = (PREFILL_TILE_K_LC + 8) * sizeof(float);

    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(seqLen, numQHeads, 1);
    MTLSize threadsPerGroup = MTLSizeMake(PREFILL_THREADS_LC, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q4_0_fused_rmsnorm_f32(void* queuePtr, void* pipelinePtr,
                                         void* x, void* normWeight, void* wMat, void* out,
                                         int n, int k, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)wMat offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&n length:sizeof(n) atIndex:4];
    [encoder setBytes:&k length:sizeof(k) atIndex:5];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:6];

    // Shared memory: K floats for x, + 8 floats scratch for reduction
    int sharedMemSize = (k + 8) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // Grid: 32 outputs per threadgroup (8 simdgroups * 4 outputs)
    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int numThreadgroups = (n + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Fused RMSNorm + Q4_0 MatVec with FP16 OUTPUT
// Identical dispatch to F32 version, just uses F16 output kernel
void metal_matvec_q4_0_fused_rmsnorm_f16_out(void* queuePtr, void* pipelinePtr,
                                              void* x, void* normWeight, void* wMat, void* out,
                                              int n, int k, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)wMat offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&n length:sizeof(n) atIndex:4];
    [encoder setBytes:&k length:sizeof(k) atIndex:5];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:6];

    int sharedMemSize = (k + 8) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int numThreadgroups = (n + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// RoPE for GQA with FP16 tensors
// ropeDim: dimensions to rotate (can be < headDim for partial RoPE like Phi-2)
// ropeNeox: 0 = LLaMA-style (interleaved pairs), 1 = NEOX-style (split pairs)
void metal_rope_gqa_f16(void* queuePtr, void* pipelinePtr,
                         void* q, void* k,
                         int seqLen, int numQHeads, int numKVHeads, int headDim,
                         int startPos, int ropeDim, float theta, int ropeNeox) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)q offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)k offset:0 atIndex:1];
    [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:2];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:3];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:4];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:5];
    [encoder setBytes:&startPos length:sizeof(startPos) atIndex:6];
    [encoder setBytes:&ropeDim length:sizeof(ropeDim) atIndex:7];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:8];
    [encoder setBytes:&ropeNeox length:sizeof(ropeNeox) atIndex:9];

    // Grid: [seqLen, max(numQHeads, numKVHeads)]
    int maxHeads = MAX(numQHeads, numKVHeads);
    MTLSize gridSize = MTLSizeMake(seqLen, maxHeads, 1);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);  // Each thread handles one position/head

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
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

// Flash Attention 2 optimized dispatch
// Uses larger tiles (64 K positions) and GQA-aware shared memory
void metal_flash_attention_2_f32(void* queuePtr, void* pipelinePtr,
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

    // Flash Attention 2 shared memory:
    // K tile: FA2_TILE_KV * headDim floats = 64 * 64 = 4096 floats = 16KB
    // V tile: FA2_TILE_KV * headDim floats = 64 * 64 = 4096 floats = 16KB
    // Total: 32KB
    int FA2_TILE_KV = 64;
    int sharedMemSize = 2 * FA2_TILE_KV * headDim * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // Dispatch grid: (seqLen, numKVHeads) threadgroups
    // Each threadgroup handles one Q position and all Q heads for one KV head
    // 8 simdgroups (256 threads) process 8 Q heads in parallel
    MTLSize threadgroups = MTLSizeMake(seqLen, numKVHeads, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);  // FA2_THREADS = 8 simdgroups * 32

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Flash Attention 2 v2 — optimized prefill kernel
// Fixes: single-pass online softmax, tiles Q across simdgroups for GQA utilization,
// TILE_KV=32 for FP32 headDim=128 within 32KB shared memory budget.
// Grid: (ceil(seqLen / tileQ), numKVHeads) where tileQ = max(1, 8/headsPerKV)
void metal_flash_attention_2_v2_f32(void* queuePtr, void* pipelinePtr,
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

    // Shared memory: K tile [TILE_KV, headDim] + V tile [TILE_KV, headDim]
    int FA2V2_TILE_KV = 32;
    int sharedMemSize = 2 * FA2V2_TILE_KV * headDim * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // Grid: (ceil(seqLen / tileQ), numKVHeads)
    // tileQ = max(1, 8 / headsPerKV): 8 for headsPerKV=1, 2 for headsPerKV=4, 1 for headsPerKV=8
    int headsPerKV = numQHeads / numKVHeads;
    int tileQ = 8 / headsPerKV;
    if (tileQ < 1) tileQ = 1;
    if (tileQ > 8) tileQ = 8;
    int numQTiles = (seqLen + tileQ - 1) / tileQ;

    MTLSize threadgroups = MTLSizeMake(numQTiles, numKVHeads, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);  // 8 simdgroups * 32 threads

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

void metal_matvec_q4_0_fused_mlp_f32(void* queuePtr, void* pipelinePtr,
                                     void* x, void* W1, void* W3, void* out,
                                     int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W1 offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W3 offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];

    // 32 outputs per threadgroup (8 simdgroups * 4 per group)
    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}
