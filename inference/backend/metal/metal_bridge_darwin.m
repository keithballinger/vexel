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
// Q4_0 MATVEC V2: llama.cpp-matched technique
// =============================================================================
// Matches llama.cpp kernel_mul_mv_q4_0_f32 approach:
//   - 64 threads (2 SGs × 32 threads), 8 outputs/TG (4 per SG)
//   - Fused nibble-masking: pre-scale activations, mask uint16 in place
//   - Thread pairs: 2 threads per Q4_0 block (16 elements each)
//   - Activation loaded once into registers, reused across NR0=4 rows
//   - Zero-point handled via sumy × -8 correction
//
// Grid: ceil(N/8) threadgroups of 64 threads
constant int MV2_NR0 = 4;   // rows per simdgroup
constant int MV2_NSG = 2;   // simdgroups per threadgroup
constant int MV2_NQ  = 16;  // blocks processed per iteration per simdgroup

kernel void matvec_q4_0_v2_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int r0 = (gid * MV2_NSG + simd_group) * MV2_NR0;
    if (r0 >= N) return;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    // Thread mapping: 2 threads per block, 16 blocks per iteration
    // ix: block index within the 16-block chunk (0..15)
    // il: element offset within block (0 = low half, 8 = high half)
    short ix = simd_lane / 2;     // 0..15
    short il = (simd_lane % 2) * 8;  // 0 or 8

    // Row pointers for NR0=4 weight rows
    device const uchar* rows[MV2_NR0];
    for (int row = 0; row < MV2_NR0; row++) {
        rows[row] = (r0 + row < N) ? B + (r0 + row) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    }

    float sumf[MV2_NR0] = {0.f, 0.f, 0.f, 0.f};

    // Process MV2_NQ=16 blocks per iteration
    for (int ib0 = ix; ib0 < numBlocks; ib0 += MV2_NQ) {
        int base_k = ib0 * Q4_BLOCK_SIZE + il;

        // Load 16 activation elements into registers (this thread's half of the block)
        // Pre-scale activations for fused nibble-masking:
        //   yl[i+0]  = yb[i]      (multiplied by low nibble mask 0x000F)
        //   yl[i+1]  = yb[i+1]/256 (multiplied by mask 0x0F00 = nibble at bit 8)
        //   yl[i+8]  = yb[i+16]/16  (multiplied by mask 0x00F0 = nibble at bit 4)
        //   yl[i+9]  = yb[i+17]/4096 (multiplied by mask 0xF000 = nibble at bit 12)
        device const float* yb = A + base_k;
        float yl[16];
        float sumy0 = 0.f, sumy1 = 0.f;

        for (short i = 0; i < 8; i += 2) {
            sumy0  += yb[i + 0] + yb[i + 1];
            yl[i+0] = yb[i + 0];
            yl[i+1] = yb[i + 1] / 256.f;

            sumy1  += yb[i + 16] + yb[i + 17];
            yl[i+8] = yb[i + 16] / 16.f;
            yl[i+9] = yb[i + 17] / 4096.f;
        }

        float sumy = sumy0 + sumy1;

        // For each of NR0=4 rows, compute dot product using fused nibble masking
        for (short row = 0; row < MV2_NR0; row++) {
            if (!rows[row]) continue;

            device const uchar* blockPtr = rows[row] + ib0 * Q4_BYTES_PER_BLOCK;
            float d = float(as_type<half>(*((device const ushort*)blockPtr)));

            // uint16_t reads: each uint16 contains 4 nibbles (2 bytes = 2 elements low + 2 elements high)
            // qs data starts at byte 2 (after 2-byte half scale); il=0/8 gives byte offset into 16-byte qs array
            device const ushort* qs = (device const ushort*)(blockPtr + 2 + il);

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (short i = 0; i < 8; i += 2) {
                acc0 += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc1 += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc2 += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc3 += yl[i + 9] * (qs[i / 2] & 0xF000);
            }

            sumf[row] += d * (sumy * -8.f + acc0 + acc1 + acc2 + acc3);
        }
    }

    // Simdgroup reduction and output
    for (short row = 0; row < MV2_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (simd_lane == 0 && r0 + row < N) {
            C[r0 + row] = tot;
        }
    }
}

// Variant of matvec_q4_0_v2_f32 that adds result to output (C[i] += dot product).
// Used to fuse W2 matmul + residual add into single dispatch.
kernel void matvec_q4_0_v2_add_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int r0 = (gid * MV2_NSG + simd_group) * MV2_NR0;
    if (r0 >= N) return;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    short ix = simd_lane / 2;
    short il = (simd_lane % 2) * 8;

    device const uchar* rows[MV2_NR0];
    for (int row = 0; row < MV2_NR0; row++) {
        rows[row] = (r0 + row < N) ? B + (r0 + row) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    }

    float sumf[MV2_NR0] = {0.f, 0.f, 0.f, 0.f};

    for (int ib0 = ix; ib0 < numBlocks; ib0 += MV2_NQ) {
        int base_k = ib0 * Q4_BLOCK_SIZE + il;
        device const float* yb = A + base_k;
        float yl[16];
        float sumy0 = 0.f, sumy1 = 0.f;

        for (short i = 0; i < 8; i += 2) {
            sumy0  += yb[i + 0] + yb[i + 1];
            yl[i+0] = yb[i + 0];
            yl[i+1] = yb[i + 1] / 256.f;
            sumy1  += yb[i + 16] + yb[i + 17];
            yl[i+8] = yb[i + 16] / 16.f;
            yl[i+9] = yb[i + 17] / 4096.f;
        }
        float sumy = sumy0 + sumy1;

        for (short row = 0; row < MV2_NR0; row++) {
            if (!rows[row]) continue;
            device const uchar* blockPtr = rows[row] + ib0 * Q4_BYTES_PER_BLOCK;
            float d = float(as_type<half>(*((device const ushort*)blockPtr)));
            device const ushort* qs = (device const ushort*)(blockPtr + 2 + il);

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (short i = 0; i < 8; i += 2) {
                acc0 += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc1 += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc2 += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc3 += yl[i + 9] * (qs[i / 2] & 0xF000);
            }
            sumf[row] += d * (sumy * -8.f + acc0 + acc1 + acc2 + acc3);
        }
    }

    // Add to output instead of overwrite (fuses residual add)
    for (short row = 0; row < MV2_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (simd_lane == 0 && r0 + row < N) {
            C[r0 + row] += tot;
        }
    }
}

// Variant of matvec_q4_0_v2_add_f32 that also adds a deferred residual.
// Used when AddRMSNorm is fused into FusedAddRMSNormMLP: the residual add
// (x += woOutput) is deferred from AddRMSNorm to this kernel.
// Final write: C[i] = C[i] + residual[i] + (A @ B^T)[i]
kernel void matvec_q4_0_v2_add2_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int r0 = (gid * MV2_NSG + simd_group) * MV2_NR0;
    if (r0 >= N) return;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    short ix = simd_lane / 2;
    short il = (simd_lane % 2) * 8;

    device const uchar* rows[MV2_NR0];
    for (int row = 0; row < MV2_NR0; row++) {
        rows[row] = (r0 + row < N) ? B + (r0 + row) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    }

    float sumf[MV2_NR0] = {0.f, 0.f, 0.f, 0.f};

    for (int ib0 = ix; ib0 < numBlocks; ib0 += MV2_NQ) {
        int base_k = ib0 * Q4_BLOCK_SIZE + il;
        device const float* yb = A + base_k;
        float yl[16];
        float sumy0 = 0.f, sumy1 = 0.f;

        for (short i = 0; i < 8; i += 2) {
            sumy0  += yb[i + 0] + yb[i + 1];
            yl[i+0] = yb[i + 0];
            yl[i+1] = yb[i + 1] / 256.f;
            sumy1  += yb[i + 16] + yb[i + 17];
            yl[i+8] = yb[i + 16] / 16.f;
            yl[i+9] = yb[i + 17] / 4096.f;
        }
        float sumy = sumy0 + sumy1;

        for (short row = 0; row < MV2_NR0; row++) {
            if (!rows[row]) continue;
            device const uchar* blockPtr = rows[row] + ib0 * Q4_BYTES_PER_BLOCK;
            float d = float(as_type<half>(*((device const ushort*)blockPtr)));
            device const ushort* qs = (device const ushort*)(blockPtr + 2 + il);

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (short i = 0; i < 8; i += 2) {
                acc0 += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc1 += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc2 += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc3 += yl[i + 9] * (qs[i / 2] & 0xF000);
            }
            sumf[row] += d * (sumy * -8.f + acc0 + acc1 + acc2 + acc3);
        }
    }

    // Add to output including deferred residual (attention output)
    for (short row = 0; row < MV2_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (simd_lane == 0 && r0 + row < N) {
            C[r0 + row] += residual[r0 + row] + tot;
        }
    }
}

// Variant of matvec_q4_0_v2_f32 that reads FP16 activations.
// Eliminates a separate ConvertF16ToF32 dispatch when SDPA outputs FP16.
// Same algorithm: 64 threads, NR0=4, fused nibble-masking.
kernel void matvec_q4_0_v2_f16in_f32(
    device const half* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int r0 = (gid * MV2_NSG + simd_group) * MV2_NR0;
    if (r0 >= N) return;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    short ix = simd_lane / 2;
    short il = (simd_lane % 2) * 8;

    device const uchar* rows[MV2_NR0];
    for (int row = 0; row < MV2_NR0; row++) {
        rows[row] = (r0 + row < N) ? B + (r0 + row) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    }

    float sumf[MV2_NR0] = {0.f, 0.f, 0.f, 0.f};

    for (int ib0 = ix; ib0 < numBlocks; ib0 += MV2_NQ) {
        int base_k = ib0 * Q4_BLOCK_SIZE + il;

        // Load FP16 activations and convert to FP32 on the fly
        device const half* yb = A + base_k;
        float yl[16];
        float sumy0 = 0.f, sumy1 = 0.f;

        for (short i = 0; i < 8; i += 2) {
            float y0 = float(yb[i + 0]);
            float y1 = float(yb[i + 1]);
            float y16 = float(yb[i + 16]);
            float y17 = float(yb[i + 17]);

            sumy0  += y0 + y1;
            yl[i+0] = y0;
            yl[i+1] = y1 / 256.f;

            sumy1  += y16 + y17;
            yl[i+8] = y16 / 16.f;
            yl[i+9] = y17 / 4096.f;
        }

        float sumy = sumy0 + sumy1;

        for (short row = 0; row < MV2_NR0; row++) {
            if (!rows[row]) continue;

            device const uchar* blockPtr = rows[row] + ib0 * Q4_BYTES_PER_BLOCK;
            float d = float(as_type<half>(*((device const ushort*)blockPtr)));

            device const ushort* qs = (device const ushort*)(blockPtr + 2 + il);

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (short i = 0; i < 8; i += 2) {
                acc0 += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc1 += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc2 += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc3 += yl[i + 9] * (qs[i / 2] & 0xF000);
            }

            sumf[row] += d * (sumy * -8.f + acc0 + acc1 + acc2 + acc3);
        }
    }

    for (short row = 0; row < MV2_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (simd_lane == 0 && r0 + row < N) {
            C[r0 + row] = tot;
        }
    }
}

// Variant of matvec_q4_0_v2_f16in_f32 that ACCUMULATES into output: C[r] += dot(A, B[r]).
// Used for Wo+Add: writes x += Wo @ attn_output, fusing the attention residual add into Wo.
// This eliminates the separate AddRMSNorm dispatch and avoids a big→small→big pipeline drain.
kernel void matvec_q4_0_v2_f16in_add_f32(
    device const half* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int r0 = (gid * MV2_NSG + simd_group) * MV2_NR0;
    if (r0 >= N) return;

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    short ix = simd_lane / 2;
    short il = (simd_lane % 2) * 8;

    device const uchar* rows[MV2_NR0];
    for (int row = 0; row < MV2_NR0; row++) {
        rows[row] = (r0 + row < N) ? B + (r0 + row) * numBlocks * Q4_BYTES_PER_BLOCK : nullptr;
    }

    float sumf[MV2_NR0] = {0.f, 0.f, 0.f, 0.f};

    for (int ib0 = ix; ib0 < numBlocks; ib0 += MV2_NQ) {
        int base_k = ib0 * Q4_BLOCK_SIZE + il;

        device const half* yb = A + base_k;
        float yl[16];
        float sumy0 = 0.f, sumy1 = 0.f;

        for (short i = 0; i < 8; i += 2) {
            float y0 = float(yb[i + 0]);
            float y1 = float(yb[i + 1]);
            float y16 = float(yb[i + 16]);
            float y17 = float(yb[i + 17]);

            sumy0  += y0 + y1;
            yl[i+0] = y0;
            yl[i+1] = y1 / 256.f;

            sumy1  += y16 + y17;
            yl[i+8] = y16 / 16.f;
            yl[i+9] = y17 / 4096.f;
        }

        float sumy = sumy0 + sumy1;

        for (short row = 0; row < MV2_NR0; row++) {
            if (!rows[row]) continue;

            device const uchar* blockPtr = rows[row] + ib0 * Q4_BYTES_PER_BLOCK;
            float d = float(as_type<half>(*((device const ushort*)blockPtr)));

            device const ushort* qs = (device const ushort*)(blockPtr + 2 + il);

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (short i = 0; i < 8; i += 2) {
                acc0 += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc1 += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc2 += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc3 += yl[i + 9] * (qs[i / 2] & 0xF000);
            }

            sumf[row] += d * (sumy * -8.f + acc0 + acc1 + acc2 + acc3);
        }
    }

    for (short row = 0; row < MV2_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (simd_lane == 0 && r0 + row < N) {
            C[r0 + row] += tot;  // ACCUMULATE, not overwrite
        }
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

// =============================================================================
// Q4_0 Simdgroup GEMM — llama.cpp-matched occupancy
// =============================================================================
// TILE_M=32, TILE_N=64, TILE_K=32, 128 threads (4 SGs), 6KB shared
// Key optimizations:
//   1. Blocked 8×8 layout with stride=8 for simdgroup_load (zero bank conflicts)
//   2. TILE_K=32 → 6KB shared memory → 5 TGs/partition (was 2 with 12KB)
//   3. Vectorized float4 activation loads (2 loads per thread vs 8 scalar)
//   4. All 128 threads do both A loading + B dequant simultaneously
//   5. SG layout: 1×4 (each SG covers 32M×16N = 8 accumulators)
//   6. Same compute:bandwidth ratio as llama.cpp (12.5 ops/byte)
//
// Shared memory (blocked 8×8):
//   sa: 32M × 32K = 16 blocks of 8×8, block_idx = bk*4+bm → 2048 bytes
//   sb: 64N × 32K = 32 blocks of 8×8, block_idx = bk*8+bn → 4096 bytes
//   Total: 6144 bytes (matches llama.cpp's 6KB)
//
constant int SMM_TILE_M = 32;
constant int SMM_TILE_N = 64;
constant int SMM_TILE_K = 32;

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

    // SG layout: 1×4 — all SGs cover full M=32, each covers 16 N-cols
    int sg_col = simd_group * 16;  // 0, 16, 32, 48

    // 8 accumulators per SG: 4 along M × 2 along N (each 8×8)
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = simdgroup_float8x8(0.0f);

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int numKTiles = (K + SMM_TILE_K - 1) / SMM_TILE_K;

    // Block indices for this SG's N-slice (constant across all k_tiles)
    int bn0 = sg_col >> 3;     // 0, 2, 4, or 6
    int bn1 = bn0 + 1;         // 1, 3, 5, or 7

    for (int k_tile = 0; k_tile < numKTiles; k_tile++) {
        int k_base = k_tile * SMM_TILE_K;

        // === Vectorized A loading: blocked 8×8 layout ===
        // 32×32 = 1024 elements, 128 threads → 8 elements/thread
        // 4 threads per M-row, each handles 8 consecutive K-cols (= one 8×8 block row)
        // Vectorized: 2× float4 load from device → 2× half4 store to shared (contiguous)
        {
            int local_m = tid >> 2;        // 0..31
            int k_group = tid & 3;         // 0..3
            int local_k = k_group * 8;     // 0, 8, 16, 24
            int global_m = tile_m + local_m;
            int global_k = k_base + local_k;

            int bm = local_m >> 3;
            int rm = local_m & 7;
            int bk = k_group;
            threadgroup half* dst_a = shared_A + (bk * 4 + bm) * 64 + rm * 8;

            if (global_m < M && global_k + 7 < K) {
                // Fast path: vectorized load of 8 consecutive floats
                float4 v0 = *(device const float4*)(A + global_m * K + global_k);
                float4 v1 = *(device const float4*)(A + global_m * K + global_k + 4);
                *(threadgroup half4*)(dst_a)     = half4(v0);
                *(threadgroup half4*)(dst_a + 4) = half4(v1);
            } else {
                // Slow path: scalar with bounds checking
                for (int j = 0; j < 8; j++) {
                    int gk = global_k + j;
                    *(dst_a + j) = (global_m < M && gk < K) ? (half)A[global_m * K + gk] : (half)0;
                }
            }
        }

        // === B dequant: blocked 8×8 layout ===
        // 64 N-cols × 1 Q4_0 block each, 128 threads → 2 threads per column
        // Thread (2n) handles bytes 2..9 (k=0..7 low, k=16..23 high)
        // Thread (2n+1) handles bytes 10..17 (k=8..15 low, k=24..31 high)
        {
            int local_n = tid >> 1;        // 0..63
            int sub = tid & 1;             // 0 or 1
            int global_n = tile_n + local_n;
            int bn = local_n >> 3;
            int cn = local_n & 7;

            if (global_n < N && k_base < K) {
                device const uchar* blockPtr = B + global_n * numBlocks * Q4_BYTES_PER_BLOCK
                                                + k_tile * Q4_BYTES_PER_BLOCK;
                ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
                half scale = as_type<half>(scale_u16);

                int byte_start = 2 + sub * 8;  // sub=0: bytes 2..9, sub=1: bytes 10..17

                for (int j = 0; j < 8; j++) {
                    uchar byte_val = blockPtr[byte_start + j];
                    int lo_k = sub * 8 + j;         // 0..7 or 8..15
                    int hi_k = 16 + sub * 8 + j;    // 16..23 or 24..31

                    int bk_lo = lo_k >> 3;
                    int rk_lo = lo_k & 7;
                    int bk_hi = hi_k >> 3;
                    int rk_hi = hi_k & 7;

                    shared_B[(bk_lo * 8 + bn) * 64 + rk_lo * 8 + cn] = scale * (half)((byte_val & 0xF) - 8);
                    shared_B[(bk_hi * 8 + bn) * 64 + rk_hi * 8 + cn] = scale * (half)(((byte_val >> 4) & 0xF) - 8);
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    int lo_k = sub * 8 + j;
                    int hi_k = 16 + sub * 8 + j;
                    int bk_lo = lo_k >> 3;
                    int rk_lo = lo_k & 7;
                    int bk_hi = hi_k >> 3;
                    int rk_hi = hi_k & 7;
                    shared_B[(bk_lo * 8 + bn) * 64 + rk_lo * 8 + cn] = (half)0;
                    shared_B[(bk_hi * 8 + bn) * 64 + rk_hi * 8 + cn] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Compute: 4 K-substeps × (4 ma + 2 mb + 8 MAC) ===
        // stride=8 for simdgroup_load (contiguous 8×8 blocks)
        // simdgroup_barrier hints improve instruction scheduling (matches llama.cpp)
        for (int ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 ma[4], mb[2];

            simdgroup_barrier(mem_flags::mem_none);

            // Load 4 A sub-matrices (32 M-rows, 8 K-cols) — shared by all SGs
            simdgroup_load(ma[0], shared_A + (ik * 4 + 0) * 64, 8);
            simdgroup_load(ma[1], shared_A + (ik * 4 + 1) * 64, 8);
            simdgroup_load(ma[2], shared_A + (ik * 4 + 2) * 64, 8);
            simdgroup_load(ma[3], shared_A + (ik * 4 + 3) * 64, 8);

            simdgroup_barrier(mem_flags::mem_none);

            // Load 2 B sub-matrices (16 N-cols of this SG, 8 K-cols)
            simdgroup_load(mb[0], shared_B + (ik * 8 + bn0) * 64, 8);
            simdgroup_load(mb[1], shared_B + (ik * 8 + bn1) * 64, 8);

            simdgroup_barrier(mem_flags::mem_none);

            // 8 MACs: mc[ia*2+ib] += ma[ia] × mb[ib]
            simdgroup_multiply_accumulate(mc[0], ma[0], mb[0], mc[0]);
            simdgroup_multiply_accumulate(mc[1], ma[0], mb[1], mc[1]);
            simdgroup_multiply_accumulate(mc[2], ma[1], mb[0], mc[2]);
            simdgroup_multiply_accumulate(mc[3], ma[1], mb[1], mc[3]);
            simdgroup_multiply_accumulate(mc[4], ma[2], mb[0], mc[4]);
            simdgroup_multiply_accumulate(mc[5], ma[2], mb[1], mc[5]);
            simdgroup_multiply_accumulate(mc[6], ma[3], mb[0], mc[6]);
            simdgroup_multiply_accumulate(mc[7], ma[3], mb[1], mc[7]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ========== Write output ==========
    int out_m = tile_m;
    int out_n = tile_n + sg_col;

    for (int i = 0; i < 8; i++) {
        int m_off = (i / 2) * 8;
        int n_off = (i % 2) * 8;
        if (out_m + m_off < M && out_n + n_off < N)
            simdgroup_store(mc[i], C + (out_m + m_off) * N + (out_n + n_off), N);
    }
}

// =============================================================================
// Q4_0 Simdgroup GEMM v2 — llama.cpp-matched architecture, flat shared memory
// =============================================================================
// Architecture from llama.cpp kernel_mul_mm:
//   1. TILE_M=64, TILE_N=32, TILE_K=32 (K matches Q4_0 block size)
//   2. 4 simdgroups × 128 threads
//   3. 8 accumulators per SG (4×2 = 32×16 output per SG, 1.33 MAC:load ratio)
//   4. 6144 bytes shared memory (vs v1's 12288) → 2× occupancy
//   5. All 128 threads participate in both A and B loading
//   6. Flat row-major shared memory (simple, efficient indexing)
//
// SG layout: 2×2 grid
//   SG 0 (sg%2=0, sg/2=0): M-rows 0-31,  N-cols 0-15
//   SG 1 (sg%2=1, sg/2=0): M-rows 32-63, N-cols 0-15
//   SG 2 (sg%2=0, sg/2=1): M-rows 0-31,  N-cols 16-31
//   SG 3 (sg%2=1, sg/2=1): M-rows 32-63, N-cols 16-31
//
// Shared memory (flat row-major):
//   sa: [64, 32] half, stride 32  → 4096 bytes
//   sb: [32, 32] half, stride 32  → 2048 bytes  (B stored as B^T[k,n])
//   Total: 6144 bytes

constant int SMM2_TILE_M = 64;
constant int SMM2_TILE_N = 32;
constant int SMM2_TILE_K = 32;  // matches Q4_0 block size
constant int SMM2_NUM_SG = 4;
constant int SMM2_THREADS = SMM2_NUM_SG * 32;  // 128

kernel void matmul_q4_0_simdgroup_v2_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    threadgroup half* shared [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int tile_m = tg_pos.y * SMM2_TILE_M;
    int tile_n = tg_pos.x * SMM2_TILE_N;
    if (tile_m >= M || tile_n >= N) return;

    // SG layout: 2×2 grid, each covers 32×16 output
    int sg_m_half = simd_group % 2;   // 0 or 1 → M-rows 0-31 or 32-63
    int sg_n_half = simd_group / 2;   // 0 or 1 → N-cols 0-15 or 16-31
    int sg_row = sg_m_half * 32;      // start row within tile
    int sg_col = sg_n_half * 16;      // start col within tile

    // 8 accumulators per SG: 4 along M × 2 along N (each 8×8)
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = simdgroup_float8x8(0.0f);

    // Shared memory: flat row-major
    threadgroup half* sa = shared;          // [64, 32] half, stride=32
    threadgroup half* sb = shared + 2048;   // [32, 32] half, stride=32

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int numKTiles = (K + SMM2_TILE_K - 1) / SMM2_TILE_K;

    for (int k_tile = 0; k_tile < numKTiles; k_tile++) {
        int k_base = k_tile * SMM2_TILE_K;

        // ============================================================
        // LOAD A (activations, float → half): all 128 threads
        // 64×32 = 2048 elements, 128 threads → 16 elements/thread
        // ============================================================
        for (int i = tid; i < 2048; i += 128) {
            int local_m = i / 32;     // i >> 5
            int local_k = i % 32;     // i & 31
            int global_m = tile_m + local_m;
            int global_k = k_base + local_k;
            half val = (global_m < M && global_k < K)
                       ? (half)A[global_m * K + global_k] : (half)0;
            *(sa + i) = val;  // sa[local_m * 32 + local_k]
        }

        // ============================================================
        // LOAD B (Q4_0 weights → half, stored as B^T[k,n])
        // 128 threads: 4 threads per Q4_0 block × 32 blocks
        // Each thread dequants 4 bytes → 8 half values
        // ============================================================
        {
            int block_n = tid >> 2;      // tid / 4 → 0..31 (N-column)
            int sub = tid & 3;           // tid % 4 → 0..3 (byte chunk)
            int global_n = tile_n + block_n;

            if (global_n < N && k_base < K) {
                device const uchar* blockPtr = B + global_n * numBlocks * Q4_BYTES_PER_BLOCK
                                                + k_tile * Q4_BYTES_PER_BLOCK;
                ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
                half scale = as_type<half>(scale_u16);

                // sub p: bytes [2+4p .. 2+4p+3] → 4 low nibbles + 4 high nibbles
                int byte_start = 2 + sub * 4;
                for (int jj = 0; jj < 4; jj++) {
                    uchar byte_val = blockPtr[byte_start + jj];
                    int lo_k = sub * 4 + jj;       // k = 0..15
                    int hi_k = 16 + sub * 4 + jj;  // k = 16..31

                    // sb[k * 32 + n] — B^T[k, n] in flat row-major
                    *(sb + lo_k * 32 + block_n) = scale * (half)((byte_val & 0xF) - 8);
                    *(sb + hi_k * 32 + block_n) = scale * (half)(((byte_val >> 4) & 0xF) - 8);
                }
            } else {
                // Zero-fill: 8 values for this thread
                int byte_start_k = sub * 4;
                for (int jj = 0; jj < 4; jj++) {
                    *(sb + (byte_start_k + jj) * 32 + block_n) = (half)0;
                    *(sb + (16 + byte_start_k + jj) * 32 + block_n) = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============================================================
        // COMPUTE: 4 K-substeps × (4 A loads + 2 B loads + 8 MACs)
        // MAC:load = 8:6 = 1.33
        // ============================================================
        for (int ik = 0; ik < 4; ik++) {
            int k_off = ik * 8;  // K offset within tile
            simdgroup_half8x8 ma[4], mb[2];

            // Load 4 A sub-matrices (32 M-rows of this SG)
            simdgroup_load(ma[0], sa + (sg_row + 0) * 32 + k_off, 32);
            simdgroup_load(ma[1], sa + (sg_row + 8) * 32 + k_off, 32);
            simdgroup_load(ma[2], sa + (sg_row + 16) * 32 + k_off, 32);
            simdgroup_load(ma[3], sa + (sg_row + 24) * 32 + k_off, 32);

            // Load 2 B sub-matrices (16 N-cols of this SG)
            simdgroup_load(mb[0], sb + k_off * 32 + sg_col, 32);
            simdgroup_load(mb[1], sb + k_off * 32 + (sg_col + 8), 32);

            // 8 MACs: mc[ia*2+ib] += ma[ia] × mb[ib]
            simdgroup_multiply_accumulate(mc[0], ma[0], mb[0], mc[0]);
            simdgroup_multiply_accumulate(mc[1], ma[0], mb[1], mc[1]);
            simdgroup_multiply_accumulate(mc[2], ma[1], mb[0], mc[2]);
            simdgroup_multiply_accumulate(mc[3], ma[1], mb[1], mc[3]);
            simdgroup_multiply_accumulate(mc[4], ma[2], mb[0], mc[4]);
            simdgroup_multiply_accumulate(mc[5], ma[2], mb[1], mc[5]);
            simdgroup_multiply_accumulate(mc[6], ma[3], mb[0], mc[6]);
            simdgroup_multiply_accumulate(mc[7], ma[3], mb[1], mc[7]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ============================================================
    // WRITE OUTPUT: 8 tiles of 8×8 per SG → 32×16 output block
    // ============================================================
    int sg_m_base = tile_m + sg_row;
    int sg_n_base = tile_n + sg_col;

    for (int ia = 0; ia < 4; ia++) {
        int m_start = sg_m_base + ia * 8;
        if (m_start >= M) continue;
        for (int ib_idx = 0; ib_idx < 2; ib_idx++) {
            int n_start = sg_n_base + ib_idx * 8;
            if (n_start >= N) continue;
            if (m_start + 8 <= M && n_start + 8 <= N) {
                simdgroup_store(mc[ia * 2 + ib_idx], C + m_start * N + n_start, N);
            } else {
                // Boundary: store via shared memory scratch
                threadgroup float* scratch = (threadgroup float*)(sa) + simd_group * 64;
                simdgroup_store(mc[ia * 2 + ib_idx], scratch, 8);
                for (int elem = simd_lane; elem < 64; elem += 32) {
                    int r = elem / 8;
                    int c = elem % 8;
                    if (m_start + r < M && n_start + c < N) {
                        C[(m_start + r) * N + (n_start + c)] = scratch[r * 8 + c];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q4_0 Simdgroup GEMM v3 — llama.cpp blocked shared memory layout
// =============================================================================
// Key insight from llama.cpp: store shared memory as contiguous 8×8 blocks
// with stride 8 for simdgroup_load, instead of flat row-major with stride 32.
// This gives ~4× better threadgroup memory read efficiency since the hardware
// can read each 8×8 tile in one burst (128 bytes) vs 8 scattered 16-byte reads.
//
// Architecture (same as v2):
//   TILE_M=64, TILE_N=32, TILE_K=32 (matches Q4_0 block size)
//   128 threads (4 simdgroups), 6144 bytes shared memory
//   8 accumulators per SG (4×2 = 32×16 output, 1.33 MAC:load ratio)
//
// Shared memory layout (DIFFERENT from v2):
//   sa: 32 blocks of 8×8 halves, block_idx = bk*8 + bm
//       sa[block_idx * 64 + row_in_block * 8 + col_in_block]
//       bk = 0..3 (K-blocks), bm = 0..7 (M-blocks)
//   sb: 16 blocks of 8×8 halves, block_idx = bk*4 + bn
//       sb[block_idx * 64 + row_in_block * 8 + col_in_block]
//       bk = 0..3 (K-blocks), bn = 0..3 (N-blocks)
//
// Compute access pattern:
//   For sgitg%2=0 (M rows 0-31), ik-th K substep:
//     lsma = sa + (ik * 8) * 64  =>  4 consecutive blocks bm=0..3
//     ma[i] = simdgroup_load(lsma + i*64, stride=8)
//   For sgitg/2=0 (N cols 0-15), ik-th K substep:
//     lsmb = sb + (ik * 4) * 64  =>  2 consecutive blocks bn=0..1
//     mb[i] = simdgroup_load(lsmb + i*64, stride=8)

kernel void matmul_q4_0_simdgroup_v3_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    threadgroup half* shared [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int tile_m = tg_pos.y * 64;
    int tile_n = tg_pos.x * 32;
    if (tile_m >= M || tile_n >= N) return;

    // SG layout: 2×2 grid, each covers 32×16 output
    int sg_m_half = simd_group & 1;    // 0 or 1
    int sg_n_half = simd_group >> 1;   // 0 or 1

    // 8 accumulators per SG: 4 along M × 2 along N (each 8×8)
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = simdgroup_float8x8(0.0f);

    // Shared memory: blocked 8×8 layout
    threadgroup half* sa = shared;          // 32 blocks × 64 halves = 4096 bytes
    threadgroup half* sb = shared + 2048;   // 16 blocks × 64 halves = 2048 bytes

    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int numKTiles = (K + 31) / 32;

    for (int k_tile = 0; k_tile < numKTiles; k_tile++) {
        int k_base = k_tile * 32;

        // ============================================================
        // LOAD A (float → half) into blocked 8×8 layout
        // 64×32 = 2048 elements, 128 threads → 16 elements/thread
        // Block index = bk*8 + bm, stored at block_idx*64 + rm*8 + ck
        // ============================================================
        for (int i = tid; i < 2048; i += 128) {
            int local_m = i >> 5;        // i / 32, range 0..63
            int local_k = i & 31;        // i % 32, range 0..31
            int global_m = tile_m + local_m;
            int global_k = k_base + local_k;

            int bm = local_m >> 3;       // local_m / 8, range 0..7
            int bk = local_k >> 3;       // local_k / 8, range 0..3
            int rm = local_m & 7;        // local_m % 8
            int ck = local_k & 7;        // local_k % 8
            int block_idx = bk * 8 + bm; // K-major block order

            half val = (global_m < M && global_k < K)
                       ? (half)A[global_m * K + global_k] : (half)0;
            *(sa + block_idx * 64 + rm * 8 + ck) = val;
        }

        // ============================================================
        // LOAD B (Q4_0 → half) into blocked 8×8 layout
        // 128 threads: 4 threads per N-column × 32 N-columns
        // Block index = bk*4 + bn, stored at block_idx*64 + rk*8 + cn
        // ============================================================
        {
            int block_n = tid >> 2;      // 0..31 (N-column)
            int sub = tid & 3;           // 0..3 (byte chunk within Q4_0 block)
            int global_n = tile_n + block_n;

            if (global_n < N && k_base < K) {
                device const uchar* blockPtr = B + global_n * numBlocks * Q4_BYTES_PER_BLOCK
                                                + k_tile * Q4_BYTES_PER_BLOCK;
                ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
                half scale = as_type<half>(scale_u16);

                int byte_start = 2 + sub * 4;
                int bn = block_n >> 3;           // N-block: 0..3
                int cn = block_n & 7;            // col within N-block

                for (int jj = 0; jj < 4; jj++) {
                    uchar byte_val = blockPtr[byte_start + jj];
                    int lo_k = sub * 4 + jj;       // k = 0..15
                    int hi_k = 16 + sub * 4 + jj;  // k = 16..31

                    // lo_k block coordinates
                    int bk_lo = lo_k >> 3;          // 0 or 1
                    int rk_lo = lo_k & 7;
                    int idx_lo = (bk_lo * 4 + bn) * 64 + rk_lo * 8 + cn;

                    // hi_k block coordinates
                    int bk_hi = hi_k >> 3;          // 2 or 3
                    int rk_hi = hi_k & 7;
                    int idx_hi = (bk_hi * 4 + bn) * 64 + rk_hi * 8 + cn;

                    *(sb + idx_lo) = scale * (half)((byte_val & 0xF) - 8);
                    *(sb + idx_hi) = scale * (half)(((byte_val >> 4) & 0xF) - 8);
                }
            } else {
                int bn = block_n >> 3;
                int cn = block_n & 7;
                int byte_start_k = sub * 4;
                for (int jj = 0; jj < 4; jj++) {
                    int lo_k = byte_start_k + jj;
                    int hi_k = 16 + byte_start_k + jj;
                    int bk_lo = lo_k >> 3;
                    int rk_lo = lo_k & 7;
                    int bk_hi = hi_k >> 3;
                    int rk_hi = hi_k & 7;
                    *(sb + (bk_lo * 4 + bn) * 64 + rk_lo * 8 + cn) = (half)0;
                    *(sb + (bk_hi * 4 + bn) * 64 + rk_hi * 8 + cn) = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============================================================
        // COMPUTE: 4 K-substeps × (4 A loads + 2 B loads + 8 MACs)
        // stride=8 for simdgroup_load (contiguous 8×8 blocks)
        // ============================================================
        // lsma points to first M-block for this SG's M-half
        threadgroup const half* lsma = sa + sg_m_half * 4 * 64;  // skip 4 blocks for M-half 1
        threadgroup const half* lsmb = sb + sg_n_half * 2 * 64;  // skip 2 blocks for N-half 1

        for (int ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 ma[4], mb[2];

            // Load 4 A sub-matrices (32 M-rows of this SG, 8 K-cols)
            simdgroup_load(ma[0], lsma + 0 * 64, 8);
            simdgroup_load(ma[1], lsma + 1 * 64, 8);
            simdgroup_load(ma[2], lsma + 2 * 64, 8);
            simdgroup_load(ma[3], lsma + 3 * 64, 8);

            // Load 2 B sub-matrices (16 N-cols of this SG, 8 K-cols)
            simdgroup_load(mb[0], lsmb + 0 * 64, 8);
            simdgroup_load(mb[1], lsmb + 1 * 64, 8);

            // 8 MACs: mc[ia*2+ib] += ma[ia] × mb[ib]
            simdgroup_multiply_accumulate(mc[0], ma[0], mb[0], mc[0]);
            simdgroup_multiply_accumulate(mc[1], ma[0], mb[1], mc[1]);
            simdgroup_multiply_accumulate(mc[2], ma[1], mb[0], mc[2]);
            simdgroup_multiply_accumulate(mc[3], ma[1], mb[1], mc[3]);
            simdgroup_multiply_accumulate(mc[4], ma[2], mb[0], mc[4]);
            simdgroup_multiply_accumulate(mc[5], ma[2], mb[1], mc[5]);
            simdgroup_multiply_accumulate(mc[6], ma[3], mb[0], mc[6]);
            simdgroup_multiply_accumulate(mc[7], ma[3], mb[1], mc[7]);

            lsma += 8 * 64;   // advance to next K-block (skip 8 M-blocks)
            lsmb += 4 * 64;   // advance to next K-block (skip 4 N-blocks)
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ============================================================
    // WRITE OUTPUT: 8 tiles of 8×8 per SG → 32×16 output block
    // ============================================================
    int sg_m_base = tile_m + sg_m_half * 32;
    int sg_n_base = tile_n + sg_n_half * 16;

    for (int ia = 0; ia < 4; ia++) {
        int m_start = sg_m_base + ia * 8;
        if (m_start >= M) continue;
        for (int ib_idx = 0; ib_idx < 2; ib_idx++) {
            int n_start = sg_n_base + ib_idx * 8;
            if (n_start >= N) continue;
            if (m_start + 8 <= M && n_start + 8 <= N) {
                simdgroup_store(mc[ia * 2 + ib_idx], C + m_start * N + n_start, N);
            } else {
                threadgroup float* scratch = (threadgroup float*)(sa) + simd_group * 64;
                simdgroup_store(mc[ia * 2 + ib_idx], scratch, 8);
                for (int elem = simd_lane; elem < 64; elem += 32) {
                    int r = elem >> 3;
                    int c = elem & 7;
                    if (m_start + r < M && n_start + c < N) {
                        C[(m_start + r) * N + (n_start + c)] = scratch[r * 8 + c];
                    }
                }
            }
        }
    }
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
constant int Q4K_NR4_OUTPUTS_PER_SG = 4;
constant int Q4K_NR4_OUTPUTS_PER_TG = 32;

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

        float d = float(as_type<half>(*((device const ushort*)blockPtr)));
        float dmin_val = float(as_type<half>(*((device const ushort*)(blockPtr + 2))));

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
        float dmn = dmin_val * float(mn);

        // Q4_K interleaved layout: groups of 64 elements share 32 qs bytes.
        // Even sub-blocks (j%2==0) use low nibble, odd use high nibble.
        int qs_base = (j >> 1) << 5;          // (j/2)*32: byte offset in qs
        int shift = (j & 1) << 2;             // 0 for even (low nib), 4 for odd (high nib)
        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);  // j * 32

        if (elem_start + 32 <= K) {
            device const float4* a_vec = (device const float4*)(A + elem_start);
            device const uint* qs32 = (device const uint*)(blockPtr + 16 + qs_base);

            for (int i = 0; i < 8; i++) {
                uint w = qs32[i];
                float4 q = float4((w >> shift) & 0xF, (w >> (shift + 8)) & 0xF,
                                  (w >> (shift + 16)) & 0xF, (w >> (shift + 24)) & 0xF);
                sum += dot(a_vec[i], dsc * q - dmn);
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

// =============================================================================
// Q4_K NR4 MATVEC — 4 outputs per simdgroup, 32 per threadgroup
// =============================================================================
// Maximizes activation reuse: each 32-element sub-block of A is loaded once
// and reused for 4 weight rows. Q4_K's 144-byte blocks (vs Q4_0's 18-byte)
// avoid the L1 cache pressure that killed Q4_0 NR4.
//
// Grid: ceil(N / Q4K_NR4_OUTPUTS_PER_TG) threadgroups of 256 threads.

kernel void matvec_q4k_nr4_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [1, N] output
    constant int& N [[buffer(3)]],                 // Number of output elements
    constant int& K [[buffer(4)]],                 // Inner dimension
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base_output = gid * Q4K_NR4_OUTPUTS_PER_TG + simd_group * Q4K_NR4_OUTPUTS_PER_SG;
    if (base_output >= N) return;

    // Determine how many valid outputs this SG handles (1..4)
    int validOutputs = min(Q4K_NR4_OUTPUTS_PER_SG, N - base_output);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;

    // Row pointers for up to 4 outputs
    device const uchar* b_row0 = B + (base_output + 0) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + (base_output + 1) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row2 = B + (base_output + 2) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row3 = B + (base_output + 3) * numBlocks * Q4K_BYTES_PER_BLOCK;

    int totalSubBlocks = numBlocks << 3;  // numBlocks * 8

    // Macro: extract scale+min for sub-block j from block at blockPtr,
    // compute dsc = d*scale, dmn = dmin*min.
    #define Q4K_NR4_EXTRACT_SCALE(blockPtr, j, dsc_var, dmn_var) \
    { \
        float d_val = float(as_type<half>(*((device const ushort*)(blockPtr)))); \
        float dmin_val = float(as_type<half>(*((device const ushort*)((blockPtr) + 2)))); \
        device const uchar* sd = (blockPtr) + 4; \
        int j_lo = (j) & 3; \
        uchar sc, mn; \
        if ((j) < 4) { \
            sc = sd[j_lo] & 0x3F; \
            mn = sd[j_lo + 4] & 0x3F; \
        } else { \
            sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4); \
            mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4); \
        } \
        dsc_var = d_val * float(sc); \
        dmn_var = dmin_val * float(mn); \
    }

    // Macro: dot product of activation float4[8] with weight nibbles from qs32[8]
    #define Q4K_NR4_DOT8(a_vec, qs32_ptr, shift, dsc_val, dmn_val, sum_var) \
    { \
        for (int i = 0; i < 8; i++) { \
            float4 a = (a_vec)[i]; \
            uint w = (qs32_ptr)[i]; \
            float4 q = float4((w >> (shift)) & 0xF, (w >> ((shift) + 8)) & 0xF, \
                              (w >> ((shift) + 16)) & 0xF, (w >> ((shift) + 24)) & 0xF); \
            sum_var += dot(a, dsc_val * q - dmn_val); \
        } \
    }

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;           // which Q4_K block
        int j = sb & 7;                    // sub-block index (0..7)

        // Q4_K interleaved layout: groups of 64 share 32 qs bytes
        int qs_base = (j >> 1) << 5;      // (j/2)*32
        int shift = (j & 1) << 2;         // 0 for even, 4 for odd
        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);  // j * 32

        if (elem_start + 32 > K) {
            // Partial sub-block: scalar fallback for boundary
            if (elem_start < K) {
                // Row 0
                {
                    device const uchar* bp = b_row0 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum0 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 1) {
                    device const uchar* bp = b_row1 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum1 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 2) {
                    device const uchar* bp = b_row2 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum2 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 3) {
                    device const uchar* bp = b_row3 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum3 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
            }
            continue;
        }

        // Full sub-block: load activation ONCE, reuse for 4 weight rows
        device const float4* a_vec = (device const float4*)(A + elem_start);

        // Row 0 (always valid)
        {
            device const uchar* bp = b_row0 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_DOT8(a_vec, qs32, shift, dsc, dmn, sum0);
        }

        // Row 1
        if (validOutputs > 1) {
            device const uchar* bp = b_row1 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_DOT8(a_vec, qs32, shift, dsc, dmn, sum1);
        }

        // Row 2
        if (validOutputs > 2) {
            device const uchar* bp = b_row2 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_DOT8(a_vec, qs32, shift, dsc, dmn, sum2);
        }

        // Row 3
        if (validOutputs > 3) {
            device const uchar* bp = b_row3 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_DOT8(a_vec, qs32, shift, dsc, dmn, sum3);
        }
    }

    #undef Q4K_NR4_EXTRACT_SCALE
    #undef Q4K_NR4_DOT8

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
// Q4_K FUSED KERNELS
// =============================================================================

// Fused RMSNorm + Q/K/V Projections for Q4_K weights, FP16 output.
// Phase 1: Cooperative RMSNorm (identical to Q4_0 version).
// Phase 2: Route threadgroup to Q, K, or V weight matrix.
// Phase 3: Q4_K sub-block strided matvec with 4 outputs per simdgroup.
// Grid: ceil(qDim/32) + ceil(kvDim/32) + ceil(kvDim/32) threadgroups.
kernel void matvec_q4k_fused_rmsnorm_qkv_f16(
    device const float* x [[buffer(0)]],           // [K] input activations
    device const float* normWeight [[buffer(1)]],  // [K] RMSNorm weights
    device const uchar* Wq [[buffer(2)]],          // [qDim, K] Q4_K
    device const uchar* Wk [[buffer(3)]],          // [kvDim, K] Q4_K
    device const uchar* Wv [[buffer(4)]],          // [kvDim, K] Q4_K
    device half* outQ [[buffer(5)]],               // [qDim] FP16
    device half* outK [[buffer(6)]],               // [kvDim] FP16
    device half* outV [[buffer(7)]],               // [kvDim] FP16
    constant int& qDim [[buffer(8)]],
    constant int& kvDim [[buffer(9)]],
    constant int& K [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    threadgroup half* shared_half [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & RMSNorm
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i];
        shared_half[i] = half(val);
        localSumSq += val * val;
    }
    localSumSq = simd_sum(localSumSq);
    threadgroup float* scratch = (threadgroup float*)(shared_half + K);
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

    // Phase 2: Route to Q, K, or V projection
    int qTGs = (qDim + 31) / 32;
    int kvTGs = (kvDim + 31) / 32;
    int localGid;
    device const uchar* W;
    device half* out;
    int N;
    if ((int)gid < qTGs) {
        localGid = gid;
        W = Wq; out = outQ; N = qDim;
    } else if ((int)gid < qTGs + kvTGs) {
        localGid = gid - qTGs;
        W = Wk; out = outK; N = kvDim;
    } else {
        localGid = gid - qTGs - kvTGs;
        W = Wv; out = outV; N = kvDim;
    }

    // Phase 3: Q4_K matvec with 4 outputs per simdgroup
    int base_output = localGid * 32 + simd_group * 4;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;

    device const uchar* row0 = (base_output + 0 < N) ? W + (base_output + 0) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* row1 = (base_output + 1 < N) ? W + (base_output + 1) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* row2 = (base_output + 2 < N) ? W + (base_output + 2) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* row3 = (base_output + 3 < N) ? W + (base_output + 3) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;

    int totalSubBlocks = numBlocks << 3;

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;
        int j = sb & 7;

        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);
        if (elem_start + 32 > K) continue;

        // Load normalized activations: x_norm = shared_half[i] * rms * normWeight[i]
        device const float4* w_ptr = (device const float4*)(normWeight + elem_start);
        threadgroup const half4* hx_ptr = (threadgroup const half4*)(shared_half + elem_start);
        float4 act[8];
        act[0] = float4(hx_ptr[0]) * rms * w_ptr[0];
        act[1] = float4(hx_ptr[1]) * rms * w_ptr[1];
        act[2] = float4(hx_ptr[2]) * rms * w_ptr[2];
        act[3] = float4(hx_ptr[3]) * rms * w_ptr[3];
        act[4] = float4(hx_ptr[4]) * rms * w_ptr[4];
        act[5] = float4(hx_ptr[5]) * rms * w_ptr[5];
        act[6] = float4(hx_ptr[6]) * rms * w_ptr[6];
        act[7] = float4(hx_ptr[7]) * rms * w_ptr[7];

        // Process each row with Q4_K dequant
        #define PROCESS_ROW_Q4K_QKV(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* blockPtr = row_ptr + block_idx * Q4K_BYTES_PER_BLOCK; \
            float d = float(as_type<half>(*((device const ushort*)blockPtr))); \
            float dmin_val = float(as_type<half>(*((device const ushort*)(blockPtr + 2)))); \
            device const uchar* sd = blockPtr + 4; \
            int j_lo = j & 3; \
            uchar sc, mn; \
            if (j < 4) { \
                sc = sd[j_lo] & 0x3F; \
                mn = sd[j_lo + 4] & 0x3F; \
            } else { \
                sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4); \
                mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4); \
            } \
            float dsc = d * float(sc); \
            float dmn = dmin_val * float(mn); \
            int qs_base = (j >> 1) << 5; \
            int shift = (j & 1) << 2; \
            device const uint* qs32 = (device const uint*)(blockPtr + 16 + qs_base); \
            for (int i = 0; i < 8; i++) { \
                uint w = qs32[i]; \
                float4 q = float4((w >> shift) & 0xF, (w >> (shift + 8)) & 0xF, \
                                  (w >> (shift + 16)) & 0xF, (w >> (shift + 24)) & 0xF); \
                sum_var += dot(act[i], dsc * q - dmn); \
            } \
        }

        PROCESS_ROW_Q4K_QKV(row0, sum0);
        PROCESS_ROW_Q4K_QKV(row1, sum1);
        PROCESS_ROW_Q4K_QKV(row2, sum2);
        PROCESS_ROW_Q4K_QKV(row3, sum3);
        #undef PROCESS_ROW_Q4K_QKV
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    if (simd_lane == 0) {
        if (base_output + 0 < N) out[base_output + 0] = half(sum0);
        if (base_output + 1 < N) out[base_output + 1] = half(sum1);
        if (base_output + 2 < N) out[base_output + 2] = half(sum2);
        if (base_output + 3 < N) out[base_output + 3] = half(sum3);
    }
}

// Fused MLP for Q4_K: SiLU(x @ W1) * (x @ W3)
// Computes two Q4_K matvecs in parallel and fuses SiLU + element-wise multiply.
// Only for decode (seqLen=1).
kernel void matvec_q4k_fused_mlp_f32(
    device const float* x [[buffer(0)]],      // [K] input
    device const uchar* W1 [[buffer(1)]],     // [N, K] Q4_K Gate
    device const uchar* W3 [[buffer(2)]],     // [N, K] Q4_K Up
    device float* out [[buffer(3)]],          // [N] output
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base_output = gid * 32 + simd_group * 4;

    float sumGate0 = 0.0f, sumGate1 = 0.0f, sumGate2 = 0.0f, sumGate3 = 0.0f;
    float sumUp0 = 0.0f, sumUp1 = 0.0f, sumUp2 = 0.0f, sumUp3 = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;

    // W1 (Gate) rows
    device const uchar* w1_row0 = (base_output + 0 < N) ? W1 + (base_output + 0) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* w1_row1 = (base_output + 1 < N) ? W1 + (base_output + 1) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* w1_row2 = (base_output + 2 < N) ? W1 + (base_output + 2) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* w1_row3 = (base_output + 3 < N) ? W1 + (base_output + 3) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;

    // W3 (Up) rows
    device const uchar* w3_row0 = (base_output + 0 < N) ? W3 + (base_output + 0) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* w3_row1 = (base_output + 1 < N) ? W3 + (base_output + 1) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* w3_row2 = (base_output + 2 < N) ? W3 + (base_output + 2) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;
    device const uchar* w3_row3 = (base_output + 3 < N) ? W3 + (base_output + 3) * numBlocks * Q4K_BYTES_PER_BLOCK : nullptr;

    int totalSubBlocks = numBlocks << 3;

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;
        int j = sb & 7;

        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);
        if (elem_start + 32 > K) continue;

        // Load 32 activations
        device const float4* x_vec = (device const float4*)(x + elem_start);
        float4 act[8];
        act[0] = x_vec[0]; act[1] = x_vec[1]; act[2] = x_vec[2]; act[3] = x_vec[3];
        act[4] = x_vec[4]; act[5] = x_vec[5]; act[6] = x_vec[6]; act[7] = x_vec[7];

        // Process Q4_K row macro
        #define PROCESS_ROW_Q4K_MLP(row_ptr, sum_var) \
        if (row_ptr) { \
            device const uchar* blockPtr = row_ptr + block_idx * Q4K_BYTES_PER_BLOCK; \
            float d = float(as_type<half>(*((device const ushort*)blockPtr))); \
            float dmin_val = float(as_type<half>(*((device const ushort*)(blockPtr + 2)))); \
            device const uchar* sd = blockPtr + 4; \
            int j_lo = j & 3; \
            uchar sc, mn; \
            if (j < 4) { \
                sc = sd[j_lo] & 0x3F; \
                mn = sd[j_lo + 4] & 0x3F; \
            } else { \
                sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4); \
                mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4); \
            } \
            float dsc = d * float(sc); \
            float dmn = dmin_val * float(mn); \
            int qs_base = (j >> 1) << 5; \
            int shift = (j & 1) << 2; \
            device const uint* qs32 = (device const uint*)(blockPtr + 16 + qs_base); \
            for (int i = 0; i < 8; i++) { \
                uint w = qs32[i]; \
                float4 q = float4((w >> shift) & 0xF, (w >> (shift + 8)) & 0xF, \
                                  (w >> (shift + 16)) & 0xF, (w >> (shift + 24)) & 0xF); \
                sum_var += dot(act[i], dsc * q - dmn); \
            } \
        }

        // Process W1 (Gate)
        PROCESS_ROW_Q4K_MLP(w1_row0, sumGate0);
        PROCESS_ROW_Q4K_MLP(w1_row1, sumGate1);
        PROCESS_ROW_Q4K_MLP(w1_row2, sumGate2);
        PROCESS_ROW_Q4K_MLP(w1_row3, sumGate3);

        // Process W3 (Up)
        PROCESS_ROW_Q4K_MLP(w3_row0, sumUp0);
        PROCESS_ROW_Q4K_MLP(w3_row1, sumUp1);
        PROCESS_ROW_Q4K_MLP(w3_row2, sumUp2);
        PROCESS_ROW_Q4K_MLP(w3_row3, sumUp3);

        #undef PROCESS_ROW_Q4K_MLP
    }

    // Reduce
    sumGate0 = simd_sum(sumGate0); sumUp0 = simd_sum(sumUp0);
    sumGate1 = simd_sum(sumGate1); sumUp1 = simd_sum(sumUp1);
    sumGate2 = simd_sum(sumGate2); sumUp2 = simd_sum(sumUp2);
    sumGate3 = simd_sum(sumGate3); sumUp3 = simd_sum(sumUp3);

    // SiLU(gate) * up
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

// Q4_K matvec with FP16 input, FP32 output.
// Eliminates separate ConvertF16ToF32 dispatch when SDPA outputs FP16.
// Uses FP16 dequant for 2× ALU throughput on Apple GPU, FP32 accumulation for precision.
kernel void matvec_q4k_f16in_f32(
    device const half* A [[buffer(0)]],            // [1, K] activations (FP16)
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [1, N] output (FP32)
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int output_idx = gid * Q4K_OUTPUTS_PER_TG + simd_group;
    if (output_idx >= N) return;

    float sum = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q4K_BYTES_PER_BLOCK;
    int totalSubBlocks = numBlocks << 3;

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;
        int j = sb & 7;

        device const uchar* blockPtr = b_row + block_idx * Q4K_BYTES_PER_BLOCK;

        float d = float(as_type<half>(*((device const ushort*)blockPtr)));
        float dmin_val = float(as_type<half>(*((device const ushort*)(blockPtr + 2))));

        device const uchar* sd = blockPtr + 4;
        int j_lo = j & 3;
        uchar sc, mn;
        if (j < 4) {
            sc = sd[j_lo] & 0x3F;
            mn = sd[j_lo + 4] & 0x3F;
        } else {
            sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4);
            mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4);
        }

        float dsc = d * float(sc);
        float dmn = dmin_val * float(mn);

        int qs_base = (j >> 1) << 5;
        int shift = (j & 1) << 2;
        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);

        if (elem_start + 32 <= K) {
            device const half4* a_vec = (device const half4*)(A + elem_start);
            device const uint* qs32 = (device const uint*)(blockPtr + 16 + qs_base);

            for (int i = 0; i < 8; i++) {
                uint w = qs32[i];
                float4 q = float4((w >> shift) & 0xF, (w >> (shift + 8)) & 0xF,
                                  (w >> (shift + 16)) & 0xF, (w >> (shift + 24)) & 0xF);
                sum += dot(float4(a_vec[i]), dsc * q - dmn);
            }
        } else if (elem_start < K) {
            device const uchar* qs = blockPtr + 16 + qs_base;
            for (int i = 0; i < 32 && elem_start + i < K; i++) {
                float q_val = float((qs[i] >> shift) & 0xF);
                sum += float(A[elem_start + i]) * (dsc * q_val - dmn);
            }
        }
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        C[output_idx] = sum;
    }
}

// Q4_K matvec that ADDS to output: C[i] += dot(A, B_row_i).
// Fuses W2 matmul + residual addition into a single dispatch.
// Uses FP16 dequant for 2× ALU throughput on Apple GPU, FP32 accumulation.
kernel void matvec_q4k_add_f32(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int output_idx = gid * Q4K_OUTPUTS_PER_TG + simd_group;
    if (output_idx >= N) return;

    float sum = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q4K_BYTES_PER_BLOCK;
    int totalSubBlocks = numBlocks << 3;

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;
        int j = sb & 7;

        device const uchar* blockPtr = b_row + block_idx * Q4K_BYTES_PER_BLOCK;

        float d = float(as_type<half>(*((device const ushort*)blockPtr)));
        float dmin_val = float(as_type<half>(*((device const ushort*)(blockPtr + 2))));

        device const uchar* sd = blockPtr + 4;
        int j_lo = j & 3;
        uchar sc, mn;
        if (j < 4) {
            sc = sd[j_lo] & 0x3F;
            mn = sd[j_lo + 4] & 0x3F;
        } else {
            sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4);
            mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4);
        }

        float dsc = d * float(sc);
        float dmn = dmin_val * float(mn);

        int qs_base = (j >> 1) << 5;
        int shift = (j & 1) << 2;
        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);

        if (elem_start + 32 <= K) {
            device const float4* a_vec = (device const float4*)(A + elem_start);
            device const uint* qs32 = (device const uint*)(blockPtr + 16 + qs_base);

            for (int i = 0; i < 8; i++) {
                uint w = qs32[i];
                float4 q = float4((w >> shift) & 0xF, (w >> (shift + 8)) & 0xF,
                                  (w >> (shift + 16)) & 0xF, (w >> (shift + 24)) & 0xF);
                sum += dot(a_vec[i], dsc * q - dmn);
            }
        } else if (elem_start < K) {
            device const uchar* qs = blockPtr + 16 + qs_base;
            for (int i = 0; i < 32 && elem_start + i < K; i++) {
                float q_val = float((qs[i] >> shift) & 0xF);
                sum += A[elem_start + i] * (dsc * q_val - dmn);
            }
        }
    }

    sum = simd_sum(sum);
    // ADD to output instead of overwrite (fuses matmul + residual add)
    if (simd_lane == 0) {
        C[output_idx] += sum;
    }
}

// =============================================================================
// Q4_K NR4 F16-input variant: same as matvec_q4k_nr4_f32 but reads FP16 activations.
// Used for Wo projection after FP16 SDPA output. 4 outputs per simdgroup.
// =============================================================================
kernel void matvec_q4k_nr4_f16in_f32(
    device const half* A [[buffer(0)]],            // [1, K] activations (FP16)
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [1, N] output (FP32)
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base_output = gid * Q4K_NR4_OUTPUTS_PER_TG + simd_group * Q4K_NR4_OUTPUTS_PER_SG;
    if (base_output >= N) return;

    int validOutputs = min(Q4K_NR4_OUTPUTS_PER_SG, N - base_output);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;

    device const uchar* b_row0 = B + (base_output + 0) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + (base_output + 1) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row2 = B + (base_output + 2) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row3 = B + (base_output + 3) * numBlocks * Q4K_BYTES_PER_BLOCK;

    int totalSubBlocks = numBlocks << 3;

    #define Q4K_NR4_F16_EXTRACT_SCALE(blockPtr, j, dsc_var, dmn_var) \
    { \
        float d_val = float(as_type<half>(*((device const ushort*)(blockPtr)))); \
        float dmin_val = float(as_type<half>(*((device const ushort*)((blockPtr) + 2)))); \
        device const uchar* sd = (blockPtr) + 4; \
        int j_lo = (j) & 3; \
        uchar sc, mn; \
        if ((j) < 4) { \
            sc = sd[j_lo] & 0x3F; \
            mn = sd[j_lo + 4] & 0x3F; \
        } else { \
            sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4); \
            mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4); \
        } \
        dsc_var = d_val * float(sc); \
        dmn_var = dmin_val * float(mn); \
    }

    #define Q4K_NR4_F16_DOT8(a_vec, qs32_ptr, shift, dsc_val, dmn_val, sum_var) \
    { \
        for (int i = 0; i < 8; i++) { \
            float4 a = float4((a_vec)[i]); \
            uint w = (qs32_ptr)[i]; \
            float4 q = float4((w >> (shift)) & 0xF, (w >> ((shift) + 8)) & 0xF, \
                              (w >> ((shift) + 16)) & 0xF, (w >> ((shift) + 24)) & 0xF); \
            sum_var += dot(a, dsc_val * q - dmn_val); \
        } \
    }

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;
        int j = sb & 7;

        int qs_base = (j >> 1) << 5;
        int shift = (j & 1) << 2;
        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);

        if (elem_start + 32 > K) {
            if (elem_start < K) {
                // Partial sub-block: scalar fallback
                {
                    device const uchar* bp = b_row0 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum0 += float(A[elem_start + i]) * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 1) {
                    device const uchar* bp = b_row1 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum1 += float(A[elem_start + i]) * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 2) {
                    device const uchar* bp = b_row2 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum2 += float(A[elem_start + i]) * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 3) {
                    device const uchar* bp = b_row3 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum3 += float(A[elem_start + i]) * (dsc * q_val - dmn);
                    }
                }
            }
            continue;
        }

        // Full sub-block: load FP16 activation ONCE, convert to float4, reuse for 4 weight rows
        device const half4* a_vec = (device const half4*)(A + elem_start);

        // Row 0 (always valid)
        {
            device const uchar* bp = b_row0 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_F16_DOT8(a_vec, qs32, shift, dsc, dmn, sum0);
        }

        if (validOutputs > 1) {
            device const uchar* bp = b_row1 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_F16_DOT8(a_vec, qs32, shift, dsc, dmn, sum1);
        }

        if (validOutputs > 2) {
            device const uchar* bp = b_row2 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_F16_DOT8(a_vec, qs32, shift, dsc, dmn, sum2);
        }

        if (validOutputs > 3) {
            device const uchar* bp = b_row3 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_F16_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_F16_DOT8(a_vec, qs32, shift, dsc, dmn, sum3);
        }
    }

    #undef Q4K_NR4_F16_EXTRACT_SCALE
    #undef Q4K_NR4_F16_DOT8

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    if (simd_lane == 0) {
        C[base_output] = sum0;
        if (base_output + 1 < N) C[base_output + 1] = sum1;
        if (base_output + 2 < N) C[base_output + 2] = sum2;
        if (base_output + 3 < N) C[base_output + 3] = sum3;
    }
}

// =============================================================================
// Q4_K NR4 Add variant: same as matvec_q4k_nr4_f32 but ADDS to output (C += A @ B^T).
// Used for fused W2 matmul + residual addition. 4 outputs per simdgroup.
// =============================================================================
kernel void matvec_q4k_nr4_add_f32(
    device const float* A [[buffer(0)]],           // [1, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_K format
    device float* C [[buffer(2)]],                 // [1, N] output (ADDS to existing)
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int base_output = gid * Q4K_NR4_OUTPUTS_PER_TG + simd_group * Q4K_NR4_OUTPUTS_PER_SG;
    if (base_output >= N) return;

    int validOutputs = min(Q4K_NR4_OUTPUTS_PER_SG, N - base_output);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;

    device const uchar* b_row0 = B + (base_output + 0) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row1 = B + (base_output + 1) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row2 = B + (base_output + 2) * numBlocks * Q4K_BYTES_PER_BLOCK;
    device const uchar* b_row3 = B + (base_output + 3) * numBlocks * Q4K_BYTES_PER_BLOCK;

    int totalSubBlocks = numBlocks << 3;

    #define Q4K_NR4_ADD_EXTRACT_SCALE(blockPtr, j, dsc_var, dmn_var) \
    { \
        float d_val = float(as_type<half>(*((device const ushort*)(blockPtr)))); \
        float dmin_val = float(as_type<half>(*((device const ushort*)((blockPtr) + 2)))); \
        device const uchar* sd = (blockPtr) + 4; \
        int j_lo = (j) & 3; \
        uchar sc, mn; \
        if ((j) < 4) { \
            sc = sd[j_lo] & 0x3F; \
            mn = sd[j_lo + 4] & 0x3F; \
        } else { \
            sc = (sd[8 + j_lo] & 0x0F) | ((sd[j_lo] >> 6) << 4); \
            mn = (sd[8 + j_lo] >> 4) | ((sd[j_lo + 4] >> 6) << 4); \
        } \
        dsc_var = d_val * float(sc); \
        dmn_var = dmin_val * float(mn); \
    }

    #define Q4K_NR4_ADD_DOT8(a_vec, qs32_ptr, shift, dsc_val, dmn_val, sum_var) \
    { \
        for (int i = 0; i < 8; i++) { \
            float4 a = (a_vec)[i]; \
            uint w = (qs32_ptr)[i]; \
            float4 q = float4((w >> (shift)) & 0xF, (w >> ((shift) + 8)) & 0xF, \
                              (w >> ((shift) + 16)) & 0xF, (w >> ((shift) + 24)) & 0xF); \
            sum_var += dot(a, dsc_val * q - dmn_val); \
        } \
    }

    for (int sb = simd_lane; sb < totalSubBlocks; sb += 32) {
        int block_idx = sb >> 3;
        int j = sb & 7;

        int qs_base = (j >> 1) << 5;
        int shift = (j & 1) << 2;
        int elem_start = block_idx * Q4K_BLOCK_SIZE + (j << 5);

        if (elem_start + 32 > K) {
            if (elem_start < K) {
                {
                    device const uchar* bp = b_row0 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum0 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 1) {
                    device const uchar* bp = b_row1 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum1 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 2) {
                    device const uchar* bp = b_row2 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum2 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
                if (validOutputs > 3) {
                    device const uchar* bp = b_row3 + block_idx * Q4K_BYTES_PER_BLOCK;
                    float dsc, dmn;
                    Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
                    device const uchar* qs = bp + 16 + qs_base;
                    for (int i = 0; i < 32 && elem_start + i < K; i++) {
                        float q_val = float((qs[i] >> shift) & 0xF);
                        sum3 += A[elem_start + i] * (dsc * q_val - dmn);
                    }
                }
            }
            continue;
        }

        // Full sub-block: load activation ONCE, reuse for 4 weight rows
        device const float4* a_vec = (device const float4*)(A + elem_start);

        {
            device const uchar* bp = b_row0 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_ADD_DOT8(a_vec, qs32, shift, dsc, dmn, sum0);
        }

        if (validOutputs > 1) {
            device const uchar* bp = b_row1 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_ADD_DOT8(a_vec, qs32, shift, dsc, dmn, sum1);
        }

        if (validOutputs > 2) {
            device const uchar* bp = b_row2 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_ADD_DOT8(a_vec, qs32, shift, dsc, dmn, sum2);
        }

        if (validOutputs > 3) {
            device const uchar* bp = b_row3 + block_idx * Q4K_BYTES_PER_BLOCK;
            float dsc, dmn;
            Q4K_NR4_ADD_EXTRACT_SCALE(bp, j, dsc, dmn);
            device const uint* qs32 = (device const uint*)(bp + 16 + qs_base);
            Q4K_NR4_ADD_DOT8(a_vec, qs32, shift, dsc, dmn, sum3);
        }
    }

    #undef Q4K_NR4_ADD_EXTRACT_SCALE
    #undef Q4K_NR4_ADD_DOT8

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    // ADD to output (fuses matmul + residual addition)
    if (simd_lane == 0) {
        C[base_output] += sum0;
        if (base_output + 1 < N) C[base_output + 1] += sum1;
        if (base_output + 2 < N) C[base_output + 2] += sum2;
        if (base_output + 3 < N) C[base_output + 3] += sum3;
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
    threadgroup float* shared_x [[threadgroup(0)]], // [K+8] shared activations + scratch
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & RMSNorm
    // F32 shared memory for this kernel (F32 output needs full precision activations)
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

    // Phase 2: MatVec using normalized x
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
    threadgroup half* shared_half [[threadgroup(0)]], // FP16 shared memory: halves usage for better occupancy
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & RMSNorm
    // Store activations as FP16 in shared memory (8 KB vs 16 KB) for better occupancy.
    // RMSNorm sum-of-squares computed in FP32 for accuracy.
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i];
        shared_half[i] = half(val);
        localSumSq += val * val;
    }

    localSumSq = simd_sum(localSumSq);

    threadgroup float* scratch = (threadgroup float*)(shared_half + K);
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

    // Phase 2: MatVec
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

        // Read activations from FP16 shared memory, convert to float4
        threadgroup const half4* hx_ptr = (threadgroup const half4*)(shared_half + base_k);
        float4 x0 = float4(hx_ptr[0]), x1 = float4(hx_ptr[1]), x2 = float4(hx_ptr[2]), x3 = float4(hx_ptr[3]);
        float4 x4 = float4(hx_ptr[4]), x5 = float4(hx_ptr[5]), x6 = float4(hx_ptr[6]), x7 = float4(hx_ptr[7]);

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

// Fused RMSNorm + QKV Projections Kernel (FP16 output)
// Computes RMSNorm once, then routes each threadgroup to Q, K, or V weight matrix.
// Saves 2 dispatches per layer vs separate Q/K/V calls (3 → 1).
// Grid: ceil(qDim/32) + ceil(kvDim/32) + ceil(kvDim/32) threadgroups.
kernel void matvec_q4_0_fused_rmsnorm_qkv_f16(
    device const float* x [[buffer(0)]],           // [K] input activations
    device const float* normWeight [[buffer(1)]],  // [K] RMSNorm weights
    device const uchar* Wq [[buffer(2)]],          // [qDim, K] Q4_0
    device const uchar* Wk [[buffer(3)]],          // [kvDim, K] Q4_0
    device const uchar* Wv [[buffer(4)]],          // [kvDim, K] Q4_0
    device half* outQ [[buffer(5)]],               // [qDim] FP16
    device half* outK [[buffer(6)]],               // [kvDim] FP16
    device half* outV [[buffer(7)]],               // [kvDim] FP16
    constant int& qDim [[buffer(8)]],
    constant int& kvDim [[buffer(9)]],
    constant int& K [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    threadgroup half* shared_half [[threadgroup(0)]],  // FP16 shared memory: halves usage for better occupancy
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & RMSNorm (identical for all TGs)
    // Store activations as FP16 in shared memory (8 KB vs 16 KB) for better occupancy.
    // RMSNorm sum-of-squares computed in FP32 for accuracy.
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i];
        shared_half[i] = half(val);
        localSumSq += val * val;
    }
    localSumSq = simd_sum(localSumSq);
    threadgroup float* scratch = (threadgroup float*)(shared_half + K);
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

    // Phase 2: Route threadgroup to Q, K, or V projection
    int qTGs = (qDim + 31) / 32;
    int kvTGs = (kvDim + 31) / 32;

    int localGid;
    device const uchar* W;
    device half* out;
    int N;

    if ((int)gid < qTGs) {
        localGid = gid;
        W = Wq;
        out = outQ;
        N = qDim;
    } else if ((int)gid < qTGs + kvTGs) {
        localGid = gid - qTGs;
        W = Wk;
        out = outK;
        N = kvDim;
    } else {
        localGid = gid - qTGs - kvTGs;
        W = Wv;
        out = outV;
        N = kvDim;
    }

    int base_output = localGid * 32 + simd_group * 4;

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

        // Read activations from FP16 shared memory, convert to float4
        threadgroup const half4* hx_ptr = (threadgroup const half4*)(shared_half + base_k);
        float4 x0 = float4(hx_ptr[0]), x1 = float4(hx_ptr[1]), x2 = float4(hx_ptr[2]), x3 = float4(hx_ptr[3]);
        float4 x4 = float4(hx_ptr[4]), x5 = float4(hx_ptr[5]), x6 = float4(hx_ptr[6]), x7 = float4(hx_ptr[7]);

        float4 a0 = x0 * rms * w0, a1 = x1 * rms * w1, a2 = x2 * rms * w2, a3 = x3 * rms * w3;
        float4 a4 = x4 * rms * w4, a5 = x5 * rms * w5, a6 = x6 * rms * w6, a7 = x7 * rms * w7;

        #define PROCESS_ROW_QKV(row_ptr, sum_var) \
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

        PROCESS_ROW_QKV(row0, sum0);
        PROCESS_ROW_QKV(row1, sum1);
        PROCESS_ROW_QKV(row2, sum2);
        PROCESS_ROW_QKV(row3, sum3);
        #undef PROCESS_ROW_QKV
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    if (simd_lane == 0) {
        if (base_output + 0 < N) out[base_output + 0] = half(sum0);
        if (base_output + 1 < N) out[base_output + 1] = half(sum1);
        if (base_output + 2 < N) out[base_output + 2] = half(sum2);
        if (base_output + 3 < N) out[base_output + 3] = half(sum3);
    }
}

// Fused RMSNorm + QKV + RoPE + KV Scatter Kernel (FP16 output, decode only)
// Combines 3 dispatches into 1: FusedQKV + RoPE + ScatterKV.
// Eliminates the 32-thread RoPE+ScatterKV dispatch that creates pipeline bubbles
// between the big QKV matvec and medium SDPA dispatch.
//
// Q threadgroups: matvec → RoPE in-register → write to outQ
// K threadgroups: matvec → RoPE in-register → scatter directly to kCache
// V threadgroups: matvec → scatter directly to vCache (no RoPE)
//
// Grid: ceil(qDim/32) + ceil(kvDim/32) + ceil(kvDim/32) threadgroups.
// Each SG produces 4 consecutive outputs. Since base_output = localGid*32 + sg*4,
// position within head is always a multiple of 4, so (sum0,sum1) and (sum2,sum3)
// form valid RoPE pairs at (posInHead, posInHead+1) and (posInHead+2, posInHead+3).
kernel void matvec_q4_0_fused_rmsnorm_qkv_rope_scatter_f16(
    device const float* x [[buffer(0)]],           // [K] input activations
    device const float* normWeight [[buffer(1)]],  // [K] RMSNorm weights
    device const uchar* Wq [[buffer(2)]],          // [qDim, K] Q4_0
    device const uchar* Wk [[buffer(3)]],          // [kvDim, K] Q4_0
    device const uchar* Wv [[buffer(4)]],          // [kvDim, K] Q4_0
    device half* outQ [[buffer(5)]],               // [qDim] FP16 Q output (RoPE applied)
    device half* kCache [[buffer(6)]],             // K cache: [numKVHeads, maxSeqLen, headDim]
    device half* vCache [[buffer(7)]],             // V cache: [numKVHeads, maxSeqLen, headDim]
    constant int& qDim [[buffer(8)]],
    constant int& kvDim [[buffer(9)]],
    constant int& K [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    constant int& headDim [[buffer(12)]],
    constant int& ropeDim [[buffer(13)]],
    constant int& startPos [[buffer(14)]],
    constant float& theta [[buffer(15)]],
    constant int& maxSeqLen [[buffer(16)]],
    constant int& seqPos [[buffer(17)]],
    threadgroup half* shared_half [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & RMSNorm (identical to FusedQKV kernel)
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i];
        shared_half[i] = half(val);
        localSumSq += val * val;
    }
    localSumSq = simd_sum(localSumSq);
    threadgroup float* scratch = (threadgroup float*)(shared_half + K);
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

    // Phase 2: Route threadgroup to Q, K, or V projection
    int qTGs = (qDim + 31) / 32;
    int kvTGs = (kvDim + 31) / 32;

    int isQ = ((int)gid < qTGs) ? 1 : 0;
    int isK = (!isQ && (int)gid < qTGs + kvTGs) ? 1 : 0;
    // isV = !isQ && !isK

    int localGid;
    device const uchar* W;
    int N;

    if (isQ) {
        localGid = gid;
        W = Wq;
        N = qDim;
    } else if (isK) {
        localGid = gid - qTGs;
        W = Wk;
        N = kvDim;
    } else {
        localGid = gid - qTGs - kvTGs;
        W = Wv;
        N = kvDim;
    }

    // Phase 3: MatVec (identical inner loop to FusedQKV)
    int base_output = localGid * 32 + simd_group * 4;

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

        threadgroup const half4* hx_ptr = (threadgroup const half4*)(shared_half + base_k);
        float4 x0 = float4(hx_ptr[0]), x1 = float4(hx_ptr[1]), x2 = float4(hx_ptr[2]), x3 = float4(hx_ptr[3]);
        float4 x4 = float4(hx_ptr[4]), x5 = float4(hx_ptr[5]), x6 = float4(hx_ptr[6]), x7 = float4(hx_ptr[7]);

        float4 a0 = x0 * rms * w0, a1 = x1 * rms * w1, a2 = x2 * rms * w2, a3 = x3 * rms * w3;
        float4 a4 = x4 * rms * w4, a5 = x5 * rms * w5, a6 = x6 * rms * w6, a7 = x7 * rms * w7;

        #define PROCESS_ROW_FQRS(row_ptr, sum_var) \
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

        PROCESS_ROW_FQRS(row0, sum0);
        PROCESS_ROW_FQRS(row1, sum1);
        PROCESS_ROW_FQRS(row2, sum2);
        PROCESS_ROW_FQRS(row3, sum3);
        #undef PROCESS_ROW_FQRS
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);

    // Phase 4: Post-process — apply RoPE + write/scatter outputs
    if (simd_lane == 0) {
        int posInHead = base_output % headDim;

        if (isQ) {
            // Q: apply RoPE in-register, write to outQ
            // Pair 1: (posInHead, posInHead+1) → j = posInHead/2
            if (base_output + 1 < N && posInHead + 1 < ropeDim) {
                int j = posInHead / 2;
                float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
                float angle = float(startPos) * freq;
                float cos_val = cos(angle);
                float sin_val = sin(angle);
                outQ[base_output + 0] = half(sum0 * cos_val - sum1 * sin_val);
                outQ[base_output + 1] = half(sum0 * sin_val + sum1 * cos_val);
            } else {
                if (base_output + 0 < N) outQ[base_output + 0] = half(sum0);
                if (base_output + 1 < N) outQ[base_output + 1] = half(sum1);
            }
            // Pair 2: (posInHead+2, posInHead+3) → j = posInHead/2 + 1
            if (base_output + 3 < N && posInHead + 3 < ropeDim) {
                int j = posInHead / 2 + 1;
                float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
                float angle = float(startPos) * freq;
                float cos_val = cos(angle);
                float sin_val = sin(angle);
                outQ[base_output + 2] = half(sum2 * cos_val - sum3 * sin_val);
                outQ[base_output + 3] = half(sum2 * sin_val + sum3 * cos_val);
            } else {
                if (base_output + 2 < N) outQ[base_output + 2] = half(sum2);
                if (base_output + 3 < N) outQ[base_output + 3] = half(sum3);
            }

        } else if (isK) {
            // K: apply RoPE in-register, scatter directly to K cache
            int head = base_output / headDim;
            int dstBase = head * maxSeqLen * headDim + seqPos * headDim + posInHead;

            // Pair 1
            if (base_output + 1 < N && posInHead + 1 < ropeDim) {
                int j = posInHead / 2;
                float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
                float angle = float(startPos) * freq;
                float cos_val = cos(angle);
                float sin_val = sin(angle);
                kCache[dstBase + 0] = half(sum0 * cos_val - sum1 * sin_val);
                kCache[dstBase + 1] = half(sum0 * sin_val + sum1 * cos_val);
            } else {
                if (base_output + 0 < N) kCache[dstBase + 0] = half(sum0);
                if (base_output + 1 < N) kCache[dstBase + 1] = half(sum1);
            }
            // Pair 2
            if (base_output + 3 < N && posInHead + 3 < ropeDim) {
                int j = posInHead / 2 + 1;
                float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
                float angle = float(startPos) * freq;
                float cos_val = cos(angle);
                float sin_val = sin(angle);
                kCache[dstBase + 2] = half(sum2 * cos_val - sum3 * sin_val);
                kCache[dstBase + 3] = half(sum2 * sin_val + sum3 * cos_val);
            } else {
                if (base_output + 2 < N) kCache[dstBase + 2] = half(sum2);
                if (base_output + 3 < N) kCache[dstBase + 3] = half(sum3);
            }

        } else {
            // V: no RoPE, scatter directly to V cache
            int head = base_output / headDim;
            int dstBase = head * maxSeqLen * headDim + seqPos * headDim + posInHead;

            if (base_output + 0 < N) vCache[dstBase + 0] = half(sum0);
            if (base_output + 1 < N) vCache[dstBase + 1] = half(sum1);
            if (base_output + 2 < N) vCache[dstBase + 2] = half(sum2);
            if (base_output + 3 < N) vCache[dstBase + 3] = half(sum3);
        }
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
// FUSED ADD + RMSNORM + MLP KERNEL
// =============================================================================
// Merges AddRMSNorm preamble into FusedMLP to eliminate 1 dispatch per layer.
// Each threadgroup: loads x + woOutput → RMSNorm → dual W1/W3 matvec → SiLU*Mul.
// Does NOT modify x in-place (residual deferred to W2+Add2+Residual kernel).
//
// Input: x[K] (residual state), woOutput[K] (attention output), normWeight[K] (FFN RMSNorm)
// Output: out[N] = SiLU(W1 @ RMSNorm(x + woOutput)) * (W3 @ RMSNorm(x + woOutput))
//
// Grid: ceil(N/32) TGs × 256 threads (8 SGs × 4 outputs/SG)
// Shared memory: K * sizeof(half) + 8 * sizeof(float)

kernel void matvec_q4_0_fused_addrmsnorm_mlp_f32(
    device const float* x [[buffer(0)]],             // [K] residual state
    device const float* woOutput [[buffer(1)]],      // [K] attention output (Wo projection)
    device const float* normWeight [[buffer(2)]],    // [K] FFN RMSNorm weight
    device const uchar* W1 [[buffer(3)]],            // [N, K] Q4_0 Gate
    device const uchar* W3 [[buffer(4)]],            // [N, K] Q4_0 Up
    device float* out [[buffer(5)]],                 // [N] output
    constant int& N [[buffer(6)]],                   // Intermediate size
    constant int& K [[buffer(7)]],                   // Hidden size
    constant float& eps [[buffer(8)]],               // RMSNorm epsilon
    threadgroup half* shared_half [[threadgroup(0)]], // K * sizeof(half) + 8 * sizeof(float)
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load, Add, & RMSNorm (identical for all TGs)
    // Compute x_new = x + woOutput, store as FP16 in shared memory, compute RMS.
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i] + woOutput[i];
        shared_half[i] = half(val);
        localSumSq += val * val;
    }
    localSumSq = simd_sum(localSumSq);
    threadgroup float* scratch = (threadgroup float*)(shared_half + K);
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

    // Phase 2: Dual W1/W3 Matvec with RMSNorm-scaled activations + SiLU activation
    int base_output = gid * 32 + simd_group * 4;

    float sumGate0 = 0.0f, sumGate1 = 0.0f, sumGate2 = 0.0f, sumGate3 = 0.0f;
    float sumUp0 = 0.0f, sumUp1 = 0.0f, sumUp2 = 0.0f, sumUp3 = 0.0f;

    int numBlocks = (K + 32 - 1) / 32;

    // W1 (Gate) rows
    device const uchar* w1_row0 = (base_output + 0 < N) ? W1 + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* w1_row1 = (base_output + 1 < N) ? W1 + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* w1_row2 = (base_output + 2 < N) ? W1 + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* w1_row3 = (base_output + 3 < N) ? W1 + (base_output + 3) * numBlocks * 18 : nullptr;

    // W3 (Up) rows
    device const uchar* w3_row0 = (base_output + 0 < N) ? W3 + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* w3_row1 = (base_output + 1 < N) ? W3 + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* w3_row2 = (base_output + 2 < N) ? W3 + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* w3_row3 = (base_output + 3 < N) ? W3 + (base_output + 3) * numBlocks * 18 : nullptr;

    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * 32;
        if (base_k >= K) break;

        // Read normWeight for this block
        device const float4* w_ptr = (device const float4*)(normWeight + base_k);
        float4 w0 = w_ptr[0], w1 = w_ptr[1], w2 = w_ptr[2], w3 = w_ptr[3];
        float4 w4 = w_ptr[4], w5 = w_ptr[5], w6 = w_ptr[6], w7 = w_ptr[7];

        // Read activations from FP16 shared memory, apply RMSNorm: a = x_new * rms * weight
        threadgroup const half4* hx_ptr = (threadgroup const half4*)(shared_half + base_k);
        float4 a0 = float4(hx_ptr[0]) * rms * w0, a1 = float4(hx_ptr[1]) * rms * w1;
        float4 a2 = float4(hx_ptr[2]) * rms * w2, a3 = float4(hx_ptr[3]) * rms * w3;
        float4 a4 = float4(hx_ptr[4]) * rms * w4, a5 = float4(hx_ptr[5]) * rms * w5;
        float4 a6 = float4(hx_ptr[6]) * rms * w6, a7 = float4(hx_ptr[7]) * rms * w7;

        // Dequant + dot product for both W1 and W3
        #define PROCESS_ROW_ANMLP(row_ptr, sum_var) \
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

        PROCESS_ROW_ANMLP(w1_row0, sumGate0);
        PROCESS_ROW_ANMLP(w1_row1, sumGate1);
        PROCESS_ROW_ANMLP(w1_row2, sumGate2);
        PROCESS_ROW_ANMLP(w1_row3, sumGate3);

        PROCESS_ROW_ANMLP(w3_row0, sumUp0);
        PROCESS_ROW_ANMLP(w3_row1, sumUp1);
        PROCESS_ROW_ANMLP(w3_row2, sumUp2);
        PROCESS_ROW_ANMLP(w3_row3, sumUp3);

        #undef PROCESS_ROW_ANMLP
    }

    // Reduce
    sumGate0 = simd_sum(sumGate0); sumUp0 = simd_sum(sumUp0);
    sumGate1 = simd_sum(sumGate1); sumUp1 = simd_sum(sumUp1);
    sumGate2 = simd_sum(sumGate2); sumUp2 = simd_sum(sumUp2);
    sumGate3 = simd_sum(sumGate3); sumUp3 = simd_sum(sumUp3);

    // Apply SiLU(gate) * up
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

// Fused RMSNorm + MLP (no Add). Reads x directly (residual already applied by Wo+Add).
// Computes RMSNorm(x) in preamble, then dual W1/W3 matvec with SiLU.
// Same structure as FusedAddRMSNormMLP but with one fewer device memory read per TG.
// Used with Wo+Add kernel to eliminate the AddRMSNorm dispatch and avoid pipeline bubbles.
kernel void matvec_q4_0_fused_rmsnorm_mlp_f32(
    device const float* x [[buffer(0)]],             // [K] residual state (already includes attn output)
    device const float* normWeight [[buffer(1)]],    // [K] FFN RMSNorm weight
    device const uchar* W1 [[buffer(2)]],            // [N, K] Q4_0 Gate
    device const uchar* W3 [[buffer(3)]],            // [N, K] Q4_0 Up
    device float* out [[buffer(4)]],                 // [N] output
    constant int& N [[buffer(5)]],                   // Intermediate size
    constant int& K [[buffer(6)]],                   // Hidden size
    constant float& eps [[buffer(7)]],               // RMSNorm epsilon
    threadgroup half* shared_half [[threadgroup(0)]], // K * sizeof(half) + 8 * sizeof(float)
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: Cooperative Load & RMSNorm (reads ONLY x — no woOutput read needed)
    float localSumSq = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float val = x[i];
        shared_half[i] = half(val);
        localSumSq += val * val;
    }
    localSumSq = simd_sum(localSumSq);
    threadgroup float* scratch = (threadgroup float*)(shared_half + K);
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

    // Phase 2: Dual W1/W3 Matvec with RMSNorm-scaled activations + SiLU activation
    int base_output = gid * 32 + simd_group * 4;

    float sumGate0 = 0.0f, sumGate1 = 0.0f, sumGate2 = 0.0f, sumGate3 = 0.0f;
    float sumUp0 = 0.0f, sumUp1 = 0.0f, sumUp2 = 0.0f, sumUp3 = 0.0f;

    int numBlocks = (K + 32 - 1) / 32;

    device const uchar* w1_row0 = (base_output + 0 < N) ? W1 + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* w1_row1 = (base_output + 1 < N) ? W1 + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* w1_row2 = (base_output + 2 < N) ? W1 + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* w1_row3 = (base_output + 3 < N) ? W1 + (base_output + 3) * numBlocks * 18 : nullptr;

    device const uchar* w3_row0 = (base_output + 0 < N) ? W3 + (base_output + 0) * numBlocks * 18 : nullptr;
    device const uchar* w3_row1 = (base_output + 1 < N) ? W3 + (base_output + 1) * numBlocks * 18 : nullptr;
    device const uchar* w3_row2 = (base_output + 2 < N) ? W3 + (base_output + 2) * numBlocks * 18 : nullptr;
    device const uchar* w3_row3 = (base_output + 3 < N) ? W3 + (base_output + 3) * numBlocks * 18 : nullptr;

    for (int block = simd_lane; block < numBlocks; block += 32) {
        int base_k = block * 32;
        if (base_k >= K) break;

        device const float4* w_ptr = (device const float4*)(normWeight + base_k);
        float4 w0 = w_ptr[0], w1 = w_ptr[1], w2 = w_ptr[2], w3 = w_ptr[3];
        float4 w4 = w_ptr[4], w5 = w_ptr[5], w6 = w_ptr[6], w7 = w_ptr[7];

        threadgroup const half4* hx_ptr = (threadgroup const half4*)(shared_half + base_k);
        float4 a0 = float4(hx_ptr[0]) * rms * w0, a1 = float4(hx_ptr[1]) * rms * w1;
        float4 a2 = float4(hx_ptr[2]) * rms * w2, a3 = float4(hx_ptr[3]) * rms * w3;
        float4 a4 = float4(hx_ptr[4]) * rms * w4, a5 = float4(hx_ptr[5]) * rms * w5;
        float4 a6 = float4(hx_ptr[6]) * rms * w6, a7 = float4(hx_ptr[7]) * rms * w7;

        #define PROCESS_ROW_RNMLP(row_ptr, sum_var) \
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

        PROCESS_ROW_RNMLP(w1_row0, sumGate0);
        PROCESS_ROW_RNMLP(w1_row1, sumGate1);
        PROCESS_ROW_RNMLP(w1_row2, sumGate2);
        PROCESS_ROW_RNMLP(w1_row3, sumGate3);

        PROCESS_ROW_RNMLP(w3_row0, sumUp0);
        PROCESS_ROW_RNMLP(w3_row1, sumUp1);
        PROCESS_ROW_RNMLP(w3_row2, sumUp2);
        PROCESS_ROW_RNMLP(w3_row3, sumUp3);

        #undef PROCESS_ROW_RNMLP
    }

    sumGate0 = simd_sum(sumGate0); sumUp0 = simd_sum(sumUp0);
    sumGate1 = simd_sum(sumGate1); sumUp1 = simd_sum(sumUp1);
    sumGate2 = simd_sum(sumGate2); sumUp2 = simd_sum(sumUp2);
    sumGate3 = simd_sum(sumGate3); sumUp3 = simd_sum(sumUp3);

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

// Fused K+V scatter: scatter both K and V into their respective caches in a single dispatch.
// Saves 1 dispatch per layer vs calling scatter_kv_f16 twice.
// srcK, srcV: [newTokens, numKVHeads, headDim] in FP16
// dstK, dstV: [numKVHeads, maxSeqLen, headDim] in FP16
kernel void scatter_kv_f16_fused(
    device const half* srcK [[buffer(0)]],
    device half* dstK [[buffer(1)]],
    device const half* srcV [[buffer(2)]],
    device half* dstV [[buffer(3)]],
    constant int& newTokens [[buffer(4)]],
    constant int& numKVHeads [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& maxSeqLen [[buffer(7)]],
    constant int& seqPos [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    int totalElements = newTokens * numKVHeads * headDim;
    if (gid >= (uint)totalElements) return;

    int srcHeadStride = numKVHeads * headDim;
    int t = gid / srcHeadStride;
    int remainder = gid % srcHeadStride;
    int h = remainder / headDim;
    int d = remainder % headDim;

    int dstOffset = h * maxSeqLen * headDim + (seqPos + t) * headDim + d;

    dstK[dstOffset] = srcK[gid];
    dstV[dstOffset] = srcV[gid];
}

// Fused RoPE + KV Scatter for FP16 decode (M=1).
// Combines 2 dispatches (RoPE + ScatterKV) into 1.
// - Applies RoPE to Q in-place
// - Applies RoPE to K and scatters directly to K cache
// - Scatters V directly to V cache (no RoPE)
// Thread grid: [max(numQHeads, numKVHeads), 1, 1]
// Each thread handles one head's worth of RoPE and/or scatter.
kernel void rope_scatter_kv_f16(
    device half* q [[buffer(0)]],             // [1, numQHeads, headDim] — Q, RoPE applied in-place
    device const half* srcK [[buffer(1)]],    // [1, numKVHeads, headDim] — K source
    device half* dstK [[buffer(2)]],          // K cache: [numKVHeads, maxSeqLen, headDim]
    device const half* srcV [[buffer(3)]],    // [1, numKVHeads, headDim] — V source
    device half* dstV [[buffer(4)]],          // V cache: [numKVHeads, maxSeqLen, headDim]
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant int& startPos [[buffer(8)]],     // Position for RoPE frequencies
    constant int& ropeDim [[buffer(9)]],      // Dimensions to rotate (can be < headDim)
    constant float& theta [[buffer(10)]],     // RoPE theta base
    constant int& maxSeqLen [[buffer(11)]],   // KV cache max sequence length
    constant int& seqPos [[buffer(12)]],      // Current position in KV cache for scatter
    uint gid [[thread_position_in_grid]]
) {
    int head = gid;
    int halfRopeDim = ropeDim / 2;

    // 1. Apply RoPE to Q head (if within Q head range)
    if (head < numQHeads) {
        int qOffset = head * headDim;  // M=1, so pos=0: (0 * numQHeads + head) * headDim
        for (int j = 0; j < halfRopeDim; j++) {
            float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
            float angle = float(startPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            // LLaMA-style interleaved: pairs are (2j, 2j+1)
            int idx0 = j * 2;
            int idx1 = j * 2 + 1;

            float q0 = float(q[qOffset + idx0]);
            float q1 = float(q[qOffset + idx1]);
            q[qOffset + idx0] = half(q0 * cos_val - q1 * sin_val);
            q[qOffset + idx1] = half(q0 * sin_val + q1 * cos_val);
        }
    }

    // 2. Apply RoPE to K head and scatter to K cache (if within KV head range)
    if (head < numKVHeads) {
        int srcKOffset = head * headDim;  // M=1: (0 * numKVHeads + head) * headDim
        int dstKOffset = head * maxSeqLen * headDim + seqPos * headDim;

        for (int j = 0; j < halfRopeDim; j++) {
            float freq = 1.0f / pow(theta, float(2 * j) / float(ropeDim));
            float angle = float(startPos) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);

            int idx0 = j * 2;
            int idx1 = j * 2 + 1;

            float k0 = float(srcK[srcKOffset + idx0]);
            float k1 = float(srcK[srcKOffset + idx1]);
            dstK[dstKOffset + idx0] = half(k0 * cos_val - k1 * sin_val);
            dstK[dstKOffset + idx1] = half(k0 * sin_val + k1 * cos_val);
        }
        // Copy non-rotated dims (if ropeDim < headDim) with RoPE applied (pass-through)
        for (int j = ropeDim; j < headDim; j++) {
            dstK[dstKOffset + j] = srcK[srcKOffset + j];
        }

        // 3. Scatter V to cache (no RoPE)
        int srcVOffset = head * headDim;
        int dstVOffset = head * maxSeqLen * headDim + seqPos * headDim;
        for (int j = 0; j < headDim; j++) {
            dstV[dstVOffset + j] = srcV[srcVOffset + j];
        }
    }
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

// =============================================================================
// Chunk-based Flash Attention SDPA decode for FP16 KV cache (v3)
//
// Key improvements over v1 (sdpa_flash_decode_f16):
//   1. Chunk-based processing: Q·K and V phases separated within C-position chunks,
//      enabling instruction-level parallelism (ILP) — 32 independent loads in flight.
//   2. Vectorized half4 loads with dot() intrinsic for Q·K computation.
//   3. Scores stored in registers (C per simdgroup) for fast softmax.
//
// Design follows llama.cpp's kernel_flash_attn_ext_vec pattern:
//   - 1 threadgroup per Q head, 8 simdgroups per TG
//   - KV positions split across simdgroups in strides (not contiguous chunks)
//   - Each simdgroup processes positions: SG_id, SG_id+8, SG_id+16, ...
//   - Within each SG, positions processed in chunks of C=32
//   - Phase 1: compute Q·K for C positions (vectorized, high ILP)
//   - Phase 2: online softmax update for the chunk
//   - Phase 3: accumulate weighted V for C positions
//
// For headDim=128: each thread loads float4 (4 half elements → 4 float), 32 threads
// cover the full 128-dim vector. simd_sum reduces to full dot product.
// =============================================================================
constant int FLASH_V3_THREADS = 256;
constant int FLASH_V3_NUM_SG = 8;
constant int FLASH_V3_CHUNK = 32;   // positions per chunk

kernel void sdpa_flash_decode_f16_v3(
    device const half* Q [[buffer(0)]],         // [numQHeads, headDim] in FP16
    device const half* K [[buffer(1)]],         // [numKVHeads, maxSeqLen, headDim] in FP16
    device const half* V [[buffer(2)]],         // [numKVHeads, maxSeqLen, headDim] in FP16
    device half* out [[buffer(3)]],             // [numQHeads, headDim] in FP16
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant int& kvHeadStride [[buffer(9)]],
    constant int& activeSGs [[buffer(10)]],     // actual launched simdgroups (avoids reading uninitialized shared mem)
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

    int elemsPerThread = headDim / 32;
    int threadBase = simd_lane * elemsPerThread;

    // Load Q into registers as float4 (assumes headDim % 128 == 0 for 4 elements/thread)
    // For headDim=128: elemsPerThread=4, one float4 per thread
    float4 q_f4;
    if (elemsPerThread == 4) {
        half4 q_h4 = *(device const half4*)(Q + qHead * headDim + threadBase);
        q_f4 = float4(q_h4);
    }
    // For larger headDim, fall back to scalar
    float q_reg[8];
    if (elemsPerThread != 4) {
        for (int e = 0; e < elemsPerThread; e++) {
            q_reg[e] = float(Q[qHead * headDim + threadBase + e]);
        }
    }

    // Per-simdgroup online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[8];
    for (int e = 0; e < 8; e++) o_i[e] = 0.0f;

    // Each SG handles a contiguous range. activeSGs tells us how many SGs were launched.
    int chunkSize = (kvLen + activeSGs - 1) / activeSGs;
    int startPos = simd_group * chunkSize;
    int endPos = min(startPos + chunkSize, kvLen);

    // Scores buffer in registers for the current chunk
    float scores[FLASH_V3_CHUNK];

    // Process positions in chunks of C=32
    for (int chunkStart = startPos; chunkStart < endPos; chunkStart += FLASH_V3_CHUNK) {
        int chunkEnd = min(chunkStart + FLASH_V3_CHUNK, endPos);
        int chunkLen = chunkEnd - chunkStart;

        // ---- Phase 1: Compute Q·K for all positions in this chunk ----
        for (int c = 0; c < chunkLen; c++) {
            int pos = chunkStart + c;
            int kBase = kvBase + pos * headDim;
            float dot_val;
            if (elemsPerThread == 4) {
                half4 k_h4 = *(device const half4*)(K + kBase + threadBase);
                dot_val = dot(q_f4, float4(k_h4));
            } else {
                dot_val = 0.0f;
                for (int e = 0; e < elemsPerThread; e++) {
                    dot_val += q_reg[e] * float(K[kBase + threadBase + e]);
                }
            }
            scores[c] = simd_sum(dot_val) * scale;
        }

        // ---- Phase 2: Online softmax for this chunk ----
        // Find max across the chunk
        float chunk_max = -INFINITY;
        for (int c = 0; c < chunkLen; c++) {
            chunk_max = max(chunk_max, scores[c]);
        }

        float m_new = max(m_i, chunk_max);
        float alpha = exp(m_i - m_new);

        // Compute exp(score - m_new) and sum
        float chunk_sum = 0.0f;
        for (int c = 0; c < chunkLen; c++) {
            scores[c] = exp(scores[c] - m_new);
            chunk_sum += scores[c];
        }

        l_i = l_i * alpha + chunk_sum;

        // Rescale old accumulator
        for (int e = 0; e < elemsPerThread; e++) {
            o_i[e] *= alpha;
        }

        // ---- Phase 3: Weighted V accumulation ----
        for (int c = 0; c < chunkLen; c++) {
            int pos = chunkStart + c;
            int vBase = kvBase + pos * headDim;
            float w = scores[c];
            if (elemsPerThread == 4) {
                half4 v_h4 = *(device const half4*)(V + vBase + threadBase);
                float4 vf4 = float4(v_h4);
                o_i[0] += w * vf4[0];
                o_i[1] += w * vf4[1];
                o_i[2] += w * vf4[2];
                o_i[3] += w * vf4[3];
            } else {
                for (int e = 0; e < elemsPerThread; e++) {
                    o_i[e] += w * float(V[vBase + threadBase + e]);
                }
            }
        }

        m_i = m_new;
    }

    // ---- Cross-simdgroup merge (same as v1) ----
    // ---- Cross-simdgroup merge (only over activeSGs launched SGs) ----
    threadgroup float* sg_maxs = shared;
    threadgroup float* sg_sums = shared + activeSGs;
    threadgroup float* sg_accs = shared + 2 * activeSGs;

    if (simd_lane == 0) {
        sg_maxs[simd_group] = m_i;
        sg_sums[simd_group] = l_i;
    }
    for (int e = 0; e < elemsPerThread; e++) {
        sg_accs[simd_group * headDim + threadBase + e] = o_i[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float global_max = sg_maxs[0];
        for (int s = 1; s < activeSGs; s++) {
            global_max = max(global_max, sg_maxs[s]);
        }

        float total_sum = 0.0f;
        float result[8];
        for (int e = 0; e < 8; e++) result[e] = 0.0f;

        for (int s = 0; s < activeSGs; s++) {
            float w = (sg_sums[s] > 0.0f) ? exp(sg_maxs[s] - global_max) : 0.0f;
            total_sum += sg_sums[s] * w;
            for (int e = 0; e < elemsPerThread; e++) {
                result[e] += sg_accs[s * headDim + threadBase + e] * w;
            }
        }

        float inv_l = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
        int outBase = qHead * headDim;
        for (int e = 0; e < elemsPerThread; e++) {
            out[outBase + threadBase + e] = half(result[e] * inv_l);
        }
    }
}

// =============================================================================
// Multi-Threadgroup (NWG) Flash Attention SDPA decode for FP16 KV cache
//
// SINGLE kernel with N threadgroups per Q head (N = ceil(kvLen / NWG_KV_PER_TG)).
// Each TG processes NWG_KV_PER_TG KV positions using v3's chunk algorithm.
// Coordination via atomic counter — last TG to finish merges all partials.
//
// Key difference from tiled (which was SLOWER due to merge dispatch overhead):
//   - Single dispatch: no separate merge kernel
//   - Atomic coordination: last TG merges in-kernel
//   - Same thread config as v3: 256 threads, 8 simdgroups
//
// Grid: (numQHeads * numTGs, 1, 1)
// Partials buffer: [numQHeads * numTGs * (2 + headDim)] floats
// Counter buffer: [numQHeads] uint32 atomics (zeroed before dispatch)
//
// Memory layout per TG partial:
//   [max_i, sum_i, acc[0], acc[1], ..., acc[headDim-1]]
// =============================================================================
constant int NWG_THREADS = 256;
constant int NWG_NUM_SG = 8;
constant int NWG_CHUNK = 32;       // positions per chunk (same as v3)
constant int NWG_KV_PER_TG = 256;  // KV positions per threadgroup

kernel void sdpa_flash_decode_f16_nwg(
    device const half* Q [[buffer(0)]],         // [numQHeads, headDim] in FP16
    device const half* K [[buffer(1)]],         // [numKVHeads, maxSeqLen, headDim] in FP16
    device const half* V [[buffer(2)]],         // [numKVHeads, maxSeqLen, headDim] in FP16
    device half* out [[buffer(3)]],             // [numQHeads, headDim] in FP16
    device float* partials [[buffer(4)]],       // [numQHeads * numTGs, 2 + headDim] temp
    device atomic_uint* counters [[buffer(5)]], // [numQHeads] atomic counters
    constant int& kvLen [[buffer(6)]],
    constant int& numQHeads [[buffer(7)]],
    constant int& numKVHeads [[buffer(8)]],
    constant int& headDim [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant int& kvHeadStride [[buffer(11)]],
    constant int& numTGs [[buffer(12)]],        // threadgroups per Q head
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Map global TG index to (Q head, TG within head)
    int qHead = gid / numTGs;
    int tgIdx = gid % numTGs;
    if (qHead >= numQHeads) return;

    // GQA mapping
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;
    int kvBase = kvHead * kvHeadStride;

    int elemsPerThread = headDim / 32;
    int threadBase = simd_lane * elemsPerThread;

    // This TG handles positions [tgStart, tgEnd)
    int tgStart = tgIdx * NWG_KV_PER_TG;
    int tgEnd = min(tgStart + NWG_KV_PER_TG, kvLen);
    if (tgStart >= kvLen) {
        // This TG has no work. Still need to participate in atomic counter.
        if (tid == 0) {
            // Write empty partial
            int partialBase = gid * (2 + headDim);
            partials[partialBase + 0] = -INFINITY; // max
            partials[partialBase + 1] = 0.0f;      // sum
            for (int e = 0; e < headDim; e++) {
                partials[partialBase + 2 + e] = 0.0f;
            }
            // Signal completion
            atomic_fetch_add_explicit(&counters[qHead], 1, memory_order_relaxed);
        }
        // Wait for merge TG if we're the last
        // Actually, empty TGs should NOT merge. Just return after signaling.
        return;
    }

    // Load Q into registers (same as v3)
    float4 q_f4;
    if (elemsPerThread == 4) {
        half4 q_h4 = *(device const half4*)(Q + qHead * headDim + threadBase);
        q_f4 = float4(q_h4);
    }
    float q_reg[8];
    if (elemsPerThread != 4) {
        for (int e = 0; e < elemsPerThread; e++) {
            q_reg[e] = float(Q[qHead * headDim + threadBase + e]);
        }
    }

    // Per-simdgroup online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[8];
    for (int e = 0; e < 8; e++) o_i[e] = 0.0f;

    // Split TG's KV range across 8 simdgroups
    int sgKVLen = tgEnd - tgStart;
    int sgChunkSize = (sgKVLen + NWG_NUM_SG - 1) / NWG_NUM_SG;
    int sgStart = tgStart + simd_group * sgChunkSize;
    int sgEnd = min(sgStart + sgChunkSize, tgEnd);

    float scores[NWG_CHUNK];

    // Process positions in chunks of 32 (same as v3)
    for (int chunkStart = sgStart; chunkStart < sgEnd; chunkStart += NWG_CHUNK) {
        int chunkEnd = min(chunkStart + NWG_CHUNK, sgEnd);
        int chunkLen = chunkEnd - chunkStart;

        // Phase 1: Q·K
        for (int c = 0; c < chunkLen; c++) {
            int pos = chunkStart + c;
            int kBase = kvBase + pos * headDim;
            float dot_val;
            if (elemsPerThread == 4) {
                half4 k_h4 = *(device const half4*)(K + kBase + threadBase);
                dot_val = dot(q_f4, float4(k_h4));
            } else {
                dot_val = 0.0f;
                for (int e = 0; e < elemsPerThread; e++) {
                    dot_val += q_reg[e] * float(K[kBase + threadBase + e]);
                }
            }
            scores[c] = simd_sum(dot_val) * scale;
        }

        // Phase 2: online softmax
        float chunk_max = -INFINITY;
        for (int c = 0; c < chunkLen; c++) {
            chunk_max = max(chunk_max, scores[c]);
        }
        float m_new = max(m_i, chunk_max);
        float alpha = exp(m_i - m_new);
        float chunk_sum = 0.0f;
        for (int c = 0; c < chunkLen; c++) {
            scores[c] = exp(scores[c] - m_new);
            chunk_sum += scores[c];
        }
        l_i = l_i * alpha + chunk_sum;
        for (int e = 0; e < elemsPerThread; e++) {
            o_i[e] *= alpha;
        }

        // Phase 3: V accumulation
        for (int c = 0; c < chunkLen; c++) {
            int pos = chunkStart + c;
            int vBase = kvBase + pos * headDim;
            float w = scores[c];
            if (elemsPerThread == 4) {
                half4 v_h4 = *(device const half4*)(V + vBase + threadBase);
                float4 vf4 = float4(v_h4);
                o_i[0] += w * vf4[0];
                o_i[1] += w * vf4[1];
                o_i[2] += w * vf4[2];
                o_i[3] += w * vf4[3];
            } else {
                for (int e = 0; e < elemsPerThread; e++) {
                    o_i[e] += w * float(V[vBase + threadBase + e]);
                }
            }
        }
        m_i = m_new;
    }

    // ---- Cross-simdgroup merge within this TG (same as v3) ----
    threadgroup float* sg_maxs = shared;
    threadgroup float* sg_sums = shared + NWG_NUM_SG;
    threadgroup float* sg_accs = shared + 2 * NWG_NUM_SG;

    if (simd_lane == 0) {
        sg_maxs[simd_group] = m_i;
        sg_sums[simd_group] = l_i;
    }
    for (int e = 0; e < elemsPerThread; e++) {
        sg_accs[simd_group * headDim + threadBase + e] = o_i[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Only simdgroup 0 does the intra-TG merge
    if (simd_group == 0) {
        float tg_max = sg_maxs[0];
        for (int s = 1; s < NWG_NUM_SG; s++) {
            tg_max = max(tg_max, sg_maxs[s]);
        }

        float tg_sum = 0.0f;
        float tg_result[8];
        for (int e = 0; e < 8; e++) tg_result[e] = 0.0f;

        for (int s = 0; s < NWG_NUM_SG; s++) {
            float w = (sg_sums[s] > 0.0f) ? exp(sg_maxs[s] - tg_max) : 0.0f;
            tg_sum += sg_sums[s] * w;
            for (int e = 0; e < elemsPerThread; e++) {
                tg_result[e] += sg_accs[s * headDim + threadBase + e] * w;
            }
        }

        // If single TG (numTGs == 1), write output directly
        if (numTGs == 1) {
            float inv_l = (tg_sum > 0.0f) ? (1.0f / tg_sum) : 0.0f;
            int outBase = qHead * headDim;
            for (int e = 0; e < elemsPerThread; e++) {
                out[outBase + threadBase + e] = half(tg_result[e] * inv_l);
            }
            return;
        }

        // Multi-TG: write partial to device memory
        int partialBase = gid * (2 + headDim);
        if (simd_lane == 0) {
            partials[partialBase + 0] = tg_max;
            partials[partialBase + 1] = tg_sum;
        }
        for (int e = 0; e < elemsPerThread; e++) {
            partials[partialBase + 2 + threadBase + e] = tg_result[e];
        }

        // Memory fence: ensure partial writes are visible before counter increment
        threadgroup_barrier(mem_flags::mem_device);

        // Atomic increment: last TG to arrive does the cross-TG merge
        if (simd_lane == 0) {
            uint completed = atomic_fetch_add_explicit(&counters[qHead], 1, memory_order_relaxed);

            // Store in shared so all threads in SG0 can see it
            sg_maxs[0] = as_type<float>(completed);
        }

        // Broadcast completed count to all threads in simdgroup 0
        uint completed = as_type<uint>(sg_maxs[0]);

        if (completed == uint(numTGs - 1)) {
            // We are the last TG — merge all partials
            // First, device memory fence to see all other TGs' writes
            // (atomic_fetch_add with relaxed ordering doesn't guarantee visibility)
            threadgroup_barrier(mem_flags::mem_device);

            float global_max = -INFINITY;
            int baseIdx = qHead * numTGs;

            // Find global max across all TG partials
            // Each thread in the simdgroup reads different partials for parallelism
            for (int t = 0; t < numTGs; t++) {
                int pBase = (baseIdx + t) * (2 + headDim);
                float t_max = partials[pBase + 0];
                global_max = max(global_max, t_max);
            }

            // Merge with online softmax correction
            float total_sum = 0.0f;
            float final_result[8];
            for (int e = 0; e < 8; e++) final_result[e] = 0.0f;

            for (int t = 0; t < numTGs; t++) {
                int pBase = (baseIdx + t) * (2 + headDim);
                float t_max = partials[pBase + 0];
                float t_sum = partials[pBase + 1];
                float w = (t_sum > 0.0f) ? exp(t_max - global_max) : 0.0f;
                total_sum += t_sum * w;
                for (int e = 0; e < elemsPerThread; e++) {
                    final_result[e] += partials[pBase + 2 + threadBase + e] * w;
                }
            }

            // Normalize and write output
            float inv_l = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
            int outBase = qHead * headDim;
            for (int e = 0; e < elemsPerThread; e++) {
                out[outBase + threadBase + e] = half(final_result[e] * inv_l);
            }

            // Self-reset: zero counter for next dispatch (eliminates external Zero kernel).
            // Safe because this TG is the last to finish — no other TG for this qHead is running.
            // The next dispatch for this qHead will see counter=0 due to memory barrier between dispatches.
            if (simd_lane == 0) {
                atomic_store_explicit(&counters[qHead], 0, memory_order_relaxed);
            }
        }
    }
}

// =============================================================================
// Tiled Flash Attention SDPA decode for FP16 KV cache (Split-K approach)
//
// Two-kernel design for better GPU occupancy at long context lengths:
//   1. sdpa_flash_decode_f16_tiled: Grid = (numQHeads, numTiles)
//      Each TG handles ONE Q head × ONE tile of KV positions.
//      All 256 threads cooperatively load K/V tiles into shared memory.
//      8 simdgroups split the tile positions with online softmax.
//      Writes per-TG partial (max, sum, accumulator) to temp buffer.
//
//   2. sdpa_flash_decode_f16_merge: Grid = numQHeads
//      Merges numTiles partials per Q head using online softmax correction.
//      Writes final output.
//
// At ctx=512 with TILE_KV=64: grid = 32×8 = 256 TGs vs current 32 TGs (8× more).
// Cooperative tile loading enables sequential burst reads (better memory pipelining)
// and shared memory access (~1 cycle vs ~200ns device latency).
// =============================================================================
constant int TILED_F16_THREADS = 256;  // 8 simdgroups × 32 threads
constant int TILED_F16_NUM_SG = 8;
constant int TILED_F16_TILE_KV = 64;  // KV positions per tile

kernel void sdpa_flash_decode_f16_tiled(
    device const half* Q [[buffer(0)]],             // [numQHeads, headDim] in FP16
    device const half* K [[buffer(1)]],             // [numKVHeads, maxSeqLen, headDim] in FP16
    device const half* V [[buffer(2)]],             // [numKVHeads, maxSeqLen, headDim] in FP16
    device float* partials [[buffer(3)]],           // [numQHeads, numTiles, 2 + headDim] temp buffer
    constant int& kvLen [[buffer(4)]],
    constant int& numQHeads [[buffer(5)]],
    constant int& numKVHeads [[buffer(6)]],
    constant int& headDim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant int& kvHeadStride [[buffer(9)]],       // Stride between KV heads
    constant int& numTiles [[buffer(10)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int qHead = gid.x;
    int tileIdx = gid.y;
    if (qHead >= numQHeads || tileIdx >= numTiles) return;

    // GQA: map Q head to KV head
    int headsPerKV = numQHeads / numKVHeads;
    int kvHead = qHead / headsPerKV;
    int kvBase = kvHead * kvHeadStride;

    // Tile boundaries
    int tileStart = tileIdx * TILED_F16_TILE_KV;
    int tileEnd = min(tileStart + TILED_F16_TILE_KV, kvLen);
    int actualTileSize = tileEnd - tileStart;
    if (actualTileSize <= 0) return;

    int elemsPerThread = headDim / 32;
    int threadBase = simd_lane * elemsPerThread;

    // Load Q into registers
    float q_reg[8];  // supports up to headDim=256
    for (int e = 0; e < elemsPerThread; e++) {
        q_reg[e] = float(Q[qHead * headDim + threadBase + e]);
    }

    // Shared memory layout:
    //   tile_data: [TILE_KV, headDim] in half (reused for K and V)
    //   sg_merge:  [2*NUM_SG + NUM_SG*headDim] in float (for cross-SG merge)
    threadgroup half* tile_data = (threadgroup half*)shared;
    int tileFloats = (TILED_F16_TILE_KV * headDim + 1) / 2;  // half → float slots
    threadgroup float* sg_merge = shared + tileFloats;

    // Split tile positions across simdgroups
    int sgChunk = (actualTileSize + TILED_F16_NUM_SG - 1) / TILED_F16_NUM_SG;
    int sgStart = simd_group * sgChunk;
    int sgEnd = min(sgStart + sgChunk, actualTileSize);

    // Per-SG online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[8];
    for (int e = 0; e < 8; e++) o_i[e] = 0.0f;

    // === Phase 1: Cooperative K tile load + Q·K scores ===
    int loadElems = actualTileSize * headDim;
    for (int idx = (int)tid; idx < loadElems; idx += TILED_F16_THREADS) {
        int localPos = idx / headDim;
        int localDim = idx % headDim;
        tile_data[idx] = K[kvBase + (tileStart + localPos) * headDim + localDim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute Q·K scores from shared memory, store in registers
    float scores[8];  // max TILE_KV/NUM_SG = 64/8 = 8 scores per SG
    int numScores = sgEnd - sgStart;
    for (int s = 0; s < numScores; s++) {
        int localPos = sgStart + s;
        float partial = 0.0f;
        for (int e = 0; e < elemsPerThread; e++) {
            partial += q_reg[e] * float(tile_data[localPos * headDim + threadBase + e]);
        }
        scores[s] = simd_sum(partial) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 2: Cooperative V tile load + weighted accumulation ===
    for (int idx = (int)tid; idx < loadElems; idx += TILED_F16_THREADS) {
        int localPos = idx / headDim;
        int localDim = idx % headDim;
        tile_data[idx] = V[kvBase + (tileStart + localPos) * headDim + localDim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax + V accumulation from shared memory
    for (int s = 0; s < numScores; s++) {
        float score = scores[s];
        float m_new = max(m_i, score);
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);
        l_i = l_i * alpha + p;

        int localPos = sgStart + s;
        for (int e = 0; e < elemsPerThread; e++) {
            o_i[e] = o_i[e] * alpha + p * float(tile_data[localPos * headDim + threadBase + e]);
        }
        m_i = m_new;
    }

    // === Cross-SG merge within this tile ===
    threadgroup float* sg_maxs = sg_merge;
    threadgroup float* sg_sums = sg_merge + TILED_F16_NUM_SG;
    threadgroup float* sg_accs = sg_merge + 2 * TILED_F16_NUM_SG;

    if (simd_lane == 0) {
        sg_maxs[simd_group] = m_i;
        sg_sums[simd_group] = l_i;
    }
    for (int e = 0; e < elemsPerThread; e++) {
        sg_accs[simd_group * headDim + threadBase + e] = o_i[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG0 merges and writes tile partial to temp buffer
    if (simd_group == 0) {
        float global_max = sg_maxs[0];
        for (int s = 1; s < TILED_F16_NUM_SG; s++) {
            global_max = max(global_max, sg_maxs[s]);
        }

        float total_sum = 0.0f;
        float result[8];
        for (int e = 0; e < 8; e++) result[e] = 0.0f;

        for (int s = 0; s < TILED_F16_NUM_SG; s++) {
            float w = (sg_sums[s] > 0.0f) ? exp(sg_maxs[s] - global_max) : 0.0f;
            total_sum += sg_sums[s] * w;
            for (int e = 0; e < elemsPerThread; e++) {
                result[e] += sg_accs[s * headDim + threadBase + e] * w;
            }
        }

        // Write partial: [max, sum, acc[0..headDim-1]]
        int partialStride = 2 + headDim;
        int partialBase = (qHead * numTiles + tileIdx) * partialStride;

        if (simd_lane == 0) {
            partials[partialBase + 0] = global_max;
            partials[partialBase + 1] = total_sum;
        }
        for (int e = 0; e < elemsPerThread; e++) {
            partials[partialBase + 2 + threadBase + e] = result[e];
        }
    }
}

// Merge kernel: combines tile partials per Q head using online softmax correction.
// Grid: numQHeads threadgroups, 256 threads each.
// Only simdgroup 0 (32 threads) does active work.
kernel void sdpa_flash_decode_f16_merge(
    device const float* partials [[buffer(0)]],    // [numQHeads, numTiles, 2 + headDim]
    device half* out [[buffer(1)]],                // [numQHeads, headDim] in FP16
    constant int& numQHeads [[buffer(2)]],
    constant int& headDim [[buffer(3)]],
    constant int& numTiles [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int qHead = gid;
    if (qHead >= numQHeads || simd_group != 0) return;

    int elemsPerThread = headDim / 32;
    int threadBase = simd_lane * elemsPerThread;
    int partialStride = 2 + headDim;

    // Find global max across all tiles
    float global_max = -INFINITY;
    for (int t = 0; t < numTiles; t++) {
        int base = (qHead * numTiles + t) * partialStride;
        float tile_max = partials[base + 0];
        // Only consider tiles that had actual data (sum > 0)
        if (partials[base + 1] > 0.0f) {
            global_max = max(global_max, tile_max);
        }
    }

    // Merge with online softmax correction
    float total_sum = 0.0f;
    float result[8];
    for (int e = 0; e < 8; e++) result[e] = 0.0f;

    for (int t = 0; t < numTiles; t++) {
        int base = (qHead * numTiles + t) * partialStride;
        float tile_sum = partials[base + 1];
        if (tile_sum <= 0.0f) continue;  // empty tile

        float tile_max = partials[base + 0];
        float w = exp(tile_max - global_max);
        total_sum += tile_sum * w;

        for (int e = 0; e < elemsPerThread; e++) {
            result[e] += partials[base + 2 + threadBase + e] * w;
        }
    }

    // Normalize and write output
    float inv_l = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    int outBase = qHead * headDim;
    for (int e = 0; e < elemsPerThread; e++) {
        out[outBase + threadBase + e] = half(result[e] * inv_l);
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

// Flash Attention F32 v2 decode: genuine split-KV with online softmax.
// Replaces sdpa_flash_decode_f32 which materialized weights[kvLen] in shared memory
// and used a sequential two-phase V accumulation that degraded with context length.
//
// Algorithm: same as sdpa_flash_decode_f16 but with float* I/O.
// - Split KV positions across 8 simdgroups
// - Each simdgroup processes its chunk with online softmax (running max + sum)
// - Cross-simdgroup merge via shared memory
// - Shared memory: O(headDim) instead of O(kvLen)
//
// 256 threads = 8 simdgroups × 32 lanes.
// Each simdgroup: 32 lanes handle headDim/32 elements each.
constant int FLASH_F32_V2_NUM_SG = 8;

kernel void sdpa_flash_decode_f32_v2(
    device const float* Q [[buffer(0)]],        // [numQHeads, headDim]
    device const float* K [[buffer(1)]],        // [numKVHeads, maxSeqLen, headDim]
    device const float* V [[buffer(2)]],        // [numKVHeads, maxSeqLen, headDim]
    device float* out [[buffer(3)]],            // [numQHeads, headDim]
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
        q_reg[e] = Q[qHead * headDim + threadBase + e];
    }

    // Per-simdgroup online softmax state
    float m_i = -INFINITY;    // running max
    float l_i = 0.0f;         // running sum of exp(score - m_i)
    float o_i[8];              // weighted V accumulator
    for (int e = 0; e < 8; e++) o_i[e] = 0.0f;

    // Split KV positions across simdgroups for parallel processing
    int chunkSize = (kvLen + FLASH_F32_V2_NUM_SG - 1) / FLASH_F32_V2_NUM_SG;
    int startPos = simd_group * chunkSize;
    int endPos = min(startPos + chunkSize, kvLen);

    // Process this simdgroup's KV chunk with online softmax
    for (int pos = startPos; pos < endPos; pos++) {
        int kBase = kvBase + pos * headDim;

        // Q·K dot product: each thread computes partial, simd_sum reduces
        float partial = 0.0f;
        for (int e = 0; e < elemsPerThread; e++) {
            partial += q_reg[e] * K[kBase + threadBase + e];
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
            o_i[e] = o_i[e] * alpha + p * V[vBase + threadBase + e];
        }

        m_i = m_new;
    }

    // ---- Cross-simdgroup merge ----
    // Shared memory layout:
    //   [0 .. NUM_SG-1]:                          per-SG max values
    //   [NUM_SG .. 2*NUM_SG-1]:                   per-SG sum values
    //   [2*NUM_SG .. 2*NUM_SG + NUM_SG*headDim-1]: per-SG output accumulators
    threadgroup float* sg_maxs = shared;
    threadgroup float* sg_sums = shared + FLASH_F32_V2_NUM_SG;
    threadgroup float* sg_accs = shared + 2 * FLASH_F32_V2_NUM_SG;

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
        for (int s = 1; s < FLASH_F32_V2_NUM_SG; s++) {
            global_max = max(global_max, sg_maxs[s]);
        }

        // Merge partial results with online softmax correction
        float total_sum = 0.0f;
        float result[8];
        for (int e = 0; e < 8; e++) result[e] = 0.0f;

        for (int s = 0; s < FLASH_F32_V2_NUM_SG; s++) {
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
            out[outBase + threadBase + e] = result[e] * inv_l;
        }
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
// Deinterleave QKV: split fused [M, qkvDim] output into separate Q, K, V
// =============================================================================
// src: [M, qkvDim] where qkvDim = qDim + 2*kvDim, row-major
// dstQ: [M, qDim], dstK: [M, kvDim], dstV: [M, kvDim]
// Each thread copies one float from the interleaved source to the correct destination.
kernel void deinterleave_qkv_f32(
    device const float* src [[buffer(0)]],
    device float* dstQ [[buffer(1)]],
    device float* dstK [[buffer(2)]],
    device float* dstV [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& qDim [[buffer(5)]],
    constant int& kvDim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    int qkvDim = qDim + 2 * kvDim;
    int totalElements = M * qkvDim;
    if ((int)gid >= totalElements) return;

    int row = (int)gid / qkvDim;
    int col = (int)gid % qkvDim;
    float val = src[gid];

    if (col < qDim) {
        dstQ[row * qDim + col] = val;
    } else if (col < qDim + kvDim) {
        dstK[row * kvDim + (col - qDim)] = val;
    } else {
        dstV[row * kvDim + (col - qDim - kvDim)] = val;
    }
}

// =============================================================================
// Deinterleave 2-way: split fused [M, dim1+dim2] output into separate A, B
// =============================================================================
// src: [M, totalDim] where totalDim = dim1 + dim2, row-major
// dstA: [M, dim1], dstB: [M, dim2]
// Each thread copies one float from the interleaved source to the correct destination.
kernel void deinterleave_2way_f32(
    device const float* src [[buffer(0)]],
    device float* dstA [[buffer(1)]],
    device float* dstB [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& dim1 [[buffer(4)]],
    constant int& dim2 [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    int totalDim = dim1 + dim2;
    int totalElements = M * totalDim;
    if ((int)gid >= totalElements) return;

    int row = (int)gid / totalDim;
    int col = (int)gid % totalDim;
    float val = src[gid];

    if (col < dim1) {
        dstA[row * dim1 + col] = val;
    } else {
        dstB[row * dim2 + (col - dim1)] = val;
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

// Q4_0 v2 matvec: llama.cpp-matched, 8 outputs per threadgroup (4 per SG × 2 SGs)
// Grid: ceil(N/8) threadgroups of 64 threads
void metal_matvec_q4_0_v2_f32(void* queuePtr, void* pipelinePtr,
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

    // 8 outputs per TG (NR0=4 × NSG=2), 64 threads (2 simdgroups × 32)
    int outputsPerTG = 8;  // MV2_NR0 * MV2_NSG
    int threadgroupSize = 64;  // MV2_NSG * 32
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 v2 matvec offset-aware: 8 outputs per TG, 64 threads, fused nibble-masking
void metal_matvec_q4_0_v2_f32_offset(void* queuePtr, void* pipelinePtr,
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

    int outputsPerTG = 8;  // MV2_NR0 * MV2_NSG
    int threadgroupSize = 64;  // MV2_NSG * 32
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

    // Tile sizes from kernel: SMM_TILE_M=32, SMM_TILE_N=64, SMM_TILE_K=32
    int TILE_M = 32;
    int TILE_N = 64;
    int TILE_K = 32;   // 1 Q4_0 block per K-tile
    int threadgroupSize = 128;  // 4 simdgroups of 32 threads

    // Shared memory: blocked 8×8 layout, half-precision A and B tiles
    int sharedMemA = TILE_M * TILE_K * 2;  // 32×32×2 = 2048 bytes (half)
    int sharedMemB = TILE_N * TILE_K * 2;  // 64×32×2 = 4096 bytes (half)

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

// V2: TILE_M=64, TILE_N=32, TILE_K=32, 4 SG × 128 threads, swizzled shared memory
void metal_matmul_q4_0_simdgroup_v2_f32(void* queuePtr, void* pipelinePtr,
                                         void* A, void* B, void* C,
                                         int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int TILE_M = 64;
    int TILE_N = 32;
    int threadgroupSize = 128;  // 4 simdgroups of 32 threads

    // 8×8 block-interleaved shared memory (matches llama.cpp):
    // sa: 32 sub-tiles × 64 halfs = 2048 halfs = 4096 bytes
    // sb: 16 sub-tiles × 64 halfs = 1024 halfs = 2048 bytes
    // Total: 6144 bytes (half of v1 → 2× occupancy)
    int sharedMemTotal = (2048 + 1024) * 2;  // 6144 bytes

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
    [encoder setThreadgroupMemoryLength:sharedMemTotal atIndex:0];

    // 2D grid: tiles in (N, M) dimensions
    int tilesN = (N + TILE_N - 1) / TILE_N;
    int tilesM = (M + TILE_M - 1) / TILE_M;
    MTLSize threadgroups = MTLSizeMake(tilesN, tilesM, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 simdgroup GEMM v3 dispatch (blocked shared memory layout)
void metal_matmul_q4_0_simdgroup_v3_f32(void* queuePtr, void* pipelinePtr,
                                         void* A, void* B, void* C,
                                         int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int TILE_M = 64;
    int TILE_N = 32;
    int threadgroupSize = 128;  // 4 simdgroups

    // Blocked 8×8 shared memory layout (same total as v2):
    // sa: 32 blocks × 64 halves = 4096 bytes
    // sb: 16 blocks × 64 halves = 2048 bytes
    // Total: 6144 bytes
    int sharedMemTotal = 6144;

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
    [encoder setThreadgroupMemoryLength:sharedMemTotal atIndex:0];

    int tilesN = (N + TILE_N - 1) / TILE_N;
    int tilesM = (M + TILE_M - 1) / TILE_M;
    MTLSize threadgroups = MTLSizeMake(tilesN, tilesM, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_0 simdgroup GEMM v3 offset-aware dispatch (blocked shared memory layout)
void metal_matmul_q4_0_simdgroup_v3_f32_offset(void* queuePtr, void* pipelinePtr,
                                                 void* A, uint64_t aOff,
                                                 void* B,
                                                 void* C, uint64_t cOff,
                                                 int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int TILE_M = 64;
    int TILE_N = 32;
    int threadgroupSize = 128;
    int sharedMemTotal = 6144;  // sa: 4096 + sb: 2048

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
    [encoder setThreadgroupMemoryLength:sharedMemTotal atIndex:0];

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

void metal_matvec_q4k_nr4_f32(void* queuePtr, void* pipelinePtr,
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

    // 32 outputs per threadgroup (8 simdgroups * 4)
    int outputsPerTG = 32;
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
    int TILE_K = 32;   // 1 Q4_0 block per K-tile
    int threadgroupSize = 128;  // 4 simdgroups

    // Blocked 8×8 layout, half-precision A and B tiles
    int sharedMemA = TILE_M * TILE_K * 2;  // 32×32×2 = 2048 bytes (half)
    int sharedMemB = TILE_N * TILE_K * 2;  // 64×32×2 = 4096 bytes (half)

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

// Deinterleave QKV: split fused [M, qkvDim] output into separate Q, K, V buffers.
// All pointers use setBuffer:offset: to support scratch-allocated (offset) buffers.
void metal_deinterleave_qkv_f32(void* queuePtr, void* pipelinePtr,
                                 void* srcBuf, uint64_t srcOff,
                                 void* dstQBuf, uint64_t dstQOff,
                                 void* dstKBuf, uint64_t dstKOff,
                                 void* dstVBuf, uint64_t dstVOff,
                                 int M, int qDim, int kvDim) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)srcBuf offset:srcOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstQBuf offset:dstQOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstKBuf offset:dstKOff atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstVBuf offset:dstVOff atIndex:3];
    [encoder setBytes:&M length:sizeof(M) atIndex:4];
    [encoder setBytes:&qDim length:sizeof(qDim) atIndex:5];
    [encoder setBytes:&kvDim length:sizeof(kvDim) atIndex:6];

    int qkvDim = qDim + 2 * kvDim;
    int totalElements = M * qkvDim;
    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    int threadsPerTG = 256;
    int numTGs = (totalElements + threadsPerTG - 1) / threadsPerTG;

    MTLSize threadgroups = MTLSizeMake(numTGs, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadsPerTG, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Deinterleave 2-way: split fused [M, dim1+dim2] output into separate A, B buffers.
// All pointers use setBuffer:offset: to support scratch-allocated (offset) buffers.
void metal_deinterleave_2way_f32(void* queuePtr, void* pipelinePtr,
                                  void* srcBuf, uint64_t srcOff,
                                  void* dstABuf, uint64_t dstAOff,
                                  void* dstBBuf, uint64_t dstBOff,
                                  int M, int dim1, int dim2) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)srcBuf offset:srcOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstABuf offset:dstAOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstBBuf offset:dstBOff atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&dim1 length:sizeof(dim1) atIndex:4];
    [encoder setBytes:&dim2 length:sizeof(dim2) atIndex:5];

    int totalDim = dim1 + dim2;
    int totalElements = M * totalDim;
    int threadsPerTG = 256;
    int numTGs = (totalElements + threadsPerTG - 1) / threadsPerTG;

    MTLSize threadgroups = MTLSizeMake(numTGs, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadsPerTG, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
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

// Fused K+V scatter: scatter both K and V into caches in a single dispatch.
void metal_scatter_kv_f16_fused(void* queuePtr, void* pipelinePtr,
                                void* srcK, void* dstK,
                                void* srcV, void* dstV,
                                int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)srcK offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstK offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)srcV offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstV offset:0 atIndex:3];
    [encoder setBytes:&newTokens length:sizeof(newTokens) atIndex:4];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:5];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:6];
    [encoder setBytes:&maxSeqLen length:sizeof(maxSeqLen) atIndex:7];
    [encoder setBytes:&seqPos length:sizeof(seqPos) atIndex:8];

    int totalElements = newTokens * numKVHeads * headDim;
    int threadgroupSize = 256;
    MTLSize gridSize = MTLSizeMake(totalElements, 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Fused RoPE + KV Scatter for FP16 decode (M=1).
// Replaces separate RoPE + ScatterKV dispatches with a single dispatch.
void metal_rope_scatter_kv_f16(void* queuePtr, void* pipelinePtr,
                                void* q, void* srcK, void* dstK,
                                void* srcV, void* dstV,
                                int numQHeads, int numKVHeads, int headDim,
                                int startPos, int ropeDim, float theta,
                                int maxSeqLen, int seqPos) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)q offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)srcK offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstK offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)srcV offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dstV offset:0 atIndex:4];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:5];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:6];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:7];
    [encoder setBytes:&startPos length:sizeof(startPos) atIndex:8];
    [encoder setBytes:&ropeDim length:sizeof(ropeDim) atIndex:9];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:10];
    [encoder setBytes:&maxSeqLen length:sizeof(maxSeqLen) atIndex:11];
    [encoder setBytes:&seqPos length:sizeof(seqPos) atIndex:12];

    int maxHeads = numQHeads > numKVHeads ? numQHeads : numKVHeads;
    MTLSize gridSize = MTLSizeMake(maxHeads, 1, 1);
    MTLSize tgSize = MTLSizeMake(maxHeads < 64 ? maxHeads : 64, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// MatVec Q4_0 v2 with FP16 activation input, FP32 output.
// Eliminates separate ConvertF16ToF32 dispatch.
void metal_matvec_q4_0_v2_f16in_f32(void* queuePtr, void* pipelinePtr,
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

    // Same dispatch geometry as v2: 64 threads per TG, NR0=4 rows per TG
    int nsg = 2;
    int nr0 = 4;
    int rows_per_tg = nsg * nr0;
    int numThreadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(64, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// MatVec Q4_0 v2 with F16 input and add-to-output.
void metal_matvec_q4_0_v2_f16in_add_f32(void* queuePtr, void* pipelinePtr,
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

    // Same dispatch geometry as v2: 64 threads per TG, NR0=4 rows per TG
    int nsg = 2;
    int nr0 = 4;
    int rows_per_tg = nsg * nr0;
    int numThreadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(64, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// MatVec Q4_0 v2 with add-to-output (fuses W2 + residual add).
void metal_matvec_q4_0_v2_add_f32(void* queuePtr, void* pipelinePtr,
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

    int nsg = 2;
    int nr0 = 4;
    int rows_per_tg = nsg * nr0;
    int numThreadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(64, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// W2+Add2+Residual: C[i] = C[i] + residual[i] + (A @ B^T)[i]
// Adds both the W2 matvec result AND a deferred attention residual to x.
void metal_matvec_q4_0_v2_add2_f32(void* queuePtr, void* pipelinePtr,
                                    void* A, void* B, void* C, void* residual,
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
    [encoder setBuffer:(__bridge id<MTLBuffer>)residual offset:0 atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];

    int nsg = 2;
    int nr0 = 4;
    int rows_per_tg = nsg * nr0;
    int numThreadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(64, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Offset variant for scratch-allocated buffers
void metal_matvec_q4_0_v2_add2_f32_offset(void* queuePtr, void* pipelinePtr,
                                            void* A, uint64_t aOff,
                                            void* B,
                                            void* C, uint64_t cOff,
                                            void* residual, uint64_t residualOff,
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
    [encoder setBuffer:(__bridge id<MTLBuffer>)residual offset:residualOff atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];

    int nsg = 2;
    int nr0 = 4;
    int rows_per_tg = nsg * nr0;
    int numThreadgroups = (N + rows_per_tg - 1) / rows_per_tg;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(64, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
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

// Chunk-based Flash Attention SDPA decode for FP16 KV cache (v3)
// Same signature as v1 but uses chunk-based processing (C=32) with separated Q·K/V phases.
// Higher ILP from 32 independent loads per chunk. Same shared memory layout as v1.
void metal_sdpa_flash_decode_f16_v3(void* queuePtr, void* pipelinePtr,
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

    // 64 threads = 2 simdgroups. Merge loop uses activeSGs to avoid reading garbage.
    int numSG = 2;
    [encoder setBytes:&numSG length:sizeof(numSG) atIndex:10];

    int sharedMemSize = (2 * numSG + numSG * headDim) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(64, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Multi-threadgroup (NWG) Flash Attention SDPA decode for FP16 KV cache.
// Single dispatch with N TGs per Q head. Last TG merges via atomic counter.
void metal_sdpa_flash_decode_f16_nwg(void* queuePtr, void* pipelinePtr,
                                      void* Q, void* K, void* V, void* out,
                                      void* partials, void* counters,
                                      int kvLen, int numQHeads, int numKVHeads, int headDim,
                                      float scale, int kvHeadStride) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    int kvPerTG = 256;  // NWG_KV_PER_TG
    int numTGsPerHead = (kvLen + kvPerTG - 1) / kvPerTG;
    if (numTGsPerHead < 1) numTGsPerHead = 1;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    // Zero the atomic counters before dispatch
    id<MTLBuffer> counterBuf = (__bridge id<MTLBuffer>)counters;
    // Use blit encoder for zeroing — but we're in a compute encoder.
    // Instead, we'll zero from Go before each call, or use a fill.
    // For now, assume counters are zeroed by caller.

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)K offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)V offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)partials offset:0 atIndex:4];
    [encoder setBuffer:counterBuf offset:0 atIndex:5];
    [encoder setBytes:&kvLen length:sizeof(kvLen) atIndex:6];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:7];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:8];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:9];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:10];
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:11];
    [encoder setBytes:&numTGsPerHead length:sizeof(numTGsPerHead) atIndex:12];

    // Shared memory: same as v3 (2*8 + 8*headDim floats)
    int numSG = 8;
    int sharedMemSize = (2 * numSG + numSG * headDim) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // Grid: numQHeads * numTGsPerHead threadgroups
    MTLSize threadgroups = MTLSizeMake(numQHeads * numTGsPerHead, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Tiled Flash Attention SDPA decode for FP16 KV cache (split-K approach).
// Two-dispatch: tile kernel computes per-tile partials, merge kernel combines them.
// Uses batch mode internally to encode both dispatches into a single command buffer.
void metal_sdpa_flash_decode_f16_tiled(void* queuePtr,
                                        void* tilePipelinePtr,
                                        void* mergePipelinePtr,
                                        void* Q, void* K, void* V, void* out,
                                        void* partials,
                                        int kvLen, int numQHeads, int numKVHeads, int headDim,
                                        float scale, int kvHeadStride) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> tilePipeline = (__bridge id<MTLComputePipelineState>)tilePipelinePtr;
    id<MTLComputePipelineState> mergePipeline = (__bridge id<MTLComputePipelineState>)mergePipelinePtr;

    int tileKV = 64;  // TILED_F16_TILE_KV
    int numTiles = (kvLen + tileKV - 1) / tileKV;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    // === Dispatch 1: Tile kernel ===
    [encoder setComputePipelineState:tilePipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)K offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)V offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)partials offset:0 atIndex:3];
    [encoder setBytes:&kvLen length:sizeof(kvLen) atIndex:4];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:5];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:6];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:7];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:8];
    [encoder setBytes:&kvHeadStride length:sizeof(kvHeadStride) atIndex:9];
    [encoder setBytes:&numTiles length:sizeof(numTiles) atIndex:10];

    // Shared memory: tile_data (TILE_KV * headDim halfs = TILE_KV * headDim * 2 bytes)
    //              + sg_merge (2*8 + 8*headDim floats)
    int numSG = 8;
    int tileDataBytes = tileKV * headDim * sizeof(short);  // half = 2 bytes
    int sgMergeBytes = (2 * numSG + numSG * headDim) * sizeof(float);
    int sharedMemSize = tileDataBytes + sgMergeBytes;
    // Align tile data to 4 bytes for float* access to sg_merge
    sharedMemSize = ((sharedMemSize + 3) / 4) * 4;
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // 2D grid: (numQHeads, numTiles)
    MTLSize tileThreadgroups = MTLSizeMake(numQHeads, numTiles, 1);
    MTLSize tileThreadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:tileThreadgroups threadsPerThreadgroup:tileThreadsPerGroup];

    // === Dispatch 2: Merge kernel ===
    [encoder setComputePipelineState:mergePipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)partials offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
    [encoder setBytes:&numQHeads length:sizeof(numQHeads) atIndex:2];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:3];
    [encoder setBytes:&numTiles length:sizeof(numTiles) atIndex:4];

    MTLSize mergeThreadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize mergeThreadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:mergeThreadgroups threadsPerThreadgroup:mergeThreadsPerGroup];

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

// Flash Attention F32 v2: split-KV online softmax, O(headDim) shared memory.
// Replaces the old sdpa_flash_decode_f32 which used O(kvLen) shared memory.
void metal_sdpa_flash_decode_f32_v2(void* queuePtr, void* pipelinePtr,
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

// Flash Attention F32 v2 with offset-aware dispatch (for scratch-allocated buffers).
void metal_sdpa_flash_decode_f32_v2_offset(void* queuePtr, void* pipelinePtr,
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

    int numSG = 8;
    int sharedMemSize = (2 * numSG + numSG * headDim) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

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

    // FP16 shared memory: k halfs for activations + 8 floats for scratch (RMSNorm reduction)
    int sharedMemSize = k * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int numThreadgroups = (n + outputsPerTG - 1) / outputsPerTG;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Fused RMSNorm + QKV FP16: single dispatch for Q, K, V projections.
// Grid: ceil(qDim/32) + 2*ceil(kvDim/32) threadgroups.
void metal_matvec_q4_0_fused_rmsnorm_qkv_f16(void* queuePtr, void* pipelinePtr,
                                               void* x, void* normWeight,
                                               void* Wq, void* Wk, void* Wv,
                                               void* outQ, void* outK, void* outV,
                                               int qDim, int kvDim, int K, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wq offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wk offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wv offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outQ offset:0 atIndex:5];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outK offset:0 atIndex:6];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outV offset:0 atIndex:7];
    [encoder setBytes:&qDim length:sizeof(qDim) atIndex:8];
    [encoder setBytes:&kvDim length:sizeof(kvDim) atIndex:9];
    [encoder setBytes:&K length:sizeof(K) atIndex:10];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:11];

    // FP16 shared memory: K halfs for activations + 8 floats for scratch (RMSNorm reduction)
    int sharedMemSize = K * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int qTGs = (qDim + outputsPerTG - 1) / outputsPerTG;
    int kvTGs = (kvDim + outputsPerTG - 1) / outputsPerTG;
    int numThreadgroups = qTGs + kvTGs + kvTGs;

    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Fused RMSNorm + Q4_0 QKV + RoPE + KV Scatter (FP16 output, decode only)
// Combines FusedQKV + RoPE + ScatterKV into a single dispatch.
// Q: matvec → RoPE → outQ. K: matvec → RoPE → kCache. V: matvec → vCache.
void metal_matvec_q4_0_fused_rmsnorm_qkv_rope_scatter_f16(void* queuePtr, void* pipelinePtr,
                                                            void* x, void* normWeight,
                                                            void* Wq, void* Wk, void* Wv,
                                                            void* outQ, void* kCache, void* vCache,
                                                            int qDim, int kvDim, int K, float eps,
                                                            int headDim, int ropeDim, int startPos,
                                                            float theta, int maxSeqLen, int seqPos) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wq offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wk offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wv offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outQ offset:0 atIndex:5];
    [encoder setBuffer:(__bridge id<MTLBuffer>)kCache offset:0 atIndex:6];
    [encoder setBuffer:(__bridge id<MTLBuffer>)vCache offset:0 atIndex:7];
    [encoder setBytes:&qDim length:sizeof(qDim) atIndex:8];
    [encoder setBytes:&kvDim length:sizeof(kvDim) atIndex:9];
    [encoder setBytes:&K length:sizeof(K) atIndex:10];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:11];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:12];
    [encoder setBytes:&ropeDim length:sizeof(ropeDim) atIndex:13];
    [encoder setBytes:&startPos length:sizeof(startPos) atIndex:14];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:15];
    [encoder setBytes:&maxSeqLen length:sizeof(maxSeqLen) atIndex:16];
    [encoder setBytes:&seqPos length:sizeof(seqPos) atIndex:17];

    int sharedMemSize = K * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int threadgroupSize = 256;
    int qTGs = (qDim + outputsPerTG - 1) / outputsPerTG;
    int kvTGs = (kvDim + outputsPerTG - 1) / outputsPerTG;
    int numThreadgroups = qTGs + kvTGs + kvTGs;

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

// Fused AddRMSNorm + MLP: eliminates 1 dispatch/layer
// Non-offset variant (all buffers at offset 0)
void metal_matvec_q4_0_fused_addrmsnorm_mlp_f32(void* queuePtr, void* pipelinePtr,
                                                  void* x, void* woOutput, void* normWeight,
                                                  void* W1, void* W3, void* out,
                                                  int N, int K, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)woOutput offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W1 offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W3 offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:5];
    [encoder setBytes:&N length:sizeof(N) atIndex:6];
    [encoder setBytes:&K length:sizeof(K) atIndex:7];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:8];

    // FP16 shared memory: K halfs for activations + 8 floats for scratch
    int sharedMemSize = K * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Offset variant for scratch-allocated buffers
void metal_matvec_q4_0_fused_addrmsnorm_mlp_f32_offset(void* queuePtr, void* pipelinePtr,
                                                         void* x, uint64_t xOff,
                                                         void* woOutput, uint64_t woOff,
                                                         void* normWeight,
                                                         void* W1, void* W3,
                                                         void* out, uint64_t outOff,
                                                         int N, int K, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:xOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)woOutput offset:woOff atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W1 offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W3 offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:5];
    [encoder setBytes:&N length:sizeof(N) atIndex:6];
    [encoder setBytes:&K length:sizeof(K) atIndex:7];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:8];

    int sharedMemSize = K * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Fused RMSNorm + MLP gate (no residual add — used when residual is already in x).
void metal_matvec_q4_0_fused_rmsnorm_mlp_f32(void* queuePtr, void* pipelinePtr,
                                               void* x, void* normWeight,
                                               void* W1, void* W3, void* out,
                                               int N, int K, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W1 offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W3 offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:4];
    [encoder setBytes:&N length:sizeof(N) atIndex:5];
    [encoder setBytes:&K length:sizeof(K) atIndex:6];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:7];

    // FP16 shared memory: K halfs for activations + 8 floats for scratch
    int sharedMemSize = K * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Offset variant for scratch-allocated buffers
void metal_matvec_q4_0_fused_rmsnorm_mlp_f32_offset(void* queuePtr, void* pipelinePtr,
                                                      void* x, uint64_t xOff,
                                                      void* normWeight,
                                                      void* W1, void* W3,
                                                      void* out, uint64_t outOff,
                                                      int N, int K, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:xOff atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W1 offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)W3 offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:outOff atIndex:4];
    [encoder setBytes:&N length:sizeof(N) atIndex:5];
    [encoder setBytes:&K length:sizeof(K) atIndex:6];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:7];

    int sharedMemSize = K * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// =============================================================================
// Q4_K FUSED C DISPATCH FUNCTIONS
// =============================================================================

void metal_matvec_q4k_fused_rmsnorm_qkv_f16(void* queuePtr, void* pipelinePtr,
                                              void* x, void* normWeight,
                                              void* Wq, void* Wk, void* Wv,
                                              void* outQ, void* outK, void* outV,
                                              int qDim, int kvDim, int K, float eps) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder = get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)normWeight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wq offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wk offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Wv offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outQ offset:0 atIndex:5];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outK offset:0 atIndex:6];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outV offset:0 atIndex:7];
    [encoder setBytes:&qDim length:sizeof(qDim) atIndex:8];
    [encoder setBytes:&kvDim length:sizeof(kvDim) atIndex:9];
    [encoder setBytes:&K length:sizeof(K) atIndex:10];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:11];

    // FP16 shared memory: K halfs for activations + 8 floats for scratch (RMSNorm reduction)
    int sharedMemSize = K * sizeof(__fp16) + 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    int outputsPerTG_qkv = 32;
    int threadgroupSize = 256;
    int qTGs = (qDim + outputsPerTG_qkv - 1) / outputsPerTG_qkv;
    int kvTGs = (kvDim + outputsPerTG_qkv - 1) / outputsPerTG_qkv;
    int numThreadgroups_qkv = qTGs + kvTGs + kvTGs;

    MTLSize threadgroups_qkv = MTLSizeMake(numThreadgroups_qkv, 1, 1);
    MTLSize threadsPerGroup_qkv = MTLSizeMake(threadgroupSize, 1, 1);

    [encoder dispatchThreadgroups:threadgroups_qkv threadsPerThreadgroup:threadsPerGroup_qkv];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q4k_fused_mlp_f32(void* queuePtr, void* pipelinePtr,
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

    int outputsPerTG_mlp = 32;
    int numThreadgroups_mlp = (N + outputsPerTG_mlp - 1) / outputsPerTG_mlp;
    MTLSize threadgroups_mlp = MTLSizeMake(numThreadgroups_mlp, 1, 1);
    MTLSize threadsPerGroup_mlp = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups_mlp threadsPerThreadgroup:threadsPerGroup_mlp];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q4k_f16in_f32(void* queuePtr, void* pipelinePtr,
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

    // Q4_K: 8 outputs per threadgroup (1 per simdgroup)
    int outputsPerTG_f16 = 8;
    int numThreadgroups_f16 = (N + outputsPerTG_f16 - 1) / outputsPerTG_f16;
    MTLSize threadgroups_f16 = MTLSizeMake(numThreadgroups_f16, 1, 1);
    MTLSize threadsPerGroup_f16 = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups_f16 threadsPerThreadgroup:threadsPerGroup_f16];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

void metal_matvec_q4k_add_f32(void* queuePtr, void* pipelinePtr,
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

    // Q4_K: 8 outputs per threadgroup (1 per simdgroup)
    int outputsPerTG_add = 8;
    int numThreadgroups_add = (N + outputsPerTG_add - 1) / outputsPerTG_add;
    MTLSize threadgroups_add = MTLSizeMake(numThreadgroups_add, 1, 1);
    MTLSize threadsPerGroup_add = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups_add threadsPerThreadgroup:threadsPerGroup_add];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_K NR4 F16-input dispatch: 32 outputs per threadgroup (4 per simdgroup)
void metal_matvec_q4k_nr4_f16in_f32(void* queuePtr, void* pipelinePtr,
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

    // NR4: 32 outputs per threadgroup (4 per simdgroup × 8 simdgroups)
    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}

// Q4_K NR4 Add dispatch: 32 outputs per threadgroup, ADDS to output
void metal_matvec_q4k_nr4_add_f32(void* queuePtr, void* pipelinePtr,
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

    // NR4: 32 outputs per threadgroup (4 per simdgroup × 8 simdgroups)
    int outputsPerTG = 32;
    int numThreadgroups = (N + outputsPerTG - 1) / outputsPerTG;
    MTLSize threadgroups = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}
