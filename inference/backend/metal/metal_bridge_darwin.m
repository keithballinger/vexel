// metal_bridge.m - Objective-C Metal implementation
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
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

// =============================================================================
// Q4_0 TILED MATMUL WITH SIMDGROUP_MATRIX
// =============================================================================
// C[M,N] = A[M,K] @ B[N,K]^T where B is Q4_0 quantized
// Uses simdgroup_matrix operations for 8x8 tiled computation
// Each threadgroup computes a TILE_M x TILE_N output tile
// Threads cooperatively load A and B tiles into threadgroup memory
//
// Tile sizes: TILE_M=32, TILE_N=32, TILE_K=32 (matches Q4_0 block size)
// Threadgroup: 256 threads = 8 simdgroups
// Each simdgroup computes 4 8x8 output tiles (2x2 grid)

constant int SMM_TILE_M = 32;
constant int SMM_TILE_N = 32;
constant int SMM_TILE_K = 32;  // Must match Q4_0 block size

kernel void matmul_q4_0_simdgroup_f32(
    device const float* A [[buffer(0)]],           // [M, K] activations
    device const uchar* B [[buffer(1)]],           // [N, K] in Q4_0 format
    device float* C [[buffer(2)]],                 // [M, N] output
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    threadgroup float* shared_A [[threadgroup(0)]],  // [TILE_M, TILE_K]
    threadgroup float* shared_B [[threadgroup(1)]],  // [TILE_K, TILE_N] (B transposed!)
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Output tile position
    int tile_m = tg_pos.y * SMM_TILE_M;
    int tile_n = tg_pos.x * SMM_TILE_N;

    // Early exit for out-of-bounds tiles
    if (tile_m >= M || tile_n >= N) return;

    // Each simdgroup computes a 16x16 portion of the output tile
    // 8 simdgroups = 2x4 grid of 16x16 tiles = 32x64
    // But we only need 32x32, so use 4 simdgroups for 2x2 grid of 16x16
    int sg_row = (simd_group / 2) * 16;  // 0 or 16
    int sg_col = (simd_group % 2) * 16;  // 0 or 16

    // Only first 4 simdgroups compute output
    bool active_sg = simd_group < 4;

    // Initialize result accumulators (4 8x8 tiles per simdgroup for 16x16)
    simdgroup_float8x8 acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    // Process K dimension in TILE_K chunks
    int numBlocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int numKTiles = (K + SMM_TILE_K - 1) / SMM_TILE_K;

    for (int k_tile = 0; k_tile < numKTiles; k_tile++) {
        int k_base = k_tile * SMM_TILE_K;

        // === Cooperative load of A tile [TILE_M, TILE_K] ===
        // 256 threads load 32*32 = 1024 elements = 4 elements per thread
        for (int i = tid; i < SMM_TILE_M * SMM_TILE_K; i += 256) {
            int local_m = i / SMM_TILE_K;
            int local_k = i % SMM_TILE_K;
            int global_m = tile_m + local_m;
            int global_k = k_base + local_k;

            float val = 0.0f;
            if (global_m < M && global_k < K) {
                val = A[global_m * K + global_k];
            }
            shared_A[local_m * SMM_TILE_K + local_k] = val;
        }

        // === Cooperative load and dequantize B tile, storing TRANSPOSED [TILE_K, TILE_N] ===
        // B is [N, K] in Q4_0 format, we store as B^T[K, N] = shared_B[k, n]
        // This allows simdgroup_multiply_accumulate to compute C += A @ B^T correctly
        int q4_block_idx = k_tile;  // Which Q4_0 block along K dimension

        for (int i = tid; i < SMM_TILE_N * SMM_TILE_K; i += 256) {
            int local_n = i / SMM_TILE_K;
            int local_k = i % SMM_TILE_K;
            int global_n = tile_n + local_n;

            float val = 0.0f;
            if (global_n < N && k_base + local_k < K) {
                // Get Q4_0 block for row global_n, block q4_block_idx
                device const uchar* blockPtr = B + global_n * numBlocks * Q4_BYTES_PER_BLOCK
                                                + q4_block_idx * Q4_BYTES_PER_BLOCK;

                // Read scale
                ushort scale_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
                float scale = q4_f16_to_f32(scale_u16);

                // Dequantize element at local_k within the 32-element block
                // Q4_0 layout: low nibbles at positions 0-15, high nibbles at 16-31
                int byte_idx = local_k < 16 ? local_k : (local_k - 16);
                uchar byte_val = blockPtr[2 + byte_idx];
                int quant_val = local_k < 16 ? (byte_val & 0xF) : (byte_val >> 4);
                val = scale * float(quant_val - 8);
            }
            // Store TRANSPOSED: shared_B[k, n] instead of shared_B[n, k]
            shared_B[local_k * SMM_TILE_N + local_n] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Compute using simdgroup_matrix ===
        if (active_sg) {
            // Process 8-wide K chunks using simdgroup_matrix
            // A is [M, K], stored as shared_A[m, k]
            // B^T is [K, N], stored as shared_B[k, n]
            // simdgroup_multiply_accumulate computes C += A @ B
            // So C[m, n] += A[m, k] @ B^T[k, n] = sum_k(A[m,k] * B[n,k]) ✓
            for (int k = 0; k < SMM_TILE_K; k += 8) {
                simdgroup_float8x8 matA0, matA1, matB0, matB1;

                // Load A tiles: A[sg_row:sg_row+8, k:k+8] and A[sg_row+8:sg_row+16, k:k+8]
                simdgroup_load(matA0, shared_A + (sg_row + 0) * SMM_TILE_K + k, SMM_TILE_K);
                simdgroup_load(matA1, shared_A + (sg_row + 8) * SMM_TILE_K + k, SMM_TILE_K);

                // Load B^T tiles: B^T[k:k+8, sg_col:sg_col+8] and B^T[k:k+8, sg_col+8:sg_col+16]
                simdgroup_load(matB0, shared_B + k * SMM_TILE_N + (sg_col + 0), SMM_TILE_N);
                simdgroup_load(matB1, shared_B + k * SMM_TILE_N + (sg_col + 8), SMM_TILE_N);

                // Multiply-accumulate: C[m,n] += A[m,k] @ B^T[k,n]
                simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
                simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
                simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
                simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store results ===
    if (active_sg) {
        int out_m = tile_m + sg_row;
        int out_n = tile_n + sg_col;

        // Store 16x16 result (4 8x8 tiles)
        if (out_m < M && out_n < N) {
            simdgroup_store(acc00, C + (out_m + 0) * N + (out_n + 0), N);
        }
        if (out_m < M && out_n + 8 < N) {
            simdgroup_store(acc01, C + (out_m + 0) * N + (out_n + 8), N);
        }
        if (out_m + 8 < M && out_n < N) {
            simdgroup_store(acc10, C + (out_m + 8) * N + (out_n + 0), N);
        }
        if (out_m + 8 < M && out_n + 8 < N) {
            simdgroup_store(acc11, C + (out_m + 8) * N + (out_n + 8), N);
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
    // Each simdgroup handles one output
    int output_idx = gid * Q4K_OUTPUTS_PER_TG + simd_group;
    if (output_idx >= N) return;

    float sum = 0.0f;

    // Q4_K row layout
    int numBlocks = (K + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    device const uchar* b_row = B + output_idx * numBlocks * Q4K_BYTES_PER_BLOCK;

    // Each thread processes elements with stride 32 within each block
    for (int block = 0; block < numBlocks; block++) {
        device const uchar* blockPtr = b_row + block * Q4K_BYTES_PER_BLOCK;

        // Parse block header
        ushort d_u16 = ((ushort)blockPtr[1] << 8) | blockPtr[0];
        ushort dmin_u16 = ((ushort)blockPtr[3] << 8) | blockPtr[2];
        float d = q4_f16_to_f32(d_u16);
        float dmin = q4_f16_to_f32(dmin_u16);

        device const uchar* scalesData = blockPtr + 4;
        device const uchar* qs = blockPtr + 16;  // 4-bit quantized values

        // Unpack 6-bit scales and mins from 12 bytes
        // Following llama.cpp layout:
        // Lower 6 bits of scales[0..7] from scalesData[0..7]
        // Upper 2 bits from scalesData[10..11]
        // Mins use bits 6-7 of scalesData[0..7] + bits from scalesData[8..9]
        uchar scales[8];
        uchar mins[8];

        scales[0] = (scalesData[0] & 0x3F);
        scales[1] = (scalesData[1] & 0x3F);
        scales[2] = (scalesData[2] & 0x3F);
        scales[3] = (scalesData[3] & 0x3F);
        scales[4] = (scalesData[4] & 0x3F);
        scales[5] = (scalesData[5] & 0x3F);
        scales[6] = (scalesData[6] & 0x3F);
        scales[7] = (scalesData[7] & 0x3F);

        mins[0] = (scalesData[0] >> 6) | ((scalesData[8] & 0x03) << 2);
        mins[1] = (scalesData[1] >> 6) | ((scalesData[8] & 0x0C) >> 0);
        mins[2] = (scalesData[2] >> 6) | ((scalesData[8] & 0x30) >> 2);
        mins[3] = (scalesData[3] >> 6) | ((scalesData[8] & 0xC0) >> 4);
        mins[4] = (scalesData[4] >> 6) | ((scalesData[9] & 0x03) << 2);
        mins[5] = (scalesData[5] >> 6) | ((scalesData[9] & 0x0C) >> 0);
        mins[6] = (scalesData[6] >> 6) | ((scalesData[9] & 0x30) >> 2);
        mins[7] = (scalesData[7] >> 6) | ((scalesData[9] & 0xC0) >> 4);

        // Additional 2 bits for each scale from bytes 10-11
        scales[0] |= (scalesData[10] & 0x03) << 4;
        scales[1] |= (scalesData[10] & 0x0C) << 2;
        scales[2] |= (scalesData[10] & 0x30);
        scales[3] |= (scalesData[10] & 0xC0) >> 2;
        scales[4] |= (scalesData[11] & 0x03) << 4;
        scales[5] |= (scalesData[11] & 0x0C) << 2;
        scales[6] |= (scalesData[11] & 0x30);
        scales[7] |= (scalesData[11] & 0xC0) >> 2;

        int base_k = block * Q4K_BLOCK_SIZE;

        // Process 8 sub-blocks of 32 elements each
        // Each thread handles elements at stride 32 within the block
        for (int elem_offset = simd_lane; elem_offset < 256 && base_k + elem_offset < K; elem_offset += 32) {
            int k_idx = base_k + elem_offset;
            float a_val = A[k_idx];

            // Determine which sub-block (0-7) and position within sub-block
            int sub_block = elem_offset / 32;
            int pos_in_sub = elem_offset % 32;  // 0-31

            float sc = float(scales[sub_block]);
            float m = float(mins[sub_block]);

            // Get 4-bit quantized value
            // Each sub-block has 16 bytes (32 elements at 4 bits each)
            // Low nibble -> positions 0-15, high nibble -> positions 16-31
            int qs_byte_idx = sub_block * 16 + (pos_in_sub % 16);
            uchar qs_byte = qs[qs_byte_idx];

            int q;
            if (pos_in_sub < 16) {
                q = qs_byte & 0x0F;  // Low nibble
            } else {
                q = (qs_byte >> 4) & 0x0F;  // High nibble
            }

            // Dequantize: d * scale * q - dmin * min
            float dequant = d * sc * float(q) - dmin * m;
            sum += a_val * dequant;
        }
    }

    // Simdgroup reduction
    sum = simd_sum(sum);

    // Lane 0 writes output
    if (simd_lane == 0) {
        C[output_idx] = sum;
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
    // [0..kvLen-1]: attention weights
    // [kvLen..kvLen+7]: warp max/sum values
    threadgroup float* weights = shared;
    threadgroup float* warpVals = shared + kvLen;

    // Phase 1a: Compute Q·K scores (convert FP16 to FP32 for computation)
    float localMax = -INFINITY;
    for (int pos = tid; pos < kvLen; pos += SDPA_F16_THREADS) {
        int kOffset = pos * numKVHeads * headDim + kvHead * headDim;
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
    int outOffset = qHead * headDim;
    for (int d = tid; d < headDim; d += SDPA_F16_THREADS) {
        float sum = 0.0f;
        for (int pos = 0; pos < kvLen; pos++) {
            int vOffset = pos * numKVHeads * headDim + kvHead * headDim;
            sum += weights[pos] * float(V[vOffset + d]);
        }
        out[outOffset + d] = half(sum);
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

// Flash Attention 2 optimized kernel for prefill
// Key optimizations:
// 1. K/V tiles loaded to shared memory, shared across Q heads (GQA)
// 2. Larger tile size (64 K positions vs 16)
// 3. 8 Q heads computed in parallel (one simdgroup each)
// 4. Vectorized loads with float4
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
    if (qHeadLocal >= headsPerKV) return;
    int qHead = qHeadBase + qHeadLocal;

    // Shared memory layout:
    // [0..FA2_TILE_KV*headDim-1]: K tile
    // [FA2_TILE_KV*headDim..2*FA2_TILE_KV*headDim-1]: V tile
    // [2*FA2_TILE_KV*headDim..]: scratch for reductions
    threadgroup float* Ktile = shared;
    threadgroup float* Vtile = shared + FA2_TILE_KV * headDim;
    threadgroup float* scratch = shared + 2 * FA2_TILE_KV * headDim;

    // Q offset for this (qPos, qHead)
    int qOffset = qPos * numQHeads * headDim + qHead * headDim;

    // Causal attention: only attend to positions <= qPos
    int maxKLen = qPos + 1;

    // Online softmax state (per Q head)
    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    // Output accumulator - each thread handles 2 dims (headDim=64, 32 threads/simdgroup)
    float acc0 = 0.0f, acc1 = 0.0f;

    // Process K/V in tiles
    for (int tileStart = 0; tileStart < maxKLen; tileStart += FA2_TILE_KV) {
        int tileEnd = min(tileStart + FA2_TILE_KV, maxKLen);
        int tileSize = tileEnd - tileStart;

        // Cooperative load: all threads load K and V tiles to shared memory
        // Each thread loads (FA2_TILE_KV * headDim) / FA2_THREADS elements
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

        // Each simdgroup computes attention scores for its Q head
        // Threads within simdgroup cooperate on the dot product

        // Phase 1: Compute Q·K scores for all K positions in tile
        // Each thread computes partial dot product, then reduce
        float tileMax = -INFINITY;
        float tileScores[FA2_TILE_KV];  // Store scores for V accumulation

        for (int k = 0; k < tileSize; k++) {
            // Compute Q·K dot product
            float dot = 0.0f;
            // Each thread handles headDim/32 = 2 dimensions
            int d0 = simd_lane * 2;
            int d1 = simd_lane * 2 + 1;
            if (d0 < headDim) dot += Q[qOffset + d0] * Ktile[k * headDim + d0];
            if (d1 < headDim) dot += Q[qOffset + d1] * Ktile[k * headDim + d1];

            // Reduce across simdgroup
            dot = simd_sum(dot);

            float score = dot * scale;
            tileScores[k] = score;
            tileMax = max(tileMax, score);
        }

        // Phase 2: Online softmax update
        float newMax = max(runningMax, tileMax);
        float rescale = exp(runningMax - newMax);

        // Rescale existing accumulator
        acc0 *= rescale;
        acc1 *= rescale;
        runningSum *= rescale;

        // Compute exp(score - newMax) and accumulate V
        float tileSum = 0.0f;
        for (int k = 0; k < tileSize; k++) {
            float expScore = exp(tileScores[k] - newMax);
            tileSum += expScore;

            // Accumulate weighted V
            int d0 = simd_lane * 2;
            int d1 = simd_lane * 2 + 1;
            if (d0 < headDim) acc0 += expScore * Vtile[k * headDim + d0];
            if (d1 < headDim) acc1 += expScore * Vtile[k * headDim + d1];
        }

        runningSum += tileSum;
        runningMax = newMax;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output (each thread writes its 2 dimensions)
    int outOffset = qPos * numQHeads * headDim + qHead * headDim;
    float invSum = 1.0f / runningSum;

    int d0 = simd_lane * 2;
    int d1 = simd_lane * 2 + 1;
    if (d0 < headDim) out[outOffset + d0] = acc0 * invSum;
    if (d1 < headDim) out[outOffset + d1] = acc1 * invSum;
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

void metal_matmul_q4_0_simdgroup_f32(void* queuePtr, void* pipelinePtr,
                                      void* A, void* B, void* C,
                                      int M, int N, int K) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelinePtr;

    // Tile sizes from kernel: SMM_TILE_M=32, SMM_TILE_N=32, SMM_TILE_K=32
    int TILE_M = 32;
    int TILE_N = 32;
    int TILE_K = 32;
    int threadgroupSize = 256;  // 8 simdgroups of 32 threads

    // Shared memory: A tile + B tile (both float)
    int sharedMemA = TILE_M * TILE_K * sizeof(float);  // 32*32*4 = 4096 bytes
    int sharedMemB = TILE_N * TILE_K * sizeof(float);  // 32*32*4 = 4096 bytes

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

    if (shouldCommit) {
        [encoder endEncoding];
        [cmdBuffer commit];
    }
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

// Q4_K multi-output matvec: 8 outputs per threadgroup using simdgroups
// Grid: ceil(N/8) threadgroups of 256 threads
void metal_matvec_q4k_multi_output_f32(void* queuePtr, void* pipelinePtr,
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
    int threadgroupSize = 256;
    int sharedMemSize = (kvLen + 8) * sizeof(float);
    [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

    // One threadgroup per Q head
    MTLSize threadgroups = MTLSizeMake(numQHeads, 1, 1);
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

// Legacy interface (kept for compatibility)
void metal_scaled_dot_product_attention(void* queuePtr,
                                        void* Q, void* K, void* V, void* out,
                                        int batchSize, int numHeads, int seqLen, int headDim,
                                        float scale, int causal) {
    // Deprecated - use metal_sdpa_decode_f32 or metal_sdpa_prefill_f32 instead
}
