// metal_bridge.h - C interface for Metal operations
#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device and queue management
void* metal_create_device(void);
void* metal_create_command_queue(void* device);
const char* metal_device_name(void* device);
void metal_release(void* obj);

// Buffer management
void* metal_alloc_buffer(void* device, size_t size);
size_t metal_buffer_size(void* buffer);
void* metal_buffer_contents(void* buffer);
void metal_copy_to_buffer(void* buffer, const void* src, size_t size);
void metal_copy_from_buffer(void* dst, void* buffer, size_t size);

// GPU-to-GPU buffer copy (uses blit encoder)
void metal_copy_buffer(void* queue, void* srcBuffer, size_t srcOffset,
                       void* dstBuffer, size_t dstOffset, size_t size);
// Batched version - integrates with command batching, no separate commit
void metal_copy_buffer_batched(void* queue, void* srcBuffer, size_t srcOffset,
                               void* dstBuffer, size_t dstOffset, size_t size);

// Shader compilation
void* metal_compile_library(void* device, const char* source);
void* metal_create_pipeline(void* device, void* library, const char* functionName);

// Synchronization
void metal_sync(void* commandQueue);

// Command Buffer Batching - reduce commit overhead by batching operations
void metal_begin_batch(void* commandQueue);
void metal_end_batch(void);
// Memory barrier within a batch — ensures all prior writes are visible to subsequent reads.
// Only effective when in batch mode (no-op otherwise, since non-batch dispatches are serialized).
void metal_memory_barrier(void);

// GPU Profiling - enable with VEXEL_GPU_PROFILE=1
void metal_get_gpu_profile(uint64_t* totalTimeNs, uint64_t* batchCount, uint64_t* kernelCount, uint64_t* syncTimeNs);
void metal_reset_gpu_profile(void);

// Kernel execution
void metal_matmul_f32(void* queue, void* pipeline,
                      void* A, void* B, void* C,
                      int M, int N, int K);

void metal_matmul_transposed_f32(void* queue, void* pipeline,
                                  void* A, void* B, void* C,
                                  int M, int N, int K);

void metal_matvec_transposed_f32(void* queue, void* pipeline,
                                  void* A, void* B, void* C,
                                  int N, int K);

// Q4_0 quantized matrix-vector: C = A @ B^T where B is Q4_0 encoded
void metal_matvec_q4_0_transposed_f32(void* queue, void* pipeline,
                                       void* A, void* B, void* C,
                                       int N, int K);

// Q4_0 multi-output matvec: 8 outputs per threadgroup for better utilization
void metal_matvec_q4_0_multi_output_f32(void* queue, void* pipeline,
                                         void* A, void* B, void* C,
                                         int N, int K);

// Q4_0 v2 matvec: llama.cpp-matched, 64 threads (2 SGs), NR0=4, fused nibble masking
void metal_matvec_q4_0_v2_f32(void* queue, void* pipeline,
                               void* A, void* B, void* C,
                               int N, int K);
void metal_matvec_q4_0_v2_f32_offset(void* queue, void* pipeline,
                                      void* A, uint64_t aOff,
                                      void* B,
                                      void* C, uint64_t cOff,
                                      int N, int K);

// Q4_0 NR2 matvec: 16 outputs per threadgroup (2 per simdgroup, better activation reuse)
void metal_matvec_q4_0_nr2_f32(void* queue, void* pipeline,
                                void* A, void* B, void* C,
                                int N, int K);

// Q4_0 NR4 matvec: 32 outputs per threadgroup (4 per simdgroup, maximum activation reuse)
void metal_matvec_q4_0_nr4_f32(void* queue, void* pipeline,
                                void* A, void* B, void* C,
                                int N, int K);

// Q4_0 collaborative matvec: llama.cpp-style 2 threads per block (better register distribution)
void metal_matvec_q4_0_collab_f32(void* queue, void* pipeline,
                                   void* A, void* B, void* C,
                                   int N, int K);

// Q4_0 optimized matvec: 32 outputs per threadgroup with shared memory activation caching
void metal_matvec_q4_0_optimized_f32(void* queue, void* pipeline,
                                      void* A, void* B, void* C,
                                      int N, int K);

// Batched Q4_0 matmul: C = A @ B^T where A is [M,K], B is [N,K] Q4_0, C is [M,N]
void metal_matmul_q4_0_batched_f32(void* queue, void* pipeline,
                                    void* A, void* B, void* C,
                                    int M, int N, int K);

// Simdgroup tiled Q4_0 matmul: uses simdgroup_matrix for prefill
void metal_matmul_q4_0_simdgroup_f32(void* queue, void* pipeline,
                                      void* A, void* B, void* C,
                                      int M, int N, int K);

// V2: TILE_M=64, TILE_N=32, TILE_K=32, swizzled shared memory, 4 SG × 128 threads
void metal_matmul_q4_0_simdgroup_v2_f32(void* queue, void* pipeline,
                                         void* A, void* B, void* C,
                                         int M, int N, int K);

// V3: TILE_M=64, TILE_N=32, TILE_K=32, blocked 8×8 shared memory (llama.cpp layout)
void metal_matmul_q4_0_simdgroup_v3_f32(void* queue, void* pipeline,
                                         void* A, void* B, void* C,
                                         int M, int N, int K);

// Q6_K multi-output matvec: for lm_head which uses Q6_K quantization
void metal_matvec_q6k_multi_output_f32(void* queue, void* pipeline,
                                        void* A, void* B, void* C,
                                        int N, int K);

// Q6_K NR2 matvec: 2 outputs per simdgroup (llama.cpp style optimization)
void metal_matvec_q6k_nr2_f32(void* queue, void* pipeline,
                               void* A, void* B, void* C,
                               int N, int K);

#include <stdint.h>

// Q4_K multi-output matvec: for attention projections using Q4_K quantization
void metal_matvec_q4k_multi_output_f32(void* queue, void* pipeline,
                                        void* A, uint64_t aOff,
                                        void* B, uint64_t bOff,
                                        void* C, uint64_t cOff,
                                        int N, int K);

// Q4_K NR2 matvec: 2 outputs per simdgroup (interleaved activations)
void metal_matvec_q4k_nr2_f32(void* queue, void* pipeline,
                               void* A, uint64_t aOff,
                               void* B, uint64_t bOff,
                               void* C, uint64_t cOff,
                               int N, int K);

// Q4_K NR4 matvec: 4 outputs per simdgroup, 32 per threadgroup
void metal_matvec_q4k_nr4_f32(void* queue, void* pipeline,
                               void* A, uint64_t aOff,
                               void* B, uint64_t bOff,
                               void* C, uint64_t cOff,
                               int N, int K);

// Q4_K batched matmul: C = A @ B^T where A is [M,K], B is [N,K] Q4_K, C is [M,N]
void metal_matmul_q4k_batched_f32(void* queue, void* pipeline,
                                   void* A, uint64_t aOff,
                                   void* B, uint64_t bOff,
                                   void* C, uint64_t cOff,
                                   int M, int N, int K);

// Q4_K simdgroup tiled matmul: uses simdgroup_matrix for prefill (M>=8)
void metal_matmul_q4k_simdgroup_f32(void* queue, void* pipeline,
                                     void* A, uint64_t aOff,
                                     void* B, uint64_t bOff,
                                     void* C, uint64_t cOff,
                                     int M, int N, int K);

// Q4_K fused RMSNorm + QKV projections (FP16 output)
void metal_matvec_q4k_fused_rmsnorm_qkv_f16(void* queue, void* pipeline,
                                              void* x, void* normWeight,
                                              void* Wq, void* Wk, void* Wv,
                                              void* outQ, void* outK, void* outV,
                                              int qDim, int kvDim, int K, float eps);

// Q4_K fused MLP: SiLU(x @ W1) * (x @ W3)
void metal_matvec_q4k_fused_mlp_f32(void* queue, void* pipeline,
                                     void* x, void* W1, void* W3, void* out,
                                     int N, int K);

// Q4_K matvec with FP16 input, FP32 output
void metal_matvec_q4k_f16in_f32(void* queue, void* pipeline,
                                 void* A, void* B, void* C,
                                 int N, int K);

// Q4_K matvec that adds to output: C += A @ B^T
void metal_matvec_q4k_add_f32(void* queue, void* pipeline,
                               void* A, void* B, void* C,
                               int N, int K);

// Q5_K multi-output matvec
void metal_matvec_q5k_multi_output_f32(void* queue, void* pipeline,
                                        void* A, uint64_t aOff,
                                        void* B, uint64_t bOff,
                                        void* C, uint64_t cOff,
                                        int N, int K);

// Q5_K NR2 matvec: 8 arguments (safe limit)
void metal_matvec_q5k_nr2_f32_v4(void* queue, void* pipeline,
                                  void* A, void* B, void* C,
                                  int N, int K, void* offsetsPtr);

void metal_rmsnorm_f32(void* queue, void* pipeline,
                       void* x, void* weight, void* out,
                       int batchSize, int dim, float eps);

// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
void metal_layernorm_f32(void* queue, void* pipeline,
                         void* x, void* weight, void* bias, void* out,
                         int batchSize, int dim, float eps);

// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void metal_gelu_f32(void* queue, void* pipeline,
                    void* x, void* out, int n);

// Fused GELU-gated multiply: out = GELU(gate) * up (GeGLU for Gemma)
void metal_gelu_mul_f32(void* queue, void* pipeline,
                        void* gate, void* up, void* out, int n);

// Add bias: out = x + bias (broadcasted across rows)
void metal_add_bias_f32(void* queue, void* pipeline,
                        void* x, void* bias, void* out,
                        int rows, int cols);

// Fused Add + RMSNorm: x = x + residual (in-place), then out = RMSNorm(x)
void metal_add_rmsnorm_f32(void* queue, void* pipeline,
                           void* x, void* residual, void* weight, void* out,
                           int batchSize, int dim, float eps);

// Fused RMSNorm + Q4_0 MatVec
void metal_matvec_q4_0_fused_rmsnorm_f32(void* queue, void* pipeline,
                                         void* x, void* normWeight, void* wMat, void* out,
                                         int n, int k, float eps);

// Fused RMSNorm + Q4_0 MatVec with FP16 OUTPUT
// Eliminates FP32->FP16 conversion after QKV projections
void metal_matvec_q4_0_fused_rmsnorm_f16_out(void* queue, void* pipeline,
                                              void* x, void* normWeight, void* wMat, void* out,
                                              int n, int k, float eps);

// Fused RMSNorm + QKV FP16: single dispatch for all 3 projections
void metal_matvec_q4_0_fused_rmsnorm_qkv_f16(void* queue, void* pipeline,
                                               void* x, void* normWeight,
                                               void* Wq, void* Wk, void* Wv,
                                               void* outQ, void* outK, void* outV,
                                               int qDim, int kvDim, int K, float eps);

// Fused MLP: SiLU(x @ W1) * (x @ W3)
// W1, W3: [N, K] Q4_0
// out: [N] F32
void metal_matvec_q4_0_fused_mlp_f32(void* queue, void* pipeline,
                                     void* x, void* W1, void* W3, void* out,
                                     int N, int K);

void metal_rope_f32(void* queue, void* pipeline,
                    void* q, void* k,
                    int batchSize, int seqLen, int numHeads, int headDim,
                    int startPos, float theta);

// ropeDim: dimensions to rotate (can be < headDim for partial RoPE like Phi-2)
// ropeNeox: 0 = LLaMA-style (interleaved pairs), 1 = NEOX-style (split pairs)
void metal_rope_gqa_f32(void* queue, void* pipeline,
                        void* q, void* k,
                        int seqLen, int numQHeads, int numKVHeads, int headDim,
                        int startPos, int ropeDim, float theta, int ropeNeox);

// RoPE for GQA with pre-computed per-dimension inverse frequencies (Gemma 2 learned RoPE)
// freqs: [headDim/2] float32 pre-computed inverse frequencies
void metal_rope_gqa_scaled_f32(void* queue, void* pipeline,
                                void* q, void* k, void* freqs,
                                int seqLen, int numQHeads, int numKVHeads, int headDim,
                                int startPos, int ropeNeox);

// RoPE for GQA with FP16 inputs/outputs
// Computation in FP32 for numerical stability, I/O in FP16
// ropeDim: dimensions to rotate (can be < headDim for partial RoPE like Phi-2)
// ropeNeox: 0 = LLaMA-style (interleaved pairs), 1 = NEOX-style (split pairs)
void metal_rope_gqa_f16(void* queue, void* pipeline,
                        void* q, void* k,
                        int seqLen, int numQHeads, int numKVHeads, int headDim,
                        int startPos, int ropeDim, float theta, int ropeNeox);

void metal_softmax_f32(void* queue, void* pipeline,
                       void* x, void* out,
                       int batchSize, int dim);

void metal_silu_f32(void* queue, void* pipeline,
                    void* x, void* out, int n);

void metal_silu_mul_f32(void* queue, void* pipeline,
                        void* gate, void* up, void* out, int n);

void metal_add_f32(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

// Offset-aware variants: use setBuffer:offset:atIndex: for sub-allocated buffers.
// These allow kernels to operate on regions of a single pre-allocated MTLBuffer.
void metal_add_f32_offset(void* queue, void* pipeline,
                          void* a, uint64_t aOff,
                          void* b, uint64_t bOff,
                          void* out, uint64_t outOff, int n);

void metal_rmsnorm_f32_offset(void* queue, void* pipeline,
                              void* x, uint64_t xOff,
                              void* weight, uint64_t weightOff,
                              void* out, uint64_t outOff,
                              int batchSize, int dim, float eps);

void metal_silu_mul_f32_offset(void* queue, void* pipeline,
                               void* gate, uint64_t gateOff,
                               void* up, uint64_t upOff,
                               void* out, uint64_t outOff, int n);

// Offset-aware Q4_0 matmul variants: A and C have offsets, B (weights) at offset 0.
void metal_matvec_q4_0_multi_output_f32_offset(void* queue, void* pipeline,
                                                void* A, uint64_t aOff,
                                                void* B,
                                                void* C, uint64_t cOff,
                                                int N, int K);

void metal_matvec_q4_0_nr2_f32_offset(void* queue, void* pipeline,
                                       void* A, uint64_t aOff,
                                       void* B,
                                       void* C, uint64_t cOff,
                                       int N, int K);

void metal_matmul_q4_0_batched_f32_offset(void* queue, void* pipeline,
                                           void* A, uint64_t aOff,
                                           void* B,
                                           void* C, uint64_t cOff,
                                           int M, int N, int K);

void metal_matmul_q4_0_simdgroup_f32_offset(void* queue, void* pipeline,
                                             void* A, uint64_t aOff,
                                             void* B,
                                             void* C, uint64_t cOff,
                                             int M, int N, int K);

// V3 offset-aware: blocked 8×8 shared memory layout, TILE_M=64, TILE_N=32
void metal_matmul_q4_0_simdgroup_v3_f32_offset(void* queue, void* pipeline,
                                                 void* A, uint64_t aOff,
                                                 void* B,
                                                 void* C, uint64_t cOff,
                                                 int M, int N, int K);

void metal_matvec_q4_0_transposed_f32_offset(void* queue, void* pipeline,
                                              void* A, uint64_t aOff,
                                              void* B,
                                              void* C, uint64_t cOff,
                                              int N, int K);

// Offset-aware RoPE: Q and K have offsets (scratch-allocated activations).
void metal_rope_gqa_f32_offset(void* queue, void* pipeline,
                                void* q, uint64_t qOff,
                                void* k, uint64_t kOff,
                                int seqLen, int numQHeads, int numKVHeads, int headDim,
                                int startPos, int ropeDim, float theta, int ropeNeox);

// Offset-aware SDPA decode: Q and out have offsets, K/V (KV cache) at offset 0.
void metal_sdpa_decode_f32_offset(void* queue, void* pipeline,
                                   void* Q, uint64_t qOff,
                                   void* K, void* V,
                                   void* out, uint64_t outOff,
                                   int kvLen, int numQHeads, int numKVHeads, int headDim,
                                   float scale, int kvHeadStride);

void metal_sdpa_flash_decode_f32_offset(void* queue, void* pipeline,
                                         void* Q, uint64_t qOff,
                                         void* K, void* V,
                                         void* out, uint64_t outOff,
                                         int kvLen, int numQHeads, int numKVHeads, int headDim,
                                         float scale, int kvHeadStride);

// Offset-aware fused kernels: activation buffers have offsets, weights at offset 0.
void metal_matvec_q4_0_fused_rmsnorm_f32_offset(void* queue, void* pipeline,
                                                  void* x, void* normWeight, void* wMat,
                                                  void* out, uint64_t outOff,
                                                  int n, int k, float eps);

void metal_add_rmsnorm_f32_offset(void* queue, void* pipeline,
                                   void* x,
                                   void* residual, uint64_t residualOff,
                                   void* weight,
                                   void* out, uint64_t outOff,
                                   int batchSize, int dim, float eps);

void metal_scatter_kv_f32_offset(void* queue, void* pipeline,
                                  void* src, uint64_t srcOff,
                                  void* dst,
                                  int newTokens, int numKVHeads, int headDim,
                                  int maxSeqLen, int seqPos);

void metal_matvec_q4_0_fused_mlp_f32_offset(void* queue, void* pipeline,
                                             void* x, uint64_t xOff,
                                             void* W1, void* W3,
                                             void* out, uint64_t outOff,
                                             int N, int K);

void metal_mul_f32(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

/// Deinterleave QKV: split fused [M, qkvDim] output into separate Q, K, V.
// All pointers use offset for scratch-allocated buffer support.
void metal_deinterleave_qkv_f32(void* queue, void* pipeline,
                                 void* srcBuf, uint64_t srcOff,
                                 void* dstQBuf, uint64_t dstQOff,
                                 void* dstKBuf, uint64_t dstKOff,
                                 void* dstVBuf, uint64_t dstVOff,
                                 int M, int qDim, int kvDim);

// Deinterleave 2-way: split fused [M, dim1+dim2] into separate A [M, dim1], B [M, dim2]
void metal_deinterleave_2way_f32(void* queue, void* pipeline,
                                  void* srcBuf, uint64_t srcOff,
                                  void* dstABuf, uint64_t dstAOff,
                                  void* dstBBuf, uint64_t dstBOff,
                                  int M, int dim1, int dim2);

// Argmax: find index of maximum value (for GPU-side greedy sampling)
void metal_argmax_f32(void* queue, void* pipeline,
                       void* input, void* result, int N);

// Compute-based memory copy (avoids blit encoder transition issues)
void metal_copy_buffer_compute(void* queue, void* pipeline,
                                void* srcBuffer, size_t srcOffset,
                                void* dstBuffer, size_t dstOffset, size_t size);

// =============================================================================
// Training Operations for Medusa Heads
// =============================================================================

// ReLU in-place: x = max(0, x)
void metal_relu_inplace_f32(void* queue, void* pipeline, void* x, int n);

// ReLU backward: dx = dx * (x > 0), masks gradients where input was <= 0
void metal_relu_backward_f32(void* queue, void* pipeline, void* x, void* dx, int n);

// Batched outer product: C[i,j] += sum_b(A[b,i] * B[b,j])
// A: [batch, M], B: [batch, N], C: [M, N]
void metal_batched_outer_product_f32(void* queue, void* pipeline,
                                      void* A, void* B, void* C,
                                      int batch, int M, int N);

// SGD weight update with weight decay: w = w*(1-lr*wd) - lr*grad
void metal_sgd_update_f32(void* queue, void* pipeline,
                          void* w, void* grad, float lr, float weightDecay, int n);

// Zero out buffer
void metal_zero_f32(void* queue, void* pipeline, void* x, int n);

void metal_embedding_f32(void* queue,
                         void* tokens, void* table, void* out,
                         int numTokens, int vocabSize, int dim);

// Attention operations
// SDPA for decode (single query position against KV cache)
// Q: [numQHeads, headDim], K/V: [numKVHeads, maxSeqLen, headDim] (head-major layout)
// kvHeadStride = maxSeqLen * headDim (stride between KV heads)
void metal_sdpa_decode_f32(void* queue, void* pipeline,
                           void* Q, void* K, void* V, void* out,
                           int kvLen, int numQHeads, int numKVHeads, int headDim,
                           float scale, int kvHeadStride);

// SDPA for prefill (batched with causal masking)
// Q/K/V: [seqLen, numHeads, headDim]
void metal_sdpa_prefill_f32(void* queue, void* pipeline,
                            void* Q, void* K, void* V, void* out,
                            int seqLen, int numQHeads, int numKVHeads, int headDim,
                            float scale);

// Flash Attention 2 optimized prefill
// Uses larger tiles (64 K positions) and GQA-aware shared memory
// K/V tiles loaded to shared memory and shared across Q heads
void metal_flash_attention_2_f32(void* queue, void* pipeline,
                                  void* Q, void* K, void* V, void* out,
                                  int seqLen, int numQHeads, int numKVHeads, int headDim,
                                  float scale);

// Flash Attention 2 v2 — optimized prefill (single-pass, tiled Q for GQA, TILE_KV=32)
void metal_flash_attention_2_v2_f32(void* queue, void* pipeline,
                                     void* Q, void* K, void* V, void* out,
                                     int seqLen, int numQHeads, int numKVHeads, int headDim,
                                     float scale);

// Flash Attention 2 with FP16 activations (2x bandwidth)
// Q, K, V, out are all FP16
void metal_flash_attention_2_f16(void* queue, void* pipeline,
                                  void* Q, void* K, void* V, void* out,
                                  int seqLen, int numQHeads, int numKVHeads, int headDim,
                                  float scale);

// Flash Decoding - parallelized SDPA for decode phase
// Uses threadgroup parallelism with online softmax reduction
// Q: [numQHeads, headDim], K/V: [numKVHeads, maxSeqLen, headDim] (head-major layout)
// kvHeadStride = maxSeqLen * headDim (stride between KV heads)
void metal_sdpa_flash_decode_f32(void* queue, void* pipeline,
                                  void* Q, void* K, void* V, void* out,
                                  int kvLen, int numQHeads, int numKVHeads, int headDim,
                                  float scale, int kvHeadStride);

// Flash Attention F32 v2: split-KV online softmax, O(headDim) shared memory.
void metal_sdpa_flash_decode_f32_v2(void* queue, void* pipeline,
                                     void* Q, void* K, void* V, void* out,
                                     int kvLen, int numQHeads, int numKVHeads, int headDim,
                                     float scale, int kvHeadStride);

void metal_sdpa_flash_decode_f32_v2_offset(void* queue, void* pipeline,
                                            void* Q, uint64_t qOff,
                                            void* K, void* V,
                                            void* out, uint64_t outOff,
                                            int kvLen, int numQHeads, int numKVHeads, int headDim,
                                            float scale, int kvHeadStride);

// SDPA decode with logit soft-capping (Gemma 2)
// Applies cap * tanh(score / cap) before softmax
void metal_sdpa_softcap_decode_f32(void* queue, void* pipeline,
                                    void* Q, void* K, void* V, void* out,
                                    int kvLen, int numQHeads, int numKVHeads, int headDim,
                                    float scale, int kvHeadStride, float softcap);

// SDPA prefill with logit soft-capping (Gemma 2)
void metal_sdpa_softcap_prefill_f32(void* queue, void* pipeline,
                                     void* Q, void* K, void* V, void* out,
                                     int seqLen, int numQHeads, int numKVHeads, int headDim,
                                     float scale, float softcap);

// Legacy interface (deprecated)
void metal_scaled_dot_product_attention(void* queue,
                                        void* Q, void* K, void* V, void* out,
                                        int batchSize, int numHeads, int seqLen, int headDim,
                                        float scale, int causal);

// =============================================================================
// FP16 (Half-Precision) Operations
// These provide 2x memory bandwidth for memory-bound operations.
// =============================================================================

void metal_add_f16(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

void metal_mul_f16(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

void metal_silu_f16(void* queue, void* pipeline,
                    void* x, void* out, int n);

void metal_silu_mul_f16(void* queue, void* pipeline,
                        void* gate, void* up, void* out, int n);

void metal_rmsnorm_f16(void* queue, void* pipeline,
                       void* x, void* weight, void* out,
                       int batchSize, int dim, float eps);

// FP32 to FP16 conversion
void metal_convert_f32_to_f16(void* queue, void* pipeline,
                               void* in, void* out, int n);
void metal_convert_f32_to_f16_offset(void* queue, void* pipeline,
                               void* in, uint64_t inOff,
                               void* out, uint64_t outOff, int n);

// FP16 to FP32 conversion
void metal_convert_f16_to_f32(void* queue, void* pipeline,
                               void* in, void* out, int n);
void metal_convert_f16_to_f32_offset(void* queue, void* pipeline,
                               void* in, uint64_t inOff,
                               void* out, uint64_t outOff, int n);

// KV Cache scatter: transpose from [newTokens, numKVHeads, headDim] to [numKVHeads, maxSeqLen, headDim]
// Used for efficient GPU-side KV cache layout transformation
void metal_scatter_kv_f16(void* queue, void* pipeline, void* src, void* dst, int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos);
void metal_scatter_kv_f32(void* queue, void* pipeline, void* src, void* dst, int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos);
void metal_scatter_kv_f32_to_f16(void* queue, void* pipeline, void* src, void* dst, int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos);

// Fused K+V scatter: scatter both K and V in a single dispatch (saves 1 dispatch/layer)
void metal_scatter_kv_f16_fused(void* queue, void* pipeline,
                                void* srcK, void* dstK,
                                void* srcV, void* dstV,
                                int newTokens, int numKVHeads, int headDim, int maxSeqLen, int seqPos);

// MatVec Q4_0 v2 with FP16 input (eliminates F16->F32 convert dispatch)
void metal_matvec_q4_0_v2_f16in_f32(void* queue, void* pipeline,
                                     void* A, void* B, void* C,
                                     int N, int K);

// MatVec Q4_0 v2 with add-to-output (fuses matmul + residual add)
void metal_matvec_q4_0_v2_add_f32(void* queue, void* pipeline,
                                   void* A, void* B, void* C,
                                   int N, int K);

// Fused RoPE + KV Scatter for FP16 decode (M=1)
void metal_rope_scatter_kv_f16(void* queue, void* pipeline,
                                void* q, void* srcK, void* dstK,
                                void* srcV, void* dstV,
                                int numQHeads, int numKVHeads, int headDim,
                                int startPos, int ropeDim, float theta,
                                int maxSeqLen, int seqPos);

// Reshape KV cache for paged attention
// Copies data from src [numTokens, numKVHeads, headDim] to paged blocks.
// pageTable: [numTokens] mapping each logical token to (blockID, offsetInBlock)
// But pageTable usually maps block_index -> block_pointer.
// Let's assume the kernel takes a list of block pointers for the sequence?
// Or a flat page table buffer?
// Standard approach (vLLM):
// Kernel takes `block_table` [max_num_blocks_per_seq] which maps logical_block_idx -> physical_block_idx.
// We are storing ONE token or a BATCH of tokens.
// If batch, we need to know which block each token goes to.
// Let's implement `metal_reshape_paged_kv` which takes:
// - src: [numTokens, numKVHeads, headDim]
// - pageTable: [numTokens] int32, where each entry is the physical block index for that token.
// - blockOffsets: [numTokens] int32, offset within the block for that token.
// - blocks: The base pointer to the block pool (or we pass array of pointers? Metal prefers one big buffer).
// Our current `KVCache` allocates one big buffer `basePtr`.
// So `blocks` is just `basePtr`.
// `pageTable` could just be `physical_block_index`.
//
// void metal_reshape_paged_kv(void* queue, void* pipeline,
//                             void* src, void* dstBase,
//                             void* pageTable, // [numTokens] int32: physical block index
//                             void* blockOffsets, // [numTokens] int32: token index within block
//                             int numTokens, int numKVHeads, int headDim, int blockSize);
void metal_reshape_paged_kv_f32(void* queue, void* pipeline,
                                void* src, void* dstBase,
                                void* pageTable,
                                void* blockOffsets,
                                int numTokens, int numKVHeads, int headDim, int blockSize, int isValue);

// Paged SDPA decode: performs attention reading K/V from paged block pool.
// Q: [numQHeads, headDim]
// kvPool: base pointer to block pool
// blockTable: [numBlocks] int32 mapping logical → physical block indices
// out: [numQHeads, headDim]
// tokensInLastBlock: number of valid tokens in the last logical block
void metal_sdpa_paged_decode_f32(void* queue, void* pipeline,
                                  void* Q, void* kvPool, void* blockTable, void* out,
                                  int numBlocks, int blockSize,
                                  int numQHeads, int numKVHeads, int headDim,
                                  float scale, int tokensInLastBlock);


// =============================================================================
// Q8_0 Quantization for KV Cache
// Q8_0 format: 34 bytes per 32 elements (2-byte f16 scale + 32 int8 values)
// =============================================================================

// Quantize FP32 to Q8_0: in [n] -> out [n/32 * 34 bytes]
void metal_quantize_f32_to_q8_0(void* queue, void* pipeline,
                                 void* in, void* out, int n);

// Dequantize Q8_0 to FP32: in [n/32 * 34 bytes] -> out [n]
void metal_dequantize_q8_0_to_f32(void* queue, void* pipeline,
                                   void* in, void* out, int n);

// SDPA decode with Q8_0 KV cache
// Q: [numQHeads, headDim] in FP32, K/V: Q8_0 format, out: [numQHeads, headDim] FP32
void metal_sdpa_decode_q8_0(void* queue, void* pipeline,
                             void* Q, void* K, void* V, void* out,
                             int kvLen, int numQHeads, int numKVHeads, int headDim,
                             float scale);

// Q4_0 matvec with FP16 activations: C = A @ B^T
// A: [1,K] in FP16, B: [N,K] in Q4_0, C: [1,N] in FP16
void metal_matvec_q4_0_f16(void* queue, void* pipeline,
                            void* A, void* B, void* C,
                            int N, int K);

// FP16 SDPA for decode - 2x KV cache bandwidth savings
// Q: [numQHeads, headDim], K/V: [numKVHeads, kvLen, headDim] (head-major layout), out: [numQHeads, headDim]
// All tensors in FP16
// kvHeadStride: stride between KV heads in elements (maxSeqLen * headDim)
void metal_sdpa_decode_f16(void* queue, void* pipeline,
                            void* Q, void* K, void* V, void* out,
                            int kvLen, int numQHeads, int numKVHeads, int headDim,
                            float scale, int kvHeadStride);

// Specialized SDPA for headDim=64 with half4 vectorization and online softmax
// Single-thread per head version - each thread processes all KV positions
void metal_sdpa_decode_f16_hd64(void* queue, void* pipeline,
                                 void* Q, void* K, void* V, void* out,
                                 int kvLen, int numQHeads, int numKVHeads,
                                 float scale, int kvHeadStride);

// SIMD version: 32 threads cooperate via simd_sum, each thread owns 2 dimensions
void metal_sdpa_decode_f16_hd64_simd(void* queue, void* pipeline,
                                      void* Q, void* K, void* V, void* out,
                                      int kvLen, int numQHeads, int numKVHeads,
                                      float scale, int kvHeadStride);

// Flash Attention SDPA decode for FP16 KV cache
// Split-KV across simdgroups with online softmax. O(headDim) shared memory.
void metal_sdpa_flash_decode_f16(void* queue, void* pipeline,
                                  void* Q, void* K, void* V, void* out,
                                  int kvLen, int numQHeads, int numKVHeads, int headDim,
                                  float scale, int kvHeadStride);

// Tiled Flash Attention SDPA F16 (split-K approach for better GPU occupancy)
void metal_sdpa_flash_decode_f16_tiled(void* queue,
                                        void* tilePipeline,
                                        void* mergePipeline,
                                        void* Q, void* K, void* V, void* out,
                                        void* partials,
                                        int kvLen, int numQHeads, int numKVHeads, int headDim,
                                        float scale, int kvHeadStride);

// Q8_0 NR2 matvec: C = A @ B^T where A is [1,K] F32, B is [N,K] Q8_0
void metal_matvec_q8_0_nr2_f32(void* queue, void* pipeline,
                                void* A, uint64_t aOff,
                                void* B, uint64_t bOff,
                                void* C, uint64_t cOff,
                                int N, int K);

// Q8_0 batched matmul: C = A @ B^T where A is [M,K] F32, B is [N,K] Q8_0, C is [M,N]
void metal_matmul_q8_0_batched_f32(void* queue, void* pipeline,
                                    void* A, uint64_t aOff,
                                    void* B, uint64_t bOff,
                                    void* C, uint64_t cOff,
                                    int M, int N, int K);

// BF16 NR2 matvec: C = A @ B^T where A is [1,K] F32, B is [N,K] BF16
void metal_matvec_bf16_nr2_f32(void* queue, void* pipeline,
                                void* A, uint64_t aOff,
                                void* B, uint64_t bOff,
                                void* C, uint64_t cOff,
                                int N, int K);

// BF16 batched matmul: C = A @ B^T where A is [M,K] F32, B is [N,K] BF16, C is [M,N]
void metal_matmul_bf16_batched_f32(void* queue, void* pipeline,
                                    void* A, uint64_t aOff,
                                    void* B, uint64_t bOff,
                                    void* C, uint64_t cOff,
                                    int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H
