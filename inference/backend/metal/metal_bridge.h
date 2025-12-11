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
void metal_copy_to_buffer(void* buffer, const void* src, size_t size);
void metal_copy_from_buffer(void* dst, void* buffer, size_t size);

// GPU-to-GPU buffer copy (uses blit encoder)
void metal_copy_buffer(void* queue, void* srcBuffer, size_t srcOffset,
                       void* dstBuffer, size_t dstOffset, size_t size);

// Shader compilation
void* metal_compile_library(void* device, const char* source);
void* metal_create_pipeline(void* device, void* library, const char* functionName);

// Synchronization
void metal_sync(void* commandQueue);

// Command Buffer Batching - reduce commit overhead by batching operations
void metal_begin_batch(void* commandQueue);
void metal_end_batch(void);

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

// Batched Q4_0 matmul: C = A @ B^T where A is [M,K], B is [N,K] Q4_0, C is [M,N]
void metal_matmul_q4_0_batched_f32(void* queue, void* pipeline,
                                    void* A, void* B, void* C,
                                    int M, int N, int K);

// Simdgroup tiled Q4_0 matmul: uses simdgroup_matrix for prefill
void metal_matmul_q4_0_simdgroup_f32(void* queue, void* pipeline,
                                      void* A, void* B, void* C,
                                      int M, int N, int K);

// Q6_K multi-output matvec: for lm_head which uses Q6_K quantization
void metal_matvec_q6k_multi_output_f32(void* queue, void* pipeline,
                                        void* A, void* B, void* C,
                                        int N, int K);

// Q4_K multi-output matvec: for attention projections using Q4_K quantization
void metal_matvec_q4k_multi_output_f32(void* queue, void* pipeline,
                                        void* A, void* B, void* C,
                                        int N, int K);

void metal_rmsnorm_f32(void* queue, void* pipeline,
                       void* x, void* weight, void* out,
                       int batchSize, int dim, float eps);

void metal_rope_f32(void* queue, void* pipeline,
                    void* q, void* k,
                    int batchSize, int seqLen, int numHeads, int headDim,
                    int startPos, float theta);

void metal_rope_gqa_f32(void* queue, void* pipeline,
                        void* q, void* k,
                        int seqLen, int numQHeads, int numKVHeads, int headDim,
                        int startPos, float theta);

void metal_softmax_f32(void* queue, void* pipeline,
                       void* x, void* out,
                       int batchSize, int dim);

void metal_silu_f32(void* queue, void* pipeline,
                    void* x, void* out, int n);

void metal_silu_mul_f32(void* queue, void* pipeline,
                        void* gate, void* up, void* out, int n);

void metal_add_f32(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

void metal_mul_f32(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

void metal_embedding_f32(void* queue,
                         void* tokens, void* table, void* out,
                         int numTokens, int vocabSize, int dim);

// Attention operations
// SDPA for decode (single query position against KV cache)
// Q: [numQHeads, headDim], K/V: [kvLen, numKVHeads, headDim]
void metal_sdpa_decode_f32(void* queue, void* pipeline,
                           void* Q, void* K, void* V, void* out,
                           int kvLen, int numQHeads, int numKVHeads, int headDim,
                           float scale);

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

// Flash Decoding - parallelized SDPA for decode phase
// Uses threadgroup parallelism with online softmax reduction
// Q: [numQHeads, headDim], K/V: [kvLen, numKVHeads, headDim]
void metal_sdpa_flash_decode_f32(void* queue, void* pipeline,
                                  void* Q, void* K, void* V, void* out,
                                  int kvLen, int numQHeads, int numKVHeads, int headDim,
                                  float scale);

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

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H
