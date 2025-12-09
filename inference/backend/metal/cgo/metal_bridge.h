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
void metal_copy_to_buffer(void* buffer, const void* src, size_t size);
void metal_copy_from_buffer(void* dst, void* buffer, size_t size);

// Shader compilation
void* metal_compile_library(void* device, const char* source);
void* metal_create_pipeline(void* device, void* library, const char* functionName);

// Synchronization
void metal_sync(void* commandQueue);

// Kernel execution
void metal_matmul_f32(void* queue, void* pipeline,
                      void* A, void* B, void* C,
                      int M, int N, int K);

void metal_matmul_transposed_f32(void* queue, void* pipeline,
                                  void* A, void* B, void* C,
                                  int M, int N, int K);

void metal_rmsnorm_f32(void* queue, void* pipeline,
                       void* x, void* weight, void* out,
                       int batchSize, int dim, float eps);

void metal_rope_f32(void* queue, void* pipeline,
                    void* q, void* k,
                    int batchSize, int seqLen, int numHeads, int headDim,
                    int startPos, float theta);

void metal_softmax_f32(void* queue, void* pipeline,
                       void* x, void* out,
                       int batchSize, int dim);

void metal_silu_f32(void* queue, void* pipeline,
                    void* x, void* out, int n);

void metal_add_f32(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

void metal_mul_f32(void* queue, void* pipeline,
                   void* a, void* b, void* out, int n);

void metal_embedding_f32(void* queue,
                         const int* tokens, void* table, void* out,
                         int numTokens, int vocabSize, int dim);

// Attention operations
void metal_scaled_dot_product_attention(void* queue,
                                        void* Q, void* K, void* V, void* out,
                                        int batchSize, int numHeads, int seqLen, int headDim,
                                        float scale, int causal);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H
