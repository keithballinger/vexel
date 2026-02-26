# Track Plan: Quantization Expansion

The engine supports Q4_0, Q4_K, Q5_K, and Q6_K with Metal GPU kernels for decode (M=1 matvec).
GGUF dequantization exists for Q4_0 and Q8_0. Prefill uses CPU-side dequantization before GPU
matmul because the quantized kernels only support M=1. This track expands format support,
adds batched quantized matmul for prefill, and benchmarks accuracy/performance tradeoffs.

## Phase 1: Batched Quantized MatMul
- [ ] Task: Q4_0 batched matmul kernel
    - Extend `MatMulQ4_0` Metal kernel to support M>1 (batch of query vectors).
    - Input: A [M, K] float32, B [N, K] Q4_0 packed. Output: C [M, N] float32.
    - Optimize threadgroup tiling for prefill batch sizes (32, 64, 128).
- [ ] Task: Q4_K batched matmul kernel
    - Extend `MatMulQ4_K` for M>1 with k-quant super-block structure.
    - Handle the 256-element super-blocks with nested 32-element sub-blocks.
- [ ] Task: Wire batched kernels into prefill path
    - Update `runtime.DecodeWithGPUKV` to use batched quantized matmul when seqLen > 1.
    - Remove CPU dequantization fallback for supported quant types during prefill.
    - Benchmark prefill throughput improvement vs CPU dequant path.

## Phase 2: Additional Dequantization Formats
- [ ] Task: Q5_0 and Q5_1 support
    - Implement `DequantizeQ5_0` and `DequantizeQ5_1` in `gguf/dequant.go`.
    - Q5_0: 22 bytes per 32 elements (f16 scale + 4-bit nibbles + 5th bit packed).
    - Q5_1: 24 bytes per 32 elements (f16 min + f16 scale + 5-bit values).
    - Add Metal matvec kernels for Q5_0 and Q5_1.
- [ ] Task: Q8_0 GPU kernel
    - Add `MatMulQ8_0` Metal kernel (simpler than Q4 — straight int8 with scale).
    - Q8_0 is used internally for KV cache quantization; a dedicated kernel avoids dequant overhead.
- [ ] Task: BF16 support
    - Add `MatMulBF16` Metal kernel leveraging hardware BF16 on M3/M4.
    - Fallback to F16 conversion on M1/M2.

## Phase 3: Benchmarking & Validation
- [ ] Task: Accuracy comparison
    - For each quant format, measure perplexity on a fixed text corpus vs F32 baseline.
    - Generate comparison table: format, model size, perplexity delta, decode tok/s.
- [ ] Task: Performance benchmarks
    - Benchmark prefill tok/s across quant formats at seqLen=128.
    - Benchmark decode tok/s across quant formats.
    - Measure memory usage (weights + KV cache) per format.
- [ ] Task: Update documentation
    - Add quantization format comparison table to README.
    - Document which formats are recommended for different use cases.
