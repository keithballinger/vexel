# Track Plan: Quantization Expansion

The engine supports Q4_0, Q4_K, Q5_K, and Q6_K with Metal GPU kernels for decode (M=1 matvec).
GGUF dequantization exists for Q4_0 and Q8_0. Prefill uses CPU-side dequantization before GPU
matmul because the quantized kernels only support M=1. This track expands format support,
adds batched quantized matmul for prefill, and benchmarks accuracy/performance tradeoffs.

## Phase 1: Batched Quantized MatMul
- [x] Task: Q4_0 batched matmul kernel
    - Simdgroup kernel already supported M>1; optimized tile layout from 32×32 to 32×64.
    - Uses all 8 simdgroups (2×4 layout of 16×16 tiles) vs previous 4 (2×2).
    - ~30% throughput improvement at M=32 (primary prefill batch size).
    - Created `q4_batched_matmul_test.go` with prefill correctness tests (M=32,64,128)
      and throughput benchmarks. All tests pass with ZERO numerical difference.
- [x] Task: Q4_K batched matmul kernel
    - Rewrote `matmul_q4k_batched_f32` from naive one-output-per-TG to NR2 pattern.
    - Each simdgroup handles 2 N outputs with simd_lane striding through super-blocks.
    - 16 outputs per threadgroup (8 simdgroups × 2). Grid: (ceil(N/16), M).
    - Old kernel wasted 242/256 threads at K=4096; new kernel fully utilizes all threads.
    - M=4: 196 GFLOPS, M=32: 312 GFLOPS, M=128: 333 GFLOPS.
    - Created `q4k_batched_matmul_test.go` with prefill correctness tests (M=4,32,64,128)
      and throughput benchmarks. All 7 tests pass with <0.001 max diff.
- [x] Task: Wire batched kernels into prefill path
    - Verified: `matMulTransposed` already dispatches Q4_0, Q4_K, Q6_K, Q5_K to GPU kernels
      for any M value (seqLen). No CPU dequantization fallback exists for weight matrices.
    - Q4_0 and Q4_K use optimized batched kernels (simdgroup/NR2) for M>1.
    - Q6_K and Q5_K use loop-based matvec for M>1 (functional, optimization deferred).
    - Loader keeps all quant types as raw quantized data on GPU — no dequant in prefill path.

## Phase 2: Additional Dequantization Formats
- [x] Task: Q5_0 and Q5_1 support
    - Implemented `DequantizeQ5_0` and `DequantizeQ5_1` in `gguf/dequant.go`.
    - Q5_0: 22 bytes per 32 elements (f16 scale + uint32 high bits + 4-bit nibbles).
    - Q5_1: 24 bytes per 32 elements (f16 scale + f16 min + uint32 high bits + 4-bit nibbles).
    - Added `BytesPerBlock()` entries for both formats.
    - Added to `Dequantize()` switch for automatic format dispatch.
    - 6 unit tests pass covering zero values, scaling, min offset, and split halves.
    - Metal kernels deferred: Q5_0/Q5_1 rarely used; loader CPU dequant→F32 handles them.
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
