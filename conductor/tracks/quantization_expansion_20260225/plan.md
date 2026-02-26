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
- [x] Task: Q8_0 GPU kernel
    - Added `matvec_q8_0_nr2_f32` (M=1) and `matmul_q8_0_batched_f32` (M>1) Metal kernels.
    - NR2 pattern: 16 outputs per threadgroup, simd_lane striding through 32-element blocks.
    - Q8_0 format: 34 bytes per 32 elements (f16 scale + 32 int8 values). Simple d*int8 dequant.
    - Wired into runtime dispatch (`matMulTransposed`) and loader (raw Q8_0 on GPU).
    - M=1: 55 GFLOPS, M=32: 371 GFLOPS, M=128: 402 GFLOPS at 4096×4096.
    - All 7 production-size correctness tests pass with max_diff < 0.000006.
    - Created `q8_0_matmul_test.go` with basic, batched, prefill, and throughput tests.
- [x] Task: BF16 support
    - Added `matvec_bf16_nr2_f32` (M=1) and `matmul_bf16_batched_f32` (M>1) Metal kernels.
    - NR2 pattern: 16 outputs per threadgroup. BF16→F32 via bit shift (works on all Apple Silicon).
    - Added `tensor.BF16` QuantProfile and wired into runtime dispatch + loader.
    - M=1: 75 GFLOPS, M=64: 654 GFLOPS, M=128: 788 GFLOPS at 4096×4096.
    - All 7 production-size correctness tests pass with ZERO numerical difference.
    - Created `bf16_matmul_test.go` with basic, batched, prefill, and throughput tests.

## Phase 3: Benchmarking & Validation
- [x] Task: Accuracy comparison
    - Perplexity measurement deferred (requires model files and text corpus).
    - Kernel-level numerical accuracy verified for all formats:
      - Q4_0/Q4_K: max_diff < 0.001 (quantization noise at K=4096).
      - Q8_0: max_diff < 0.000006 (near-F32 precision).
      - BF16: max_diff = 0.0 (lossless BF16→F32 conversion).
- [x] Task: Performance benchmarks
    - Created `quant_benchmark_test.go` with unified comparison report.
    - Kernel-level GFLOPS at 4096×4096 (Apple Silicon M-series):
      - Q4_0: 112 (M=1), 1191 (M=32), 2790 (M=128) — simdgroup kernel.
      - Q4_K:  46 (M=1),  315 (M=32),  330 (M=128) — NR2 kernel.
      - Q8_0:  57 (M=1),  382 (M=32),  402 (M=128) — NR2 kernel.
      - BF16:  81 (M=1),  571 (M=32),  810 (M=128) — NR2 kernel.
      - F32:   82 (M=1),  641 (M=32),  848 (M=128) — simdgroup kernel.
    - Model size estimates for 7B: Q4=3.9GB, Q8=7.4GB, BF16=14.0GB, F32=28.0GB.
- [x] Task: Update documentation
    - Benchmark results embedded in plan.md (above).
    - Each format test file includes throughput report test functions.
    - README update deferred to separate documentation track.
