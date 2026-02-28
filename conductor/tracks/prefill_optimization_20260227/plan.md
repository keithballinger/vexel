# Prefill Pipeline Optimization (P9)

> **Goal:** Close the 4-5x prefill throughput gap to llama.cpp
> **Baseline:** Vexel Q4_0 200 tok/s @ seqLen=128, llama.cpp 803 tok/s
> **Target:** ≥500 tok/s @ seqLen=128 (2.5x improvement, close to 60% of llama.cpp)
> **Hardware:** M3 Max 128GB, LLaMA 2 7B

## Root Cause Analysis

Five bottlenecks identified (priority ordered by estimated impact):

1. **GEMM kernel dequantizes to FP32 shared memory (2-3x)**
   - B tile inflated 7x: 1152 bytes (Q4_0) → 8192 bytes (FP32) in shared mem
   - Scalar byte-by-byte dequant, conditional branch per nibble
   - Write-read shared memory roundtrip for B kills latency
   - Fix: dequantize to registers, vectorized nibble extraction

2. **FA2 prefill wastes 7/8 simdgroups on LLaMA 2 7B (1-1.5x)**
   - LLaMA 2 7B has headsPerKV=1 (no GQA), so only simdgroup 0 is active
   - 87.5% of compute wasted per threadgroup
   - Fix: tile multiple Q positions per threadgroup, or restructure dispatch

3. **Per-layer command buffers — 32 CB create/commit cycles (0.5-1x)**
   - Each layer gets own BeginBatch/EndBatch → 32 command buffers
   - llama.cpp encodes ALL layers into single CB with barriers
   - Fix: cross-layer batching with memory barriers

4. **TILE_K=32 — one Q4_0 block per K-tile iteration (0.3-0.5x)**
   - 128 threadgroup barriers for K=4096
   - llama.cpp uses TILE_K=64+, processing 2 blocks per iteration
   - Fix: increase TILE_K to 64

5. **Separate Q/K/V projections — 3 reads of normOut (0.2-0.4x)**
   - Prefill does RMSNorm → MatMul(Wq) → MatMul(Wk) → MatMul(Wv)
   - 3 × 2MB = 6MB wasted bandwidth per layer × 32 layers = 192MB
   - Fix: fused QKV GEMM with concatenated weights

## Implementation Plan

### Phase 1: GEMM Kernel — Register-Based Dequantization

Target: 2-3x improvement on GEMM-dominated prefill path.

- [ ] Task 1.1: Write benchmark test for isolated Q4_0 simdgroup GEMM throughput
    - Test at M=128, N=4096, K=4096 (typical prefill projection)
    - Measure current GFLOPS and memory bandwidth utilization
- [ ] Task 1.2: Implement register-dequant Q4_0 simdgroup kernel
    - Each simdgroup loads A from shared memory, dequantizes B from device to registers
    - Eliminate shared_B entirely (save 8KB threadgroup memory)
    - Vectorized nibble extraction: load uint4, extract 8 quants with bitwise ops
    - Keep TILE_M=32, TILE_N=64, increase TILE_K to 64 (2 Q4_0 blocks per iteration)
    - Use simdgroup_load for A, manual register construction for B tiles
- [ ] Task 1.3: Run correctness tests (parity with current kernel)
- [ ] Task 1.4: Benchmark isolated GEMM and full-model prefill
    - Target: ≥400 tok/s at seqLen=128

### Phase 2: Cross-Layer Command Buffer Batching

Target: 0.5-1x improvement from eliminating CB overhead.

- [ ] Task 2.1: Implement cross-layer batch mode in metal backend
    - Single command buffer spans all 32 layers
    - Memory barriers between dependent dispatches (same as within-layer)
    - BeginBatch at DecodeWithGPUKV start, EndBatch at end
- [ ] Task 2.2: Correctness tests (TestFusionCorrectness must still pass)
- [ ] Task 2.3: Benchmark prefill and decode (decode must not regress)

### Phase 3: FA2 Prefill Multi-Query Tiling

Target: 1-1.5x improvement on attention portion (smaller impact since matmuls dominate).

- [ ] Task 3.1: Profile attention vs GEMM time split during prefill
    - If attention < 15% of prefill time, deprioritize this phase
- [ ] Task 3.2: Implement multi-query FA2 for non-GQA models
    - Each simdgroup handles a different KV head (not Q head)
    - Dispatch (ceil(seqLen/tilesPerTG), numKVHeads) threadgroups
- [ ] Task 3.3: Correctness and benchmark

### Phase 4: Q4_K Prefill GEMM (apply Phase 1 technique)

- [ ] Task 4.1: Port register-dequant approach to Q4_K simdgroup kernel
- [ ] Task 4.2: Benchmark Q4_K_M prefill improvement

### Phase 5: Integration & Benchmarks

- [ ] Task 5.1: Full benchmark suite (Q4_0 + Q4_K_M, all sequence lengths)
- [ ] Task 5.2: Verify decode throughput not regressed
- [ ] Task 5.3: Update RESULTS.md
- [ ] Task 5.4: Update README.md

---

## Reference: Key Files

- GEMM kernel: `inference/backend/metal/metal_bridge_darwin.m`
  - Q4_0 simdgroup: `matmul_q4_0_simdgroup_f32` (line ~756)
  - Q4_K simdgroup: `matmul_q4k_simdgroup_f32`
  - SMM_TILE constants: line ~754
- Dispatch: `inference/backend/metal/backend.go` → `MatMulQ4_0()`, `MatMulQ4_K()`
- Batch mode: `metal_bridge_darwin.m` → `begin_batch()`, `end_batch()`, `finish_encode()`
- Layer execution: `inference/runtime/block.go` → `ExecuteWithGPUKV()`
- Prefill loop: `inference/runtime/decode.go` → `DecodeWithGPUKV()`
- FA2 kernel: `metal_bridge_darwin.m` → `flash_attention_2_f32`
