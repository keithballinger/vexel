# Prefill Pipeline Optimization (P9)

> **Goal:** Close the 4-5x prefill throughput gap to llama.cpp
> **Baseline:** Vexel Q4_0 200 tok/s @ seqLen=128, llama.cpp 803 tok/s
> **After Phase 1:** Vexel Q4_0 377 tok/s @ seqLen=128 (1.88x improvement)
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

### Phase 1: GEMM Kernel — Half-Precision + Block Dequantization ✅ DONE (f4dd160)

**Achieved:** 2.62x isolated GEMM improvement, 1.88x full-model prefill at seqLen=128.

Key techniques (final kernel):
- Half-precision shared memory (`threadgroup half*`) for A and B tiles
- TILE_K=64 (2 Q4_0 blocks per K-tile, halves barrier count 128→64)
- Block-based B dequant: threads 0-127 each handle one Q4_0 block (32 values),
  single scale load per block (32x fewer scale reads vs flat approach)
- `simdgroup_half8x8` loads → `simdgroup_float8x8` accumulators (mixed-precision MAC)
- 12KB total shared memory (4KB A + 8KB B) — allows 2 TGs per compute unit

Approaches explored (M=128, N=K=4096):
| Configuration | GFLOPS | vs Baseline |
|---|---|---|
| Baseline (float, TILE_K=32) | 2033 | 1.00x |
| Half shared mem only (TILE_K=32) | 1510 | 0.74x (regression) |
| 128 threads (4 SG) | 2079 | 1.02x (no gain) |
| Half + TILE_K=64 flat dequant | 3397 | 1.67x |
| Half + TILE_K=128 block dequant | 4466 | 2.20x (24KB kills occupancy) |
| **Half + TILE_K=64 block dequant** | **5327** | **2.62x** |

Full-model prefill results:
| seqLen | Before | After | Improvement |
|--------|--------|-------|-------------|
| 5 | 96.2 tok/s | 94.2 tok/s | Same (NR2 kernel path) |
| 32 | 145.2 tok/s | 337.2 tok/s | 2.32x |
| 128 | 200.2 tok/s | 376.9 tok/s | 1.88x |
| 385 | 151.7 tok/s | 223.2 tok/s | 1.47x |

- [x] Task 1.1: Baseline GEMM benchmark (2033 GFLOPS)
- [x] Task 1.2: Implement optimized kernel (half + TILE_K=64 + block dequant)
- [x] Task 1.3: Correctness tests pass (max_diff 0.000689)
- [x] Task 1.4: Full-model prefill: 377 tok/s (target was ≥400, close)

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
