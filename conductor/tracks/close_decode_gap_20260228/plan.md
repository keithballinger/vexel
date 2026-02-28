# Track Plan: Close the 15% Decode Gap to llama.cpp

Vexel Q4_0 decode is 64.8 tok/s vs llama.cpp's 76.3 tok/s (-15.1%) on M3 Max.
BW utilization: 69.7% vs 82.0%. The gap is entirely in Q4_0 matvec kernel efficiency.

**Target: ≥73 tok/s decode throughput (≤5% gap to llama.cpp).**

## Background

The current M=1 decode path uses `matvec_q4_0_multi_output_f32`:
- 8 outputs per threadgroup (1 per simdgroup), 256 threads
- Each thread in simdgroup handles blocks with stride 32
- Vectorized float4 activation loads, simd_sum reduction
- NR2 (16 outputs/TG) and NR4 (32 outputs/TG) benchmarked: no benefit
  due to Q4_0's 18-byte block layout causing L1 cache pressure

The `matvec_q4_0_optimized_f32` variant exists with shared-memory activation
caching but is not used in the main decode path.

## Phase 0: Decode Profiling Baseline
- [x] Task 0.1: Per-operation decode profiling at ctx=16
    - Test: TestDecodeProfileBaseline in throughput_bench_test.go
    - Baseline: **62.3 tok/s** (16.0 ms/token median), BW util 55.5% (222 GB/s)
    - Per-op breakdown (sync-inflated, but relative proportions valid):
      FusedMLP=16.6%, FusedRMSNorm+QKV_F16=16.3%, W2=13.0%,
      SDPA=10.8%, RoPE=9.6%, Wo=9.5%, AddRMSNorm=8.2%, KVCache=8.1%, Add2=7.7%
    - 419 dispatches/token: MatMulQ4_0(64) + FusedRMSNorm+MatMul(96) + FusedMLP(32) +
      ScatterKV(64) + Convert(32) + RoPE(32) + SDPA(32) + Add(32) + AddRMSNorm(32) + misc
    - **Matmuls dominate** (7 per layer = 224 dispatches, 53.5%)
    - FP16 KV path active: adds 32 Convert dispatches/token
- [x] Task 0.2: Per-layer matmul bandwidth measurement
    - Overall: 3.56 GB × 62.3 tok/s = 222 GB/s (55.5% of 400 GB/s)
    - Theoretical ceiling: 400/3.56 = 112 tok/s
    - Gap analysis: 76.3 llama.cpp → 82.0% BW, Vexel 62.3 → 55.5%
    - Remaining bandwidth gap: 26.5 percentage points
- [x] Task 0.3: Profile command buffer sync overhead
    - Cross-layer batching IS active (line 476 of decode.go wraps all layers in single CB)
    - 2 batches/token (outer cross-layer + final sync)
    - 40µs avg per dispatch within batch (16.75ms / 419 dispatches)
    - Sync overhead NOT the bottleneck — the batch path is working correctly

## Phase 1: Decode Command Buffer Optimization
- [ ] Task 1.1: Audit decode command buffer batching
    - Verify whether decode already uses the batch-encode path from
      commit 4fc581c, or still uses per-dispatch synchronization
    - If batching IS active: measure remaining sync overhead
    - If batching is NOT active for M=1: wire it up
- [ ] Task 1.2: Minimize per-token CPU overhead
    - Profile Go→CGo→Metal dispatch overhead per decode token
    - Consider fusing consecutive matmuls into single dispatch where possible
    - Benchmark with reduced sync points

## Phase 2: Matvec Kernel Optimization
- [ ] Task 2.1: Shared-memory activation caching kernel
    - Test `matvec_q4_0_optimized_f32` variant that pre-loads activations
      to shared memory before computing multiple output rows
    - Key hypothesis: shared-memory activation reuse reduces device BW
    - Benchmark vs current multi_output at all decode dimensions
- [ ] Task 2.2: Vectorized Q4_0 block extraction
    - Current: byte-by-byte nibble extraction with masks
    - Try: uint16_t packed reads (2 bytes → 4 elements) matching llama.cpp
    - Try: uint32_t reads (4 bytes → 8 elements) for wider extraction
- [ ] Task 2.3: Tune output parallelism
    - Current: 8 outputs/TG (1 per simdgroup)
    - Test: 4 outputs/TG with 2 simdgroups per output (more reduction parallelism)
    - Test: 16 outputs/TG with shared memory activation broadcast
    - Measure occupancy tradeoffs at each configuration

## Phase 3: Integration Benchmarks
- [ ] Task 3.1: End-to-end decode benchmark
    - Measure at ctx=16, 128, 512 with temperature=0, 50 tokens
    - Compare to llama.cpp baseline
    - Target: ≥73 tok/s at short context
- [ ] Task 3.2: Verify no prefill regression
    - Confirm prefill throughput still ≥700 tok/s at seqLen=128
- [ ] Task 3.3: Update tracking docs
    - Update RESULTS.md with new decode numbers
    - Update COMPETITORS.md gap analysis

## Reference: Key Files

- Matvec kernel: `inference/backend/metal/metal_bridge_darwin.m` (lines 238-318)
- NR2 kernel: same file (lines 329-451)
- Optimized kernel: same file (lines 653-737)
- Dispatch routing: `inference/backend/metal/backend.go` (lines 595-631)
- SDPA decode: `inference/backend/metal/metal_bridge_darwin.m` (lines 4624-4747)
