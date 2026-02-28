# Track Plan: Close the 2x Prefill Gap to llama.cpp

Vexel Q4_0 prefill at seqLen=128 is 377 tok/s vs llama.cpp's 803 tok/s (2.13x gap).

**Target: ≥700 tok/s at seqLen=128, LLaMA 2 7B Q4_0, M3 Max.**

## Phase 0: Baseline Profiling
- [x] Task 0.1: Per-dimension GEMM throughput benchmark
    - N=4096/K=4096: 5,400 GFLOPS (attention — slowest, low occupancy)
    - N=11008/K=4096: 7,450 GFLOPS (MLP gate/up)
    - N=4096/K=11008: 6,636 GFLOPS (MLP down)
    - N=32000/K=4096: 8,667 GFLOPS (lm_head — fastest, best occupancy)
    - Matmul-only time: 261ms → 491 tok/s theoretical ceiling
- [x] Task 0.2: Per-operation forward pass profiling
    - **CRITICAL FINDING: SDPA = 31.4% of total time (130ms / 416ms profiled)**
    - Matmuls = 58% (240ms), Other = 10.5% (46ms)
    - FA2 kernel has 3 major issues:
      1. Two-pass QK computation (computes Q·K twice per tile)
      2. GQA waste (87.5% threads idle: headsPerKV=1 but 8 simdgroups allocated)
      3. FP32 shared memory (2× needed vs FP16)

## Phase 1: Prefill SDPA Rewrite [checkpoint: 987c252]
- [x] Task 1.1: Write new FA2v2 prefill kernel (`flash_attention_2_v2_f32`)
    - Single-pass online softmax (eliminates 2-pass Q·K recomputation)
    - Tiles Q positions across simdgroups: tileQ = max(1, 8/headsPerKV)
    - TILE_KV=32 for FP32 headDim=128 within 32KB shared memory budget
    - Strided dimension mapping for bank-conflict-free access
    - Grid: (ceil(seqLen/tileQ), numKVHeads) — 16×32=512 TGs vs old 128×32=4096
- [x] Task 1.2: Wire FA2v2 into dispatch path
    - C dispatch function: `metal_flash_attention_2_v2_f32`
    - Pipeline: `flashAttention2V2Pipeline`
    - Go methods: `SDPAPrefillFA2V2`, updated `SDPAPrefill` with v2 ≥48 tok threshold
    - FP16 KV cache: route prefill through FP32 FA2v2 (kPtr/vPtr still FP32 in scratch)
- [x] Task 1.3: Correctness tests
    - `TestFA2V2_Correctness_vsCPU`: 12 configs (1h, 4h, 32h, GQA 32q/8kv, 32q/4kv, edge cases)
    - Max diff: 8.9e-8 (float32 epsilon), all PASS
    - `TestFA2V2_vs_V1`: 4 configs, max diff 6.0e-8, all PASS
- [x] Task 1.4: SDPA benchmark + integration throughput
    - **Isolated SDPA kernel speedup (A/B):**
      | seq32 | 0.48x (v1 faster) | seq64 | 6.3x | seq128 | 11.4x | seq256 | 15.0x |
    - **Per-op profile (seq128, profiling mode):**
      SDPA: 130ms → 22ms (5.9x), Total: 416ms → 305ms
    - **End-to-end prefill throughput:**
      | seqLen | Before | After | Improvement |
      |--------|--------|-------|-------------|
      | 5      | 95.4   | 96.2  | ~same       |
      | 32     | 343.5  | 315.1 | -8% (v1 still used < 48) |
      | 128    | 381.8  | 562.2 | **+47%**    |
      | 385    | 225.7  | 518.0 | **+129%**   |
    - **Regression tests:** TestPrefillThenDecode_LLaMA2_7B PASS
    - **Decode throughput:** Unchanged (~66 tok/s)

## Remaining Gap Analysis (after Phase 1)

| Metric | Before | After Phase 1 | llama.cpp | Gap |
|--------|--------|---------------|-----------|-----|
| Prefill seq128 | 377 tok/s | **562 tok/s** | 803 tok/s | **1.43x** |
| Prefill seq385 | 106 tok/s | **518 tok/s** | ~793 tok/s | **1.53x** |
| SDPA time/call | 4,088 µs | 701 µs | — | — |

Gap narrowed from 2.13x → 1.43x. Remaining gap is now entirely in GEMM throughput.
GEMM: ~240ms/pass (58%) = 5,500 GFLOPS vs llama.cpp ~13,000 GFLOPS.

## Phase 2: GEMM Kernel Optimization [checkpoint: cd22b51]
- [x] Task 2.1: Analyze llama.cpp kernel_mul_mm for reference
    - llama.cpp uses TILE_M=64, TILE_N=32, TILE_K=32, 128 threads (4 SGs), 6KB shared
    - Blocked 8×8 shared memory layout with stride=8 for simdgroup_load
    - Key insight: their large tile dimension (64) is for weights, not activations
    - Their bandwidth ratio: 12.5 ops/byte (5248 bytes/tile for 65536 ops)
- [x] Task 2.2: Double-buffered shared memory
    - **REJECTED**: 24KB shared memory (2× buffers) reduces occupancy from 2→1 TG/partition
    - Measured: 2796 GFLOPS (vs 3863 without) — 28% regression
    - Apple GPU handles latency differently from NVIDIA; occupancy > double buffering
- [x] Task 2.3: Tile size tuning
    - Tested V1-blocked (32×64×64, 256 threads, 12KB) → 534 tok/s
    - Tested V3 (64×32×32, 128 threads, 6KB) → fast at N≤4096 but slow at N>4096
    - Tested hybrid V3+V1 dispatch → 466 tok/s (pipeline transition overhead)
    - Tested 256-thread B loading → 511 tok/s (contention regression)
    - **WINNER: TILE_M=32, TILE_N=64, TILE_K=32** — 128 threads (4 SGs), 6KB shared
      - Matches llama.cpp's 12.5 ops/byte and 6KB shared memory
      - 2.5× occupancy improvement (5 TGs/partition vs 2)
      - Vectorized float4 activation loads (2 loads vs 8 scalar)
      - simdgroup_barrier hints for instruction scheduling (+4%)
      - Result: **717 tok/s** at seqLen=128
- [x] Task 2.4: Correctness verification + benchmarks
    - All 6 GEMM correctness configs pass (max_diff=0.000689, identical to pre-change)
    - Fusion correctness: bit-identical prefill logits, 20/20 decode tokens match
    - Per-dimension GFLOPS (M=128):
      | Dimension | Before (V1-blocked) | After (TK=32) | Improvement |
      |-----------|---------------------|----------------|-------------|
      | 4096×4096 | ~3,863 | ~6,500 | **+68%** |
      | 11008×4096 | 6,928 | 9,000 | **+30%** |
      | 4096×11008 | 6,077 | 8,382 | **+38%** |
      | 32000×4096 | 7,924 | 10,651 | **+34%** |

## Phase 3: Fused Projections [checkpoint: cd22b51]
- [x] Task 3.1: Fused QKV projection
    - Concatenate Wq+Wk+Wv → Wqkv [12288,4096] at load time
    - FuseQKVWeights() in loader.go + deinterleave kernel
    - Split output via DevicePtrOffset for Q, K, V pointers
    - Free original Wq/Wk/Wv to avoid 2× memory
- [x] Task 3.2: Fused gate_up projection
    - Added W1W3 field to BlockRuntime
    - Concatenate W1+W3 → W1W3 [22016,4096] at load time
    - FuseGateUpWeights() in loader.go + split kernel
    - Single matmul → split into gate [M,11008] and up [M,11008]
- [x] Task 3.3: Correctness verification
    - TestFusedQKVCorrectness: bit-identical prefill, 20/20 decode match
    - TestFusedGateUpCorrectness: bit-identical prefill, 20/20 decode match
    - TestFusedAllCorrectness (both fused): bit-identical prefill, 20/20 decode match
    - Fusion speedup: ~1% at seqLen=128 (689→696 tok/s) — marginal

## Phase 4: Integration Benchmarks [checkpoint: cd22b51]
- [x] Task 4.1: End-to-end prefill benchmark
    - **TARGET MET: 717 tok/s at seqLen=128** (target ≥700)
    - Full results (M3 Max, LLaMA 2 7B Q4_0, 94% battery charging):
      | seqLen | Before (Phase 0) | After Phase 1 | After Phase 2 | Improvement |
      |--------|-------------------|---------------|---------------|-------------|
      | 5      | 95.4              | 96.2          | 96.8          | +1.5%       |
      | 32     | 343.5             | 315.1         | 400.3         | **+16.5%**  |
      | 128    | 377               | 562           | **717.1**     | **+90.2%**  |
      | 385    | 225.7             | 518           | 636.7         | **+182%**   |
    - Gap to llama.cpp: 803/717 = **1.12×** (was 2.13×)
    - Decode throughput: unchanged (~66 tok/s)
    - All existing tests pass (90+ backend, all runtime except 2 pre-existing M=2 failures)
- [x] Task 4.2: Update tracking docs
    - RESULTS.md: Updated prefill table (717 tok/s), added campaign summary,
      per-dimension GFLOPS table, P10 roadmap entry, closed prefill gap note
    - COMPETITORS.md: Updated Vexel numbers (decode 65, prefill 717),
      updated vs MLX and vs llama.cpp gap analysis
