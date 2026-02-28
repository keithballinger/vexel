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

## Phase 2: GEMM Kernel Optimization
- [ ] Task 2.1: Analyze llama.cpp kernel_mul_mm for reference
- [ ] Task 2.2: Double-buffered shared memory
- [ ] Task 2.3: Tile size tuning
- [ ] Task 2.4: Correctness verification + benchmarks

## Phase 3: Fused Projections
- [ ] Task 3.1: Fused QKV projection
- [ ] Task 3.2: Fused gate_up projection
- [ ] Task 3.3: Correctness verification

## Phase 4: Integration Benchmarks
- [ ] Task 4.1: End-to-end prefill benchmark
- [ ] Task 4.2: Update tracking docs
