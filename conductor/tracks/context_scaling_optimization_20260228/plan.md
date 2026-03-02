# Track Plan: Context Scaling Optimization (Revised)

Vexel decode degrades severely at real-world context lengths vs llama.cpp's flat scaling.

**Measured 2026-03-01 (LLaMA 2 7B Q4_0, M3 Max):**

| Context | Vexel | llama.cpp | Gap |
|---------|-------|-----------|-----|
| 16      | 71.2 tok/s | 72.9 | -2.3% |
| 128     | 70.3 | 74.2 | -5.3% |
| 512     | 63.1 | 72.7 | -13.2% |
| 1024    | 56.3 | 72.4 | -22.2% |
| 2048    | 45.6 | 71.5 | -36.2% |

**Target: ≤5% degradation at ctx=2048 (matching llama.cpp).**

## Root Cause Analysis

llama.cpp's SDPA overhead from ctx=0→2048 is only ~0.3ms across 32 layers.
Our SDPA overhead is ~8ms at ctx=2048. Theoretical bandwidth minimum for
32 KV heads × 2048 positions × 128 dim × 2 bytes × 2 (K+V) × 32 layers
= 1GB at 400 GB/s = 2.5ms minimum.

llama.cpp achieves <0.3ms overhead — this means they overlap SDPA with matvec.

### Dependency Analysis (Task 1.1 — COMPLETED)

Per-layer Q4_0 FP16 decode (LLaMA SwiGLU) has 7 operations in strict serial chain:
```
FusedRMSNorm+QKV → RoPE+ScatterKV → SDPA → Wo → AddRMSNorm → FusedMLP → W2+Add2
                                     ^^^^
                                     expensive at long ctx
```

**Cross-layer overlap is BLOCKED** because AddRMSNorm does `xPtr += attnOut`
in-place (read-modify-write of xPtr). The next layer's FusedRMSNorm+QKV reads
xPtr, creating a WAW+RAW dependency. Double-buffering xPtr would fix this but
requires significant runtime restructuring (separate track).

**Concurrent dispatch within a layer is also blocked**: SDPA→Wo→AddRMSNorm→FusedMLP
is a strict dependency chain. No independent operations can overlap with SDPA.

**Revised strategy**: Focus on making SDPA kernel itself faster (NWG multi-TG),
and investigate double-buffered cross-layer pipelining as a separate track.

### What we've tried (and results)
- **SDPA v3 (chunk-based, C=32, vectorized half4)**: 1.15-1.9x faster in isolation,
  but only ~1.6 tok/s E2E improvement at ctx=512. Kernel is better but not enough.
- **Tiled split-K SDPA (2-kernel merge)**: SLOWER at all contexts (-27% to -88%).
  Merge kernel dispatch overhead dominates.
- **KV cache consolidation (64→2 buffers)**: Only 0.02ms savings. Not the bottleneck.
- **Concurrent dispatch analysis**: Blocked by serial data dependencies (see above).

## Phase 1: SDPA Multi-Threadgroup NWG Kernel [checkpoint: xxxxxxx]

Use multiple threadgroups per Q head with in-kernel coordination for merge.
Unlike the tiled approach (which used a separate merge kernel and was slower),
this uses device-memory atomics so the last TG to finish does the merge.

**Key difference from tiled (which failed)**:
- Tiled: 2 separate kernel dispatches (tile + merge). Merge dispatch overhead killed it.
- NWG: Single kernel dispatch. TGs coordinate via atomic counter in device memory.
  Last TG to increment counter == numTGs does the final merge in-kernel.

- [x] Task 1.1: Dependency DAG analysis (see above)
- [x] Task 1.2: Write end-to-end baseline test (TDD Red) — 34.8% degradation confirmed
    - TestDecodeContextScalingTarget at ctx=16, 128, 512, 1024, 2048
    - Assert ≤5% degradation at ctx=2048 vs ctx=16
    - Generates actual tokens, measures wall-clock throughput

- [x] Task 1.3: Write sdpa_flash_decode_f16_nwg kernel
    - N threadgroups per Q head where N = max(1, kvLen / 256)
    - Each TG: 8 simdgroups, 256 threads (same as v3)
    - Each TG handles ceil(kvLen/N) positions using v3's chunk algorithm
    - TGs write partial (max, sum, acc[headDim]) to device memory
    - atomic_fetch_add on counter; TG that reads counter == numTGs-1 merges
    - Single dispatch — no merge kernel

- [x] Task 1.4: Write C dispatch function + pipeline registration
    - metal_sdpa_flash_decode_f16_nwg with extra buffer for partials + counter
    - Pipeline registration in backend.go
    - Go dispatch method SDPAF16NWG
    - Lazy-allocated scratch buffers (partials + counters) in Backend struct

- [x] Task 1.5: Correctness tests
    - 15 test cases against CPU reference (all pass, max_diff < 0.002)
    - v3 vs NWG comparison at 10 context lengths (max_diff < 0.002)
    - Edge cases: kvLen=1, kvLen=257, kvLen=500, kvLen=1000
    - GQA configs: 32/8, 32/4, 32/32
    - headDim=64 and headDim=128

- [x] Task 1.6: Benchmark NWG vs v3 at real-world contexts
    - Isolated 32-layer batched results:
      ctx=16: NWG 1.16x faster, ctx=512: 1.45x, ctx=1024: 2.37x, ctx=2048: 2.86x
    - SDPA overhead reduced from ~8ms to ~2.7ms at ctx=2048 (target was ≤2ms)
    - Context scaling overhead reduced from 1020% to 355% (16→2048)

- [x] Task 1.7: Wire NWG into SDPAF16 dispatch (replace v3 at long ctx)
    - NWG used when kvLen > 256, v3 for short contexts
    - Zero + barrier pattern for atomic counter initialization
    - E2E results with NWG:
      ctx=16: 71.9, ctx=128: 71.9, ctx=512: 66.1, ctx=1024: 64.2, ctx=2048: 56.6
    - Degradation reduced from 34.8% to 19.5% (target: ≤5%)
    - Still above target — remaining gap requires Phase 2 (cross-layer pipelining)

## Phase 2: Double-Buffered Cross-Layer Pipelining [checkpoint: xxxxxxx]

If NWG alone doesn't close the gap, add cross-layer overlap via double-buffering.

- [ ] Task 2.1: Implement xPtr double-buffering in decode loop
    - Alternate between xBufA and xBufB across layers
    - Layer N writes to xBuf[N%2], layer N+1 reads from xBuf[N%2]
    - Enables AddRMSNorm of layer N to overlap with QKV of layer N+1

- [ ] Task 2.2: Split per-layer dispatch into two command buffers
    - CB1: FusedRMSNorm+QKV → RoPE+ScatterKV → SDPA → Wo (attention)
    - CB2: AddRMSNorm → FusedMLP → W2+Add2 (MLP)
    - CB1 of layer N+1 can overlap with CB2 of layer N

- [ ] Task 2.3: Correctness tests + benchmark

## Phase 3: Integration & Verification [checkpoint: xxxxxxx]

- [ ] Task 3.1: Final decode benchmark at all context lengths
    - ctx=16, 128, 256, 512, 1024, 2048
    - Compare to llama.cpp at matching contexts
    - Target: ≤5% gap at all context lengths

- [ ] Task 3.2: Verify no regression at short contexts (ctx<128)
- [ ] Task 3.3: Commit and update tracking docs

## Reference: Key Files

- Decode SDPA v1: `metal_bridge_darwin.m` (search `sdpa_flash_decode_f16`)
- Decode SDPA v3: same file (search `sdpa_flash_decode_f16_v3`)
- Tiled SDPA (slower, reference only): same file (search `sdpa_flash_decode_f16_tiled`)
- Go dispatch: `backend.go` (search `SDPAF16`)
- Block execution: `inference/runtime/block.go` (per-layer dispatch flow)
- KV cache: `inference/runtime/gpu_kv_cache.go`
- Decode loop: `inference/runtime/decode.go`
- v3 tests: `sdpa_flash_f16_v3_test.go`
- Profile tests: `profile_test.go`
