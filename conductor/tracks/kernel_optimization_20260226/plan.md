# Track Plan: Kernel Optimization & Server Hardening

Close the 89% single-stream decode gap against llama.cpp by optimizing Metal kernel
efficiency, reducing dispatch overhead, and unblocking batched benchmarks. Vexel currently
achieves ~9% of M3 Max theoretical memory bandwidth (8.3 tok/s) vs llama.cpp's ~84%
(78.1 tok/s). The matmul kernel is the dominant bottleneck.

Optimization targets from `benchmarks/RESULTS.md` (post P0+P1 analysis):

| Priority | Fix | Expected Impact |
|----------|-----|-----------------|
| **Server timeout** | Configurable request timeout | Unblocks batched benchmarks |
| **Matmul kernel tuning** | Tiling, threadgroup sizing, coalesced reads | Largest single-stream impact |
| **P2: Fused KV scatter** | Single dispatch for K+V per layer | +5-8% decode throughput |
| **P3: Fused attention+norm** | Fused SDPA+RoPE, RMSNorm+residual | +10-15% decode throughput |
| **P4: Command buffer batching** | Encode all per-layer ops in one buffer | +3-5% decode throughput |

---

## Phase 1: Server Timeout Fix [checkpoint: cdec69e]

The `/generate` endpoint has a hardcoded 30-second timeout (`inference/serve/server.go:110`)
that prevents batched throughput benchmarks from completing. This is a quick fix that
unblocks the `run_batched.sh` harness.

- [x] Task: Make server request timeout configurable
    - Added `--timeout` flag to serve subcommand (default 120s, 0 = no timeout).
    - Added `serve.Config` with `RequestTimeout` and `NewServerWithConfig` constructor.
    - Replaced hardcoded `30*time.Second` with configurable value.
    - Fixed concurrent sequence ID collisions (atomic counter replaces `time.Now().UnixNano()`).
    - Files: `inference/serve/server.go`, `inference/cmd/vexel/commands.go`, `inference/cmd/vexel/cli.go`.
- [x] Task: Add /health endpoint
    - Returns 200 OK with `{"status":"ok"}` JSON body and `application/json` content type.
    - Registered at `/health` route, logged on server startup.
    - Used by `run_batched.sh` to detect running servers.
    - File: `inference/serve/server.go`.
- [x] Task: Test and verify
    - 6 new unit tests: default timeout, custom timeout, zero timeout (unlimited),
      short timeout (triggers 408), health response, health content-type.
    - 3 CLI flag tests: --timeout 60, default 120, --timeout 0.
    - Pre-existing flakey E2E concurrent test now passes (atomic counter fix).
    - All serve + CLI tests pass.

## Phase 2: Matmul Kernel Tuning [checkpoint: 6d854d3]

The matmul kernel is the dominant bottleneck. Vexel achieves ~9% memory bandwidth
utilization vs llama.cpp's ~84%. For LLaMA 2 7B Q4_0 decode (M=1, batch=1), the
hot path is `MatMulQ4_0` which dispatches to either `matmul_q4_0_simdgroup_f32` or
`matmul_q4_0_batched_f32` depending on M.

Key code paths:
- Go dispatch: `inference/backend/metal/backend.go:541-577` (MatMulQ4_0)
- C bridge: `metal_bridge_darwin.m` (metal_matmul_q4_0_batched_f32, etc.)
- Metal kernels: embedded in .m file or .h headers
- Pipeline states: `matmulQ4BatchedPipeline`, `matmulQ4SimdgroupPipeline`

Tuning targets:
- Threadgroup sizing for M3 Max (may differ from M1/M2 optimal)
- Memory access patterns (coalesced reads from quantized weight buffers)
- Tiling strategy for L1/L2 cache locality
- Pipeline overlap (prefetch next layer weights)

- [x] Task: Profile matmul kernel and identify dispatch overhead
    - Profiling revealed the dominant bottleneck is NOT the matmul kernel algorithm
      but per-dispatch synchronization: `finish_encode()` calls `[cmdBuf waitUntilCompleted]`
      on every non-batched dispatch (~320 waits/token).
    - Additional overhead: unconditional `b.backend.Sync()` at end of each layer (32/token).
    - Q4_0 decode used multi_output kernel (8 outputs/TG) instead of NR2 (16 outputs/TG).
- [x] Task: Switch Q4_0 decode to NR2 kernel
    - Changed `MatMulQ4_0` and `MatMulQ4_0Offset` to dispatch NR2 for M=1.
    - NR2: 16 outputs per threadgroup (2 per simdgroup), amortizes activation loads.
    - Added `metal_matvec_q4_0_nr2_f32_offset` C bridge for scratch allocator path.
    - Matches approach used by Q6_K, Q4_K, Q8_0, and BF16 formats.
- [x] Task: Remove per-layer Sync() overhead
    - Removed unconditional `b.backend.Sync()` at block.go:1825.
    - Eliminates 32 empty barrier buffer create+commit+wait per token.
    - Safe: Metal FIFO ordering guarantees layer N writes complete before layer N+1 reads.
    - Final sync point remains in DecodeWithGPUKV after all layers complete.
- [x] Task: Investigate async dispatch (reverted)
    - Attempted removing `[cmdBuf waitUntilCompleted]` from `finish_encode()`.
    - Would have eliminated ~320 synchronous waits per token.
    - Failed: memory coherency between separate command buffers requires explicit wait.
    - Proper fix is command buffer batching (Phase 5), not per-dispatch async.
- [ ] Task: Re-benchmark and measure improvement
    - Run decode throughput (200 tokens, LLaMA 2 7B Q4_0, M3 Max).
    - Run prefill at 12, 124, 385 tokens.
    - Compare against baseline 8.3 tok/s decode, 41/152/107 tok/s prefill.
    - Update `benchmarks/RESULTS.md` with new numbers.

## Phase 3: Fused KV Scatter (P2)

KV cache uses separate K and V buffers per layer (64 total for 32 layers). Each
decode step dispatches 2 scatter kernels per layer (one for K, one for V). Fusing
into a single dispatch halves the kernel dispatch count.

Key code:
- `inference/runtime/gpu_kv_cache.go:155-242` (AppendKV — 2x ScatterKV calls)
- `inference/backend/metal/backend.go:1449-1481` (ScatterKV, ScatterKVF16, ScatterKVF32ToF16)
- Metal kernels: `scatter_kv_f32`, `scatter_kv_f16`, `scatter_kv_f32_to_f16`

Design: Interleave K/V in a single buffer per layer `[K_row0, V_row0, K_row1, V_row1, ...]`
or pass both K and V pointers to a single kernel that writes both.

- [ ] Task: Create fused scatter kernel (Metal + C bridge)
    - New Metal kernel: `scatter_kv_fused_f32` — takes K, V source and K, V dest pointers.
    - Single dispatch writes both K and V for all heads in one pass.
    - Add corresponding C bridge function and Go wrapper.
    - Support both F32 and FP16 variants.
- [ ] Task: Wire fused scatter into GPUKVCache.AppendKV
    - Add backend interface method for fused scatter.
    - Update AppendKV to call fused scatter when available, fallback to 2x scatter.
    - Preserve existing FP16 and Q8_0 KV cache paths.
- [ ] Task: Test correctness and benchmark
    - Unit test: fused scatter output matches 2x separate scatter output.
    - E2E: identical model output with fused vs unfused scatter.
    - Benchmark: measure per-token overhead reduction (expect ~32-64 fewer dispatches).

## Phase 4: Fused Attention+Norm Kernels (P3)

Attention, RoPE, layer norm, and residual add are separate kernel dispatches.
Fusing reduces memory I/O by keeping intermediates in registers/threadgroup memory.

Key code:
- `inference/runtime/block.go:1100-1140` — RMSNorm+QKV projection (already partially fused for Q4_0)
- `inference/runtime/block.go:1237-1246` — mid-layer sync point
- Per-layer ops: RMSNorm, Q/K/V matmul, RoPE, SDPA, output projection, residual add, FFN

Currently fused (for decode, Q4_0 only):
- `MatMulQ4_0_FusedRMSNorm` — RMSNorm + MatMul in one dispatch
- `MatMulQ4_0_FusedRMSNormF16` — same but outputs FP16

Still separate:
- RoPE applied after Q/K projection
- Residual add after attention output projection
- FFN norm + gate/up projection

- [ ] Task: Fuse RoPE into attention projection
    - Current: QKV projection, then separate RoPE kernel on Q and K.
    - New: `FusedRMSNorm+MatMul+RoPE` kernel that applies RoPE in the same dispatch.
    - Or: fuse RoPE into SDPA kernel input path.
    - Choose approach based on profiling data from Phase 2.
- [ ] Task: Fuse residual add + RMSNorm for FFN
    - Current: separate `Add(residual, attnOut)` then `RMSNorm(ffnInput)`.
    - New: `FusedAddRMSNorm` kernel — single read of both inputs, one write.
    - File: new Metal kernel + C bridge + Go backend method.
- [ ] Task: Test correctness and benchmark
    - Unit test: fused kernel output matches unfused pipeline (within FP32 tolerance).
    - E2E: identical model output with fused vs unfused.
    - Benchmark: expect +10-15% decode throughput from reduced memory I/O.

## Phase 5: Command Buffer Batching (P4) [checkpoint: 4fc581c]

Vexel submitted separate command buffers for each dispatch (~320/token), each
calling `[cmdBuf waitUntilCompleted]`. This CPU-GPU roundtrip dominated all
other overhead. The fix: encode all operations per layer into one command buffer
with `memoryBarrierWithScope:MTLBarrierScopeBuffers` between dependent dispatches.

Key changes:
- `metal_bridge_darwin.m`: Added `metal_memory_barrier()` (memoryBarrierWithScope)
- `backend.go`: Added `MemoryBarrier()` to Metal backend
- `backend/backend.go`: Added `MemoryBarrier()` to Batcher interface
- `block.go`: Removed scratch-disables-batching guard, inserted ~7-8 barrier()
  calls per layer at all scratch write→read dependency points, replaced mid-layer
  EndBatch/BeginBatch with a single barrier.

**Result: 8x decode speedup (7.7 → 61.3 tok/s), 2x prefill (48 → 103 tok/s)**

- [x] Task: Re-enable command buffer batching with scratch allocator
    - Used `[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]` instead of MTLFence.
    - Barrier inserted at every scratch write→read dependency point.
    - Output correctness verified: identical text output pre/post batching.
    - All Metal backend tests pass (53 tests including paged SDPA).
- [x] Task: Replace mid-layer batch split with memory barrier
    - Old approach: EndBatch/BeginBatch mid-layer (commits + waits on first half).
    - New approach: single barrier() call — stays in same command buffer.
    - Eliminated unnecessary batch split overhead.
- [x] Task: Benchmark command buffer batching impact
    - Decode: 7.7 → 61.3 tok/s (8x improvement, 65.9% BW utilization)
    - Prefill (~12 tok): ~48 → ~103 tok/s (2x improvement)
    - Gap to llama.cpp: -89.4% → -21.5%
    - Variance: ±0.2 tok/s over 5 runs (very consistent)

## Phase 6: Final Benchmarks & Reporting

- [ ] Task: Run full competitive benchmark suite
    - Decode throughput: Vexel vs llama.cpp vs Ollama (same harness as Phase 5 benchmarks).
    - Prefill throughput at 12, 124, 385 tokens.
    - Model load time.
    - Batched throughput with `run_batched.sh` (now unblocked by Phase 1).
- [ ] Task: Update RESULTS.md and README.md
    - Update performance table with post-optimization numbers.
    - Update optimization roadmap status.
    - Update competitive positioning based on new gap.
