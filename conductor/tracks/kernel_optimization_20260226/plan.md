# Track Plan: Kernel Optimization & Server Hardening

Close the single-stream decode gap against llama.cpp by optimizing Metal kernel
efficiency, reducing dispatch overhead, and unblocking batched benchmarks.

**Current state (post Phase 5):** Vexel achieves **61.3 tok/s** (65.9% BW utilization)
vs llama.cpp's 78.1 tok/s (84.0%). Gap: **-21.5%** (down from -89.4%).

Optimization targets from `benchmarks/RESULTS.md` (post P0+P1 analysis):

| Priority | Fix | Expected Impact | Status |
|----------|-----|-----------------|--------|
| **Server timeout** | Configurable request timeout | Unblocks batched benchmarks | ✅ DONE |
| **Matmul kernel tuning** | NR2 tested (neutral), per-layer sync removed | Profile insights | ✅ DONE |
| ~~P2: Fused KV scatter~~ | ~~Single dispatch for K+V per layer~~ | ~~+5-8%~~ | ⏭️ SKIPPED |
| ~~P3: Fused attention+norm~~ | ~~Fused SDPA+RoPE, RMSNorm+residual~~ | ~~+10-15%~~ | ⏭️ SKIPPED |
| **P4: Command buffer batching** | Encode all per-layer ops in one buffer | **8x decode speedup** | ✅ DONE |

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
- [x] Task: Re-benchmark and measure improvement
    - Superseded by Phase 5 benchmarking: decode 7.7→61.3 tok/s, prefill ~48→~103 tok/s.
    - NR2 kernel was neutral vs multi_output for Q4_0 (both 7.6-7.7 tok/s pre-batching).
    - Per-layer Sync removal was neutral (within noise).
    - Real bottleneck was per-dispatch synchronization (fixed in Phase 5).

## Phase 3: Fused KV Scatter (P2) — SKIPPED

**Decision: Not worth implementing.** With command buffer batching (Phase 5), the 64
scatter dispatches per token are effectively free (~0.5-1µs CPU encoding per dispatch,
~16-32µs total). Fusion would save below the noise floor.

Analysis:
- Pre-batching: each dispatch cost ~10-50µs (waitUntilCompleted roundtrip) → 64 × 50µs = 3.2ms
- Post-batching: each dispatch costs ~0.5µs (encode only) → 64 × 0.5µs = 32µs
- `TestFusionABComparison` confirmed: dispatch count reduction does NOT translate to
  measurable throughput improvement. Bottleneck is memory bandwidth in Q4_0 matmul kernels.

- [x] Task: Analyze whether fusion is worth implementing → NO
    - Research concluded dispatches are free with batching.
    - The intermediate memory I/O saved (K+V scatter: ~64 KB/layer) is negligible
      compared to weight reads (~114 MB/layer).

## Phase 4: Fused Attention+Norm Kernels (P3) — PARTIALLY DONE / DEPRIORITIZED

Two of three fusion targets were already implemented before this track:
- ✅ `AddRMSNorm` — fused Add1 + RMSNorm2 (saves 1 dispatch/layer = 32 total)
- ✅ `FusedMLP` — fused SiLU(x@W1)*(x@W3) (saves 2 dispatches/layer = 64 total)

**Decision: Remaining fusion (RoPE into matmul) deprioritized.** Analysis shows:
- RoPE intermediate memory I/O: ~128 KB per layer × 32 = 4 MB total
- Weight reads: ~3.56 GB per token → intermediates are 0.1% of total I/O
- `TestFusionABComparison` proved fusion has NO measurable throughput impact
- The remaining 21.5% gap is entirely in Q4_0 matmul kernel bandwidth utilization
  (65.9% vs llama.cpp's 84%), not in dispatch overhead or intermediate memory I/O

Current decode pipeline (12 dispatches/layer × 32 layers + 3 = 387 total):
1. 3× FusedRMSNorm+MatMul (Q, K, V)
2. RoPE
3. 2× ScatterKV (K, V)
4. SDPA
5. MatMulQ4_0 (Wo)
6. AddRMSNorm (fused Add1+RMSNorm2) ← already fused
7. FusedMLP (fused W1+W3+SiLUMul) ← already fused
8. MatMulQ4_0 (W2)
9. Add (residual)

- [x] Task: Fuse residual add + RMSNorm for FFN → ALREADY DONE
    - `AddRMSNorm` kernel: x += attn_output, normOut = RMSNorm(x) in single dispatch.
    - Implemented in prior track, wired in block.go for SwiGLU models.
- [~] Task: Fuse RoPE into attention projection → DEPRIORITIZED
    - Feasible: apply rotation in matvec kernel after computing output pairs.
    - Estimated savings: ~4 MB memory I/O (~0.01ms at 400 GB/s) = negligible.
    - Not worth the code complexity given the minimal throughput impact.
- [x] Task: Test correctness and benchmark
    - `TestFusionABComparison`: fused (387 dispatches) vs unfused (451 dispatches).
    - Result: no measurable throughput difference.
    - `TestFusionCorrectness`: token-for-token identical output between fused and unfused paths.

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

Run comprehensive benchmarks to establish final competitive positioning.
Post-batching decode: 61.3 tok/s (-21.5% vs llama.cpp). Prefill, batched, and
longer-context benchmarks still need measurement.

- [ ] Task: Run full competitive benchmark suite
    - Decode throughput: Vexel vs llama.cpp vs Ollama (200 tokens, M3 Max).
    - Prefill throughput at 12, 124, 385 tokens (Vexel vs llama.cpp).
    - Model load time comparison.
    - Decode throughput at varying context lengths (50, 200, 500, 1000 tokens).
- [ ] Task: Update RESULTS.md
    - Fill in TBD prefill numbers (124, 385 tokens).
    - Add context length scaling data.
    - Update optimization roadmap with final status (P2/P3 skipped, P4 done).
- [ ] Task: Update README.md performance section
    - Update headline performance numbers.
    - Update competitive positioning (from -89% to -21.5%).
    - Document the optimization journey and key insights.
