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

## Phase 1: Server Timeout Fix

The `/generate` endpoint has a hardcoded 30-second timeout (`inference/serve/server.go:110`)
that prevents batched throughput benchmarks from completing. This is a quick fix that
unblocks the `run_batched.sh` harness.

- [ ] Task: Make server request timeout configurable
    - Add `--timeout` flag to serve subcommand (default 120s, 0 = no timeout).
    - Wire through `serve.Config` to `handleGenerate` and `handleStream`.
    - Replace hardcoded `30*time.Second` with configurable value.
    - Files: `inference/serve/server.go`, `inference/cmd/vexel/commands.go`.
- [ ] Task: Add /health endpoint
    - Return 200 OK with `{"status":"ok"}` JSON body.
    - Used by `run_batched.sh` to detect running servers.
    - File: `inference/serve/server.go`.
- [ ] Task: Test and verify
    - Unit test for configurable timeout (mock scheduler, verify context deadline).
    - Unit test for /health endpoint.
    - E2E: start server with `--timeout 120`, run `run_batched.sh` with N=1,2 clients.

## Phase 2: Matmul Kernel Tuning

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

- [ ] Task: Profile matmul kernel with Metal GPU profiler
    - Use `VEXEL_GPU_PROFILE=1` to capture per-kernel timing.
    - Use Xcode Metal System Trace (or `metal-trace`) to identify:
      - GPU occupancy and threadgroup utilization
      - Memory bandwidth utilization per kernel
      - Stall reasons (ALU-bound vs memory-bound)
    - Document baseline: latency per matmul dispatch, total matmul time per token.
- [ ] Task: Benchmark threadgroup size variations
    - Current: uses `pipeline.maxTotalThreadsPerThreadgroup` (likely 1024).
    - Test 64, 128, 256, 512, 1024 threadgroup sizes for Q4_0 matvec (M=1).
    - Measure tok/s for each configuration.
    - Find optimal threadgroup size for M3 Max architecture.
- [ ] Task: Optimize Q4_0 matvec memory access pattern
    - Audit the Metal kernel source for coalesced memory reads.
    - Q4_0 format: 32 values packed in 18 bytes (16 bytes data + 2 bytes scale).
    - Ensure threads within a SIMD group read contiguous memory addresses.
    - Consider double-buffering: load next tile while computing current tile.
- [ ] Task: Implement tiled matmul for prefill (M>1)
    - Current: `matmul_q4_0_batched_f32` used for M<8, simdgroup for M>=8.
    - For prefill workloads (M=12..512), tiling for L2 cache can improve throughput.
    - Target: match or exceed current 152 tok/s at 124-token prefill.
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

## Phase 5: Command Buffer Batching (P4)

Vexel submits separate command buffers for each of the ~7 operations per layer.
At 32 layers, that's ~224 command buffer submissions per token. Each submission has
~0.5us overhead on M3 Max.

Key code:
- `inference/runtime/block.go:969-985` — batching setup (disabled when scratch active)
- `inference/backend/metal/backend.go:440-450` — BeginBatch/EndBatch
- `metal_bridge_darwin.m` — `metal_begin_batch` / `metal_end_batch`

Current status: batching exists but is disabled when scratch allocator is active
(line 978) due to Metal memory hazards with shared buffer. Need either:
(a) Use MTLFence between dependent operations within a single command buffer, or
(b) Use separate command encoders with explicit resource dependencies.

- [ ] Task: Re-enable command buffer batching with scratch allocator
    - Add MTLFence-based barriers between dependent operations within a batch.
    - Or: use `makeComputeCommandEncoder(descriptor:)` with resource barriers.
    - Ensure scratch allocator's single MTLBuffer has proper memory barriers.
    - Test: output correctness with scratch + batching enabled.
- [ ] Task: Optimize batch boundaries
    - Currently: entire layer in one batch with mid-layer sync (line 1243).
    - Profile: determine if mid-layer sync can be eliminated with proper fencing.
    - Test: `VEXEL_SKIP_MID_SYNC=1` with fence-based batching for correctness.
- [ ] Task: Benchmark command buffer batching impact
    - Measure per-token latency with batching ON vs OFF.
    - Expect +3-5% from reduced command submission overhead.
    - Test at both decode (seqLen=1) and prefill (seqLen=128) batch sizes.

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
