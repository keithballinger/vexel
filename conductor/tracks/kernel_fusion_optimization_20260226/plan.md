# Track Plan: Kernel Fusion & GPU Optimization

Vexel's forward pass already uses unified memory (MTLResourceStorageModeShared) with zero
per-token CPU↔GPU copies. The 40% bandwidth utilization gap vs MLX (~44 tok/s vs ~115 tok/s
on M3 Max for LLaMA 2 7B Q4_0) comes from two sources:

1. **Kernel dispatch overhead**: ~30-50 individual Metal kernel launches per token, each with
   pipeline state setup, buffer binding, and inter-kernel GPU stalls. MLX fuses adjacent ops
   into single kernels via lazy graph evaluation.

2. **Per-allocation MTLBuffer creation**: The arena allocator calls `[device newBufferWithLength:]`
   for each temporary activation instead of sub-allocating from a pre-allocated buffer pool.

Theoretical max on M3 Max (400 GB/s ÷ 3.6 GB weights) = ~111 tok/s.
Target: reach 70-80% bandwidth utilization (~78-89 tok/s), competitive with llama.cpp.

## Phase 1: Profiling & Measurement [checkpoint: eedb1ba]
- [x] Task: Instrument kernel dispatch count
    - Created `dispatch_profiler.go` with DispatchProfiler struct and ForwardPassProfile.
    - Instrumented all 48 kernel dispatch methods in `backend.go` with RecordDispatch calls.
    - Instrumented `Alloc()` with allocation tracking (pool hit detection, timing).
    - 10 unit tests in `dispatch_profiler_test.go` + integration tests for real Backend.
- [x] Task: Identify fusion candidates
    - Created `fusion_analysis_test.go` mapping the full decode and prefill kernel sequences.
    - Decode: 14 dispatches/layer × 32 layers + 3 = 451 total.
    - Prefill: 16 dispatches/layer × 32 layers + 3 = 515 total.
    - Identified 4 fusion targets for 35% dispatch reduction (451→291):
      - RoPE+ScatterKV (saves 64 dispatches, 14%)
      - SiLUMul+W2 (saves 32, 7%)
      - Wo+Add (saves 32, 7%)
      - W2+Add (saves 32, 7%)
- [x] Task: Measure memory allocation overhead
    - Profiled buffer pool reuse: 100% hit rate on second pass with ResetPool.
    - Fresh allocation: ~5.54 µs/alloc, pooled: ~0.02 µs/alloc (277× faster).
    - 288 allocs per pass × 5.54 µs = ~1.6 ms total fresh allocation overhead.
    - Pool reuse eliminates virtually all allocation overhead (~6 µs total).
    - Conclusion: allocation overhead is minor vs compute time; kernel dispatch
      count is the primary optimization target.

## Phase 2: GPU Memory Pool [checkpoint: bd6f856]
- [x] Task: Implement sub-allocating scratch buffer
    - Created `scratch_allocator.go` with ScratchAllocator (bump allocator from single MTLBuffer).
    - 256-byte alignment for Metal optimal access.
    - WriteAt/ReadAt for CPU-side data access to sub-regions.
    - Integrated into Backend: InitScratch(), ScratchAlloc(), ScratchReset().
    - Fallback to pool-based Alloc when scratch is exhausted or not initialized.
    - 8 unit tests covering creation, bump alloc, alignment, reset, OOM, data isolation.
- [x] Task: Update Metal kernel dispatch for offset-based buffers
    - Added `metal_buffer_contents` to C bridge for CPU-side scratch access.
    - Added offset-aware C bridge functions: `metal_add_f32_offset`,
      `metal_rmsnorm_f32_offset`, `metal_silu_mul_f32_offset`.
    - Added Go methods: AddOffset, RMSNormOffset, SiLUMulOffset.
    - Existing ~40+ kernels retain zero-offset dispatch (pool path unchanged).
    - Integration tests prove offset-aware dispatch produces correct results:
      - TestScratchAllocatorAddKernel: a+b with shared buffer ✓
      - TestScratchAllocatorRMSNormKernel: RMSNorm with mixed scratch/pool ✓
- [x] Task: Benchmark allocation improvement
    - Pool-based (second pass): 0.02 µs/alloc, 100% hit rate.
    - Scratch-based: zero overhead (bump pointer, no profiler tracking).
    - Both approaches eliminate fresh MTLBuffer creation overhead (5.54 µs/alloc).
    - Confirms Phase 1 finding: allocation is not the bottleneck.
    - Kernel dispatch count (451) remains the primary optimization target for Phase 3.

## Phase 3: Fused Transformer Block Kernels
- [ ] Task: Fused RMSNorm + MatMul kernel
    - Single Metal kernel: read input, compute RMSNorm in threadgroup shared memory,
      immediately use normalized output as matmul input.
    - Saves: 1 kernel launch + 1 intermediate buffer write/read (hidden_size * 4 bytes).
    - Apply to all 4 attention projections (Q, K, V, O) and MLP projections.
    - For quantized weights: fused RMSNorm + dequant matvec.
- [ ] Task: Fused Gate-Up + SiLU×Mul kernel
    - Current: gate_proj → up_proj → SiLU(gate) × up → down_proj (4 kernels).
    - Fused: single kernel computes gate_proj and up_proj in parallel, applies SiLU×Mul
      inline, outputs directly to down_proj input buffer.
    - Saves: 3 kernel launches + 2 intermediate buffers.
- [ ] Task: Fused RoPE + KV scatter kernel
    - Current: RoPE rotation → separate KV cache scatter (2 kernels per K and V).
    - Fused: rotate Q/K and scatter K/V into cache in a single kernel.
    - Saves: 2 kernel launches.

## Phase 4: Attention Fusion
- [ ] Task: Fused attention pre-processing
    - Combine: Q/K/V projections + RoPE + KV cache update into single dispatch.
    - Input: hidden state. Output: rotated Q (for SDPA) + updated KV cache.
    - This is the most impactful fusion — replaces ~8-10 kernel launches with 1.
- [ ] Task: Fused post-attention
    - Combine: O projection + residual add + feed-forward RMSNorm.
    - Input: SDPA output. Output: normalized residual for MLP.
    - Saves: 3 kernel launches + 2 intermediate buffers.
- [ ] Task: Full fused transformer block
    - Ultimate goal: entire transformer block as 3-4 kernel launches instead of ~15-20.
    - Decomposition: [fused_attn_preprocess] → [flash_attention] → [fused_attn_postprocess] → [fused_mlp].
    - FlashAttention-2 stays as its own kernel (already fused and complex).

## Phase 5: Verification & Benchmarking
- [ ] Task: Correctness verification
    - Fused kernels must produce bit-identical output to unfused path.
    - Run deterministic generation (temp=0) and compare token-for-token.
    - Add toggle: `VEXEL_FUSED_KERNELS=0` to disable fusion for debugging.
- [ ] Task: Performance benchmarks
    - Measure decode tok/s with each fusion applied incrementally.
    - Report: kernel count per token, bandwidth utilization %, tok/s.
    - Target: 70-80% bandwidth utilization (~78-89 tok/s on M3 Max).
    - Compare against llama.cpp on same hardware and model.
- [ ] Task: Prefill impact
    - Measure prefill tok/s improvement from fused kernels (batched operations).
    - Fusion should help prefill even more than decode (more compute per dispatch).
