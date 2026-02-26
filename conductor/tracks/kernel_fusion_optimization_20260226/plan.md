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

## Phase 1: Profiling & Measurement
- [ ] Task: Instrument kernel dispatch count
    - Count exact number of Metal kernel dispatches per forward pass for LLaMA 2 7B.
    - Measure time spent in dispatch overhead vs actual compute.
    - Use `VEXEL_GPU_PROFILE=1` to get per-kernel timing, identify the longest and shortest ops.
    - Profile MTLBuffer allocation: count `newBufferWithLength` calls per token.
- [ ] Task: Identify fusion candidates
    - Map the full forward pass kernel sequence (e.g., RMSNorm → Q_proj matmul → RoPE → ...).
    - Identify adjacent ops that read/write the same tensor without intervening dependencies.
    - Priority fusion targets (highest dispatch-overhead-to-compute ratio):
      - RMSNorm + MatMul (norm output feeds directly into projection)
      - SiLU×Mul + Down projection (fused activation output feeds into matmul)
      - RoPE + KV cache scatter (rotated vectors go directly to cache)
    - Estimate dispatch reduction (e.g., 40 kernels → 15 fused kernels).
- [ ] Task: Measure memory allocation overhead
    - Profile `backend.Alloc()` calls per forward pass: count, sizes, cumulative time.
    - Compare: current per-allocation MTLBuffer creation vs sub-allocation from pre-allocated pool.
    - Determine if allocation overhead is significant relative to compute time.

## Phase 2: GPU Memory Pool
- [ ] Task: Implement sub-allocating scratch buffer
    - Pre-allocate a single large MTLBuffer for scratch space (sized via `ModelConfig.ScratchBytes`).
    - `Alloc()` returns offsets into this buffer instead of creating new MTLBuffer objects.
    - `ResetPool()` resets the offset to 0 (same as current arena reset).
    - Kernels receive `(buffer, offset)` pairs instead of individual buffer pointers.
- [ ] Task: Update Metal kernel dispatch for offset-based buffers
    - Modify `setBuffer:offset:atIndex:` calls to use base buffer + offset.
    - All temporary activations share one underlying MTLBuffer.
    - Eliminates per-allocation `[device newBufferWithLength:]` overhead.
- [ ] Task: Benchmark allocation improvement
    - Measure tok/s improvement from eliminating per-allocation overhead.
    - Profile remaining dispatch overhead to quantify fusion opportunity.

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
