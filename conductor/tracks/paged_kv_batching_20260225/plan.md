# Track Plan: Paged KV Cache & Continuous Batching

The PagedKVCache (`inference/kv/paged_cache.go`) is fully implemented with block allocation,
reference counting, and fragment caching. The scheduler can route to either PagedKVCache or
simple GPUKVCache. However, batched decode with paged KV is stubbed (TODO in scheduler line
~326). This track completes the integration to enable true multi-sequence concurrent generation.

## Phase 1: Investigation & Profiling [checkpoint: 5969018]
- [x] Task: Profile current paged KV path
    - Created `paged_kv_profile_test.go` with TestPagedKVRoundtripProfile.
    - Measured GPU↔CPU transfer overhead at kvLen = 64, 256, 512, 1024, 2048.
    - GPU→CPU (new 32KB K/V): ~34 µs constant (negligible).
    - CPU→GPU (full K/V): grows linearly — 53 µs (2MB) → 1148 µs (67MB) at kvLen=2048.
    - SDPA decode: 306 µs (kvLen=64) → 986 µs (kvLen=2048).
    - ReshapePagedKV scatter: ~370 µs constant (includes sync overhead).
    - Per-token roundtrip overhead (×32 layers):
      - kvLen=64:   23% of total (2.9 ms roundtrip vs 12.6 ms total)
      - kvLen=256:  32% of total (6.8 ms roundtrip vs 20.9 ms total)
      - kvLen=512:  36% of total (11.8 ms roundtrip vs 33.1 ms total)
      - kvLen=1024: 43% of total (18.4 ms roundtrip vs 42.6 ms total)
      - kvLen=2048: 55% of total (37.8 ms roundtrip vs 69.3 ms total)
    - Projected speedup from GPU-native path: 1.2× at kvLen=1024, 1.6× at kvLen=2048.
    - CRITICAL: CPU→GPU transfer of full K/V dominates at long sequences.
      The paged path currently re-transfers the ENTIRE KV history every decode step.
    - NOTE: profiling uses per-kernel Sync() which overstates reshape cost.
      In production, reshape+SDPA would be serialized without sync.
- [x] Task: Audit Metal attention kernels
    - Created TestPagedAttentionKernelAudit with functional verification + audit report.
    - EXISTING: reshape_paged_kv_f32 — scatter K/V into GPU block pool (verified correct).
    - EXISTING: sdpa_gqa_f32, sdpa_flash_decode_f32 — decode SDPA (contiguous K/V only).
    - EXISTING: sdpa_prefill_f32, flash_attention_2_f32 — prefill SDPA (contiguous K/V only).
    - MISSING: sdpa_paged_decode_f32 — decode SDPA with block table lookup.
      Interface: Q [numQHeads, headDim], kvPool base ptr, blockTable [numBlocks] int32,
      out [numQHeads, headDim]. Strategy: flash-decode with online softmax across blocks.
    - MISSING (optional): flash_attention_2_paged_f32 — paged prefill (lower priority,
      single-sequence prefill can gather into contiguous buffer first).
    - Block layout: K [blockSize, numKVHeads, headDim] + V [...] per physical block.
      LLaMA 7B with blockSize=16: 512 KB/block.
    - Current data flow: K,V [GPU] → Sync → ToHost → CPU → ToDevice → Sync → SDPA.
    - Target data flow:  K,V [GPU] → ReshapePagedKV → pool [GPU] → PagedSDPA [GPU].

## Phase 2: Implementation
- [ ] Task: Implement paged attention kernel dispatch
    - Write `sdpa_paged_decode_f32` Metal shader with block table indirection.
    - Each threadgroup processes one Q head, iterates K/V blocks with online softmax.
    - Support `tokensInLastBlock` for partial blocks.
    - Add C bridge `metal_sdpa_paged_decode_f32` and Go method `SDPAPagedDecode`.
- [ ] Task: Implement GPU-resident block pool manager
    - Create `GPUBlockPool` struct managing a single large MTLBuffer for all blocks.
    - Block allocation/deallocation with free list (replaces CPU-side PagedKVCache storage).
    - Page table maintained on GPU as int32 buffer, updated via ToDevice on new block alloc.
    - Wire into ExecuteWithPagedKV: replace ToHost/ToDevice with ReshapePagedKV + SDPAPagedDecode.
- [ ] Task: Implement batched decode with paged KV
    - Complete the TODO in scheduler: wire `runDecodeStep` to use paged KV for multi-sequence batches.
    - Each sequence in the batch uses its own block table for K/V retrieval.
    - Handle block allocation/deallocation as sequences start and finish.
- [ ] Task: Implement prefix caching
    - Leverage existing fragment cache in PagedKVCache for shared prompt prefixes.
    - When a new sequence shares a prefix with an existing one, reuse KV blocks (copy-on-write via refcount).
    - Add metrics for cache hit rate.

## Phase 3: Verification
- [ ] Task: Correctness tests
    - Verify paged KV decode produces identical output to simple GPU KV for single sequences.
    - Test multi-sequence concurrent generation (2, 4, 8 sequences) for output correctness.
    - Test prefix sharing: two sequences with the same prefix produce correct independent continuations.
- [ ] Task: Throughput benchmarks
    - Benchmark concurrent throughput: N sequences generating simultaneously.
    - Compare paged vs simple KV cache under concurrent load.
    - Measure prefix cache hit rate and its impact on TTFT for shared-prefix workloads.
