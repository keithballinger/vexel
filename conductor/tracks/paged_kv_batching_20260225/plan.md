# Track Plan: Paged KV Cache & Continuous Batching

The PagedKVCache (`inference/kv/paged_cache.go`) is fully implemented with block allocation,
reference counting, and fragment caching. The scheduler can route to either PagedKVCache or
simple GPUKVCache. However, batched decode with paged KV is stubbed (TODO in scheduler line
~326). This track completes the integration to enable true multi-sequence concurrent generation.

## Phase 1: Investigation & Profiling
- [ ] Task: Profile current paged KV path
    - Trace the scheduler's `runDecodeStep` and `runBatchedPrefill` with PagedKVCache enabled.
    - Identify which Metal kernels need multi-sequence block table support.
    - Document the gap between simple GPU KV (contiguous) and paged KV (block table lookup).
- [ ] Task: Audit Metal attention kernels
    - Check if FlashAttention-2 kernel supports block table indirection.
    - Identify what changes are needed in `metal_bridge_darwin.m` for paged attention.
    - Document the kernel interface contract (block_table pointer, block_size, num_blocks).

## Phase 2: Implementation
- [ ] Task: Implement paged attention kernel dispatch
    - Add Metal kernel variant (or modify existing) to perform attention with block table lookup.
    - K/V values read via `block_table[seq][block_idx] * block_size + offset` indirection.
    - Support variable sequence lengths per batch element.
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
