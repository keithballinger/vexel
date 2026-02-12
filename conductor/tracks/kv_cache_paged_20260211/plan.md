# Track Plan: PagedAttention Support

## Phase 1: Implementation
- [ ] Task: Implement Paging Data Structure
    - Define `PageTable` in `inference/kv/cache.go`.
    - Implement `Allocate()` and `Free()` for managing KV blocks.
- [ ] Task: Metal Integration
    - Update `metal_bridge.h` and `backend.go` to support paged KV lookup.
    - Create a Metal kernel (`reshape_kv_cache`) to handle paged writes.

## Phase 2: Verification
- [ ] Task: Unit Tests
    - Verify allocation/deallocation logic.
    - Test `Cache.Read()` and `Cache.Write()` with paged layout.
- [ ] Task: Benchmark
    - Compare memory fragmentation vs. contiguous allocation.
