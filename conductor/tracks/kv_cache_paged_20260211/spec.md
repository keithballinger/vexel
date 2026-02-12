# Track Spec: PagedAttention Support

## Overview
Implement memory management for the Key-Value (KV) cache using a paging system. This avoids large contiguous memory allocations and allows the inference engine to handle variable-length sequences without fragmentation.

## Goals
1.  **Eliminate Fragmentation:** Enable efficient use of memory for KV storage.
2.  **Support Batching:** Allow multiple requests with varying sequence lengths to coexist in the cache.
3.  **Metal Optimization:** Implement GPU-side paging for fast access during attention computation.

## Technical Details
-   **Page Size:** Configurable (e.g., 16 or 32 tokens per block).
-   **Page Table:** A CPU-side map or array tracking virtual-to-physical block mappings.
-   **Integration:** Update `inference/kv` and `inference/backend/metal` to pass page tables to kernels.
