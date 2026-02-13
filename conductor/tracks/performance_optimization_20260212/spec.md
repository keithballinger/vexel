# Track Spec: Performance Optimization

## Overview
Optimize key inference kernels and scheduling logic to improve throughput and reduce latency, particularly for long-context generation on Apple Silicon.

## Goals
1.  **FlashAttention Optimization:** Tune the Metal FlashAttention kernel for better performance on M-series GPUs.
2.  **Kernel Fusing:** Identify and implement opportunities for fusing common operation sequences (beyond just MLP).
3.  **Scheduler Efficiency:** Reduce overhead in the inference scheduler to handle more concurrent requests efficiently.

## Technical Details
-   **Kernels:** `inference/backend/metal/*.m` (specifically `flash_attention` and `matmul`).
-   **Scheduler:** `inference/scheduler/scheduler.go` (batching, overhead reduction).
-   **Profiling:** Use `xcrun xctrace` or similar tools to pinpoint bottlenecks.
