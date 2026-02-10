# Track Spec: Metal Performance Optimization (Phi-2)

## Overview
Move the Phi-2 inference path from the current CPU-based implementation to a fully Metal-accelerated pipeline. This track aims to achieve high-performance inference on Apple Silicon by implementing specialized, optimized kernels for Phi-2 specific operations and integrating them into the `BlockRuntime`.

## Goals
1.  **High Throughput:** Achieve tokens/sec comparable to llama.cpp on Metal hardware.
2.  **Scalability:** Enable larger batch sizes for inference compared to the CPU implementation.
3.  **Full Offload:** Ensure all Phi-2 operations run on the GPU to minimize CPU-GPU synchronization overhead.

## Functional Requirements
1.  **New Metal Kernels:** Implement highly optimized kernels from scratch for:
    *   `LayerNorm` (with mean subtraction and bias support)
    *   `GELU` (tanh approximation)
    *   `RoPE-NeoX` (partial dimension support, split-pair rotation)
    *   `AddBias` (fused or standalone, as appropriate for memory coalescing)
2.  **Runtime Integration:** Update `BlockRuntime` to detect Metal backend availability and dispatch to these new kernels for Phi-2 models.
3.  **Weight Loading:** Ensure Phi-2 specific weights (including biases and combined QKV tensors) are correctly loaded and resident on the GPU.

## Non-Functional Requirements
1.  **Correctness:**
    *   Outputs must match the CPU reference implementation (within FP32/FP16 tolerance).
    *   Outputs must match llama.cpp reference values.
    *   Statistical properties (mean, variance) must be preserved.
2.  **Performance:** No regression in latency compared to CPU; target >10x speedup for prompt processing (prefill).

## Out of Scope
*   Optimization for non-Apple Silicon GPUs (CUDA/Vulkan).
*   Changes to the LLaMA inference path.
*   Quantized kernel support (Q4_K/Q6_K) for Phi-2 (this is a separate future track; we will use F32/F16 for now).

## Acceptance Criteria
*   Unit tests for `LayerNorm`, `GELU`, and `RoPE` pass on Metal backend.
*   `TestPhi2Parity` passes when configured to use the Metal backend.
*   Benchmarks show significant throughput improvement over CPU baseline.
