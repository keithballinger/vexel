# Track Spec: Fused MLP Kernels

## Overview
Combine the separate matrix multiplications in the MLP block (W1, W3) and the activation function into a single kernel pass. This reduces memory bandwidth by avoiding intermediate reads/writes.

## Goals
1.  **Reduce Memory I/O:** Perform multiple operations per memory fetch.
2.  **Increase Throughput:** Speed up the feed-forward network (FFN) computation.

## Technical Details
-   **Operations:** `Out = SiLU(X @ W1.T) * (X @ W3.T)`
-   **Kernel:** A new Metal compute kernel that computes both dot products in parallel threads or SIMD groups.
