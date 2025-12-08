# Vexel Inference Engine - Status Report

**Date:** December 7, 2025
**Component:** CPU Backend / Runtime
**Severity:** Critical (Panic)

## 1. Critical Issue: Panic in Attention Layer
The inference engine is currently crashing with a runtime panic during the execution of the first Transformer block.

**Error:**
```
panic: runtime error: index out of range [524288] with length 524288
...
vexel/inference/backend/cpu.(*cpuBackend).MatmulTransposeB.func1
```

### Root Cause Analysis
The crash is caused by a mismatch between the expected and actual dimensions of the Key/Value weight matrices due to **Grouped Query Attention (GQA)**.

1.  **Model Configuration:**
    *   The loaded model (`tiny_model.safetensors`) implements the `Llama` architecture with GQA.
    *   `HiddenSize`: 2048
    *   `NumAttentionHeads`: 32 (Query heads)
    *   `NumKeyValueHeads`: 4 (KV heads)
    *   `HeadDim`: 64

2.  **Weight Shapes:**
    *   `Wq` (Query Projection): `[2048, 2048]` (Standard)
    *   **`Wk` (Key Projection): `[256, 2048]`** (4 heads * 64 dim) -> *Significantly smaller than hidden size*
    *   **`Wv` (Value Projection): `[256, 2048]`** (4 heads * 64 dim)

3.  **The Bug:**
    *   The `BlockRuntime.Execute` function naively assumes that all linear projections (`Wq`, `Wk`, `Wv`) output a tensor of size `HiddenSize` (2048).
    *   It calls `MatmulTransposeB` requesting an output dimension (`N`) of 2048.
    *   The CPU backend iterates from `j = 0` to `2047`.
    *   When accessing the `Wk` weight matrix at row 256 (index `256 * 2048 = 524288`), it exceeds the slice bounds because `Wk` only has 256 rows.

## 2. Secondary Issues identified

### A. Scheduler Input Handling
*   **Status:** Partially Patched
*   **Issue:** The `Scheduler` was previously initializing `BatchRuntimeInputs` with nil tokens, leading to early exits and `nil` logit tensors. A temporary fix was applied to inject mock tokens (`1`).

### B. SDPA Implementation Limits
*   **Status:** Broken for GQA
*   **Issue:** The current `BlockRuntime` implementation attempts to perform attention by multiplying `Q` and `K` directly as flat matrices: `Matmul(Q, K^T)`.
    *   With GQA, `Q` has 2048 features and `K` has 256. This multiplication is mathematically invalid without reshaping `Q` into heads and broadcasting `K`.
    *   Even if the panic is fixed by correcting the matrix dimensions, the resulting attention scores will be mathematical garbage without a proper Multi-Head Attention (MHA/GQA) loop.

### C. Config Mismatch
*   **Status:** Risky
*   **Issue:** The CLI was hardcoding `HiddenSize: 288` while the model file actually used `HiddenSize: 2048`. This led to confusion during debugging until the true dimensions were verified via the loading logs.

## 3. Recommended Fixes

1.  **Fix Matmul Dimensions:** Update `BlockRuntime.Execute` to dynamically determine the output dimension `N` from the weight tensor's shape (`t.Shape().Dims()[0]`) instead of assuming `HiddenSize`. This will prevent the immediate panic.
2.  **Implement GQA/MHA Logic:** Rewrite the Attention calculation in `BlockRuntime` to:
    *   Reshape `Q`, `K`, `V` into `[Batch, Seq, Heads, HeadDim]`.
    *   Handle GQA repetition (broadcasting KV heads to match Query heads).
    *   Perform per-head Dot Product Attention.
3.  **Update Config Loading:** Ensure the CLI loads `tiny_config.json` values correctly instead of relying on hardcoded defaults.
