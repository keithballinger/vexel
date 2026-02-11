# Track Spec: Enable Batching for Phi-2

## Overview
Command buffer batching (`useBatching`) is currently disabled for Phi-2 in `BlockRuntime`. This feature allows multiple GPU commands (MatMul, RoPE, etc.) to be encoded into a single Metal command buffer, reducing CPU dispatch overhead and synchronization latency. It was disabled due to suspected Read-After-Write (RAW) hazards in the parallel residual architecture of Phi-2.

## Goals
1.  **Enable Batching:** Safely enable `useBatching = true` for Phi-2.
2.  **Resolve Hazards:** Identify and fix any RAW hazards that occur due to the parallel residual graph (where `MLP` and `Attention` run in parallel and sum into the residual).
3.  **Stability:** Ensure `TestPhi2MetalParity` passes consistently with batching enabled.

## Technical Details
*   **Parallel Residual:** `x_out = x_in + Attn(x_norm) + MLP(x_norm)`.
*   **Hazard:** If `Attn` and `MLP` are encoded in the same command buffer, they might try to write to `x_out` concurrently, or one might read `x_norm` while another is writing?
    *   Actually, `Attn` and `MLP` read `x_norm`. This is safe (Read-After-Read).
    *   They write to separate outputs? Or sum in-place?
    *   Vexel implementation: `Attn` writes to `attnOut`. `MLP` writes to `mlpOut`.
    *   Then `Add` sums them.
    *   The hazard might be in the `AppendKV` or intermediate buffer reuse?

## Acceptance Criteria
*   `TestPhi2MetalParity` passes with `useBatching=true`.
*   Performance (prefill especially) improves slightly due to reduced dispatch overhead.
