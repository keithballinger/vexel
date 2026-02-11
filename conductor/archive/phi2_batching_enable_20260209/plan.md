# Track Plan: Enable Batching for Phi-2

## Phase 1: Enable & Reproduce
- [x] Task: Enable Batching
    - Modify `phi2_block.go` (or config) to set `useBatching = true`.
- [x] Task: Run Parity Test
    - Run `TestPhi2MetalParity` to see if it fails.
    - If it fails, capture debug logs to identify the mismatched layer/tensor.

## Phase 2: Debug & Fix
- [x] Task: Analyze Hazards
    - Review `BlockRuntime` execution graph.
    - Identify where synchronization (batch commit) is missing.
- [x] Task: Implement Sync Points
    - Insert `b.batcher.EndBatch()` / `BeginBatch()` at critical points (e.g., before `AppendKV`, or before `Add`).
    - Minimize sync points to maximize batching benefit.

## Phase 3: Verification
- [x] Task: Verify Correctness
    - Ensure `TestPhi2MetalParity` passes consistently.
