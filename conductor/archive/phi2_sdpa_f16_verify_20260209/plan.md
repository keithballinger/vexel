# Track Plan: Verify SDPAF16 Correctness

## Phase 1: Verification
- [x] Task: Unit Test SDPAF16 with HeadDim 80
    - Create a unit test `inference/runtime/sdpa_test.go` that calls `SDPAF16` directly with `headDim=80`.
    - Compare output against CPU reference or `SDPA` (F32).
- [x] Task: Check BlockRuntime Logic
    - Ensure `BlockRuntime` selects `SDPAF16` when appropriate (e.g., when `useFP16Path` is true, or `useFP16KVCache` is true).

## Phase 2: Fix & Enable
- [x] Task: Fix Kernel (if needed)
    - If unit test fails, modify `metal_bridge_darwin.m` to handle `headDim=80` (e.g., improve loop bounds or padding).
    - *Action:* Fixed shared memory offset bug in `sdpa_decode_f16_vec` to handle dynamic `headDim`.
- [x] Task: Enable in Runtime
    - Update `BlockRuntime` to allow `SDPAF16` for Phi-2 configuration.
    - *Action:* `backend.go` updated to use vectorized kernel for `headDim%4 == 0`.

## Phase 3: Validation
- [x] Task: Verify Correctness
    - Run `TestPhi2MetalParity`.
