# Track Plan: Speculative Decoding Refinement

## Phase 1: Implementation
- [x] Task: Incremental KV Cache Update
    - Correctly handle KV updates (`cache.Append()`) during draft and verify steps.
    - Implement rollback/reset logic if speculation fails.
- [x] Task: Scheduler Logic
    - Update `inference/scheduler/speculative.go` to handle partial acceptances.

## Phase 2: Verification
- [x] Task: Correctness Tests
    - Verify identical output with and without speculative decoding.
- [x] Task: Speedup Measurement
    - Benchmark token generation speed vs. standard decoding.
