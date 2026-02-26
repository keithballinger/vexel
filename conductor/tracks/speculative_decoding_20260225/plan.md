# Track Plan: Speculative Decoding

`scheduler/speculative.go` implements the core SpeculativeDecoder with draft generation,
verification, and probability-based acceptance. `scheduler/medusa_scheduler.go` wraps the
base scheduler with Medusa head support and online training infrastructure. However,
`getLogitsSlice()` is stubbed (returns nil) and the full scheduler integration is incomplete.
This track completes the integration and enables speculative decoding for production use.

## Phase 1: Core Integration
- [x] Task: Implement `getLogitsSlice` (472d2e1)
    - Replace the nil stub in `speculative.go` with actual tensor-to-CPU extraction.
    - Read GPU logits via `backend.Sync()` + `backend.ToHost()`, convert to `[]float32`.
    - Handle vocabulary dimension correctly (slice to vocabSize).
- [x] Task: Wire SpeculativeDecoder into scheduler (30da016)
    - Add `SpeculativeConfig` field to `scheduler.Config`.
    - When a draft model is provided, scheduler uses `GenerateDraftTokens` + `VerifyDraftTokens`
      instead of single-token decode.
    - Update `runDecodeStep` to handle multi-token acceptance (advance position by accepted count).
- [x] Task: KV cache management for speculation (d6d13eb)
    - After verification rejects draft tokens, roll back KV cache to the last accepted position.
    - For simple GPUKVCache: overwrite entries at rejected positions on next forward pass.
    - For PagedKVCache: deallocate blocks beyond the accepted position.

Checkpoint: d6d13eb

## Phase 2: Self-Speculative Decoding
- [x] Task: Implement early-exit drafting (4f3cd7a)
    - Use `SelfSpeculativeConfig` (already defined): run only the first N layers as the draft model.
    - Share weights between draft and target (no separate model needed).
    - Add early-exit projection head: take hidden state at layer N, project to vocab logits.
- [x] Task: Draft model loading (4f65427)
    - Support loading a separate small draft model (e.g., TinyLlama as draft for LLaMA 2 7B).
    - Both models share the same tokenizer.
    - Draft model uses its own KV cache but shared Metal backend.
- [x] Task: Adaptive draft length (9334f41)
    - Track acceptance rate over a rolling window.
    - Increase `NumDraftTokens` when acceptance is high (>80%), decrease when low (<40%).
    - Clamp to [1, 8] range.

Checkpoint: 786dc4d

## Phase 3: Medusa Heads
- [ ] Task: Medusa head architecture
    - Implement Medusa head: small MLP that predicts K future tokens from a hidden state.
    - Architecture: `hidden -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, vocab)` per head.
    - Support 2-4 Medusa heads (predicting positions +1, +2, +3, +4).
- [ ] Task: Online training integration
    - Wire MedusaScheduler's ring buffer collection to GPU trainer.
    - After each decode step, store (hidden_state, next_token) pairs.
    - Periodically train Medusa heads on collected pairs (background goroutine).
- [ ] Task: Tree-based verification
    - Implement tree attention for verifying multiple Medusa candidates simultaneously.
    - Build candidate tree from top-k predictions of each Medusa head.
    - Single forward pass through target model verifies all candidates.

## Phase 4: Verification
- [ ] Task: Correctness tests
    - Verify speculative decoding produces identical output to standard decoding (temp=0).
    - Test draft rejection and rollback: force rejection by using a very different draft model.
    - Test adaptive draft length convergence.
- [ ] Task: Speedup benchmarks
    - Measure tokens/second with and without speculative decoding.
    - Report acceptance rate, average accepted tokens per step, and effective speedup.
    - Benchmark self-speculative vs separate draft model approaches.
