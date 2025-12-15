# Vexel Next Steps

**Date:** 2024-12-15
**Status:** Mistral 7B working at ~80% of llama.cpp

## Current Performance

| Model | Prefill | Decode | vs llama.cpp |
|-------|---------|--------|--------------|
| TinyLlama 1.1B | 121 tok/s | 179 tok/s | - |
| Mistral 7B | 40 tok/s | 44 tok/s | 80% |

## Priority 1: Tokenizer Robustness (High Leverage)

The `-tokenizer` flag unblocks immediate use, but the system is still fragile.

### Must Do
- [ ] **Vocab size compatibility check at startup**
  - Tokenizer vocab size must match model vocab size (lm_head dim or GGUF metadata)
  - Fail fast with explicit error on mismatch
  - Print resolved tokenizer path and vocab size in logs

### Nice to Have
- [ ] Accept `--model-dir` and load tokenizer relative to that directory
- [ ] Extract tokenizer from GGUF metadata if available
- [ ] Auto-detect model family from GGUF architecture field

## Priority 2: Close the 20% Performance Gap (7B)

Now that Mistral works, optimize with real profiles instead of abstract tuning.

### A) Validate FuseW1W3 on 7B
- [ ] Verify `plan.Fusion.FuseW1W3=true` is active in large regime
- [ ] Measure decode tok/s before/after
- [ ] Get per-op breakdown (GateUp, W2, Adds, SDPA)

### B) Optimize W2 Mapping
- [ ] Verify W2 uses nr0=4 effectively for 7B
- [ ] Profile W2 specifically - it's the other big MLP bucket
- [ ] Consider activation reuse patterns

### C) Fuse Residual Adds / Epilogues
- [ ] Fuse adds into projection epilogues where safe
- [ ] This eliminates separate kernel launches and memory round-trips

## Priority 3: Testing & Guardrails

### FA2 headDim>64 Test
- [ ] Add unit test for attention with head_dim=128
- [ ] Confirm output is finite and stable for fixed seed
- [ ] Add to CI if possible

### Model Validation
- [ ] Test Phi-2 (1.7B) - different architecture
- [ ] Test Qwen2 (0.5B) - different architecture
- [ ] Test Gemma (2B) - different architecture

## Priority 4: Command Buffer Batching (Deferred)

Batching was disabled due to Metal memory hazards. Proper fix requires:
- Understanding exact data dependencies
- Adding proper MTLFence or barrier synchronization
- Could yield 10-20% speedup when fixed

## Priority 5: Speculative Decoding (Future)

Medusa infrastructure exists. Could give 2-3x throughput by:
- Using TinyLlama as draft model for Mistral
- Or training Medusa heads on target model

---

## Expert Feedback Summary

Key insights from expert review:

1. **Mistral correctness proved it wasn't the math path** - the kernels are working correctly
2. **80% of llama.cpp is a "real engine" milestone** - validates the architecture
3. **Don't leave tokenizer fragile** - vocab size check is high leverage
4. **TinyLlama numbers vary** - don't worry about absolute values, worry about repeatable baselines
5. **MLP dominated in earlier profiling** - FuseW1W3 and W2 are the optimization targets
6. **Add guardrail test for FA2** - prevents silent regression on tricky kernel

## Immediate Next Action

```
1. Add vocab-size mismatch fail-fast checks
2. Re-run 7B profile with FuseW1W3 enabled
3. Confirm W2 is the next bottleneck
4. Optimize W2 nr0=4 mapping
5. Consider residual-add fusion
```
