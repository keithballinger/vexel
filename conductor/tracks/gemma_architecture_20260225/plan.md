# Track Plan: Gemma Architecture Support

Google's Gemma family (Gemma 1, Gemma 2) has distinctive features vs LLaMA:
- **GeGLU activation**: GELU-gated linear unit (different from SwiGLU and plain GELU).
- **Large vocabulary**: 256,128 tokens.
- **Gemma 2 specifics**: Logit soft-capping in attention, alternating sliding window + global
  attention layers, pre AND post attention/MLP norms, learnable RoPE scaling.

The Vexel runtime is ~75% ready: RMSNorm, RoPE, GQA, GGUF parsing, and bias handling all
work. The main gaps are GeGLU, architecture detection, and Gemma 2's attention variants.

## Phase 1: Gemma 1 Support [checkpoint: 9e8a645]
- [x] Task: Add Gemma architecture detection
    - Added `"gemma"/"gemma2"` case to `ModelConfigFromGGUF` switch in `runtime/config.go`.
    - Set: NormRMSNorm, MLPGeGLU, HasBias=false, ParallelResidual=false, RoPENeox=false.
    - Added `Gemma2B()` and `Gemma7B()` hardcoded configs with correct dimensions.
    - Architecture detection verified for all 9 architectures including gemma/gemma2.
- [x] Task: Implement GeGLU activation
    - Added `MLPGeGLU` variant to `MLPType` enum in `config.go`.
    - Implemented fused `gelu_mul_f32` Metal kernel (GELU activation + multiply).
    - Added `GELUMul` to `GELUOps` interface with Metal + CPU backend dispatch.
    - Updated all 3 block Execute variants with GeGLU routing.
    - Guarded fused MLP to SwiGLU-only (fused kernel uses SiLU).
    - GPU kernel tests: basic (n=4096), edge cases, production size (n=16384), maxDiff < 8e-6.
- [x] Task: Gemma 1 correctness tests — DEFERRED
    - No Gemma 2B Q4_0 GGUF model file available on this machine.
    - Config-level detection and GeGLU kernel thoroughly tested.
    - Full forward pass correctness deferred until model file is available.

## Phase 2: Gemma 2 Attention Variants
- [x] Task: Logit soft-capping
    - Added `AttentionLogitSoftCap float32` to `ModelConfig` (0=disabled, 30.0 for gemma2).
    - Created `SoftCapAttentionOps` optional interface in `backend/backend.go`.
    - Implemented `sdpa_softcap_decode_f32` and `sdpa_softcap_prefill_f32` Metal kernels.
    - Added Metal + CPU backend dispatch with `SDPASoftCap` and `SDPAPrefillSoftCap` methods.
    - Wired conditional soft-cap routing in all 3 block Execute variants.
    - GPU tests: 4 subtests (zero-cap identity, bounds, GQA typical, prefill), maxDiff < 2.1e-7.
    - Architecture detection test verifies gemma2 → softcap=30.0.
- [~] Task: Alternating sliding window + global attention
    - Add `AttentionWindowPattern` field to `ModelConfig`: "global", "sliding", or "alternating".
    - For alternating: even layers use full context, odd layers use sliding window.
    - Implement per-layer attention mask selection in the block execution loop.
    - Requires passing layer index to attention dispatch.
- [ ] Task: Pre and post norms
    - Add `PostNorm` boolean to `ModelConfig` (Gemma 2 applies RMSNorm after attention AND MLP too).
    - Current flow: norm -> attn -> residual. New flow: norm -> attn -> post_norm -> residual.
    - Load post-norm weights from GGUF (separate weight tensors).

## Phase 3: Learnable RoPE & Verification
- [ ] Task: Learnable RoPE scaling
    - Add `RoPEFreqScales []float32` field to `ModelConfig` for per-dimension learned frequencies.
    - Parse from GGUF if present (Gemma 2 stores learned inv_freq values).
    - Modify RoPE Metal kernel to accept frequency array buffer instead of computing from theta.
    - Fallback to standard theta-based computation when scales are not provided.
- [ ] Task: Gemma 2 correctness tests
    - Load Gemma 2 2B and Gemma 2 9B GGUF models.
    - Verify architecture auto-detection enables all Gemma 2 features (soft-cap, alternating window, post-norm).
    - Compare output token-for-token with llama.cpp at temp=0.
- [ ] Task: Performance benchmarks
    - Benchmark Gemma 2B and 7B: prefill tok/s, decode tok/s, TTFT.
    - Compare with LLaMA 2 7B at equivalent quantization (Q4_0).
    - Verify large vocabulary (256K) doesn't cause memory issues in logit computation.
