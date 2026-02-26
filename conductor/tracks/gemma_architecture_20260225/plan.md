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

## Phase 2: Gemma 2 Attention Variants [checkpoint: 7295d6b]
- [x] Task: Logit soft-capping
    - Added `AttentionLogitSoftCap float32` to `ModelConfig` (0=disabled, 30.0 for gemma2).
    - Created `SoftCapAttentionOps` optional interface in `backend/backend.go`.
    - Implemented `sdpa_softcap_decode_f32` and `sdpa_softcap_prefill_f32` Metal kernels.
    - Added Metal + CPU backend dispatch with `SDPASoftCap` and `SDPAPrefillSoftCap` methods.
    - Wired conditional soft-cap routing in all 3 block Execute variants.
    - GPU tests: 4 subtests (zero-cap identity, bounds, GQA typical, prefill), maxDiff < 2.1e-7.
    - Architecture detection test verifies gemma2 → softcap=30.0.
- [x] Task: Alternating sliding window + global attention
    - Added `AttentionWindowType` enum (Global, Sliding, Alternating) to config.
    - Gemma 2 auto-detects as `WindowAlternating`: even layers=global, odd layers=sliding.
    - Added `useSlidingWindow(layerIdx)` and `effectiveKVLen(layerIdx, totalKVLen)` helpers.
    - Updated `ExecuteWithPagedKV` existing sliding window to respect alternating pattern.
    - Added KV pointer offset in `ExecuteWithGPUKV` FP32 decode path for sliding window.
    - Comprehensive unit tests: 5 subtests covering alternating, global, sliding, and edge cases.
    - Architecture detection test verifies gemma2 → WindowAlternating.
- [x] Task: Pre and post norms
    - Added `HasPostNorms bool` to `ModelConfig`, auto-set for gemma2.
    - Added `PostAttnNorm` and `PostFFNNorm` weight tensor fields to `BlockRuntime`.
    - GGUF loader maps `attn_post_norm.weight` and `ffn_post_norm.weight` tensors.
    - Post-attn RMSNorm applied after Wo projection, before residual add (all 3 Execute variants).
    - Post-FFN RMSNorm applied after MLP output, before residual add (all 3 Execute variants).
    - Architecture detection test verifies gemma2 → HasPostNorms=true.

## Phase 3: Learnable RoPE & Verification [checkpoint: 902f947]
- [x] Task: Learnable RoPE scaling
    - Added `RoPEFreqScales []float32` field to `ModelConfig` for per-dimension learned frequencies.
    - Added `ScaledRoPEOps` interface to `backend.go` with `RoPEWithFreqs` method.
    - Implemented `rope_gqa_scaled_f32` Metal kernel (reads freqs from buffer, not theta).
    - Added CPU backend `RoPEWithFreqs` implementation.
    - Added Metal C dispatch and Go binding with pipeline initialization.
    - Added `applyRoPE` helper to block.go — dispatches scaled vs standard based on freq buffer presence.
    - Added `SetupRoPEFreqs` to ModelRuntime for device-side frequency upload.
    - GPU kernel tests: 4 subtests (theta equivalence, custom freqs, NEOX, production), maxDiff < 5e-7.
    - Fallback to standard theta-based computation when scales are not provided.
- [x] Task: Gemma 2 correctness tests — DEFERRED
    - No Gemma 2B/9B GGUF model files available on this machine.
    - All kernel-level and config-level tests pass (GeGLU, soft-cap SDPA, alternating window, post-norms, scaled RoPE).
    - Full forward pass correctness deferred until model files are available.
- [x] Task: Performance benchmarks — DEFERRED
    - No Gemma GGUF model files available on this machine.
    - Deferred until model files are available for end-to-end benchmarking.
