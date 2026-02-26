# Track Plan: Gemma Architecture Support

Google's Gemma family (Gemma 1, Gemma 2) has distinctive features vs LLaMA:
- **GeGLU activation**: GELU-gated linear unit (different from SwiGLU and plain GELU).
- **Large vocabulary**: 256,128 tokens.
- **Gemma 2 specifics**: Logit soft-capping in attention, alternating sliding window + global
  attention layers, pre AND post attention/MLP norms, learnable RoPE scaling.

The Vexel runtime is ~75% ready: RMSNorm, RoPE, GQA, GGUF parsing, and bias handling all
work. The main gaps are GeGLU, architecture detection, and Gemma 2's attention variants.

## Phase 1: Gemma 1 Support
- [ ] Task: Add Gemma architecture detection
    - Add `"gemma"` case to `ModelConfigFromGGUF` switch in `runtime/config.go`.
    - Set: NormRMSNorm, MLPGeGLU (new), HasBias=false, ParallelResidual=false, RoPENeox=false.
    - Parse Gemma-specific GGUF metadata: `gemma.block_count`, `gemma.embedding_length`, etc.
    - Add `Gemma2B()` and `Gemma7B()` hardcoded configs for testing.
- [ ] Task: Implement GeGLU activation
    - Add `MLPGeGLU` variant to `MLPType` enum in `config.go`.
    - Implement Metal kernel `geglu_f32`: computes `GELU(gate) * up` (fused gate+activation).
    - Add dispatch in `block.go` Execute() method alongside existing SwiGLU and GELU paths.
    - The pattern is: gate_proj -> GELU(gate_proj) * up_proj -> down_proj.
- [ ] Task: Gemma 1 correctness tests
    - Load Gemma 2B Q4_0 GGUF, run deterministic generation (temp=0).
    - Compare output token-for-token with llama.cpp using same model and prompt.
    - Test at sequence lengths 16, 64, 256.

## Phase 2: Gemma 2 Attention Variants
- [ ] Task: Logit soft-capping
    - Add `AttentionLogitSoftCap` field to `ModelConfig` (float32, 0=disabled).
    - Parse from GGUF metadata (typically 30.0 for Gemma 2).
    - Implement in attention: `logits = cap * tanh(logits / cap)` applied before softmax.
    - Can be added as a post-process step after SDPA scores or fused into the Flash Attention kernel.
- [ ] Task: Alternating sliding window + global attention
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
