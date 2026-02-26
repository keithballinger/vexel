# Track Plan: Multi-Model Validation

`runtime/config.go` auto-detects architecture from GGUF metadata (LLaMA, Phi, GPT-NeoX, Mistral,
Qwen2) and selects the correct norm type, MLP type, RoPE variant, and bias settings. However,
only LLaMA 2 7B and TinyLlama have been tested end-to-end. This track validates correctness
across all supported architectures using llama.cpp as the reference.

## Phase 1: LLaMA Family Validation
- [x] Task: LLaMA 2 7B correctness
    - Config auto-detection verified: architecture, NormType, MLPType, bias, RoPE settings.
    - Forward pass produces valid logits (no NaN/Inf, correct [1,32000] shape).
    - Deterministic: bitwise identical outputs across runs (maxDiff=0).
    - Greedy decode from BOS: 16 tokens at 38.9 tok/s, non-degenerate output.
    - Prefill: 8-token batch at 29 tok/s, 128-token batch at 195 tok/s.
    - Prefill+decode pipeline verified end-to-end.
    - Decode throughput: 45.2 tok/s (M=1).
    - GGUF tensor index: 291 tensors (225 Q4_0, 65 F32, 1 Q6_K).
    - llama.cpp token-for-token comparison deferred (requires matching llama.cpp build).
- [ ] Task: Mistral 7B correctness
    - Config auto-detection verified: mistral→RMSNorm, SwiGLU, no bias, no parallel, no RoPENeox.
    - Sliding window config propagation tested (SlidingWindow=4096).
    - Forward pass deferred: model file not available (needs mistral-7b.Q4_0.gguf).
- [ ] Task: TinyLlama correctness
    - Config uses same llama arch path (verified).
    - Forward pass deferred: model file not available (needs tinyllama-1.1b.Q4_0.gguf).

## Phase 2: Phi Architecture Validation
- [ ] Task: Phi-2 correctness
    - Config auto-detection verified: phi2→LayerNorm, GELU, HasBias, ParallelResidual, RoPENeox.
    - Partial RoPE tested: 32 of 80 dims (40% rotated).
    - Forward pass deferred: model file not available (needs phi-2.Q4_0.gguf).
- [ ] Task: Phi-3 correctness
    - Config auto-detection verified: phi3→LayerNorm, GELU, HasBias, ParallelResidual, RoPENeox.
    - Forward pass deferred: model file not available (needs phi-3-mini.Q4_0.gguf).

## Phase 3: Sliding Window & GQA Testing
- [x] Task: Sliding window attention
    - Config propagation validated for llama (0), mistral (4096), phi3 (2048).
    - Sliding window masking in SDPA kernel already implemented.
    - Long-sequence testing deferred: requires Mistral/Phi-3 model files.
- [x] Task: GQA correctness sweep
    - Config-level GQA ratios verified: MHA 1:1, GQA 4:1, GQA 8:1, MQA 32:1.
    - Head dimension calculations validated for all ratios.
    - Architecture detection maps 7 architectures to correct GQA settings.
    - Runtime GQA broadcasting verified through LLaMA 2 7B forward pass (MHA path).
    - GQA 4:1/8:1 forward pass deferred: requires 70B/Llama3-8B model files.
