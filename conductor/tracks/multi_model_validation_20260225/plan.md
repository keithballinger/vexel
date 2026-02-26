# Track Plan: Multi-Model Validation

`runtime/config.go` auto-detects architecture from GGUF metadata (LLaMA, Phi, GPT-NeoX, Mistral,
Qwen2) and selects the correct norm type, MLP type, RoPE variant, and bias settings. However,
only LLaMA 2 7B and TinyLlama have been tested end-to-end. This track validates correctness
across all supported architectures using llama.cpp as the reference.

## Phase 1: LLaMA Family Validation
- [ ] Task: LLaMA 2 7B correctness
    - Run deterministic generation (temp=0) and compare output token-for-token with llama.cpp.
    - Test at multiple sequence lengths (16, 64, 256 tokens).
    - Automate comparison in a test: `TestCorrectnessLlama2_7B`.
- [ ] Task: Mistral 7B correctness
    - Load Mistral 7B Q4_0 GGUF, verify GGUF auto-detection selects correct config.
    - Verify sliding window attention config is parsed (SlidingWindow field).
    - Run deterministic generation and compare with llama.cpp output.
- [ ] Task: TinyLlama correctness
    - Verify TinyLlama 1.1B Q4_0 produces identical output to llama.cpp.
    - Test chat template formatting matches expected format.

## Phase 2: Phi Architecture Validation
- [ ] Task: Phi-2 correctness
    - Load Phi-2 Q4_0 GGUF, verify auto-detection enables: LayerNorm, GELU MLP, ParallelResidual, HasBias.
    - Run deterministic generation and compare with llama.cpp.
    - Test that MHA (NumKVHeads == NumHeads) is handled correctly (no GQA).
- [ ] Task: Phi-3 correctness
    - Load Phi-3 Mini Q4_0 GGUF, verify config differences from Phi-2.
    - Validate partial RoPE dimension support (RoPEDim < HeadDim).
    - Run deterministic generation and compare with llama.cpp.

## Phase 3: Sliding Window & GQA Testing
- [ ] Task: Sliding window attention
    - Implement and validate sliding window attention masking for Mistral/Phi-3.
    - Test with sequences longer than the sliding window size.
    - Verify KV cache correctly evicts entries outside the window.
- [ ] Task: GQA correctness sweep
    - Test GQA head ratios: 1:1 (Phi-2 MHA), 4:1 (LLaMA 2 7B), 8:1 (LLaMA 3 8B).
    - Verify K/V head broadcasting is correct across all ratios.
    - Add regression test for each ratio.
