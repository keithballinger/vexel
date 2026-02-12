# Track Plan: Model Compatibility

## Phase 1: Implementation
- [x] Task: Model Loader Refactor
    - Make `inference/runtime/loader.go` generic or interface-based to handle different GGUF structures.
    - Identify architecture-specific parameters (e.g., LLaMA, Phi, Mistral).
- [x] Task: Runtime Support
    - Implement missing kernels or adjustments for new architectures (e.g., sliding window attention).

## Phase 2: Verification
- [x] Task: Integration Tests
    - Load and infer with LLaMA 2/3, Mistral, Gemma, etc.
- [x] Task: Correctness Check
    - Verify against llama.cpp outputs.
