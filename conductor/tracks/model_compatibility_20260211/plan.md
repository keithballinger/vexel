# Track Plan: Model Compatibility

## Phase 1: Implementation
- [ ] Task: Model Loader Refactor
    - Make `inference/runtime/loader.go` generic or interface-based to handle different GGUF structures.
    - Identify architecture-specific parameters (e.g., LLaMA, Phi, Mistral).
- [ ] Task: Runtime Support
    - Implement missing kernels or adjustments for new architectures (e.g., sliding window attention).

## Phase 2: Verification
- [ ] Task: Integration Tests
    - Load and infer with LLaMA 2/3, Mistral, Gemma, etc.
- [ ] Task: Correctness Check
    - Verify against llama.cpp outputs.
