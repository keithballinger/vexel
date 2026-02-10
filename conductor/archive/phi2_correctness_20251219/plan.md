# Track Plan: Phi-2 Correctness Parity

## Phase 1: Tokenizer Parity [checkpoint: f8152f6]
- [x] Task: Write failing unit tests for Phi-2 tokenization using a prompt set verified against llama.cpp [43093b3]
- [x] Task: Debug and fix ByteLevel BPE tokenizer implementation for Phi-2 [f409834]
- [x] Task: Verify tokenizer parity and 80% Go code coverage [153e5b6]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Tokenizer Parity' (Protocol in workflow.md) [f8152f6]

## Phase 2: Math Path Correctness [checkpoint: 5d694c7]
- [x] Task: Write unit tests for Phi-2 specific math operations (LayerNorm, GELU, bias, parallel residual, RoPE-NeoX) [1e89f2a]
- [x] Task: Debug and fix Phi-2 specific math path implementations in Metal kernels and Go bindings [cd446c9]
- [x] Task: Verify math operation correctness against known reference values [1e89f2a]
- [x] Task: Conductor - User Manual Verification 'Phase 2: Math Path Correctness' (Protocol in workflow.md)

## Phase 3: End-to-End Inference Parity [checkpoint: 3c5853f]
- [x] Task: Write integration test for greedy output parity on "Hello!" prompt [5d694c7]
- [x] Task: Final debugging of inference logic to achieve long-prefix parity with llama.cpp [e68e79b]
- [x] Task: Run performance harness to confirm fix and document throughput [5620737]
- [x] Task: Conductor - User Manual Verification 'Phase 3: End-to-End Inference Parity' (Protocol in workflow.md) [3c5853f]
