# Track Spec: Phi-2 Correctness Parity

## Problem Statement
Phi-2 currently experiences a hard correctness failure in Vexel. Text similarity is extremely low (~0.01) and token prefix matching is 0/9.

## Known Issues
1. **Tokenizer Mismatch:** Phi-2 uses ByteLevel BPE. Vexel encodes "unit testing" as 17 tokens, while llama.cpp encodes it as 14.
2. **Phi-2 Specific Math Paths:** Potential bugs in LayerNorm, GELU, bias handling, parallel residuals, or RoPE-NeoX partial dimensions.

## Success Criteria
- **Tokenization Parity:** Vexel's Phi-2 tokenizer matches llama.cpp exactly for a specified set of test prompts.
- **Inference Parity:** Vexel's greedy output matches a long token prefix (at least 20 tokens) of llama.cpp output for "Hello!" in completion mode.
