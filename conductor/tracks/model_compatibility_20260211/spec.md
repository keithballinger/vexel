# Track Spec: Model Compatibility

## Overview
Expand the inference engine to support a wider range of Large Language Model (LLM) architectures (e.g., LLaMA, Mistral, Gemma). This involves generalizing the model loader and implementing architecture-specific features.

## Goals
1.  **Broad Support:** Load and run popular open-source models.
2.  **Correctness:** Match the behavior/outputs of reference implementations (e.g., llama.cpp).

## Technical Details
-   **Loader:** Parse architecture-specific metadata from GGUF.
-   **Runtime:** Handle variations in attention (RoPE, ALiBi), activation functions (GELU vs. SiLU), and layer norms.
