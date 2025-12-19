# Vexel User Guide

## 1. Project Overview

Vexel is a high-performance Go library designed to enable efficient inference for transformer-based large language models (LLMs) on a single GPU. It is specifically optimized for smaller models (under 10 billion parameters) and prioritizes low-latency streaming for interactive workloads and high throughput through continuous batching.

## 2. Goals and Key Features

The primary goal of Vexel is to provide a Go-native, production-ready solution for integrating local LLM inference into Go applications and services. Key features include:

*   **High Performance:** Competitive throughput and latency for single-GPU inference, leveraging optimized CUDA/Metal backends with fused kernels and graph execution.
*   **Memory Efficiency:** Paged KV cache for optimal memory utilization and support for long context windows.
*   **Concurrency:** Idiomatic Go concurrency patterns for orchestrating inference tasks.
*   **Flexible Quantization:** Support for various data types including FP16, BF16, INT4, and INT8.
*   **Streaming Output:** Low-latency token streaming for interactive user experiences.
*   **Continuous Batching:** Efficiently handles many concurrent sequences to maximize GPU utilization.

## 3. Target Audience

Vexel is primarily intended for **Go developers** who wish to integrate local LLM inference capabilities directly into their Go applications or services. This includes developers building:

*   Real-time AI-powered features.
*   Applications requiring on-premise or edge inference.
*   Services where data privacy or low network latency is critical.
*   Projects where deep integration with the Go ecosystem is desired.

## 4. How to Use Vexel (High-Level)

Developers will interact with Vexel by:

1.  **Initializing a Model Runtime:** Loading a pre-trained transformer model (e.g., Llama-3-style 8B) into the Vexel runtime, specifying quantization profiles and target device (CUDA/Metal).
2.  **Sending Inference Requests:** Submitting prompts to Vexel's scheduler via HTTP/gRPC endpoints (or directly through the Go API).
3.  **Handling Responses:** Receiving generated tokens, either as a complete response or a real-time stream.
4.  **Managing Sequences:** The internal scheduler handles continuous batching, admission control, and resource management for concurrent inference requests.

## 5. Architectural Overview

Vexel's architecture is modular, comprising:

*   **`serve/`**: Handles HTTP/gRPC endpoints, token streaming, authentication, and observability.
*   **`scheduler/`**: Manages continuous batching, admission control, QoS, and timeouts for inference requests.
*   **`runtime/`**: Executes Block IR, compiled graphs, sampling, and interacts with the paged KV cache.
*   **`tensor/, memory/, kv/`**: Core components for tensor operations, memory arenas, quantization profiles, and paged KV cache structures.
*   **`backend/`**: Provides device-specific implementations (CUDA/Metal) for optimized kernel execution.

This structure allows Go developers to leverage high-performance LLM inference while maintaining an idiomatic Go development experience for orchestration and service integration.