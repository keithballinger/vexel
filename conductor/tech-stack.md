# Technology Stack

## Core Technologies
- **Go:** The primary language for the inference engine's control plane, API bindings, and the performance harness.
- **Metal:** Used for high-performance GPU-accelerated LLM kernels on Apple Silicon.
- **Fused Kernels:** Optimized Metal compute kernels combining multiple operations (e.g., MLP projection + activation) to reduce memory bandwidth.
- **Async Scheduling:** Native Go inference scheduler for managing multiple concurrent requests and continuous batching.
- **Real-time Streaming:** SSE and gRPC streaming support for low-latency token generation.

## Supported Architectures
- **LLaMA Family:** Support for RMSNorm and SwiGLU architectures (LLaMA 2/3, Mistral).
- **Phi Family:** Support for LayerNorm, GELU, and parallel residual architectures (Phi-2, Phi-3).

## Development & Infrastructure
- **CLI Tool:** Unified `vexel` binary with subcommands for running models, serving APIs, and benchmarking.
- **llama.cpp:** Used as the reference implementation for correctness verification and performance benchmarking.
- **Makefile:** Orchestrates the build process for both the Go components and any native dependencies.
- **Shell Scripts:** Used for automated performance and correctness regression testing.

## Testing & Quality Assurance
- **Go Testing:** Standard unit and integration tests using the `testing` package.
- **Custom Performance Harness:** A specialized tool for comparing throughput and output parity between Vexel and llama.cpp.
