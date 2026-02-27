# Main Project Plan

This plan tracks all major tracks for the project. Each track has its own detailed plan in its respective folder.

---

- [x] **Track: Fused MLP Kernels: (Archived)**
- [x] **Track: Server & Scheduler Integration: (Archived)**
- [x] **Track: CLI Tool: (Archived)**

- [x] **Track: gRPC Streaming: Production gRPC server with streaming, TLS, interceptors, and client library.**
*Link: [./tracks/grpc_streaming_20260225/](./tracks/grpc_streaming_20260225/)*

- [x] **Track: Go Client Library: Develop a high-level Go client for Vexel inference.**
*Link: [./tracks/go_client_library_20260212/](./tracks/go_client_library_20260212/)*

- [x] **Track: Performance Optimization: Optimize Metal kernels and scheduler.**
*Link: [./tracks/performance_optimization_20260212/](./tracks/performance_optimization_20260212/)*

- [x] **Track: Competitive Benchmarking: Rigorous comparison against MLX, llama.cpp, MLC-LLM, vllm-mlx on Apple Silicon.**
*Link: [./tracks/competitive_benchmarking_20260225/](./tracks/competitive_benchmarking_20260225/)*
*Status: All phases complete. P0 OOM fixed, P1 scratch sub-allocation shipped. Vexel 89% slower than llama.cpp on single-stream decode — primary bottleneck is matmul kernel efficiency. Batched benchmarks limited by server timeout (30s). See RESULTS.md for full analysis.*

- [x] **Track: Documentation & Examples: Comprehensive docs and usage examples.**
*Link: [./tracks/documentation_examples_20260212/](./tracks/documentation_examples_20260212/)*
*Status: Phase 1 complete. Examples (client, server, direct generate), godoc for client and scheduler packages, README updated with competitive benchmarks.*

- [x] **Track: Kernel Optimization & Server Hardening: Close the decode gap against llama.cpp via command buffer batching, matmul kernel analysis, and server timeout fix.**
*Link: [./tracks/kernel_optimization_20260226/](./tracks/kernel_optimization_20260226/)*
*Status: All 6 phases complete. Decode gap closed from -89% to -15% (64.8 vs 76.3 tok/s). 8.4x speedup from command buffer batching with memory barriers. Kernel fusion analysis showed dispatch overhead is negligible post-batching. Remaining gap is Q4_0 matmul BW utilization (69.7% vs 82.0%). Model load 20% faster than llama.cpp.*
