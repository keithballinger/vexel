# Vexel Project Plan

## Guiding Principles
This project adheres to the Conductor methodology, with a strong emphasis on Test-Driven Development (TDD), high code coverage, and deliberate architectural planning. All tasks will follow the Standard Task Workflow.

## Phase 1: Foundation and Core Components

### Task: Set up Project Structure and Basic Go Modules
- [x] Write Failing Tests: For basic module structure and compilation.
- [x] Implement Feature: Create `inference/tensor/`, `inference/memory/`, `inference/kv/`, `inference/backend/cpu/`, `inference/ir/`, `inference/runtime/`, `inference/scheduler/`, `inference/serve/`, `inference/cmd/` directories. (Started: 2025-12-07 12:00, Completed: 2025-12-07 12:05)
- [x] Write Failing Tests: For Go module initialization. (Started: 2025-12-07 12:10, Completed: 2025-12-07 12:12)
- [x] Implement Feature: Initialize Go module and manage dependencies (`go mod init`, `go mod tidy`). (Started: 2025-12-07 12:15, Completed: 2025-12-07 12:17)

### Task: Implement Tensor and Memory System (`inference/tensor/`, `inference/memory/`)
- [x] Write Failing Tests: For `DType` and `Location` enumerations and their methods (e.g., `SizeBytes`, `BitSize`). (Started: 2025-12-07 12:20, Completed: 2025-12-07 12:22)
- [x] Implement Feature: Define `DType` and `Location` types and associated methods. (Started: 2025-12-07 12:25, Completed: 2025-12-07 12:27)
- [x] Write Failing Tests: For `Shape` creation, `NumElements`, `Rank`, `Equal`, and `StridesRowMajor`. (Started: 2025-12-07 12:30, Completed: 2025-12-07 12:32)
- [x] Implement Feature: Define `Shape` type and its methods. (Started: 2025-12-07 12:35, Completed: 2025-12-07 12:37)
- [x] Write Failing Tests: For `DevicePtr` structure and basic initialization. (Started: 2025-12-07 12:40, Completed: 2025-12-07 12:42)
- [x] Implement Feature: Define `DevicePtr` and `Device` types. (Started: 2025-12-07 12:45, Completed: 2025-12-07 12:47)
- [x] Write Failing Tests: For `Tensor` struct initialization and field access. (Started: 2025-12-07 12:50, Completed: 2025-12-07 12:52)
- [x] Implement Feature: Define `Tensor` struct. (Started: 2025-12-07 12:55, Completed: 2025-12-07 12:57)
- [x] Write Failing Tests: For `ArenaKind` enumerations. (Started: 2025-12-07 13:00, Completed: 2025-12-07 13:02)
- [x] Implement Feature: Define `ArenaKind`. (Started: 2025-12-07 13:05, Completed: 2025-12-07 13:07)
- [x] Write Failing Tests: For `Arena` creation (`NewArena`), allocation (`Alloc`), and reset (`Reset`). (Started: 2025-12-07 13:10, Completed: 2025-12-07 13:12)
- [x] Implement Feature: Implement `Arena` struct and methods for memory management. (Started: 2025-12-07 13:15, Completed: 2025-12-07 13:17)
- [x] Write Failing Tests: For `InferenceContext` initialization and field access. (Started: 2025-12-07 13:20, Completed: 2025-12-07 13:22)
- [x] Implement Feature: Define `InferenceContext` struct. (Started: 2025-12-07 13:25, Completed: 2025-12-07 13:27)

### Task: Implement Quantization Profiles
- [x] Write Failing Tests: For `QuantProfile` enumerations. (Started: 2025-12-07 13:30, Completed: 2025-12-07 13:32)
- [x] Implement Feature: Define `QuantProfile`. (Started: 2025-12-07 13:35, Completed: 2025-12-07 13:37)
- [x] Write Failing Tests: For `QuantizedTensor` struct initialization. (Started: 2025-12-07 13:40, Completed: 2025-12-07 13:42)
- [x] Implement Feature: Define `QuantizedTensor` struct. (Started: 2025-12-07 13:45, Completed: 2025-12-07 13:47)

### Task: Implement Paged KV Cache Core Structures (`inference/kv/`)
- [x] Write Failing Tests: For `KVConfig` struct and initialization. (Started: 2025-12-07 13:50, Completed: 2025-12-07 13:52)
- [x] Implement Feature: Define `KVConfig` struct. (Started: 2025-12-07 13:55, Completed: 2025-12-07 13:57)
- [x] Write Failing Tests: For `KVCache` struct initialization and basic page management (conceptual). (Started: 2025-12-07 14:00, Completed: 2025-12-07 14:02)
- [x] Implement Feature: Define `KVCache` struct. (Started: 2025-12-07 14:05, Completed: 2025-12-07 14:07)
- [x] Write Failing Tests: For `PageIndex` and `SeqKVHandle` structs. (Started: 2025-12-07 14:10, Completed: 2025-12-07 14:12)
- [x] Implement Feature: Define `PageIndex` and `SeqKVHandle` structs. (Started: 2025-12-07 14:15, Completed: 2025-12-07 14:17)

### Task: Implement Basic CPU Backend (`inference/backend/cpu/`)
- [x] Write Failing Tests: For the `Backend` interface definition (as an empty interface at this stage, focusing on structural correctness). (Started: 2025-12-07 14:20, Completed: 2025-12-07 14:22)
- [x] Implement Feature: Define the `Backend` interface. (Started: 2025-12-07 14:25, Completed: 2025-12-07 14:27)
- [x] Write Failing Tests: For a basic `cpuBackend` implementation (e.g., `CreateStream` returning a dummy stream, `Device` returning `DeviceCPU`). (Started: 2025-12-07 14:30, Completed: 2025-12-07 14:32)
- [x] Implement Feature: Create a placeholder `cpuBackend` that implements the `Backend` interface for initial development and testing without GPU dependencies. (Started: 2025-12-07 14:35, Completed: 2025-12-07 14:37)

## Phase 2: Intermediate Representation and Runtime

### Task: Implement Block IR (`inference/ir/`)
- [x] Write Failing Tests: For `TensorID`, `OpKind` enumerations, and `OpNode` struct initialization. (Started: 2025-12-07 14:40, Completed: 2025-12-07 14:42)
- [x] Implement Feature: Define `TensorID`, `OpKind`, and `OpNode`. (Started: 2025-12-07 14:45, Completed: 2025-12-07 14:47)
- [x] Write Failing Tests: For `BlockIR` struct initialization, including inputs, outputs, and nodes. (Started: 2025-12-07 14:50, Completed: 2025-12-07 14:52)
- [x] Implement Feature: Define `BlockIR` struct. (Started: 2025-12-07 14:55, Completed: 2025-12-07 14:57)
- [x] Write Failing Tests: For a conceptual fusion pass (e.g., a mock function that takes a `BlockIR` and returns a modified one, checking basic structure). (Started: 2025-12-07 15:00, Completed: 2025-12-07 15:02)
- [x] Implement Feature: Implement basic fusion pass mechanisms (conceptual, to be detailed in Phase 3). (Started: 2025-12-07 15:05, Completed: 2025-12-07 15:07)

### Task: Implement Model Runtime Core (`inference/runtime/`)
- [x] Write Failing Tests: For `ModelConfig` struct and initialization (e.g., `Llama3_8B` configuration). (Started: 2025-12-07 15:10, Completed: 2025-12-07 15:12)
- [x] Implement Feature: Define `ModelConfig` struct. (Started: 2025-12-07 15:15, Completed: 2025-12-07 15:17)
- [x] Write Failing Tests: For `BlockRuntime` struct initialization. (Started: 2025-12-07 15:20, Completed: 2025-12-07 15:22)
- [x] Implement Feature: Define `BlockRuntime` struct. (Started: 2025-12-07 15:25, Completed: 2025-12-07 15:27)
- [ ] Write Failing Tests: For `ModelRuntime` struct initialization (requires `Backend`, `InferenceContext`, `KVCache`).
- [ ] Implement Feature: Define `ModelRuntime` struct (initial structure).
- [ ] Write Failing Tests: For `BatchRuntimeInputs` struct initialization.
- [ ] Implement Feature: Define `BatchRuntimeInputs` struct.
- [ ] Write Failing Tests: For the high-level `DecodeStep` function signature and its return types.
- [ ] Implement Feature: Implement `DecodeStep` function (high-level structure, integrating with IR and backend conceptually).

## Phase 3: GPU Backends and Optimization

### Task: Implement CUDA Backend (`inference/backend/cuda/`)
- [ ] Write Failing Tests: For `cudaBackend` implementing the `Backend` interface, focusing on `CreateStream`, `RecordEvent`, `WaitEvent`, `SynchronizeStream`, and `Device`.
- [ ] Implement Feature: Implement `cudaBackend` struct and its core stream/event management methods.
- [ ] Write Failing Tests: For `CompileBlockGraph` using mock IR and weights (checking return types and error handling).
- [ ] Implement Feature: Implement `CompileBlockGraph` for CUDA, integrating with `nvcc` and CUDA Graphs.
- [ ] Write Failing Tests: For `RunGraph` with mock inputs and stream.
- [ ] Implement Feature: Implement `RunGraph` for CUDA.
- [ ] Write Failing Tests: For `HostToDevice` and `DeviceToHost` operations.
- [ ] Implement Feature: Implement host-device memory transfer functions for CUDA.

### Task: Implement Metal Backend (`inference/backend/metal/`)
- [ ] Write Failing Tests: For `metalBackend` implementing the `Backend` interface, focusing on `CreateStream`, `RecordEvent`, `WaitEvent`, `SynchronizeStream`, and `Device`.
- [ ] Implement Feature: Implement `metalBackend` struct and its core stream/event management methods.
- [ ] Write Failing Tests: For `CompileBlockGraph` using mock IR and weights (checking return types and error handling).
- [ ] Implement Feature: Implement `CompileBlockGraph` for Metal, integrating with `metallib` pipelines.
- [ ] Write Failing Tests: For `RunGraph` with mock inputs and stream.
- [ ] Implement Feature: Implement `RunGraph` for Metal.
- [ ] Write Failing Tests: For `HostToDevice` and `DeviceToHost` operations.
- [ ] Implement Feature: Implement host-device memory transfer functions for Metal.

### Task: Develop and Integrate Optimized Fused Kernels
- [ ] Write Failing Tests: For specific fusion pass identification (e.g., detecting Matmul→SiLU patterns).
- [ ] Implement Feature: Develop and integrate specific fusion passes within the IR (e.g., Matmul→SiLU→Matmul→Mul→Matmul → GatedMLP).
- [ ] Write Failing Tests: For compiled fused kernels executing correctly on CUDA and Metal backends.
- [ ] Implement Feature: Implement optimized fused kernels for both CUDA and Metal backends.

## Phase 4: Scheduler and Serving Layer

### Task: Implement Scheduler Core (`inference/scheduler/`)
- [ ] Write Failing Tests: For `Sequence` struct initialization and state transitions.
- [ ] Implement Feature: Define `Sequence` struct and state machine.
- [ ] Write Failing Tests: For `Scheduler` struct initialization.
- [ ] Implement Feature: Define `Scheduler` struct.
- [ ] Write Failing Tests: For the `Run` and `step` loop's basic execution flow and ticker management.
- [ ] Implement Feature: Implement `Run` and `step` methods for the scheduler's main loop.
- [ ] Write Failing Tests: For `collectReady()` accurately identifying sequences in Pending/Decoding states.
- [ ] Implement Feature: Implement `collectReady()` method.
- [ ] Write Failing Tests: For `formBatches()` creating batches based on priority and configurable policies.
- [ ] Implement Feature: Implement `formBatches()` method.
- [ ] Write Failing Tests: For `runDecodeStep()` orchestrating the `ModelRuntime.DecodeStep` and handling errors.
- [ ] Implement Feature: Implement `runDecodeStep()` method.

### Task: Implement Serving Layer (`inference/serve/`)
- [ ] Write Failing Tests: For HTTP `/generate` endpoint handling requests and returning non-streaming responses.
- [ ] Implement Feature: Implement HTTP `/generate` endpoint.
- [ ] Write Failing Tests: For HTTP `/stream` endpoint handling requests and providing token-by-token streaming.
- [ ] Implement Feature: Implement HTTP `/stream` endpoint.
- [ ] Write Failing Tests: For gRPC `/generate` and `/stream` endpoints.
- [ ] Implement Feature: Implement gRPC `/generate` and `/stream` endpoints.
- [ ] Write Failing Tests: For request admission control and sequence registration with the scheduler.
- [ ] Implement Feature: Implement request admission control and sequence registration.

## Phase 5: Integration, Testing, and Documentation

### Task: End-to-End Integration Testing
- [ ] Write Failing Tests: For loading and running a Llama-3-style 8B model from end-to-end.
- [ ] Implement Feature: Perform end-to-end integration of all modules with a reference model.
- [ ] Write Failing Tests: For key performance indicators (prefill, decode step, TtFT, active sequences).
- [ ] Implement Feature: Implement performance benchmarking tools and scripts.

### Task: Performance Benchmarking and Validation
- [ ] Write Failing Tests: For `ModelConfig.ApproxParams()` calculation.
- [ ] Implement Feature: Implement `ModelConfig.ApproxParams()`.
- [ ] Write Failing Tests: For `ModelConfig.WeightsBytes()` calculation across different quantization profiles.
- [ ] Implement Feature: Implement `ModelConfig.WeightsBytes()`.
- [ ] Write Failing Tests: For `KVBytes()` calculation based on model config, KV config, and active sequences.
- [ ] Implement Feature: Implement `KVBytes()`.
- [ ] Write Failing Tests: For `ScratchBytes()` calculation based on model config and max batch.
- [ ] Implement Feature: Implement `ScratchBytes()`.
- [ ] Write Failing Tests: For `ModelConfig.MemoryPlan()` aggregating all memory calculations.
- [ ] Implement Feature: Implement `ModelConfig.MemoryPlan()`.
- [ ] Implement Feature: Conduct comprehensive performance benchmarking against defined targets for throughput, latency, and VRAM usage on both CUDA and Metal.

### Task: Documentation and Code Quality Refinement
- [ ] Implement Feature: Add comprehensive GoDoc comments for all public types, functions, and methods.
- [ ] Implement Feature: Review and refine `user_guide.md` and `architecture.md` to reflect implementation details and any evolved design decisions.
- [ ] Implement Feature: Ensure all code adheres to the selected Go code style guide and passes linting/static analysis checks.
- [ ] Implement Feature: Review and ensure high code coverage (>80%) for all modules.

