# Vexel Technical Architecture

## 1. Introduction
Vexel is a high-performance Go-native inference engine designed for transformer-based language models, specifically targeting single-GPU deployments. Its architecture is crafted to deliver competitive throughput and latency for models under 10 billion parameters, with a strong emphasis on memory efficiency, low-latency streaming, and idiomatic Go concurrency.

## 2. High-Level Architecture
The Vexel architecture is modular, organized into distinct layers responsible for specific aspects of the inference pipeline:

*   **`serve/`**: Handles external communication via HTTP/gRPC endpoints, manages token streaming, authentication, and observability.
*   **`scheduler/`**: Orchestrates inference requests through continuous batching, admission control, Quality of Service (QoS), and timeouts.
*   **`runtime/`**: Executes the Block Intermediate Representation (IR), manages compiled computational graphs, performs sampling, and interacts with the paged KV cache.
*   **`tensor/`, `memory/`, `kv/`**: These foundational layers manage tensor data types, memory allocation (arenas), quantization profiles, and the paged KV cache structures.
*   **`backend/`**: Provides the hardware-specific implementations for GPU execution, supporting CUDA for NVIDIA GPUs and Metal for Apple Silicon, leveraging optimized fused kernels and graph execution.

```
 ┌─────────────────────────────────────────────────────────────┐
 │                           serve/                           │
 │   HTTP/gRPC endpoints, token streaming, auth, observability │
 ├─────────────────────────────────────────────────────────────┤
 │                        scheduler/                           │
 │   Continuous batching, admission control, QoS, timeouts     │
 ├─────────────────────────────────────────────────────────────┤
 │                        runtime/                             │
 │   Block IR execution, compiled graphs, sampler, paged KV    │
 ├─────────────────────────────────────────────────────────────┤
 │                   tensor/, memory/, kv/                     │
 │    Tensors, quant profiles, arenas, paged KV structures     │
 ├─────────────────────────────────────────────────────────────┤
 │                      backend/                               │
 │   CUDA/Metal backends, streams, CUDA Graphs, fused kernels  │
 └─────────────────────────────────────────────────────────────┘
```

## 3. Core Components and Concepts

### 3.1 Tensor and Memory System
Vexel defines robust types for tensor manipulation:
*   **`DType`**: Enumerates supported data types (Float32, Float16, BFloat16, Int8, Int4, Uint32).
*   **`Location`**: Specifies whether a tensor resides on Host or Device.
*   **`Shape`**: Represents tensor dimensions and provides methods for element counting and stride calculation.
*   **`Tensor`**: The core data structure for N-dimensional arrays, encapsulating shape, data type, location, and underlying data pointers (`HostData` or `DevPtr`). Tensors do not implicitly move between host and device.
*   **`DevicePtr`**: Abstracts device memory pointers, including `Base` address, `Size`, and the `Device` type (CPU, CUDA, Metal).

### 3.2 Memory Arenas and Lifetimes
Memory management is handled via arenas to optimize allocations:
*   **`ArenaKind`**: Categorizes arenas (ArenaWeights, ArenaKVPages, ArenaScratch).
*   **`Arena`**: Manages a contiguous block of memory on a specific device. It provides `Alloc` for sub-allocations and `Reset` to efficiently clear memory for reuse without deallocation.
*   **`InferenceContext`**: Groups the `Weights`, `KVPages`, and `Scratch` arenas for a given inference session.

### 3.3 Quantization Profiles
Vexel supports various quantization schemes to optimize model size and performance:
*   **`QuantProfile`**: Defines quantization types (QNone, Q4_K, Q8, FP8_E4M3).
*   **`QuantizedTensor`**: Wraps a `Tensor` with a specific `QuantProfile` and optional metadata, allowing backend kernels to specialize execution based on the quantization.

### 3.4 Block IR (Intermediate Representation)
The Block IR provides an abstract, graph-based representation of model computations:
*   **`TensorID`**: Unique identifier for tensors within the IR.
*   **`OpKind`**: Enumerates supported operations (Matmul, RMSNorm, RoPE, FusedAttention, GatedMLP, Add).
*   **`OpNode`**: Represents a single operation in the graph, with its kind, input/output tensor IDs, and attributes.
*   **`BlockIR`**: Comprises inputs, outputs, and a sequence of `OpNode`s, representing a computational block. Fusion passes can rewrite common operation sequences for optimization.

### 3.5 Backend Interface
The `backend/` layer defines a polymorphic `Backend` interface, enabling hardware abstraction:
*   **`Backend` Interface**: Specifies methods for stream and event management (`CreateStream`, `RecordEvent`, `WaitEvent`, `SynchronizeStream`), graph compilation (`CompileBlockGraph`), graph execution (`RunGraph`), and host-device memory transfers (`HostToDevice`, `DeviceToHost`).
*   Implementations: Dedicated CUDA and Metal backends adhere to this interface, utilizing respective hardware-specific features like CUDA Graphs or Metal command buffers for maximum performance.

### 3.6 Paged KV Cache
Vexel employs a paged Key-Value (KV) cache for efficient memory usage and handling of long context windows:
*   **`KVConfig`**: Configures the KV cache with parameters like number of layers, KV heads, head dimension, page size, and maximum sequence length.
*   **`KVCache`**: Manages the underlying `pages` tensor, which is structured as `[num_layers][2 (K/V)][num_pages][num_kv_heads][page_size][head_dim]`, backed by a dedicated `pageArena`.
*   **`SeqKVHandle`**: Tracks the page table for individual sequences, mapping logical sequence positions to physical pages in the cache.

## 4. Model Runtime
The `runtime/` component orchestrates model execution:
*   **`ModelConfig`**: Defines model hyperparameters (VocabSize, HiddenSize, NumLayers, etc.), maximum sequence length, batch size, quantization profile, and target device.
*   **`ModelRuntime`**: Encapsulates the `ModelConfig`, chosen `Backend`, `InferenceContext`, compiled `BlockRuntime`s (one per block/layer), `KVCache`, and a `Stream` for execution.
*   **`DecodeStep`**: The core inference function, which for each layer constructs `GraphInputs`, runs the compiled graph, updates hidden states, projects to vocabulary, samples tokens, and appends KV entries.

## 5. Inference Orchestration

### 5.1 Scheduler and Continuous Batching
The `scheduler/` is central to managing concurrent inference requests and maximizing GPU utilization:
*   **`Sequence`**: Represents an individual inference request, tracking its state (Pending, Prefilling, Decoding, Finished, Error), priority, prompt, generated tokens, KV handle, and context.
*   **`Scheduler`**:
    *   **Main Loop**: Runs periodically (e.g., `DecodeInterval`) to execute inference steps.
    *   **`collectReady()`**: Identifies sequences ready for processing.
    *   **`formBatches()`**: Groups ready sequences into batches based on a configurable policy.
    *   **`runDecodeStep()`**: Executes the `ModelRuntime.DecodeStep` for a given batch, streams tokens back to users, and updates sequence states and KV cache.

## 6. Serving Layer
The `serve/` layer provides external access to Vexel's capabilities:
*   **HTTP/gRPC Endpoints**: Offers `/generate` for non-streaming inference and `/stream` for token-by-token streaming, suitable for interactive applications.
*   Each incoming request is converted into a `Sequence` and registered with the scheduler.

## 7. Build System
Vexel's build system supports conditional compilation for different hardware backends:
*   **Directory Structure**: Organizes source code into logical modules (e.g., `inference/tensor/`, `inference/backend/cpu/`, `inference/backend/cuda/`, `inference/backend/metal/`).
*   **Backend Compilation**: Uses `nvcc` for CUDA and `metallib` pipelines for Metal.
*   **Go Build Tags**: Leverages standard Go build tags (`-tags=cuda`, `-tags=metal`) to enable compilation of specific backend implementations.

## 8. Performance Targets and Reference Model
Vexel aims for ambitious performance targets:
*   **Reference Model**: Llama-3-style 8B (32 layers, 4096 hidden size, 32 attention heads, 8 KV heads, 128 head dim, 14336 FFN dim, 128k vocabulary, 8192 max context).
*   **Performance Goals (on A100/L40/4090)**:
    *   Prefill 512 tokens: < 25 ms.
    *   Decode step: < 6 ms/token (Q4_K).
    *   Time-to-first-token: < 15 ms.
    *   Active streaming sequences: 2,000–4,000.
*   **VRAM Usage**: Approximately 4–6 GB for Q4_K weights, with KV memory bounded by page configuration.

### Memory Planning API
Vexel includes a `MemoryPlan` API to estimate resource requirements:
*   `MemoryPlan` struct: Provides estimates for `WeightsBytes`, `KVBytes`, `ScratchBytes`, and `ApproxParams`.
*   `ModelConfig.MemoryPlan(kv KVConfig, maxActiveSeqs int)`: A function to calculate memory usage based on model configuration, KV cache configuration, and maximum active sequences. This includes detailed calculations for parameter count, weights memory, paged KV memory, and scratch memory.
