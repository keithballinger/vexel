# Vexel: a Go LLM Inference Engine

### High-performance single-GPU Transformer runtime with continuous batching and paged KV

---

## 0. Goals and Scope

This project implements a high-performance inference engine for transformer-based language models (<10B parameters) with the following priorities:

* Single-GPU throughput and latency competitive with modern engines.
* Low-latency streaming for interactive workloads.
* Continuous batching across many concurrent sequences.
* Paged KV cache for efficient memory use and long contexts.
* Go-native orchestration using idiomatic concurrency patterns.
* CUDA/Metal backends providing optimized fused kernels and graph execution.

The first performance target is Llama-like architectures in FP16/BF16/INT4/INT8 with context lengths up to 8k.

---

# 1. High-Level Architecture

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

---

# 2. Tensor and Memory System

## 2.1 DTypes and Locations

```go
type DType uint8

const (
    Float32 DType = iota
    Float16
    BFloat16
    Int8
    Int4
    Uint32
)

func (d DType) SizeBytes() int
func (d DType) BitSize() int

type Location uint8

const (
    Host Location = iota
    Device
)
```

## 2.2 Shape and Strides

```go
type Shape []int

func (s Shape) NumElements() int
func (s Shape) Rank() int
func (s Shape) Equal(other Shape) bool
func (s Shape) StridesRowMajor() []int
```

## 2.3 Tensor Representation

```go
type Tensor struct {
    Shape    Shape
    DType    DType
    Loc      Location

    HostData []byte
    DevPtr   DevicePtr

    Offset   int
    Strides  []int

    arenaID  uint64
}
```

Tensors never implicitly move between host and device.

## 2.4 Device Pointers

```go
type Device uint8

const (
    DeviceCPU Device = iota
    DeviceCUDA
    DeviceMetal
)

type DevicePtr struct {
    Base   uintptr
    Size   int
    Device Device
}
```

---

# 3. Memory Arenas and Lifetimes

## 3.1 Arena Types

```go
type ArenaKind uint8

const (
    ArenaWeights ArenaKind = iota
    ArenaKVPages
    ArenaScratch
)

type Arena struct {
    id       uint64
    kind     ArenaKind
    device   Device
    basePtr  DevicePtr
    hostBuf  []byte
    offset   int
    capacity int
}

func NewArena(kind ArenaKind, device Device, capacity int) *Arena
func (a *Arena) Alloc(size, align int) (DevicePtr, []byte)
func (a *Arena) Reset()
func (a *Arena) ID() uint64
```

Each arena reset increments its epoch. Tensors reference the epoch and may assert validity in debug mode.

## 3.2 Inference Context

```go
type InferenceContext struct {
    Weights *Arena
    KVPages *Arena
    Scratch *Arena
    Device  Device
}
```

---

# 4. Quantization Profiles

```go
type QuantProfile uint8

const (
    QNone QuantProfile = iota
    Q4_K
    Q8
    FP8_E4M3
)

type QuantizedTensor struct {
    Base    *Tensor
    Profile QuantProfile
    Meta    any
}
```

Backend kernels are specialized based on `(DType, QuantProfile)`.

---

# 5. Block IR (Intermediate Representation)

## 5.1 IR Types

```go
type TensorID int

type OpKind string

const (
    OpMatmul    OpKind = "matmul"
    OpRMSNorm   OpKind = "rmsnorm"
    OpRoPE      OpKind = "rope"
    OpFAttn     OpKind = "fused_attention"
    OpGatedMLP  OpKind = "gated_mlp"
    OpAdd       OpKind = "add"
)

type OpNode struct {
    Kind    OpKind
    Inputs  []TensorID
    Outputs []TensorID
    Attrs   map[string]any
}

type BlockIR struct {
    Inputs  []TensorID
    Outputs []TensorID
    Nodes   []OpNode
}
```

## 5.2 Example Block Structure (conceptual)

```
X → RMSNorm → Hn → FusedAttention → A
X ───────────────┬──────────────────┘
                 └── Add → H1 → RMSNorm → H2 → GatedMLP → M
H1 ─────────────────────────┬───────────────────────────────┘
                             └── Add → Y
```

Fusion passes rewrite common sequences (e.g., Matmul→SiLU→Matmul→Mul→Matmul → GatedMLP).

---

# 6. Backend Interface

The backend executes compiled block graphs on device streams.

```go
type Stream struct{ handle uintptr }
type Event struct{ handle uintptr }

type GraphInputs struct {
    Hidden   DevicePtr
    PosIDs   DevicePtr
    KVHandle DevicePtr
}

type BlockGraph struct {
    handle uintptr
}

type Backend interface {
    CreateStream(priority int) (Stream, error)
    RecordEvent(s Stream) (Event, error)
    WaitEvent(s Stream, e Event) error
    SynchronizeStream(s Stream) error

    CompileBlockGraph(ir *BlockIR, weights []*QuantizedTensor) (BlockGraph, error)
    RunGraph(graph BlockGraph, inputs GraphInputs, stream Stream) error

    HostToDevice(dst DevicePtr, src []byte) error
    DeviceToHost(dst []byte, src DevicePtr) error

    Device() Device
}
```

Backend implementations use CUDA Graphs or Metal command buffers for performance.

---

# 7. Paged KV Cache

## 7.1 KV Layout

Paged KV uses fixed-size KV pages:

```
[num_layers][2 (K/V)][num_pages][num_kv_heads][page_size][head_dim]
```

```go
type KVConfig struct {
    NumLayers      int
    NumKVHeads     int
    HeadDim        int
    PageSizeTokens int
    MaxSeqLen      int
}

type KVCache struct {
    cfg       KVConfig
    pages     *Tensor
    pageArena *Arena
}
```

## 7.2 Sequence Page Tables

```go
type PageIndex struct {
    Layer int
    IsKey bool
    Page  int
}

type SeqKVHandle struct {
    SeqID     int64
    PageTable [][]PageIndex
    Length    int
}
```

Appending new tokens allocates pages as needed and writes K/V into the appropriate slots.

---

# 8. Model Runtime

## 8.1 ModelConfig

```go
type ModelConfig struct {
    VocabSize   int
    HiddenSize  int
    NumLayers   int
    NumHeads    int
    NumKVHeads  int
    HeadDim     int
    IntermSize  int

    MaxSeqLen   int
    MaxBatch    int

    QuantProfile QuantProfile
    Device       Device
}
```

## 8.2 Runtime Structure

```go
type BlockRuntime struct {
    Graph BlockGraph
}

type ModelRuntime struct {
    Cfg      ModelConfig
    Backend  Backend
    Ctx      *InferenceContext

    Blocks   []BlockRuntime
    KV       *KVCache
    Stream   Stream

    Sampler   *Sampler
    TokenProj *QuantizedTensor
}
```

## 8.3 Batch Runtime Inputs

```go
type BatchRuntimeInputs struct {
    BatchSize  int
    Hidden     *Tensor
    PosIDs     *Tensor
    KVHandles  []*SeqKVHandle
    Sequences  []*Sequence
}
```

## 8.4 Decode Step

```go
func (rt *ModelRuntime) DecodeStep(br *BatchRuntimeInputs) ([]int, error)
```

Per layer:

1. Construct `GraphInputs`.
2. Run compiled graph.
3. Update hidden state.
4. After all layers, project to vocab and sample tokens.
5. Append KV entries via paged KV.

---

# 9. Scheduler and Continuous Batching

## 9.1 Sequence State

```go
type SeqState int

const (
    SeqStatePending SeqState = iota
    SeqStatePrefilling
    SeqStateDecoding
    SeqStateFinished
    SeqStateError
)

type Sequence struct {
    ID            int64
    State         SeqState
    Priority      int
    Deadline      time.Time

    PromptTokens  []int
    Generated     []int

    KVHandle      *SeqKVHandle
    Hidden        *Tensor
    Pos           int

    Ctx           context.Context
    Cancel        context.CancelFunc

    StreamCh      chan string
    Err           error
}
```

## 9.2 Scheduler Structure

```
 API → request admission → seq_registry → decode loop
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                collect_ready()        form_batches()
                          │                       │
                          └───────────┬────────────┘
                                      ▼
                               run_decode_step()
```

## 9.3 Main Loop

```go
func (s *Scheduler) Run() {
    ticker := time.NewTicker(s.cfg.DecodeInterval)
    defer ticker.Stop()

    for {
        select {
        case <-s.ctx.Done():
            return
        case <-ticker.C:
            s.step()
        }
    }
}
```

```go
func (s *Scheduler) step() {
    ready := s.collectReady()
    if len(ready) == 0 {
        return
    }

    batches := s.formBatches(ready)
    for _, batch := range batches {
        if err := s.runDecodeStep(batch); err != nil {
            s.handleBatchError(batch, err)
        }
    }
}
```

## 9.4 Collecting Ready Sequences

```go
func (s *Scheduler) collectReady() []*Sequence
```

Sequences in `Pending` or `Decoding` state are included.

## 9.5 Forming Batches

```go
func (s *Scheduler) formBatches(ready []*Sequence) [][]*Sequence
```

Simple priority-based batching; can be replaced with more advanced policies.

## 9.6 Running a Decode Step

```go
func (s *Scheduler) runDecodeStep(batch []*Sequence) error
```

1. Construct batch inputs.
2. Execute model decode step.
3. Stream tokens back to users.
4. Update KV and sequence state.

---

# 10. Serving Layer

Provides HTTP/gRPC endpoints:

* `/generate` for non-streaming generation
* `/stream` for token-by-token streaming

Each request becomes a `Sequence` registered with the scheduler.

---

# 11. Build System

Directory structure:

```
inference/
  tensor/
  memory/
  kv/
  backend/
    cpu/
    cuda/
    metal/
  ir/
  runtime/
  scheduler/
  serve/
  cmd/
```

Backends built with:

* CUDA: nvcc + CUDA Graphs
* Metal: metallib pipelines

Go build tags:

* `-tags=cuda`
* `-tags=metal`

---

# 12. Performance Targets

For a 7–8B model on a modern GPU (A100/L40/4090):

* Prefill 512 tokens: < 25 ms
* Decode step: < 6 ms/token (Q4_K)
* Time-to-first-token: < 15 ms
* Active streaming sequences: 2k–4k
* VRAM usage:

  * ~4–6 GB (Q4_K weights)
  * KV bounded by page config

---

# 13. Reference Model and Memory Planning

## 13.1 Reference Model

The reference target model is **Llama-3-style 8B** with:

* Layers: 32
* Hidden size: 4096
* Attention heads: 32
* KV heads: 8
* Head dim: 128
* FFN dim: 14 336
* Vocabulary: 128k
* Max context: 8192

Configuration:

```go
var Llama3_8B = ModelConfig{
    VocabSize:   128_000,
    HiddenSize:  4096,
    NumLayers:   32,
    NumHeads:    32,
    NumKVHeads:  8,
    HeadDim:     128,
    IntermSize:  14_336,
    MaxSeqLen:   8_192,
    MaxBatch:    512,
    QuantProfile: Q4_K,
    Device:       DeviceCUDA,
}
```

---

## 13.2 Memory Plan API

```go
type MemoryPlan struct {
    WeightsBytes int64
    KVBytes      int64
    ScratchBytes int64
    ApproxParams float64
}

func (c ModelConfig) MemoryPlan(kv KVConfig, maxActiveSeqs int) MemoryPlan
```

---

## 13.3 Parameter Count

```go
func (c ModelConfig) ApproxParams() float64 {
    V, D := float64(c.VocabSize), float64(c.HiddenSize)
    L := float64(c.NumLayers)
    Dff := float64(c.IntermSize)
    H := float64(c.NumHeads)
    Dh := float64(c.HeadDim)

    emb := V * D
    final := V * D

    qkv := 3 * D * H * Dh
    wo  := D * D
    mlp := 3 * D * Dff

    perLayer := qkv + wo + mlp
    return emb + final + L*perLayer
}
```

---

## 13.4 Weights Memory

```go
func (c ModelConfig) WeightsBytes() int64 {
    params := c.ApproxParams()
    var bpp float64

    switch c.QuantProfile {
    case QNone:
        bpp = 2
    case Q4_K:
        bpp = 0.7
    case Q8:
        bpp = 1.1
    case FP8_E4M3:
        bpp = 1
    }

    return int64(params * bpp)
}
```

---

## 13.5 KV Memory (Paged)

```go
func KVBytes(c ModelConfig, kv KVConfig, maxActiveSeqs int) int64 {
    L := int64(c.NumLayers)
    Hkv := int64(c.NumKVHeads)
    Dh := int64(c.HeadDim)
    pageSize := int64(kv.PageSizeTokens)

    maxTokens := int64(kv.MaxSeqLen) * int64(maxActiveSeqs)
    numPages := (maxTokens + pageSize - 1) / pageSize

    elems := L * 2 * numPages * Hkv * pageSize * Dh
    return elems * 2 // FP16 KV
}
```

---

## 13.6 Scratch Memory

```go
func ScratchBytes(c ModelConfig, maxBatch int) int64 {
    B := int64(maxBatch)
    D := int64(c.HiddenSize)
    Dff := int64(c.IntermSize)

    elems := 4*B*D + 2*B*Dff
    return elems * 2 * 2 // FP16 + safety factor
}
```

---

## 13.7 Combined Memory Plan

```go
func (c ModelConfig) MemoryPlan(kv KVConfig, maxActiveSeqs int) MemoryPlan {
    return MemoryPlan{
        WeightsBytes: c.WeightsBytes(),
        KVBytes:      KVBytes(c, kv, maxActiveSeqs),
        ScratchBytes: ScratchBytes(c, c.MaxBatch),
        ApproxParams: c.ApproxParams(),
    }
}
````

