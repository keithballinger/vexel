# Model Flexibility Analysis

## Current Architecture Support

Vexel currently supports **LLaMA-family decoder-only transformers**:
- LLaMA 2/3, TinyLlama, Mistral, Codestral
- Architecture: Pre-norm with RMSNorm, GQA attention, SwiGLU FFN, RoPE

## Hardcoded Assumptions

### 1. Normalization: RMSNorm Only
**File**: `runtime/block.go:249, 309`
```go
b.backend.RMSNorm(xPtr, b.AttnNorm.DevicePtr(), normOutPtr, ...)
```
- No LayerNorm support
- **Models affected**: Phi-2/3 (LayerNorm), GPT-2/NeoX (LayerNorm)

### 2. FFN Structure: SwiGLU (3 projections)
**File**: `runtime/block.go:314-327`
```go
// Gate projection: gate = SiLU(normOut @ W1^T)
// Up projection: up = normOut @ W3^T
// SiLU+Mul: gate = silu(gate) * up
// Down projection: result = gate @ W2^T
```
- Requires W1 (gate), W2 (down), W3 (up) matrices
- **Models affected**:
  - GPT-2/NeoX: GELU MLP with 2 projections (W_fc, W_proj)
  - Falcon: 2-projection MLP

### 3. Position Encoding: RoPE (Split Pairs)
**File**: `runtime/block.go:288`
```go
b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, pos, float32(b.RoPETheta))
```
- Only LLaMA-style split pairs: `(0, 32), (1, 33), ...`
- **Models affected**:
  - GPT-NeoX: Interleaved pairs `(0, 1), (2, 3), ...`
  - ALiBi models: No RoPE (Bloom, MPT)
  - Sinusoidal: Older GPT-2 style

### 4. No Bias Terms
**File**: `runtime/block.go`
- Wq, Wk, Wv, Wo, W1, W2, W3 are weight-only
- **Models affected**: Phi-2 (has attention biases), GPT-2

### 5. Tensor Naming: HuggingFace + GGML Style
**File**: `runtime/loader.go:484-515`, `pkg/gguf/loader.go:85-137`
- Supports: `model.layers.X.self_attn.q_proj.weight` → `blk.X.attn_q.weight`
- **Models affected**: Models with different naming conventions

## What's Already Configurable

| Parameter | Config Field | Example Values |
|-----------|--------------|----------------|
| Hidden size | `HiddenSize` | 2048, 4096 |
| FFN size | `IntermediateSize` | 5632, 14336 |
| Num layers | `NumHiddenLayers` | 22, 32 |
| Q heads | `NumAttentionHeads` | 32 |
| KV heads | `NumKeyValueHeads` | 4, 8, 32 |
| Vocab size | `VocabSize` | 32000, 128256 |
| RoPE theta | `RoPETheta` | 10000, 500000 |
| RMSNorm eps | `RMSNormEPS` | 1e-5, 1e-6 |

## Recommendations for Extensibility

### Option A: Architecture Enum + Switch (Quick Win)
Add `Architecture` field to `ModelConfig`, switch on it in Execute:
```go
type Architecture string
const (
    ArchLLaMA   Architecture = "llama"
    ArchPhi     Architecture = "phi"
    ArchFalcon  Architecture = "falcon"
)

func (b *BlockRuntime) Execute(...) {
    switch b.config.Architecture {
    case ArchLLaMA:
        return b.executeLLaMA(...)
    case ArchPhi:
        return b.executePhi(...)  // LayerNorm, biases
    }
}
```
**Pros**: Simple, explicit, fast
**Cons**: Code duplication, N×M problem with optimizations

### Option B: Composable Components (Medium Effort)
Define interfaces for each component:
```go
type Normalizer interface {
    Normalize(input, weights, output DevicePtr, ...)
}
type FFN interface {
    Forward(input, output DevicePtr, weights []Tensor, ...)
}
type PositionEncoder interface {
    Encode(q, k DevicePtr, pos int, ...)
}

type BlockConfig struct {
    AttnNorm  Normalizer  // RMSNorm or LayerNorm
    FFNNorm   Normalizer
    FFN       FFN         // SwiGLU or GELU-MLP
    PosEnc    PositionEncoder // RoPE or ALiBi
}
```
**Pros**: Flexible, composable, testable
**Cons**: Interface overhead, more complex config

### Option C: Config-Driven Execution (Full Flexibility)
Model config specifies execution graph:
```go
type LayerOp struct {
    Type    string  // "rmsnorm", "layernorm", "matmul", "rope", "gelu", "silu"
    Inputs  []string
    Outputs []string
    Params  map[string]interface{}
}
```
**Pros**: Maximum flexibility, declarative
**Cons**: Complex, hard to optimize, overkill for now

## Recommended Next Steps

### Phase 1: Quick Wins (Immediate)
1. Add `Architecture` field to GGUF parsing (already parsed as `general.architecture`)
2. Add validation: warn if architecture != "llama"
3. Document supported models in README

### Phase 2: LayerNorm Support (If Needed)
1. Add `LayerNorm` kernel to backend interface
2. Add `NormType` enum to config (`RMSNorm` vs `LayerNorm`)
3. Switch in `Execute` based on config

### Phase 3: FFN Variants (If Needed)
1. Add `FFNType` enum (`SwiGLU`, `GELU`, `GEGLU`)
2. Implement GELU MLP path (2 projections)
3. Handle missing W3 gracefully

## Model Compatibility Matrix

| Model | Architecture | Norm | FFN | RoPE | Bias | Status |
|-------|--------------|------|-----|------|------|--------|
| TinyLlama 1.1B | llama | RMSNorm | SwiGLU | Split | No | ✅ Working |
| Mistral 7B | llama | RMSNorm | SwiGLU | Split | No | ✅ Working |
| LLaMA 2/3 | llama | RMSNorm | SwiGLU | Split | No | ✅ Should work |
| Codestral | llama | RMSNorm | SwiGLU | Split | No | ✅ Should work |
| Phi-2 | phi | LayerNorm | GELU | Split | Yes | ❌ Not supported |
| Falcon | falcon | LayerNorm | GELU | ALiBi | Yes | ❌ Not supported |
| Qwen 2 | qwen2 | RMSNorm | SwiGLU | Split | No | 🔶 Likely works |
| Gemma | gemma | RMSNorm | GEGLU | Split | No | 🔶 Close |

## Conclusion

Current architecture is optimized for LLaMA-family models. Extending to other architectures requires:
1. **Minimal**: Add architecture validation + warnings
2. **Moderate**: Add LayerNorm + bias support for Phi-2
3. **Major**: Refactor to composable components for full flexibility

Recommendation: Start with Phase 1 (validation), add specific support as users request it.
