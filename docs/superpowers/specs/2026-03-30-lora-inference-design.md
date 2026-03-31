# LoRA Inference (Phase 1) — Design Spec

## Goal

Load and apply LoRA adapters during inference, enabling `vexel generate --model model.gguf --lora ./adapter/` to produce adapted output. No training in this phase — adapters are loaded from files (e.g., pre-trained adapters from HuggingFace converted to safetensors).

## Architecture

```
CLI: --lora flag
  → LoRA loader (reads safetensors adapter files)
  → Adapter struct (A/B matrices per layer, stored on GPU)
  → Injection into BlockRuntime forward pass
  → Modified Q/V projections: y = W@x + scale * A@(B@x)
```

### Components

**1. `inference/lora/adapter.go` — Adapter data structure + loading**

```go
type Adapter struct {
    Layers []LayerAdapter
    Rank   int
    Alpha  float32
    Scale  float32 // = Alpha / Rank
}

type LayerAdapter struct {
    // Q projection LoRA
    QA tensor.DevicePtr // [rank, hidden] — LoRA A matrix
    QB tensor.DevicePtr // [qDim, rank]   — LoRA B matrix
    // V projection LoRA
    VA tensor.DevicePtr // [rank, hidden]
    VB tensor.DevicePtr // [vDim, rank]
}
```

Standard LoRA convention: A is initialized randomly (Kaiming), B is initialized to zero. The product A@B starts as zero, so the adapter initially has no effect on the base model. The `Scale` factor (alpha/rank) controls the adapter's contribution magnitude.

Weight dimensions follow the HuggingFace PEFT convention:
- `A` (lora_A): [rank, in_features] — projects input down to rank
- `B` (lora_B): [out_features, rank] — projects rank back up to output

**2. `inference/lora/safetensors.go` — Safetensors file I/O**

Read adapter weights from safetensors format. HuggingFace PEFT saves LoRA adapters as:
```
adapter_model.safetensors:
  base_model.model.layers.0.self_attn.q_proj.lora_A.weight  [rank, hidden]
  base_model.model.layers.0.self_attn.q_proj.lora_B.weight  [qDim, rank]
  base_model.model.layers.0.self_attn.v_proj.lora_A.weight  [rank, hidden]
  base_model.model.layers.0.self_attn.v_proj.lora_B.weight  [vDim, rank]
  ...per layer...

adapter_config.json:
  {"r": 16, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"], ...}
```

The loader reads adapter_config.json for rank/alpha/target_modules, then loads the corresponding weight tensors from safetensors, copies them to GPU via `AllocPermanent`, and constructs the `Adapter` struct.

**3. Forward pass injection in `inference/runtime/block.go`**

In `ExecuteWithGPUKV`, after each frozen Q/V matmul, add the LoRA contribution. The LoRA forward for a single projection is:

```
q_lora = scale * (B @ (A @ x))     where x is normOut [seqLen, hidden]
Q = Q_frozen + q_lora
```

This requires two small FP32 matmuls per LoRA-adapted projection:
1. `intermediate = A @ x` → [seqLen, rank] (rank is small: 4-64)
2. `lora_out = B @ intermediate` → [seqLen, qDim]
3. `Q += scale * lora_out` (or fuse scale into step 2)

For M=1 decode, these are tiny matvecs (rank=16 means 16-element dot products). Overhead is minimal.

**4. Metal kernel: `lora_forward_f32`**

Fused kernel that computes `out += scale * B @ (A @ x)` in a single dispatch:
- Input x: [1, hidden] (decode) or [seqLen, hidden] (prefill)
- A: [rank, hidden]
- B: [outDim, rank]
- out: [1, outDim] or [seqLen, outDim] — accumulated in-place

For M=1 decode (the hot path), this is two small matvecs that fit entirely in registers/shared memory. For prefill (M>1), use the existing FP32 MatMulTransposed for A@x and B@intermediate separately.

**5. CLI integration**

```
vexel generate --model model.gguf --lora ./adapter/ --prompt "Hello"
vexel serve --model model.gguf --lora ./adapter/
vexel chat --model model.gguf --lora ./adapter/
```

The `--lora` flag accepts a directory path containing `adapter_model.safetensors` and `adapter_config.json`. The adapter is loaded after the base model and injected into the runtime before generation begins.

### Data Flow

```
1. Load base model (GGUF, quantized)
2. Load LoRA adapter (safetensors, FP32)
3. Copy A/B matrices to GPU (AllocPermanent)
4. Attach adapter to ModelRuntime
5. During forward pass:
   For each layer:
     normOut = RMSNorm(x)
     Q = Wq @ normOut                    // existing quantized matmul
     Q += scale * B_q @ (A_q @ normOut)  // NEW: LoRA contribution
     V = Wv @ normOut                    // existing quantized matmul
     V += scale * B_v @ (A_v @ normOut)  // NEW: LoRA contribution
     K = Wk @ normOut                    // unchanged (no LoRA on K)
     ... rest of layer unchanged ...
```

### What This Enables

1. Load HuggingFace PEFT LoRA adapters (after converting to safetensors if needed)
2. Apply LoRA at inference time with minimal overhead (~2-5% decode speed impact for rank 16)
3. Hot-swap adapters without reloading the base model
4. Stack multiple adapters (future, not v1)

### Testing

1. **Unit test:** Create a small LoRA adapter with known A/B values, verify the forward pass output matches manual computation
2. **Integration test:** Load a real LoRA adapter (train one with PEFT on a small model), verify output differs from base model in the expected direction
3. **Performance test:** Verify decode speed regression is <5% for rank 16 LoRA
4. **Safetensors test:** Round-trip save/load of adapter weights, verify bit-exact

### File Structure

```
inference/lora/
  adapter.go          — Adapter struct, GPU weight management
  safetensors.go      — Safetensors reader for adapter weights
  config.go           — adapter_config.json parser
  lora_test.go        — Unit tests

inference/runtime/
  block.go            — Modified: LoRA injection in forward pass
  model.go            — Modified: AttachLoRA method

inference/backend/metal/
  metal_bridge_darwin.m — New: lora_forward_f32 kernel
  backend.go            — New: LoRAForward dispatch

inference/cmd/vexel/
  commands.go         — Modified: --lora flag, adapter loading
  cli.go              — Modified: --lora in GlobalFlags
```

### Non-Goals (Phase 1)

- Training (Phase 2+3)
- LoRA on K, O, or FFN projections (easy to add later, same mechanism)
- GGUF-format LoRA adapters (safetensors only for now)
- Adapter merging (baking LoRA into base weights)
- Multiple simultaneous adapters
