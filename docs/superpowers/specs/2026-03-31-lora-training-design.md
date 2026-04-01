# LoRA Training (Phase 1) — Design Spec

## Goal

Add LoRA fine-tuning to vexel, enabling `vexel train --model model.gguf --data train.jsonl --rank 16 --lr 1e-4 --output ./adapter/` to train a LoRA adapter on Apple Silicon via Metal. The trained adapter is saved in HuggingFace PEFT format and can be loaded for inference with `--lora`.

## Scope

- **Single-example SGD training** — one example per step, small datasets (hundreds of examples), interactive use on Mac
- **Q and V projections only** — matches the existing inference injection points
- **SGD optimizer** — reuses the existing `SGDUpdate` kernel
- **JSONL data** — auto-detects `{"text"}` (full-sequence loss) vs `{"prompt", "completion"}` (completion-only loss)
- **Live loss output** — prints loss per step to stdout
- **Ctrl-C checkpoint** — graceful shutdown saves adapter to output directory

## Non-Goals (for this phase)

- Batch training (multiple examples per step)
- AdamW or other optimizers
- K/O/FFN target modules
- Gradient accumulation
- Evaluation / validation splits
- Distributed training
- Activation checkpointing

## Architecture

```
CLI ("vexel train" subcommand)
  → Data loader (JSONL → tokens + loss mask)
  → Trainer loop:
      Forward pass (existing ModelRuntime + activation saving)
      → Cross-entropy loss
      → Full backward pass (layer-by-layer, reverse order)
      → SGD weight update
      → Print loss
  → Checkpoint writer (safetensors format)
```

## Training Forward Pass

The training forward pass reuses the existing inference forward pass with one addition: at each layer, intermediate activations needed for the backward pass are saved.

**Saved per layer:**
- `normOut` — RMSNorm output feeding Q/K/V projections `[seqLen, hidden]`
- `Q, K, V` — post-RoPE query/key/value tensors
- `attnWeights` — attention weight matrix `[nHeads, seqLen, seqLen]`
- `gate, up` — FFN gating intermediates `[seqLen, ffnHidden]`
- RMSNorm RMS statistics (scalar per position per norm)

**Memory estimate** (Qwen 0.5B, seqLen=256):
- normOut: 24 × 256 × 896 × 4 = 21 MB
- Q/K/V: 24 × 3 × 256 × 896 × 4 = 63 MB
- attnWeights: 24 × 14 × 256 × 256 × 4 = 88 MB
- gate/up: 24 × 2 × 256 × 4864 × 4 = 227 MB
- Total: ~400 MB (comfortable on any Apple Silicon Mac)

## Loss Computation

Cross-entropy loss with loss masking:

```
For each position t where mask[t] == 1:
  loss += -log(softmax(logits[t])[target[t+1]])
loss /= num_masked_positions
```

**Loss masking by data format:**
- `{"text": "..."}` — mask is all 1s, loss on every token
- `{"prompt": "...", "completion": "..."}` — mask is 0 for prompt tokens, 1 for completion tokens

Gradient of cross-entropy w.r.t. logits:
```
dLogits[t][j] = (softmax(logits[t])[j] - (1 if j == target[t+1] else 0)) * mask[t]
```

Computed by a fused `CrossEntropyLossForwardBackward` kernel for numerical stability (log-sum-exp trick).

## Backward Pass

Full backward through all layers. Frozen layers compute activation gradients only (no weight gradients). LoRA layers additionally compute weight gradients for A and B matrices.

```
// Phase 1: Loss → hidden state gradient
dLogits = CrossEntropyBackward(logits, targets, mask)
dLastHidden = dLogits @ Wunembed^T
dResidual = RMSNormBackward(dLastHidden, finalNormState)

// Phase 2: Layer-by-layer backward (N-1 down to 0)
For layer i = N-1 down to 0:

  // FFN backward (activation gradients only)
  dFFNOut = dResidual
  dFFNMid = MatMul(dFFNOut, Wdown^T)
  dGate, dUp = SiLUMulBackward(dFFNMid, gate_i, up_i)
  dFFNInput = MatMul(dGate, Wgate^T) + MatMul(dUp, Wup^T)
  dFFNNorm = RMSNormBackward(dFFNInput, ffnNormState_i)
  dResidual += dFFNNorm

  // Attention backward (activation gradients only)
  dAttnOut = dResidual
  dAttnProj = MatMul(dAttnOut, Wo^T)
  dQ, dK, dV = SDPABackward(dAttnProj, Q_i, K_i, V_i, attnWeights_i)
  dQ = RoPEBackward(dQ)
  dK = RoPEBackward(dK)

  // LoRA weight gradients
  normOut = activations[i]

  // Q LoRA: forward was Q += scale * B_q @ (A_q @ normOut^T)
  inter_q = normOut @ A_q^T                              // recompute [seqLen, rank]
  dB_q += scale * (dQ^T @ inter_q)                       // [qDim, rank]
  dA_q += scale * (B_q^T @ dQ)^T @ normOut               // [rank, hidden]

  // V LoRA: same pattern
  inter_v = normOut @ A_v^T
  dB_v += scale * (dV^T @ inter_v)
  dA_v += scale * (B_v^T @ dV)^T @ normOut

  // Continue residual gradient
  dNormOut = MatMul(dQ, Wq^T) + MatMul(dK, Wk^T) + MatMul(dV, Wv^T)
  dResidual = RMSNormBackward(dNormOut, attnNormState_i)
  dResidual += dAttnOut
```

## New Metal Kernels

| Kernel | Signature | Notes |
|--------|-----------|-------|
| `CrossEntropyLossForwardBackward` | logits `[S,V]`, targets `[S]`, mask `[S]` → loss (scalar), dLogits `[S,V]` | Fused softmax + log-sum-exp for stability |
| `RMSNormBackward` | dOut `[S,H]`, input `[S,H]`, weight `[H]`, rms `[S]` → dInput `[S,H]` | Needs saved RMS per position |
| `SDPABackward` | dOut `[S,D]`, Q, K, V, attnWeights → dQ, dK, dV | Per-head, standard attention backward |
| `SiLUMulBackward` | dOut `[S,F]`, gate `[S,F]`, up `[S,F]` → dGate, dUp `[S,F]` | Element-wise `d(silu(gate) * up)` |
| `RoPEBackward` | dOut `[S,D]`, freqs → dInput `[S,D]` | Inverse rotation (RoPE is orthogonal) |

**Reused existing kernels:**
- `MatMulTransposed` — all matmul backward ops (different operand order)
- `Add` — residual gradient accumulation
- `SGDUpdate` — weight updates
- `Zero` — gradient buffer reset
- `ScaleBuffer` — LoRA scale application

## Optimizer

SGD with momentum. Per-parameter state: one momentum buffer (same shape as parameter).

```
momentum = momentum_coeff * momentum + gradient
weight -= lr * momentum + lr * weight_decay * weight
```

Uses the existing `SGDUpdate` kernel. Momentum buffers allocated as permanent GPU memory alongside LoRA weights.

**CLI flags:**
- `--lr` — learning rate (default: 1e-4)
- `--momentum` — SGD momentum coefficient (default: 0.9)
- `--weight-decay` — L2 regularization (default: 0.0)

## Data Loading

- Read entire JSONL into memory at startup (small datasets assumed)
- Auto-detect format from first line: presence of `"prompt"` + `"completion"` fields vs `"text"` field
- Validate all examples parse correctly before training begins
- Shuffle example order each epoch
- One example per step (no batching)
- Truncate to model max context length with warning

**Chat template for prompt/completion format:**
1. Tokenize prompt with model's chat template → `prompt_tokens`
2. Tokenize completion → `completion_tokens`
3. Concatenate → `tokens = prompt_tokens + completion_tokens`
4. Mask: 0 for `0..len(prompt_tokens)-1`, 1 for remaining positions

## LoRA Weight Initialization

- A matrices: Kaiming uniform (standard PEFT convention)
- B matrices: zeros (adapter starts as no-op)
- Scale: `alpha / rank` (pre-computed, same as inference)

## Checkpoint Format

Standard HuggingFace PEFT format for interoperability:

```
output_dir/
  adapter_config.json         — {"r": 16, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": "..."}
  adapter_model.safetensors   — LoRA A/B weights per layer
```

Requires a new **safetensors writer** (inverse of the existing loader): 8-byte header length + JSON metadata + packed FP32 tensor data.

## Signal Handling

Register SIGINT handler. On Ctrl-C:
1. Finish current training step
2. Save checkpoint to output directory
3. Print checkpoint path and final loss
4. Exit cleanly

## CLI Interface

New `train` subcommand:

```
vexel train \
  --model model.gguf \
  --data train.jsonl \
  --output ./my-adapter/ \
  --rank 16 \
  --alpha 16 \
  --lr 1e-4 \
  --momentum 0.9 \
  --weight-decay 0.0 \
  --epochs 3
```

**Live output:**

```
LoRA training: rank=16, alpha=16, lr=1e-4, SGD momentum=0.9
Model: qwen2.5-0.5b-instruct (24 layers, hidden=896)
Data: 150 examples (83 text, 67 prompt/completion)

epoch 1/3
  step 1/150    loss=3.4521
  step 2/150    loss=3.2187
  ...
  step 150/150  loss=1.8934

epoch 2/3
  step 151/300  loss=1.8412
  ^C
Saving checkpoint to ./my-adapter/ ... done.
```

## Package Structure

```
inference/lora/
  config.go              — (existing) adapter config
  loader.go              — (existing) safetensors reader
  gpu_adapter.go         — (existing) GPU upload
  writer.go              — NEW: safetensors writer for checkpoints

inference/lora/train/
  trainer.go             — training loop, signal handling, live output
  backward.go            — backward pass (layer-by-layer gradient computation)
  data.go                — JSONL loading, auto-detect, loss masking
  init.go                — LoRA weight initialization (Kaiming A, zero B)

inference/backend/backend.go
  — extend TrainingOps interface with backward methods

inference/backend/metal/
  backend.go             — Go dispatch wrappers for new kernels
  metal_bridge_darwin.m  — new kernel dispatch functions
  kernels/               — .metal shader source for backward kernels

inference/cmd/vexel/
  commands.go            — add "train" subcommand
  cli.go                 — add training flags

inference/runtime/
  block.go               — save activations when training mode enabled
  model.go               — training forward path
```
