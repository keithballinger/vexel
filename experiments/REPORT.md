# LoRA Training Experiments Report

## Executive Summary

We implemented LoRA fine-tuning for vexel from scratch — Metal backward kernels, SGD with momentum, full backward pass through frozen layers — and tested it across four use cases on Mistral 7B Instruct. The key finding: **LoRA is highly effective for behavioral shifts (teaching a model to output commands instead of explanations) but struggles with exact fact memorization.** The most promising approach is teaching models to call knowledge lookup tools rather than memorizing facts directly.

## Infrastructure Built

| Component | Status |
|-----------|--------|
| 5 Metal backward kernels (CE, RMSNorm, SDPA, SiLU, RoPE) | Verified correct (gradient checks pass) |
| Full backward pass through frozen layers | Working, GQA-aware |
| SGD with momentum optimizer | Working (critical for convergence) |
| Pool-based activation memory management | Working (fixed 7B OOM) |
| GPU attention weight computation | Working (replaced CPU bottleneck) |
| Chat template alignment (train ↔ inference) | Working (critical for adapter effect) |
| Q/K/V/O target module support | Implemented (Q/V verified, K/O needs tuning) |
| Safetensors checkpoint read/write | Round-trip verified |
| Batch eval tool with KV cache reset | Working |

## Experiments

### Scenario 1: Personal Knowledge Base

**Goal:** Teach the model 49 workplace facts (manager name, WiFi password, port numbers, team members).

**Best config:** rank=16, Q/V only, lr=5e-5, momentum=0.9, 50 epochs

**Training:** Loss converged from 2.5 → 0.002 over 50 epochs.

**Results:**

| Question | Base Model | With LoRA | Correct? |
|----------|-----------|-----------|----------|
| Who is Marcus? | Hunger Games character | Lead engineer, payments team | **Yes** (role correct) |
| What CI system? | Jenkins | **GitHub Actions** | **Exact match** |
| Sprint cadence? | Two weeks | Two weeks + planning/review/retro | **Yes** (with detail) |
| Standup time? | Generic | 9:00 AM every day | **Close** (training: 9:15 CT) |
| Manager? | John Doe | John Doe (+ contact details) | No |
| WiFi password? | password123 | SecureNetwork123 | No (plausible hallucination) |
| Staging port? | 3000 | 8001 | No (different wrong port) |

**Assessment:** Strong style shift from generic assistant to workplace-specific. 3/15 exact facts, several more with correct structure but wrong values. The model learned *what kind* of answer to give but not all specific values.

### Scenario 2: Domain Jargon (Biotech)

**Goal:** Teach 36 Veridian Bio-specific terms (CRISPR-VEX, VPI, ToxPredict, etc.).

**Best config:** rank=16, Q/V, lr=1e-4, momentum=0.9, 30 epochs (on earlier run)

**Results:** The model shifted from generic definitions to biotech/pharma domain responses. Mentioned real companies (Veracyte, Insilico Medicine) and plausible technical details. But did NOT learn Veridian-specific terminology — it learned the *domain* but not the *company*.

**Assessment:** Domain shift works. Specific term memorization needs more capacity or data repetition.

### Scenario 3: Tool Routing

**Goal:** Teach 57 natural language → shell command mappings.

**Best config:** rank=16, Q/V, lr=1e-4, momentum=0.9, 30 epochs

**Training:** Loss converged to 0.002.

**Results (best scenario):**

| Query | Base Model | With LoRA |
|-------|-----------|-----------|
| Check if API is healthy | Generic explanation | `curl --silent --head https://api.example.com/healthcheck` |
| How many pods running? | Generic explanation | `kubectl get pods --namespace default \| grep Running \| wc -l` |
| Restart the API | Generic explanation | `docker restart api` |
| Cache status? | "75% capacity" | `redis-cli -h cache-server -p 6379 info` |
| Disk usage? | Generic explanation | `df -h /mnt/data` |
| Cluster events? | Generic explanation | `kubectl get events --field-selector type=Normal` |

**Assessment:** Strongest results of all scenarios. The model completely shifted from "explain concepts" to "output commands." The exact URLs/hostnames aren't from training data but the command structure is correct. This demonstrates LoRA's strength: behavioral/format shifts.

### Scenario 4: Tool-Use Knowledge Lookup

**Goal:** Instead of memorizing facts, teach the model to call `<tool_call>lookup_person("manager")</tool_call>` etc.

**Status:** Training in progress.

**Hypothesis:** This should work as well as tool-routing since it's the same pattern — mapping natural language to structured output. The key difference: the lookup tools provide exact facts at runtime, so the LoRA adapter only needs to learn *when* to call each tool, not memorize the facts themselves.

## Failed Experiments

| Config | Issue | Root Cause |
|--------|-------|-----------|
| Rank 64 + Q/K/V/O, 49 examples | Empty outputs | Too much capacity for too little data — adapter destabilized |
| 5x augmented data, lr=5e-5 | Loss diverged to 7+ | LR too high for larger dataset (more gradient steps per epoch) |
| 5x augmented data, lr=1e-5 | Degenerate repetition | Catastrophic forgetting — adapter overwrote language ability despite low loss |
| Qwen 0.5B, any config | Garbage outputs | Model too small — insufficient capacity for LoRA to be effective |
| Gemma 2 2B, lr=1e-4 | Loss diverged | Architecture-specific (post-norms, softcap) needs lower LR |
| Any model without chat template | No adapter effect | Train/inference format mismatch — LoRA weights trained on wrong token patterns |

## Key Findings

### 1. Chat template alignment is mandatory
Training data MUST be tokenized with the same chat template used at inference. Without this, the LoRA adapter trains on raw text patterns but inference presents chat-templated tokens — the adapter never activates. This was the single most impactful fix.

### 2. Momentum is essential for single-example SGD
Pure SGD without momentum oscillates and never converges on 7B models. Momentum β=0.9 smooths gradients across steps, enabling loss to drop from 2.5 to 0.002.

### 3. LoRA excels at behavioral shifts, not fact memorization
Teaching a model to output `kubectl` commands instead of prose explanations works extremely well. Teaching it that the WiFi password is "maple-thunder-42" does not — the model's pretrained distribution is too strong for rank-16 Q/V LoRA to override on specific tokens.

### 4. The sweet spot is narrow
The working config (rank 16, Q/V, lr=5e-5, momentum 0.9, ~50 examples) sits in a narrow stable region. Increasing rank, adding modules, or scaling data all push past it into divergence or catastrophic forgetting without careful LR adjustment. More capacity ≠ better results.

### 5. Pool-based memory management is critical for large models
`AllocPermanent` for saved activations caused OOM on 7B models after ~400 training steps. Switching to pool-allocated buffers with explicit `ResetPool` cycles fixed this — no memory growth over 2500+ steps.

### 6. GPU kernels dramatically accelerate backward pass
Replacing CPU-side attention weight recomputation (download Q/K → CPU softmax → re-upload) with a Metal kernel eliminated the main backward pass bottleneck.

## Recommended Configuration

For Mistral 7B Instruct on Apple Silicon:

```
Rank: 16
Target modules: q_proj, v_proj
Learning rate: 5e-5
Momentum: 0.9
Weight decay: 0.0
Alpha: 32
Epochs: 40-50
Training examples: 40-100
```

Scale LR down proportionally if increasing data size or rank.

## Architecture for Production Use

Based on findings, the recommended architecture for a personal AI agent is:

```
User query → LoRA-adapted model → tool calls → tool execution → response
```

- **LoRA handles:** routing queries to the right tool, formatting tool calls, setting response tone/style
- **Tools handle:** exact fact lookup, database queries, API calls
- **NOT in LoRA:** specific facts, passwords, URLs, exact names

This plays to LoRA's strength (behavioral shifts) while avoiding its weakness (fact memorization).

## Bugs Found and Fixed During Development

| Bug | Impact | Fix |
|-----|--------|-----|
| Buffer pool recycling mask/targets | CE kernel read zeros | Upload after forward pass |
| SDPA backward GQA overrun | Zero gradients | Use numKVHeads for dK/dV sizes |
| SDPA backward GQA indexing | Wrong K/V head mapping | Add numKVHeads param to kernel |
| AllocPermanent memory leak | OOM on 7B at epoch 8 | Switch to pool-based allocation |
| Train/inference format mismatch | Adapter has no effect | Apply chat template during training |
| Shell pipe SIGPIPE | Training killed mid-epoch | Redirect to log file, no pipes |
| Metal CE kernel zero output | Loss always 0 | Buffer recycling fix (same as #1) |

## Files

```
inference/lora/train/
  trainer.go          — training loop, SGD+momentum, checkpointing
  backward.go         — full backward pass, LoRA gradient extraction
  data.go             — JSONL loader, loss masking
  init.go             — Kaiming/zero weight initialization
  gradcheck_test.go   — numerical gradient verification
  sdpa_backward_test.go — SDPA kernel unit test
  rmsnorm_backward_test.go
  silu_backward_test.go
  rope_backward_test.go
  matmul_backward_test.go
  lora_grad_test.go
  attn_weights_test.go

inference/backend/metal/
  metal_bridge_darwin.m — 7 new Metal kernels
  metal_bridge.h       — C declarations
  backend.go           — Go dispatch wrappers

experiments/
  personal-kb/         — workplace Q&A scenario
  domain-jargon/       — biotech terminology scenario
  tool-routing/        — intent → command mapping scenario
  tool-use-kb/         — tool call routing scenario
  eval_batch.go        — batch evaluation tool
```
