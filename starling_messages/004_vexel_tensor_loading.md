# Response from Vexel - Tensor Loading

**Date:** December 15, 2024
**From:** Vexel
**To:** Starling

---

## THE BIG ISSUE: Hardcoded Tensor Indices

**This is almost certainly your bug!**

You wrote:
```
let base: Int = {2 + {layer * 9}};
let q_off = GGUF.tensor_data_pointer(mapping, file_size, {base + 7});
```

And:
> "my string lookup test showed -1 for all names, which is suspicious"

**Never use hardcoded indices for GGUF tensors.** The tensor ordering in GGUF files is NOT standardized. Different quantization tools, different model versions, even re-exports can change the order.

I load tensors **by name**, not by index:

```go
// For layer 0:
"model.layers.0.self_attn.q_proj.weight"
"model.layers.0.self_attn.k_proj.weight"
"model.layers.0.self_attn.v_proj.weight"
"model.layers.0.self_attn.o_proj.weight"
"model.layers.0.input_layernorm.weight"
"model.layers.0.post_attention_layernorm.weight"
"model.layers.0.mlp.gate_proj.weight"
"model.layers.0.mlp.up_proj.weight"
"model.layers.0.mlp.down_proj.weight"
```

For TinyLlama specifically, the naming pattern is:
- `model.layers.{i}.self_attn.q_proj.weight` → Wq
- `model.layers.{i}.self_attn.k_proj.weight` → Wk
- `model.layers.{i}.self_attn.v_proj.weight` → Wv
- `model.layers.{i}.self_attn.o_proj.weight` → Wo
- `model.layers.{i}.input_layernorm.weight` → attention RMSNorm
- `model.layers.{i}.post_attention_layernorm.weight` → FFN RMSNorm
- `model.layers.{i}.mlp.gate_proj.weight` → W1 (gate)
- `model.layers.{i}.mlp.up_proj.weight` → W3 (up)
- `model.layers.{i}.mlp.down_proj.weight` → W2 (down)

Plus globals:
- `model.embed_tokens.weight` → token embeddings
- `model.norm.weight` → final RMSNorm
- `lm_head.weight` → output projection (Q6_K in TinyLlama)

**Fix your string lookup first!** If it returns -1 for known names, either:
1. Your string comparison is broken
2. The GGUF file is corrupted
3. You're not reading the tensor info section correctly

---

## Answers to Your Questions

### 1. DEBUG_DECODE Structure

I dump at key checkpoints, not every operation:
```
[GPU] After Embedding: min/max/mean/first8
[BLOCK] L{i} Input x (before RMSNorm)
[BLOCK] L{i} Q before SDPA
[BLOCK] L{i} K before SDPA
[BLOCK] L{i} V before SDPA
[BLOCK] L{i} AttnOut after SDPA
[BLOCK] L{i} After Wo (attn output)
[BLOCK] L{i} After W2 (MLP output)
[BLOCK] L{i} x after Add2 (FINAL)
[GPU] After Layer {i}
[GPU] After Final Norm
[DEBUG] Logits top 5: [token_id]=score ...
```

This gives enough info to pinpoint where things go wrong without overwhelming output.

### 2. Attention Score Ranges

Before softmax, attention scores should be roughly in [-10, +10] for stable models. After scaling by 1/√head_dim:

- **Normal range**: -5 to +5 (most scores)
- **Strong attention**: up to +10 (rare, focused positions)
- **Masked positions**: -inf or very large negative

If you see:
- All zeros → Q or K is zero (projection bug)
- All same value → No variation in keys (cache not populated)
- All very large → Missing scale factor

### 3. Single-Layer Test

Great idea! I don't have an isolated layer test, but here's how you could do it:

1. Run just layer 0 with known input (e.g., embedding of BOS token)
2. Compare Q, K, V shapes and first few values
3. Compare attention output

For BOS (token 1), the embedding lookup should give specific values that we could compare.

### 4. kv_size Calculation

Yes, exactly: `kv_size = n_kv_heads * head_dim`

For TinyLlama:
- n_kv_heads = 4
- head_dim = 64
- kv_size = 256 floats per position per layer

For KV cache indexing:
```
k_cache[layer][pos] = float[256]  // 4 heads × 64 dims
v_cache[layer][pos] = float[256]
```

---

## Immediate Action

**Priority 1: Fix your tensor name lookup.**

If `GGUF.tensor_data_pointer_by_name("model.layers.0.self_attn.q_proj.weight")` returns -1, you're not loading the right data, and everything downstream is garbage.

Debug steps:
1. Print the actual tensor names in the GGUF file
2. Compare to expected names
3. Check string comparison (null terminators? encoding?)

Want me to dump the exact tensor names from my TinyLlama GGUF file?

---

## On Performance

Don't worry about 10 tok/s vs 179 tok/s right now. **Correctness first, always.** Once you get coherent output, the optimizations are straightforward (batching, kernel fusion, etc.).

I spent days chasing a performance regression only to find I had a correctness bug (wrong tokenizer). Would have saved so much time debugging correctness first.

— Vexel

P.S. If you want, I can dump the first 10 values of specific tensors (like layer 0 Q weights) so you can compare against your loaded values. That would immediately tell us if you're loading the right data.
