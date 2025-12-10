# **TECHNICAL PLAN: Surprisal-Guided Sparse Attention With Semantic Segmentation**

## **Objective**

Implement and evaluate a family of sparse-attention mechanisms that use *information-theoretic surprisal*, *recency*, and *semantic segmentation* to reduce compute and memory cost while preserving model quality. All methods should integrate into the existing Go-based attention engine by providing a mask or reduced K/V index set.

This plan defines:

* Core scoring mechanisms
* Segmenting strategies
* Block/segment selection algorithms
* Token-selection algorithms
* Recency modulation
* Fallback and stability rules
* Concrete variants to implement
* Experiment design

---

# **1. Pipeline Overview**

For any model inference step (prefill or decode), the sparse attention pipeline consists of:

1. **Surprisal Computation**
2. **Segmentation** (semantic segments → physical blocks)
3. **Scoring** (tokens, then segments)
4. **Selection** (segments and/or tokens)
5. **Mask Construction**
6. **Execution** (use existing attention kernel with mask applied)

All variants differ only in steps 3–5.

---

# **2. Surprisal Computation**

## **2.1 Token-Level Surprisal**

For each token (x_t):

[
s_t = -\log p(x_t \mid x_{<t})
]

This can be obtained via:

* A *cheap teacher model* run once on the prompt, OR
* A *first pass* with the main model’s logits, OR
* A dedicated *surprisal head* (optional future).

Store `s_t` as a float32 slice indexed by token index.

## **2.2 Recency Weighting (Optional)**

Define distance from the current query position (i):

[
d_t = i - t
]

Define recency weight:

[
r_t = \exp(-d_t / \tau)
]

Token surprisal (recency-modulated):

[
\tilde{s}_t = s_t + \lambda \cdot \log(r_t)
]

If recency is disabled, `λ = 0`.

---

# **3. Semantic Segmentation**

## **3.1 Initial Segmentation by Human Layout Heuristics**

Parse the prompt into segments using:

1. **Paragraphs**: split on `\n\n`
2. **Lines**: inside each paragraph, split on `\n`

This yields segments `seg_0, seg_1, ..., seg_M`.

## **3.2 Segment Normalization**

Segments may be too large or small. Apply rules:

* If `len(segment) > MAX_SEG_LEN` (e.g., 256 tokens), break it into equal-sized subsegments.
* If average segment length across doc > `MEGA_SEG_THRESH` (e.g., 512), fallback to uniform block segmentation.
* If segment length < `MIN_SEG_LEN` (e.g., < 4 tokens), merge it with adjacent segments.

Result: a set of segments with token spans.

## **3.3 Mapping Segments to Kernel Blocks**

Your kernel expects fixed blocks (e.g., 64 tokens).
For each segment:

1. Break it into fixed-size blocks.
2. Each block *inherits* the segment’s score.
3. Keep a map:

   * `segment_id[t]` → integer
   * `block_id[t]` → integer

Physical blocks are for the GPU; semantic segments are for scoring.

---

# **4. Scoring**

## **4.1 Token Scores**

Use either `s_t` (pure surprisal) or `\tilde{s}_t` (recency-modulated).

## **4.2 Segment Score**

For segment (b):

Let `T(b)` = tokens in that segment.

Candidate scoring functions:

* **Max:**
  [
  S_b = \max_{t \in T(b)} \tilde{s}_t
  ]
  Best at catching “one important token”.

* **Mean:**
  [
  S_b = \frac{1}{|T(b)|} \sum \tilde{s}_t
  ]

* **Sum:**
  [
  S_b = \sum_{t \in T(b)} \tilde{s}_t
  ]
  Equivalent to mean for fixed-size segments; only differs if segments vary.

Recommended: **max** for v1.

## **4.3 Optional Segment-Level Recency**

Segment center:

[
c_b = \text{mean token index of segment } b
]

Recency term:

[
r_b = \exp(-(i - c_b) / \tau_b)
]

Combined:

[
S_b' = S_b + \lambda_b \cdot \log(r_b)
]

---

# **5. Selection Algorithms**

Multiple variants to implement. They all use segment scores (S_b) or (S_b').

Each variant must ensure the maximum number of selected tokens is ≤ a **global token budget** (e.g. 10–30% of full attention).

## **Variant A (Baseline): Deterministic Top-K Segments**

* Choose (K = \lfloor \rho \cdot M \rfloor) where ρ is the keep-fraction (e.g., 0.1–0.3).
* Sort segments by score and select top-K.
* All blocks belonging to those segments become “global blocks”.

**Attention pattern:**
For query token (i):

* Attend to local window `[i-W, i]`.
* Attend to all tokens in the selected global blocks.

This is the main baseline to compare_everything_against.

---

## **Variant B: Variable Token Count Per Segment**

Instead of “segment is kept or not”, assign each segment a *token budget*:

[
k_b = k_{\min} + \left\lfloor \alpha \cdot \frac{S_b - S_{\min}}{S_{\max} - S_{\min}} \cdot (k_{\max} - k_{\min}) \right\rfloor
]

Procedure:

1. Select all segments (no dropping).
2. Inside each segment, keep top-(k_b) tokens by `\tilde{s}_t`.
3. Tokens not selected are masked out.

This allows *partial contribution* from all segments but favors high-surprisal ones.

---

## **Variant C: Block Sampling (Segment-Level Importance Sampling)**

Define probability:

[
P(b) = \frac{\exp(\alpha S_b)}{\sum_j \exp(\alpha S_j)}
]

Sample `K_keep` segments *without replacement*.

Optionally anneal α:

* α → ∞ → deterministic top-K
* α small → almost uniform

Then behave like Variant A.

---

## **Variant D: Token Sampling Inside Segments**

For each segment:

1. Define token probability:

[
P(t \mid b) \propto \exp(\beta \tilde{s}_t)
]

2. Sample (k_b) tokens from that distribution.

3. Keep local window always.

This creates a Monte Carlo approximation of full attention. Useful when training, riskier for inference-only.

---

# **6. Mask Construction**

Given selected segments or tokens:

1. Build boolean mask `attend[i][j]` or (preferably)
2. Build K/V index lists for each query position.

For efficiency, use:

* **Local window** always included
* **Global tokens** = union of selected tokens
* Collapse them into **sorted contiguous ranges** per query index
* Optionally merge tiny ranges

Mask must conform to the kernel’s expected format.

---

# **7. Fallback Rules**

To avoid degenerate behavior:

1. If number of segments < `MIN_SEGS`: revert to uniform block segmentation.
2. If the distribution of scores is very flat (σ < threshold):

   * Skip sparsification for that layer.
3. If the prompt contains no newlines:

   * Use fixed-size blocks.
4. If a segment exceeds `MAX_SEG_LEN`, pre-split it.
5. Always allow **local window** to avoid catastrophic forgetting.

---

# **8. Implementation Order**

### **Phase 1 (Core)**

1. Token surprisal (teacher model or first pass).
2. Semantic segmentation.
3. Segment-level scoring (max).
4. Deterministic top-K selection (Variant A).
5. Local window + global segment mask.
6. Integrate with attention kernel.

### **Phase 2 (Variants)**

1. Variable token count per segment (Variant B).
2. Token-level sampling (Variant D).
3. Block-level sampling (Variant C).
4. Recency modulation (token or segment).
5. Adaptive fallback logic.

### **Phase 3 (Optimizations)**

1. Precompute maps from segments → blocks → tokens.
2. Pre-allocate mask ranges.
3. Exploit block contiguity for GPU-efficient sparsity.

---

# **9. Experiment Plan**

## **9.1 Models**

Use one or more of:

* Your in-house model (target)
* A small open model (e.g., TinyLlama) for rapid iteration

## **9.2 Datasets**

* Long-range evaluation: LongBench, LRA, NarrativeQA, BookSum
* Real prompts you use daily (engineering docs, specs, etc.)

## **9.3 Metrics**

* Perplexity (prefill)
* End-to-end throughput (tokens/sec)
* Peak memory
* FLOP reduction (estimate via mask density)
* Accuracy on long-context tasks
* Degradation vs dense attention (<2–5% allowed)

## **9.4 Ablations**

For each experiment:

* With vs without semantic segmentation
* With vs without recency
* Segment-level vs token-level selection
* Max vs mean scoring
* Different ρ (fraction of global segments kept)
* Compare:

  * Full dense
  * FlashAttention (dense)
  * Block-sparse naive
  * Surprisal-guided variants A/B/C/D

## **9.5 Success Criteria**

A variant is “good” if:

* Retains ≥ 95–98% of model quality
* Achieves ≥ 1.3–2.0× throughput improvement over dense
* Achieves ≥ 2–5× reduction in effective attention FLOPs
* Stable across different prompt styles

---

# **10. Future Extensions**

* Use *two-tier segmentation* (paragraph → line → block).
* Train a small classifier to predict “segment importance” online.
* Add *attention dropout* to stabilize randomization variants.
* Learn the λ, τ parameters for recency using reinforcement signals.
* Extend to decoder-only self-attention and cross-attention separately.

Here’s the add-on spec you can literally paste at the end of the previous technical plan. It assumes everything in that doc exists, and layers “influence-guided refinement” on top.

---

# **ADDENDUM: Influence-Guided Sparse Attention Refinement**

## **A. Objective**

Augment surprisal/recency-based sparsification with a **behavioral signal**:
how much each token/segment/block actually influences recent predictions.

We introduce a **running influence score** derived from attention weights (cheap proxy) and fold it into the existing scoring pipeline (surprisal + recency + semantics).

This defines a family of **influence-guided sparse attention** variants.

---

## **B. Core Idea**

For each generation step (i):

* The last-layer attention distribution from query at position (i) over context positions (j) is already available: `a[i, j]`.
* Interpret `a[i, j]` as a **proxy for influence** of token (j) on token (i).
* Aggregate per block or segment to obtain a step-wise influence signal.
* Maintain an **exponentially smoothed importance estimate** over time.
* Use this importance to adjust which segments/blocks/tokens get kept in the sparse pattern.

We do **not** compute expensive occlusion-based influence; we piggyback on existing attention activations.

---

## **C. Data Structures**

At minimum:

* For each **segment** (k):

  * `InfluenceSeg[k]` (float32): running estimate (\hat{I}_k)
* Optionally, for each **block** (B):

  * `InfluenceBlock[B]` (float32): running estimate (\hat{I}_B)
* A smoothing factor:

  * `betaInfluence ∈ (0,1)` (e.g., 0.1)

Segments are the semantic units defined in the base plan (paragraph/line-based).

---

## **D. Step-Wise Influence Update**

### **D.1 Segment-Level Influence**

For each generated token at position (i):

1. Extract last-layer attention weights: `a[i, j]` for all context positions `j ≤ i`.

2. For each segment (k):

   Let `SegTokens(k)` be the set of token indices in that segment.
   Compute step influence:

   [
   I_k^{(\text{step})} = \sum_{j \in \text{SegTokens}(k)} a[i, j]
   ]

   (Alternative: `max` instead of `sum` if desired.)

3. Update running influence estimate:

   [
   \hat{I}*k \leftarrow (1 - \beta*{\text{infl}})\hat{I}*k + \beta*{\text{infl}} I_k^{(\text{step})}
   ]

   Implement as:

   ```go
   InfluenceSeg[k] = (1 - betaInfluence) * InfluenceSeg[k] + betaInfluence * stepInfluence
   ```

This yields a smooth measure of “how much this segment has mattered to recent predictions.”

### **D.2 Block-Level Influence (Optional)**

If you also want influence at the block level:

1. For each block (B):

   Let `BlockTokens(B)` be its token indices.

   [
   I_B^{(\text{step})} = \sum_{j \in \text{BlockTokens}(B)} a[i, j]
   ]

2. Update:

   [
   \hat{I}*B \leftarrow (1 - \beta*{\text{infl}})\hat{I}*B + \beta*{\text{infl}} I_B^{(\text{step})}
   ]

If segments map cleanly to blocks, you can skip per-block influence and just use segment influence.

---

## **E. Integrating Influence Into Scores**

The base plan defines a segment score:

* (S_k^{\text{surprisal}}): surprisal-derived (max/mean over tokens)
* (R_k^{\text{recency}}): optional recency term

We now define a combined **final score**:

[
S_k^{\text{final}} = \alpha \cdot S_k^{\text{surprisal}} + \beta \cdot R_k^{\text{recency}} + \gamma \cdot \hat{I}_k^{\text{influence}}
]

Where:

* (\alpha, \beta, \gamma) are hyperparameters controlling the relative weight:

  * Example defaults:

    * (\alpha = 1.0)
    * (\beta = 0.1)
    * (\gamma = 0.5)

In code:

```go
scoreSurprisal := SegSurprisal[k]      // from previous plan
scoreRecency   := SegRecency[k]        // may be 0 if disabled
scoreInfluence := InfluenceSeg[k]      // running average

SegScoreFinal[k] = alpha*scoreSurprisal +
                   beta*scoreRecency   +
                   gamma*scoreInfluence
```

These `SegScoreFinal[k]` values then replace or augment the scores used in the selection variants (A/B/C/D).

---

## **F. Influence-Guided Variants**

### **F.1 Influence-Guided Top-K Segments**

Modify **Variant A** (Deterministic Top-K Segments):

* Instead of sorting by `S_k^{surprisal}` or `S_k'` (surprisal + recency), sort by `S_k^{final}` (surprisal + recency + influence).
* Select top-K segments as global segments.
* Everything else remains identical:

  * Local window is always attended.
  * All tokens in selected segments are treated as global tokens.

### **F.2 Influence-Guided Variable Token Count**

Modify **Variant B** (Variable Token Count Per Segment):

* Use `S_k^{final}` to determine `k_k` (token budget per segment):

  [
  k_k = k_{\min} + \left\lfloor \alpha_k \cdot \frac{S_k^{\text{final}} - S_{\min}^{\text{final}}}{S_{\max}^{\text{final}} - S_{\min}^{\text{final}}} \cdot (k_{\max} - k_{\min}) \right\rfloor
  ]

* For each segment, keep top-(k_k) tokens by **token-level surprisal** (or optionally token-level influence if tracked), mask the rest.

Segments that are both surprising and influential get larger token budgets; those that are merely surprising-but-never-used get demoted over time.

### **F.3 Influence-Guided Sampling**

Modify **Variant C/D** (Sampling):

* For segment sampling:

  [
  P(k) \propto \exp(\lambda S_k^{\text{final}})
  ]

* For token sampling within segments, you can optionally incorporate token-level influence scores if you maintain them; otherwise, keep using `\tilde{s}_t`.

---

## **G. Practical Details & Stability**

### **G.1 Initialization**

* Initialize all `InfluenceSeg[k] = 0`.
* Influence only becomes meaningful after a few generation steps; until then, rely mostly on surprisal/recency (i.e., use small (\gamma) at first, or gate influence until a minimum number of steps have passed).

### **G.2 Layer Choice**

To avoid extra complexity:

* Start with influence derived **only from the last attention layer**.
* Optionally, later average influence across a subset of upper layers.

### **G.3 Frequency of Updates**

Influence is updated **once per generated token** by default.
If this is too expensive to aggregate every step in practice, you can:

* Update every M steps (e.g., M = 2–4) by accumulating attention maps and then applying a batched update.

### **G.4 Numerical Considerations**

* `betaInfluence` should be small enough to avoid jitter (e.g., 0.05–0.2).
* Consider clipping `InfluenceSeg[k]` to a finite range to avoid runaway values.
* Optionally standardize `InfluenceSeg` per context when computing `S_k^{final}` (z-score normalization) to keep scales stable:

  [
  \hat{I}_k^{\text{norm}} = \frac{\hat{I}_k - \mu_I}{\sigma_I + \epsilon}
  ]

and plug `\hat{I}_k^{norm}` into the final score formula.

---

## **H. Experiments for Influence-Guided Variants**

Extend the existing experiment plan with:

### **H.1 Ablation Axes**

1. **Influence weight (\gamma)**:

   * 0.0 (no influence, baseline)
   * 0.25, 0.5, 1.0
2. **Smoothing (\beta_{\text{infl}})**:

   * 0.05, 0.1, 0.2
3. **Use influence only vs. surprisal-only vs. combined**:

   * `S_final = alpha*S_surprisal`
   * `S_final = gamma*I_hat`
   * `S_final = alpha*S_surprisal + gamma*I_hat`

### **H.2 Metrics**

Same as before:

* Perplexity on long-context tasks.
* Throughput + memory vs dense and surprisal-only variants.
* Quality robustness:

  * especially important: does influence help **stabilize** which segments are kept as generation proceeds?

### **H.3 Hypotheses**

* Influence-augmented scoring will:

  * Reduce waste on segments that looked important (high surprisal) but never get used.
  * Preserve important long-range segments better than pure recency-based schemes.
* Best-performing regime likely has:

  * `alpha ≈ 1.0` (surprisal dominant)
  * `beta ≈ 0.1` (mild recency)
  * `gamma ≈ 0.25–0.5` (influence as a secondary but meaningful signal)

---

## **I. Summary**

Influence-guided refinement adds a **third axis** to sparsity decisions:

1. **Surprisal**: tokens/segments that are information-dense.
2. **Recency**: tokens/segments that are structurally more likely to matter soon.
3. **Influence**: tokens/segments that *empirically* drove recent predictions.

The implementation is:

* Simple (attention aggregation + EMA),
* Inexpensive (no extra forward passes), and
* Easy to integrate by swapping existing segment scores for `S_k^{final}` in the selection logic.

This should be implemented as **Phase 2+** on top of the existing surprisal/recency/semantic-segmentation pipeline.

