# Fixing matvec_q4_0_multi_output_f32 Performance on Apple GPUs

> **Historical reference** (early 2025). Documents the threadgroup memory staging optimization for Q4_0 matvec kernels. The optimization described here has been implemented.

## Goal

Bring **Vexel’s Metal kernel performance to parity with (or close to) llama.cpp** by fixing the dominant bandwidth inefficiency: **redundant activation (A / x) loads from device memory**.

Right now, the kernel reloads the same activation block **once per simdgroup** (8× per threadgroup). This document describes how to **stage activations into threadgroup memory once per block** and reuse them across all simdgroups, eliminating that waste.

This is the **highest-ROI fix** before any nibble-unpack or micro-ALU optimizations.

---

## Key Observations (Ground Truth)

- Each threadgroup has **8 simdgroups**
- Each simdgroup computes **1 output row**
- All simdgroups:
  - Iterate over `block` in **lockstep**
  - Use the **same activation vector A**
  - Differ *only* in which weight row they read
- Current behavior:
  - Each simdgroup loads the same 32 FP32 activations from device memory
  - Result: **8× redundant global memory traffic**

This is the core reason Vexel is ~1.6× slower than llama.cpp on the same hardware.

---

## High-Level Fix

**Stage the activation block (`A`) once per threadgroup into threadgroup memory, then reuse it across all simdgroups.**

This turns:
```

A loads per block = 8

```
into:
```

A loads per block = 1

```

---

## Constraints & Assumptions

These are already true in the current kernel:

- `numBlocks` is constant across simdgroups
- All simdgroups start at the same `simd_lane`
- No divergence in block loop
- All simdgroups are synchronized by construction

Therefore:
- Cooperative loading is safe
- A single threadgroup barrier is sufficient

---

## Implementation Plan

### 1. Add Threadgroup Storage for Activations

Each Q4 block corresponds to **32 FP32 activations**, currently loaded as:

- 8 × `float4` loads

Declare threadgroup storage:

```metal
threadgroup float4 A_sh[8];
````

This is only **128 bytes** of threadgroup memory.

---

### 2. Restructure the Block Loop (Minimal Change)

Inside the existing block loop:

```metal
for (int block = simd_lane; block < numBlocks; block += 32) {
    int base_k = block * 32;

    // Stage activations once
    if (simd_group == 0 && simd_lane < 8) {
        A_sh[simd_lane] =
            ((device const float4*)(A + base_k))[simd_lane];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All simdgroups now read activations from A_sh
    float4 a0 = A_sh[0];
    float4 a1 = A_sh[1];
    float4 a2 = A_sh[2];
    float4 a3 = A_sh[3];
    float4 a4 = A_sh[4];
    float4 a5 = A_sh[5];
    float4 a6 = A_sh[6];
    float4 a7 = A_sh[7];

    // Existing weight unpack + accumulation logic goes here
}
```

### Notes

* Only **simdgroup 0** performs the global loads
* Only **lanes 0–7** load `float4`s
* All simdgroups synchronize before use
* This preserves existing row-per-simdgroup structure

---

### 3. Barriers (Important)

* **First barrier** (after loading A_sh): **required**
* **Second barrier** (before next block):

  * Optional if block loop is strictly lockstep
  * Include initially for correctness
  * Remove later if profiling shows it matters

Safer initial version:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
// compute
threadgroup_barrier(mem_flags::mem_threadgroup);
```

---

## What This Fix Accomplishes

### Before

* Per block, per threadgroup:

  * 8 × 32 FP32 loads = **1024 bytes**
* Completely bandwidth-wasteful

### After

* Per block, per threadgroup:

  * 1 × 32 FP32 loads = **128 bytes**
* **8× reduction in activation bandwidth**

This alone can plausibly recover the entire observed 1.6× slowdown.

---

## What NOT to Do Yet (Defer These)

Do **not** start here — these are second-order:

* Nibble-unpack micro-optimizations
* `uint16` vs `uint8` loads
* Pre-scaled activations to remove shifts
* FP16 activation storage
* Changing block packing
* Rewriting the kernel into a different shape

Those only matter **after** redundant activation loads are eliminated.

---

## Validation Checklist

After implementing:

1. **Functional correctness**

   * Outputs match previous kernel bit-for-bit (or within FP error)

2. **Performance measurement**

   * Measure tokens/sec on the same model & prompt
   * Expect a **large jump**, not a subtle one

3. **Profiling sanity**

   * Confirm:

     * Fewer global loads from `A`
     * Increased arithmetic intensity
     * No unexpected sync stalls

If performance does *not* improve materially:

* The bottleneck is likely **framework overhead** (command buffers, encoder transitions, mid-layer sync), not kernel math.

---

## Next Steps (After This Works)

Only after confirming a big win:

1. Consider **multi-row-per-simdgroup** to keep `A` in registers
2. Consider **FP16 activations + FP32 accumulation**
3. Match llama.cpp’s:

   * `uint16` packed weight loads
   * Mask-based nibble extraction
   * Pre-scaled activations
4. Reduce command buffers / encoder transitions

But **do not skip this step** — this is the foundation.

---

## Summary (One Sentence)

> The kernel is slow because it reloads the same activation vector 8×; stage it once into threadgroup memory and reuse it across simdgroups to recover most of the lost performance.

Once this is implemented and benchmarked, come back with:

* new tokens/sec numbers
* any unexpected regressions
* whether barriers showed up as significant

We’ll decide the next move based on real data.
