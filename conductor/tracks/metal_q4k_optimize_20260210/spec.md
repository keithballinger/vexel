# Track Spec: Optimize Q4_K Kernels (NR2)

## Overview
`Q4_K` is the primary quantization format used in Vexel for most weight matrices, including the massive `mlp.fc1` (up/gate projection) in Phi-2. Currently, Vexel uses a generic `matvec_q4k_multi_output_f32` kernel that computes one output per SIMD group. On M4 Pro, this underutilizes the hardware. Implementing an `NR2` (2 outputs per SIMD group) variant will allow better instruction interleaving and shared activation access.

## Goals
1.  **Implement NR2 Kernel:** Create `matvec_q4k_nr2_f32` in Metal.
2.  **Dispatch Support:** Update `backend.go` to select the NR2 kernel when M=1.
3.  **Performance Improvement:** Target >26 tokens/sec (up from 23.3) for Phi-2.

## Technical Details
*   **Target Layers:** `mlp.fc1` (10240 outputs), `self_attn.o_proj` (2560 outputs).
*   **Format:** `Q4_K` super-blocks of 256 elements (144 bytes).
*   **Optimization:** Interleave dot-product computation for two rows to reuse activation values (`A`) loaded into registers or shared memory.

## Acceptance Criteria
*   `TestPhi2MetalParity` passes with NR2 kernel enabled.
*   Benchmark shows measurable reduction in `ms/token`.
