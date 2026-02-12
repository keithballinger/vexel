# Track Spec: Speculative Decoding Refinement

## Overview
Improve the speculative decoding scheduler (`inference/scheduler/speculative.go`) to correctly handle the KV cache state during draft generation and verification.

## Goals
1.  **Correctness:** Ensure output matches standard decoding (no hallucination/errors from broken cache).
2.  **Performance:** Achieve speedups by accepting valid draft tokens.
3.  **Efficiency:** Minimize unnecessary cache copies/resets.

## Technical Details
-   **Draft Cache:** Maintain a tentative KV cache for drafts.
-   **Verification:** Commit valid tokens to the main cache.
-   **Rollback:** Revert state efficiently on rejection.
