# Project Status

**Date:** 2025-12-08
**Status:** 🟢 On Track

## Current Phase
Phase 5: Integration, Testing, and Documentation (Real Inference)

## Current Task
Task: Real Inference Support

## Progress
- [x] Implement Feature: Implement `Tokenizer`.

## Next Action
Write Failing Tests: For `mmap` based weight loading.

## Blockers
None

## Notes
- Basic tokenizer that loads vocab from `tokenizer.json` is implemented.
- BPE logic is extremely naive (direct map lookup), which works for whole words but will fail for subwords. For "Real Inference", this is a known limitation but sufficient to start the pipe.
