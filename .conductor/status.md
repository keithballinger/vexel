# Project Status

**Date:** 2025-12-07
**Status:** 🟢 On Track

## Current Phase
Phase 4: Scheduler and Serving Layer

## Current Task
Write Failing Tests: For `runDecodeStep()` orchestrating the `ModelRuntime.DecodeStep` and handling errors.

## Progress
- [x] Write Failing Tests: For `formBatches()`.
- [x] Implement Feature: Implement `formBatches()`.

## Next Action
Write failing tests for `runDecodeStep()` in `inference/scheduler/exec_test.go`.

## Blockers
None

## Notes
- Scheduler now has `collectReady` and `formBatches` logic.
- Next is linking these to the actual `ModelRuntime`.
