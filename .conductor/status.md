# Project Status

**Date:** 2025-12-07
**Status:** 🟢 On Track

## Current Phase
Phase 4: Scheduler and Serving Layer

## Current Task
Write Failing Tests: For `collectReady()` accurately identifying sequences in Pending/Decoding states.

## Progress
- [x] Write Failing Tests: For `Scheduler.Run` loop.
- [x] Implement Feature: Implement `Scheduler.Run` and `step` loop.

## Next Action
Write failing tests for `collectReady()` in `inference/scheduler/collect_test.go`.

## Blockers
None

## Notes
- Basic scheduler loop is operational (but empty).
- Need to implement internal sequence management logic (`collectReady`, `formBatches`).
