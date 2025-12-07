# Project Status

**Date:** 2025-12-07
**Status:** 🟢 On Track

## Current Phase
Phase 4: Scheduler and Serving Layer

## Current Task
Write Failing Tests: For `formBatches()` creating batches based on priority and configurable policies.

## Progress
- [x] Write Failing Tests: For `collectReady()`.
- [x] Implement Feature: Implement `collectReady()`.

## Next Action
Write failing tests for `formBatches()` in `inference/scheduler/batch_test.go`.

## Blockers
None

## Notes
- `collectReady` logic implemented and tested.
- Scheduler now tracks sequences in a map.