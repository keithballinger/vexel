# Project Status

**Date:** 2025-12-07
**Status:** 🟢 On Track

## Current Phase
Phase 4: Scheduler and Serving Layer

## Current Task
Write Failing Tests: For `Scheduler` struct initialization.

## Progress
- [x] Write Failing Tests: For `Sequence` struct initialization.
- [x] Implement Feature: Define `Sequence` struct and state machine.

## Next Action
Write failing tests for `Scheduler` struct in `inference/scheduler/scheduler_test.go`.

## Blockers
None

## Notes
- `Sequence` struct and states (`Pending`, `Prefill`, `Decoding`, `Finished`) implemented.
- Scheduler will manage these sequences.
