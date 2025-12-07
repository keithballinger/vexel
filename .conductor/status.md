# Project Status

**Date:** 2025-12-07
**Status:** 🟢 On Track

## Current Phase
Phase 4: Scheduler and Serving Layer

## Current Task
Write Failing Tests: For request admission control and sequence registration with the scheduler.

## Progress
- [x] Write Failing Tests: For gRPC `/generate` and `/stream` endpoints.
- [x] Implement Feature: Implement gRPC `/generate` and `/stream` endpoints.

## Next Action
Write failing tests for admission control in `inference/serve/admission_test.go` (verifying that HTTP/gRPC handlers actually call `Scheduler.AddSequence`).

## Blockers
None

## Notes
- All endpoints (HTTP/gRPC) are implemented as mocks.
- Next step is wiring them to the real `Scheduler`.
