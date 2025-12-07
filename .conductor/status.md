# Project Status

**Date:** 2025-12-07
**Status:** 🟢 On Track

## Current Phase
Phase 4: Scheduler and Serving Layer

## Current Task
Write Failing Tests: For gRPC `/generate` and `/stream` endpoints.

## Progress
- [x] Write Failing Tests: For `/stream` endpoint.
- [x] Implement Feature: Implement `/stream` endpoint (SSE).

## Next Action
Write failing tests for request admission control and sequence registration with the scheduler. (Skipping gRPC for now to focus on wiring up HTTP to Scheduler first, unless strictly required by plan).

Wait, strict adherence says I should follow plan. The plan lists gRPC next.
However, I might want to prioritize "Request admission control" to actually make the HTTP endpoints functional before adding *another* protocol layer.

Let's follow the plan order but acknowledge that gRPC might be a stub too.

## Blockers
None

## Notes
- HTTP `/generate` and `/stream` are implemented (mocks).