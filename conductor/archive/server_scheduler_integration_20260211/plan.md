# Track Plan: Server & Scheduler Integration

## Phase 1: Implementation
- [x] Task: Create Scheduler Integration
    - Update `inference/serve/server.go` to initialize and subscribe to `inference/scheduler`.
    - Handle incoming requests (gRPC/HTTP) and submit to scheduler.
- [x] Task: Response Streaming
    - Stream token outputs from the scheduler back to the client.

## Phase 2: Verification
- [x] Task: End-to-End Tests
    - Submit requests via `curl` or gRPC client.
    - Verify correct responses and handling of multiple concurrent requests.
