# Track Plan: Go Client Library

## Phase 1: Implementation
- [x] Task: Create Client Package
    - Initialize `vexel/client` package.
    - Implement `Client` struct and configuration (e.g., baseURL, timeout).
- [x] Task: Generate Method
    - Implement `Generate(ctx, prompt, options)` for simple text completion via HTTP POST.
- [x] Task: Stream Method
    - Implement `Stream(ctx, prompt, options)` for streaming token responses via SSE.

## Phase 2: Verification
- [x] Task: Unit Tests
    - Write unit tests for `Client` methods, mocking the server responses.
- [x] Task: Integration Tests
    - Run integration tests against a live (or mocked) Vexel server instance.
- [x] Task: Examples
    - Create a simple `examples/client` directory demonstrating usage.
