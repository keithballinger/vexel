# Track Plan: Go Client Library

## Phase 1: Implementation
- [ ] Task: Create Client Package
    - Initialize `vexel/client` package.
    - Implement `Client` struct and configuration (e.g., baseURL, timeout).
- [ ] Task: Generate Method
    - Implement `Generate(ctx, prompt, options)` for simple text completion via HTTP POST.
- [ ] Task: Stream Method
    - Implement `Stream(ctx, prompt, options)` for streaming token responses via SSE.

## Phase 2: Verification
- [ ] Task: Unit Tests
    - Write unit tests for `Client` methods, mocking the server responses.
- [ ] Task: Integration Tests
    - Run integration tests against a live (or mocked) Vexel server instance.
- [ ] Task: Examples
    - Create a simple `examples/client` directory demonstrating usage.
