# Track Plan: Documentation & Examples

## Phase 1: Implementation
- [x] Task: Create `examples/`
    - Add basic Go examples for usage (CLI wrapper, API client if implemented).
    - Added `examples/server/main.go`: Embedding Vexel HTTP server with scheduler.
    - Added `examples/generate/main.go`: Direct inference loop with greedy sampling.
    - Existing `examples/client/main.go`: HTTP client (Generate + Stream).
- [x] Task: Update README
    - Enhance `README.md` with new features (Streaming, Scheduling) and usage instructions.
    - Added: Direct Inference API section, Embedding the Server section, Examples table, Architecture overview, Performance benchmarks.
- [x] Task: API Docs
    - Ensure clear `godoc` comments on public packages (`inference/client`, `inference/scheduler`).
    - Added `client/doc.go`: Package-level godoc with Quick Start, Generation Options, and Thread Safety sections.
    - Added `inference/scheduler/doc.go`: Package-level godoc with Architecture, Usage, Metrics, and Sequence Lifecycle sections.
