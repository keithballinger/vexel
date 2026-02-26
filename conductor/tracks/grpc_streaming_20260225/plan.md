# Track Plan: gRPC Streaming

Proto definitions exist (`serve/pb/inference.proto`) with `Generate` and `StreamGenerate` RPCs.
Generated Go stubs are in place. `serve/grpc.go` has a `GRPCServer` struct implementing
the interface but both methods return mock data. A basic test (`grpc_test.go`) verifies the
server starts. This track completes the implementation and adds production features.

## Phase 1: Implementation
- [x] Task: Implement `Generate` RPC
    - Wire to scheduler: create Sequence, AddSequence, collect tokens from TokenChan, return response.
    - Add timeout handling via context deadline.
    - Return proper gRPC status codes (InvalidArgument, DeadlineExceeded, Internal).
- [x] Task: Implement `StreamGenerate` RPC
    - Wire to scheduler: create Sequence, AddSequence, stream tokens via `server.Send()`.
    - Each token is sent as a `GenerateResponse{text: token}`.
    - Handle client cancellation (context.Done) and clean up sequence.
- [x] Task: Expand proto schema
    - Add sampling parameters to `GenerateRequest`: temperature, top_k, top_p, max_tokens.
    - Add metadata to `GenerateResponse`: token_count, finish_reason (eos, max_tokens, cancelled).
    - Add `ModelInfo` RPC returning model name, architecture, quantization, max context length.
    - Regenerate Go stubs.

Checkpoint: pending

## Phase 2: Production Features
- [ ] Task: TLS support
    - Add TLS certificate loading to GRPCServer.
    - Support both mutual TLS and server-only TLS.
    - Add `--grpc-tls-cert` and `--grpc-tls-key` flags to CLI serve subcommand.
- [ ] Task: Request metadata and interceptors
    - Add unary and stream interceptors for logging (request ID, duration, token count).
    - Propagate request ID through scheduler for tracing.
    - Add keepalive configuration for long-running streams.
- [ ] Task: Unified server startup
    - Serve both HTTP (SSE) and gRPC from the CLI `serve` subcommand.
    - HTTP on `--port` (default 8080), gRPC on `--grpc-port` (default 9090).
    - Shared scheduler instance between both servers.

## Phase 3: Testing
- [ ] Task: Integration tests
    - Test `Generate` RPC end-to-end with a mock scheduler.
    - Test `StreamGenerate` receives correct token sequence.
    - Test cancellation: cancel context mid-stream, verify sequence cleanup.
    - Test timeout: set short deadline, verify DeadlineExceeded response.
- [ ] Task: Client library
    - Add gRPC client to `client/` package alongside the HTTP client.
    - `GRPCClient` with `Generate()` and `Stream()` matching the HTTP client interface.
    - Auto-detection: try gRPC first, fall back to HTTP.
- [ ] Task: Load testing
    - Benchmark gRPC vs HTTP/SSE latency for single-token responses.
    - Benchmark concurrent streaming throughput (10, 50, 100 concurrent streams).
    - Measure gRPC overhead vs direct scheduler access.
