# Track Plan: CLI Tool

## Phase 1: Structure & Refactor
- [x] Task: Project Layout
    - Create `inference/cmd/vexel/internal/cli/` structure.
    - Move current logic (run mode) to `run` subcommand handler.
- [x] Task: Main Dispatcher
    - Implement `main.go` using `flag.NewFlagSet` for `run`, `serve`, `benchmark` subcommands.

## Phase 2: Implementation
- [x] Task: Run Subcommand
    - Refine existing chat loop logic into `run` command.
- [x] Task: Serve Subcommand
    - Implement `serve` command to start the HTTP/gRPC server (using `inference/serve`).
- [x] Task: Benchmark Subcommand
    - Implement `benchmark` command (using existing `bench_run` logic or similar).

## Phase 3: Polish
- [x] Task: Help & Usage
    - Ensure clear help text for all commands (`vexel help`).
