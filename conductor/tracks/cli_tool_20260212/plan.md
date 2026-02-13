# Track Plan: CLI Tool

## Phase 1: Structure & Refactor
- [ ] Task: Project Layout
    - Create `inference/cmd/vexel/internal/cli/` structure.
    - Move current logic (run mode) to `run` subcommand handler.
- [ ] Task: Main Dispatcher
    - Implement `main.go` using `flag.NewFlagSet` for `run`, `serve`, `benchmark` subcommands.

## Phase 2: Implementation
- [ ] Task: Run Subcommand
    - Refine existing chat loop logic into `run` command.
- [ ] Task: Serve Subcommand
    - Implement `serve` command to start the HTTP/gRPC server (using `inference/serve`).
- [ ] Task: Benchmark Subcommand
    - Implement `benchmark` command (using existing `bench_run` logic or similar).

## Phase 3: Polish
- [ ] Task: Configuration Loading
    - Implement simple config file support (optional/future phase, but basic flag parsing here).
- [ ] Task: Help & Usage
    - Ensure clear help text for all commands (`vexel help`).
