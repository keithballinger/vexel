# Track Plan: CLI Tool Completion

The README documents `./vexel serve` but no unified CLI binary exists. Individual tools
(`cmd/bench`, `cmd/vexel-debug`, `cmd/tokenize`) are scattered. The REPL lives in
`cmd/vexel/internal/repl.go` but has no main entry point. This track creates a single
`vexel` binary with subcommands.

## Phase 1: Core CLI Framework [checkpoint: a930e71]
- [x] Task: Create `cmd/vexel/main.go`
    - Implement subcommand dispatcher using `flag` (no external deps per product-guidelines).
    - Subcommands: `serve`, `generate`, `chat`, `bench`, `tokenize`.
    - Global flags: `--model`, `--verbose`.
    - Print usage/help when invoked with no subcommand.
    - Files: `main.go`, `cli.go` (parsing), `commands.go` (runners), `cli_test.go` (10 tests).
- [x] Task: Implement `serve` subcommand
    - Wire up Metal backend init, GGUF model loading, scheduler creation, and HTTP server.
    - Flags: `--port` (default 8080), `--max-tokens`, `--max-batch-size`.
    - Match the interface documented in README: `./vexel serve --model MODEL --port 8080`.
    - Graceful shutdown on SIGINT/SIGTERM.
    - Shared `initModel()` helper for model loading (reused by chat/generate).

## Phase 2: Generation Subcommands [checkpoint: pending]
- [x] Task: Implement `generate` subcommand
    - One-shot text generation from a prompt (no HTTP, direct runtime).
    - Flags: `--prompt`, `--max-tokens`, `--temperature`.
    - Print tokens as they are generated (streaming to stdout).
    - Verbose mode prints prefill/decode tok/s metrics.
- [x] Task: Implement `chat` subcommand
    - Wire up existing REPL from `cmd/vexel/internal/repl.go`.
    - Flags: `--system-prompt`, `--no-chat-template`.
    - Uses scheduler for inference via `internal.RunChatLoopWithConfig`.
    - Moved `printUsage` from Metal-tagged main.go to cli.go for testability.

## Phase 3: Utility Subcommands
- [ ] Task: Integrate `bench` subcommand
    - Wrap existing `cmd/bench/main.go` logic as a subcommand.
    - Flags: `--batch`, `--seq-len`, `--num-seqs`.
- [ ] Task: Integrate `tokenize` subcommand
    - Wrap existing `cmd/tokenize/main.go` as a subcommand.
    - Flags: `--input`, `--tokenizer`.
- [ ] Task: Update Makefile and README
    - Update `make build` to produce `./vexel` binary.
    - Update README usage examples to reflect final CLI interface.
