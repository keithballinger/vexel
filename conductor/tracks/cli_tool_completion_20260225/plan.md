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

## Phase 2: Generation Subcommands [checkpoint: e80c244]
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

## Phase 3: Utility Subcommands [checkpoint: pending]
- [x] Task: Integrate `bench` subcommand
    - Wrapped existing `cmd/bench/main.go` logic as a subcommand.
    - Uses `runtime.Llama3_8B()` config and `MemoryPlan` for estimation.
    - Reports memory usage, duration, total tokens, and throughput.
    - Flags: `--batch`, `--seq-len`, `--num-seqs`.
- [x] Task: Integrate `tokenize` subcommand
    - Already implemented in Phase 1 (`runTokenize` in commands.go).
    - Loads tokenizer from `--tokenizer` path or infers from `--model`.
    - Reads from `--input` flag or stdin, outputs JSON-encoded token IDs.
    - Flags: `--input`, `--tokenizer`.
- [x] Task: Update Makefile and README
    - `make build` now produces `./vexel` binary with Metal tags.
    - Added `build-all`, `test-metal`, `vet-metal` targets.
    - README updated with full CLI reference (all 5 subcommands).
    - Architecture section updated with cmd/vexel entry.
