# Track Spec: CLI Tool

## Overview
Transform the current `inference/cmd/vexel` executable into a polished, subcommand-based CLI application. This tool will serve as the primary user interface for running models, starting the API server, and benchmarking.

## Goals
1.  **Subcommands:** Implement `run`, `serve`, `benchmark` subcommands.
2.  **Configuration:** Support loading configuration (e.g., model path, thread count) from a file or environment variables.
3.  **User Experience:** Improve CLI output (loading progress, chat interface).
4.  **Consistency:** Standardize flags and arguments across subcommands.

## Technical Details
-   **Library:** Use standard `flag` package or a lightweight alternative (`spf13/cobra` if complexity warrants, but `flag` subcommands prefered for simplicity).
-   **Structure:** `inference/cmd/vexel/main.go` dispatching to `internal/cli/run.go`, `internal/cli/serve.go`, etc.
-   **Config:** Simple YAML/JSON/TOML config loading.
