//go:build metal && darwin && cgo

// Command vexel is the unified CLI for the Vexel inference engine.
//
// Usage:
//
//	vexel [global flags] <subcommand> [subcommand flags]
//
// Subcommands:
//
//	serve      Start the HTTP inference server
//	generate   One-shot text generation from a prompt
//	chat       Interactive chat REPL
//	bench      Run scheduling benchmarks
//	tokenize   Tokenize text to token IDs
//
// Global flags:
//
//	--model    Path to GGUF model file
//	--verbose  Enable verbose logging
//
// Examples:
//
//	vexel --model models/llama-2-7b.Q4_0.gguf serve --port 8080
//	vexel --model models/llama-2-7b.Q4_0.gguf generate --prompt "Hello!"
//	vexel --model models/llama-2-7b.Q4_0.gguf chat
package main

import (
	"fmt"
	"os"
)

func main() {
	cmd, globals, err := parseArgs(os.Args)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		printUsage(os.Stderr)
		os.Exit(2)
	}

	switch cmd {
	case "serve":
		if err := runServe(globals, os.Args); err != nil {
			fmt.Fprintf(os.Stderr, "serve: %v\n", err)
			os.Exit(1)
		}
	case "generate":
		if err := runGenerate(globals, os.Args); err != nil {
			fmt.Fprintf(os.Stderr, "generate: %v\n", err)
			os.Exit(1)
		}
	case "chat":
		if err := runChat(globals, os.Args); err != nil {
			fmt.Fprintf(os.Stderr, "chat: %v\n", err)
			os.Exit(1)
		}
	case "bench":
		if err := runBench(globals, os.Args); err != nil {
			fmt.Fprintf(os.Stderr, "bench: %v\n", err)
			os.Exit(1)
		}
	case "tokenize":
		if err := runTokenize(globals, os.Args); err != nil {
			fmt.Fprintf(os.Stderr, "tokenize: %v\n", err)
			os.Exit(1)
		}
	}
}

