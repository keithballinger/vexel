package main

import (
	"flag"
	"fmt"
	"io"
)

// GlobalFlags holds flags shared across all subcommands.
type GlobalFlags struct {
	Model      string
	DraftModel string // Optional: path to draft model for speculative decoding
	Verbose    bool
}

// validSubcommands lists the recognized subcommand names.
var validSubcommands = map[string]bool{
	"serve":    true,
	"generate": true,
	"chat":     true,
	"bench":    true,
	"tokenize": true,
}

// parseArgs extracts global flags and the subcommand from os.Args.
// It returns the subcommand name, parsed global flags, and any remaining args.
//
// The expected format is: vexel [--model M] [--verbose] <subcommand> [subcommand flags...]
func parseArgs(args []string) (string, GlobalFlags, error) {
	if len(args) < 2 {
		return "", GlobalFlags{}, fmt.Errorf("no subcommand specified")
	}

	var globals GlobalFlags

	// Scan for global flags before the subcommand.
	// We parse manually to allow flags before the subcommand name.
	i := 1
	for i < len(args) {
		switch args[i] {
		case "--model":
			if i+1 >= len(args) {
				return "", GlobalFlags{}, fmt.Errorf("--model requires a value")
			}
			globals.Model = args[i+1]
			i += 2
		case "--draft-model":
			if i+1 >= len(args) {
				return "", GlobalFlags{}, fmt.Errorf("--draft-model requires a value")
			}
			globals.DraftModel = args[i+1]
			i += 2
		case "--verbose":
			globals.Verbose = true
			i++
		default:
			// Not a global flag — must be the subcommand
			goto foundCmd
		}
	}

foundCmd:
	if i >= len(args) {
		return "", GlobalFlags{}, fmt.Errorf("no subcommand specified")
	}

	cmd := args[i]
	if !validSubcommands[cmd] {
		return "", GlobalFlags{}, fmt.Errorf("unknown subcommand: %q", cmd)
	}

	return cmd, globals, nil
}

// subcommandArgs returns the arguments after the subcommand name.
func subcommandArgs(args []string) []string {
	for i := 1; i < len(args); i++ {
		if validSubcommands[args[i]] {
			return args[i+1:]
		}
	}
	return nil
}

// ServeFlags holds flags for the serve subcommand.
type ServeFlags struct {
	Port         int
	MaxTokens    int
	MaxBatchSize int
}

// parseServeFlags parses serve-specific flags from the remaining args.
func parseServeFlags(args []string) (ServeFlags, error) {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	sf := ServeFlags{}
	fs.IntVar(&sf.Port, "port", 8080, "HTTP port to listen on")
	fs.IntVar(&sf.MaxTokens, "max-tokens", 256, "Max tokens to generate per request")
	fs.IntVar(&sf.MaxBatchSize, "max-batch-size", 1, "Max batch size for scheduler")
	if err := fs.Parse(args); err != nil {
		return ServeFlags{}, err
	}
	return sf, nil
}

// GenerateFlags holds flags for the generate subcommand.
type GenerateFlags struct {
	Prompt      string
	MaxTokens   int
	Temperature float64
}

// parseGenerateFlags parses generate-specific flags.
func parseGenerateFlags(args []string) (GenerateFlags, error) {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	gf := GenerateFlags{}
	fs.StringVar(&gf.Prompt, "prompt", "", "Input prompt (required)")
	fs.IntVar(&gf.MaxTokens, "max-tokens", 64, "Maximum tokens to generate")
	fs.Float64Var(&gf.Temperature, "temperature", 0.0, "Sampling temperature (0=greedy)")
	if err := fs.Parse(args); err != nil {
		return GenerateFlags{}, err
	}
	return gf, nil
}

// ChatFlags holds flags for the chat subcommand.
type ChatFlags struct {
	SystemPrompt   string
	NoChatTemplate bool
}

// parseChatFlags parses chat-specific flags.
func parseChatFlags(args []string) (ChatFlags, error) {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	cf := ChatFlags{}
	fs.StringVar(&cf.SystemPrompt, "system-prompt", "You are a helpful assistant.", "System prompt for chat mode")
	fs.BoolVar(&cf.NoChatTemplate, "no-chat-template", false, "Disable chat template formatting")
	if err := fs.Parse(args); err != nil {
		return ChatFlags{}, err
	}
	return cf, nil
}

// BenchFlags holds flags for the bench subcommand.
type BenchFlags struct {
	BatchSize int
	SeqLen    int
	NumSeqs   int
}

// parseBenchFlags parses bench-specific flags.
func parseBenchFlags(args []string) (BenchFlags, error) {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	bf := BenchFlags{}
	fs.IntVar(&bf.BatchSize, "batch", 128, "Max batch size")
	fs.IntVar(&bf.SeqLen, "seq-len", 128, "Sequence length")
	fs.IntVar(&bf.NumSeqs, "num-seqs", 256, "Total sequences to process")
	if err := fs.Parse(args); err != nil {
		return BenchFlags{}, err
	}
	return bf, nil
}

// TokenizeFlags holds flags for the tokenize subcommand.
type TokenizeFlags struct {
	Input     string
	Tokenizer string
}

// parseTokenizeFlags parses tokenize-specific flags.
func parseTokenizeFlags(args []string) (TokenizeFlags, error) {
	fs := flag.NewFlagSet("tokenize", flag.ContinueOnError)
	tf := TokenizeFlags{}
	fs.StringVar(&tf.Input, "input", "", "Text to tokenize (reads stdin if empty)")
	fs.StringVar(&tf.Tokenizer, "tokenizer", "", "Path to tokenizer.json")
	if err := fs.Parse(args); err != nil {
		return TokenizeFlags{}, err
	}
	return tf, nil
}

// printUsage writes the top-level usage message to w.
func printUsage(w io.Writer) {
	fmt.Fprintln(w, `Usage: vexel [global flags] <subcommand> [flags]

Subcommands:
  serve      Start the HTTP inference server
  generate   One-shot text generation from a prompt
  chat       Interactive chat REPL
  bench      Run scheduling benchmarks
  tokenize   Tokenize text to token IDs

Global flags:
  --model        Path to GGUF model file (required for serve/generate/chat)
  --draft-model  Path to draft model for speculative decoding (optional)
  --verbose      Enable verbose logging

Examples:
  vexel --model model.gguf serve --port 8080
  vexel --model model.gguf generate --prompt "Hello!"
  vexel --model model.gguf --draft-model draft.gguf generate --prompt "Hello!"
  vexel --model model.gguf chat`)
}
