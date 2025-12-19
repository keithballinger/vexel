package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"vexel/inference/pkg/tokenizer"
)

func main() {
	tokenizerPath := flag.String("tokenizer", "", "Path to tokenizer.json")
	text := flag.String("text", "", "Text to tokenize (default: read stdin)")
	flag.Parse()

	if *tokenizerPath == "" {
		fmt.Fprintln(os.Stderr, "missing required -tokenizer flag")
		os.Exit(2)
	}

	tok, err := tokenizer.Load(*tokenizerPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load tokenizer: %v\n", err)
		os.Exit(1)
	}

	input := *text
	if input == "" {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to read stdin: %v\n", err)
			os.Exit(1)
		}
		input = string(b)
	}

	ids, err := tok.Encode(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode failed: %v\n", err)
		os.Exit(1)
	}

	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(ids); err != nil {
		fmt.Fprintf(os.Stderr, "failed to write json: %v\n", err)
		os.Exit(1)
	}
}
