package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestParseSubcommand(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantCmd string
		wantErr bool
	}{
		{
			name:    "serve subcommand",
			args:    []string{"vexel", "serve", "--model", "model.gguf"},
			wantCmd: "serve",
		},
		{
			name:    "generate subcommand",
			args:    []string{"vexel", "generate", "--model", "model.gguf"},
			wantCmd: "generate",
		},
		{
			name:    "chat subcommand",
			args:    []string{"vexel", "chat", "--model", "model.gguf"},
			wantCmd: "chat",
		},
		{
			name:    "bench subcommand",
			args:    []string{"vexel", "bench"},
			wantCmd: "bench",
		},
		{
			name:    "tokenize subcommand",
			args:    []string{"vexel", "tokenize"},
			wantCmd: "tokenize",
		},
		{
			name:    "no subcommand shows help",
			args:    []string{"vexel"},
			wantCmd: "",
			wantErr: true,
		},
		{
			name:    "unknown subcommand",
			args:    []string{"vexel", "foobar"},
			wantCmd: "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd, _, err := parseArgs(tt.args)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got cmd=%q", cmd)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if cmd != tt.wantCmd {
				t.Errorf("got cmd=%q, want %q", cmd, tt.wantCmd)
			}
		})
	}
}

func TestUsageOutput(t *testing.T) {
	var buf bytes.Buffer
	printUsage(&buf)
	output := buf.String()

	// Must list all subcommands
	for _, cmd := range []string{"serve", "generate", "chat", "bench", "tokenize"} {
		if !strings.Contains(output, cmd) {
			t.Errorf("usage output missing subcommand %q", cmd)
		}
	}

	// Must contain the binary name
	if !strings.Contains(output, "vexel") {
		t.Error("usage output missing binary name")
	}
}

func TestGlobalFlags(t *testing.T) {
	// Global flags should be parsed before the subcommand
	args := []string{"vexel", "--model", "/path/to/model.gguf", "--verbose", "serve", "--port", "9090"}
	cmd, globals, err := parseArgs(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cmd != "serve" {
		t.Errorf("got cmd=%q, want serve", cmd)
	}
	if globals.Model != "/path/to/model.gguf" {
		t.Errorf("got model=%q, want /path/to/model.gguf", globals.Model)
	}
	if !globals.Verbose {
		t.Error("expected verbose=true")
	}
}

func TestServeFlags(t *testing.T) {
	args := []string{"vexel", "--model", "m.gguf", "serve", "--port", "9090", "--max-tokens", "512"}
	cmd, _, err := parseArgs(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cmd != "serve" {
		t.Fatalf("got cmd=%q, want serve", cmd)
	}

	sf, err := parseServeFlags([]string{"--port", "9090", "--max-tokens", "512"})
	if err != nil {
		t.Fatalf("unexpected error parsing serve flags: %v", err)
	}
	if sf.Port != 9090 {
		t.Errorf("got port=%d, want 9090", sf.Port)
	}
	if sf.MaxTokens != 512 {
		t.Errorf("got max-tokens=%d, want 512", sf.MaxTokens)
	}
}

func TestServeFlagsDefaults(t *testing.T) {
	sf, err := parseServeFlags(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sf.Port != 8080 {
		t.Errorf("default port: got %d, want 8080", sf.Port)
	}
	if sf.MaxTokens != 256 {
		t.Errorf("default max-tokens: got %d, want 256", sf.MaxTokens)
	}
	if sf.MaxBatchSize != 1 {
		t.Errorf("default max-batch-size: got %d, want 1", sf.MaxBatchSize)
	}
}

func TestGenerateFlags(t *testing.T) {
	gf, err := parseGenerateFlags([]string{"--prompt", "Hello!", "--max-tokens", "100", "--temperature", "0.7"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gf.Prompt != "Hello!" {
		t.Errorf("got prompt=%q, want Hello!", gf.Prompt)
	}
	if gf.MaxTokens != 100 {
		t.Errorf("got max-tokens=%d, want 100", gf.MaxTokens)
	}
	if gf.Temperature != 0.7 {
		t.Errorf("got temperature=%f, want 0.7", gf.Temperature)
	}
}

func TestChatFlags(t *testing.T) {
	cf, err := parseChatFlags([]string{"--system-prompt", "Be brief.", "--no-chat-template"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cf.SystemPrompt != "Be brief." {
		t.Errorf("got system-prompt=%q, want Be brief.", cf.SystemPrompt)
	}
	if !cf.NoChatTemplate {
		t.Error("expected no-chat-template=true")
	}
}

func TestBenchFlags(t *testing.T) {
	bf, err := parseBenchFlags([]string{"--batch", "64", "--seq-len", "256", "--num-seqs", "100"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bf.BatchSize != 64 {
		t.Errorf("got batch=%d, want 64", bf.BatchSize)
	}
	if bf.SeqLen != 256 {
		t.Errorf("got seq-len=%d, want 256", bf.SeqLen)
	}
	if bf.NumSeqs != 100 {
		t.Errorf("got num-seqs=%d, want 100", bf.NumSeqs)
	}
}

func TestTokenizeFlags(t *testing.T) {
	tf, err := parseTokenizeFlags([]string{"--input", "Hello world", "--tokenizer", "/path/tok.json"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tf.Input != "Hello world" {
		t.Errorf("got input=%q, want Hello world", tf.Input)
	}
	if tf.Tokenizer != "/path/tok.json" {
		t.Errorf("got tokenizer=%q, want /path/tok.json", tf.Tokenizer)
	}
}

func TestSubcommandArgs(t *testing.T) {
	args := []string{"vexel", "--model", "m.gguf", "--verbose", "serve", "--port", "9090"}
	sub := subcommandArgs(args)
	if len(sub) != 2 || sub[0] != "--port" || sub[1] != "9090" {
		t.Errorf("got subcommandArgs=%v, want [--port 9090]", sub)
	}
}

func TestDraftModelFlag(t *testing.T) {
	args := []string{"vexel", "--model", "/target.gguf", "--draft-model", "/draft.gguf", "generate"}
	cmd, globals, err := parseArgs(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cmd != "generate" {
		t.Errorf("got cmd=%q, want generate", cmd)
	}
	if globals.Model != "/target.gguf" {
		t.Errorf("got model=%q, want /target.gguf", globals.Model)
	}
	if globals.DraftModel != "/draft.gguf" {
		t.Errorf("got draft-model=%q, want /draft.gguf", globals.DraftModel)
	}
}

func TestDraftModelFlagMissingValue(t *testing.T) {
	args := []string{"vexel", "--draft-model"}
	_, _, err := parseArgs(args)
	if err == nil {
		t.Error("expected error for --draft-model without value")
	}
}

func TestUsageContainsDraftModel(t *testing.T) {
	var buf bytes.Buffer
	printUsage(&buf)
	output := buf.String()
	if !strings.Contains(output, "--draft-model") {
		t.Error("usage output missing --draft-model flag")
	}
}

func TestSubcommandArgsNoSubcmd(t *testing.T) {
	args := []string{"vexel", "--model", "m.gguf"}
	sub := subcommandArgs(args)
	if sub != nil {
		t.Errorf("expected nil, got %v", sub)
	}
}
