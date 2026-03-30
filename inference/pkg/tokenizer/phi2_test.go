package tokenizer_test

import (
	"os"
	"testing"
	"vexel/inference/pkg/tokenizer"
)

func TestPhi2TokenizerParity(t *testing.T) {
	path := "../../../models/phi2_tokenizer.json"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("Phi-2 tokenizer not found at " + path)
	}

	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	tests := []struct {
		name     string
		input    string
		wantLen  int
		wantText string // Optional: check decoded text
	}{
		{
			name:    "Unit Testing Mismatch (Full Prompt)",
			input:   "Describe the benefits of unit testing in Go in three concise sentences.",
			wantLen: 14, // llama.cpp produces 14 tokens for this full prompt
		},
		{
			name:    "Hello World",
			input:   "Hello world",
			wantLen: 2, // "Hello" (1) + " world" (1) = 2 usually
		},
		{
			name:    "Special Chars",
			input:   "<|endoftext|>",
			wantLen: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ids, err := tok.Encode(tt.input)
			if err != nil {
				t.Fatalf("Encode failed: %v", err)
			}

			t.Logf("Input: %q", tt.input)
			t.Logf("IDs: %v", ids)
			t.Logf("Length: %d", len(ids))

			if len(ids) != tt.wantLen {
				t.Errorf("Length mismatch: got %d, want %d", len(ids), tt.wantLen)
			}

			// Verify round-trip decoding
			decoded, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode failed: %v", err)
			}
			if decoded != tt.input {
				t.Errorf("Round-trip mismatch: got %q, want %q", decoded, tt.input)
			}
		})
	}
}
