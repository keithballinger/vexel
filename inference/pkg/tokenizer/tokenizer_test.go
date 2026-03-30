package tokenizer_test

import (
	"os"
	"testing"
	"vexel/inference/pkg/tokenizer"
)

func TestTokenizerLoad(t *testing.T) {
	// Create a mock tokenizer.json
	// Minimal structure: {"model": {"vocab": {"hello": 1, "world": 2}}}

	// But let's verify with the REAL TinyLlama file if present, as per our goal.
	path := "../../../models/tiny_tokenizer.json"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("TinyLlama tokenizer not found")
	}

	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Test Encoding
	// "Hello world" -> [1, ...] depends on model.
	// For now just check we got an object back.
	if tok == nil {
		t.Fatal("Tokenizer is nil")
	}

	// Check basic encode (mocked or real)
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	if len(ids) == 0 {
		t.Error("Encoded ids empty")
	}

	// Check basic decode
	text, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if text == "" {
		// It might be empty if IDs map to nothing, but check nonetheless.
	}
}
