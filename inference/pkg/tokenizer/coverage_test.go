package tokenizer_test

import (
	"os"
	"testing"
	"vexel/inference/pkg/tokenizer"
)

func TestTinyLlamaTokenizer(t *testing.T) {
	path := "../../../models/tiny_tokenizer.json"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("TinyLlama tokenizer not found")
	}

	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Verify SentencePiece behavior (should use ▁)
	// "Hello world" -> "Hello" (1) + " world" (1) (prefixed with ▁)
	input := "Hello world"
	ids, err := tok.Encode(input)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	
	if decoded != input {
		t.Errorf("Round trip failed. Got %q, want %q", decoded, input)
	}

	// Test accessor methods
	if tok.BOS() <= 0 {
		t.Error("Invalid BOS")
	}
	if tok.EOS() <= 0 {
		t.Error("Invalid EOS")
	}
	if !tok.AddBOS() {
		t.Error("TinyLlama should AddBOS")
	}
	if tok.VocabSize() == 0 {
		t.Error("VocabSize is 0")
	}

	// Test special token splitting
	specialInput := "<s>Hello</s>"
	specialIds, err := tok.Encode(specialInput)
	if err != nil {
		t.Fatalf("Special encode failed: %v", err)
	}
	// <s> (1) + Hello (id) + </s> (2)
	if len(specialIds) < 3 {
		t.Errorf("Special ids too short: %v", specialIds)
	}
	if specialIds[0] != tok.BOS() {
		t.Errorf("First token should be BOS")
	}
	if specialIds[len(specialIds)-1] != tok.EOS() {
		t.Errorf("Last token should be EOS")
	}

	// Test fallback to byte tokens (simulated by using a char not likely in vocab?)
	// TinyLlama vocab is usually complete for bytes, but let's try a control char 0x07 (BEL)
	// if it's not in vocab, it should be encoded as <0x07>
	// Actually most BPE vocabularies include all 256 bytes or handle them.
	// But let's try to verify decoding of <0x0A> manually to hit decodeSpecialChars regex.
	
	// Manually construct IDs for <0x0A> if possible, or just test the decodeSpecialChars function indirectly via Decode
	// if we can force it.
	// Let's rely on Decode test for regex coverage.
	// "Hello\nWorld" -> \n might be encoded as <0x0A> or as a token containing newline.
	
	// Test byte token decoding directly by injecting a byte token into Decode if possible.
	// We can't easily inject arbitrary IDs that aren't in vocab without mocking, but Load() fills vocab from file.
	
	// Let's try to encode a string with newline and see if it hits the byte logic or vocab match.
	newlineInput := "Line1\nLine2"
	nlIds, _ := tok.Encode(newlineInput)
	nlDecoded, _ := tok.Decode(nlIds)
	if nlDecoded != newlineInput {
		t.Errorf("Newline round trip failed: %q != %q", nlDecoded, newlineInput)
	}

	// Test byte fallback
	byteInput := "\x00"
	byteIds, _ := tok.Encode(byteInput)
	// Should produce <0x00> ID if available, or fallback to something else
	if len(byteIds) > 0 {
		decodedByte, _ := tok.Decode(byteIds)
		// Decode might return <0x00> or the byte depending on logic
		// Just ensure it doesn't crash
		t.Logf("Byte decode: %q", decodedByte)
	}
}

func TestChatTemplates(t *testing.T) {
	// Test TinyLlama Template
	tl := tokenizer.TinyLlamaChatTemplate()
	formatted := tl.FormatChat("Sys", "UserMsg")
	expected := "<|system|>\nSys</s>\n<|user|>\nUserMsg</s>\n<|assistant|>\n"
	if formatted != expected {
		t.Errorf("TinyLlama format mismatch. Got %q, want %q", formatted, expected)
	}

	// Test Llama2 Template
	l2 := tokenizer.Llama2ChatTemplate()
	formattedL2 := l2.FormatChat("Sys", "UserMsg")
	expectedL2 := "[INST] <<SYS>>\nSys\n<</SYS>>\n\nUserMsg [/INST] "
	if formattedL2 != expectedL2 {
		t.Errorf("Llama2 format mismatch. Got %q, want %q", formattedL2, expectedL2)
	}
}
