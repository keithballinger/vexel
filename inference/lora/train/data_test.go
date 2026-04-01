package train

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDataText(t *testing.T) {
	dir := t.TempDir()
	jsonl := `{"text": "Hello world"}
{"text": "Second example"}
`
	os.WriteFile(filepath.Join(dir, "train.jsonl"), []byte(jsonl), 0644)

	examples, err := LoadData(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadData: %v", err)
	}
	if len(examples) != 2 {
		t.Fatalf("got %d examples, want 2", len(examples))
	}
	if examples[0].Text != "Hello world" {
		t.Errorf("text=%q, want %q", examples[0].Text, "Hello world")
	}
	if examples[0].Format != FormatText {
		t.Errorf("format=%v, want FormatText", examples[0].Format)
	}
}

func TestLoadDataPromptCompletion(t *testing.T) {
	dir := t.TempDir()
	jsonl := `{"prompt": "What is 2+2?", "completion": "4"}
{"prompt": "Capital of France?", "completion": "Paris"}
`
	os.WriteFile(filepath.Join(dir, "train.jsonl"), []byte(jsonl), 0644)

	examples, err := LoadData(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadData: %v", err)
	}
	if len(examples) != 2 {
		t.Fatalf("got %d examples, want 2", len(examples))
	}
	if examples[0].Prompt != "What is 2+2?" {
		t.Errorf("prompt=%q", examples[0].Prompt)
	}
	if examples[0].Completion != "4" {
		t.Errorf("completion=%q", examples[0].Completion)
	}
	if examples[0].Format != FormatPromptCompletion {
		t.Errorf("format=%v, want FormatPromptCompletion", examples[0].Format)
	}
}

func TestLoadDataEmpty(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "empty.jsonl"), []byte(""), 0644)

	_, err := LoadData(filepath.Join(dir, "empty.jsonl"))
	if err == nil {
		t.Error("expected error for empty file")
	}
}

func TestLoadDataMixedFormat(t *testing.T) {
	dir := t.TempDir()
	jsonl := `{"text": "Hello"}
{"prompt": "Q?", "completion": "A"}
`
	os.WriteFile(filepath.Join(dir, "mixed.jsonl"), []byte(jsonl), 0644)

	_, err := LoadData(filepath.Join(dir, "mixed.jsonl"))
	if err == nil {
		t.Error("expected error for mixed formats")
	}
}

func TestBuildLossMaskText(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5}
	mask := BuildLossMask(tokens, FormatText, 0)
	expected := []float32{1, 1, 1, 1, 0}
	if len(mask) != len(expected) {
		t.Fatalf("mask len=%d, want %d", len(mask), len(expected))
	}
	for i, v := range expected {
		if mask[i] != v {
			t.Errorf("mask[%d]=%f, want %f", i, mask[i], v)
		}
	}
}

func TestBuildLossMaskPromptCompletion(t *testing.T) {
	tokens := []int32{10, 20, 30, 40, 50}
	promptLen := 3
	mask := BuildLossMask(tokens, FormatPromptCompletion, promptLen)
	expected := []float32{0, 0, 0, 1, 0}
	if len(mask) != len(expected) {
		t.Fatalf("mask len=%d, want %d", len(mask), len(expected))
	}
	for i, v := range expected {
		if mask[i] != v {
			t.Errorf("mask[%d]=%f, want %f", i, mask[i], v)
		}
	}
}
