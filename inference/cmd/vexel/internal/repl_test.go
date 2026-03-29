package internal_test

import (
	"bytes"
	"strings"
	"testing"
	"vexel/inference/cmd/vexel/internal"
)

func TestChatLoop(t *testing.T) {
	// Mock Stdin with two inputs
	input := "Hello\nexit\n"
	r := strings.NewReader(input)
	var w bytes.Buffer

	// We can't easily mock the Scheduler/Runtime here without dependency injection.
	// We'll verify that the loop reads input and prints prompts.

	// Since we don't have the scheduler mock handy in this package easily,
	// we will just test the "REPL" logic if possible, or skip scheduler interaction for now.

	err := internal.RunChatLoop(r, &w, nil) // nil scheduler

	output := w.String()
	if !strings.Contains(output, ">>>") {
		t.Error("Expected prompt '>>>' in output")
	}

	// If it fails on nil scheduler, that's expected behavior we can assert on,
	// or we mock a simple interface.
	if err == nil {
		// If it succeeded with nil scheduler, it probably did nothing.
	}
}

func TestChatLoopClear(t *testing.T) {
	// Test that /clear command works without crashing
	input := "/clear\nexit\n"
	r := strings.NewReader(input)
	var w bytes.Buffer

	config := internal.REPLConfig{
		ChatMode:     true,
		SystemPrompt: "You are helpful.",
	}

	err := internal.RunChatLoopWithConfig(r, &w, nil, config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := w.String()
	if !strings.Contains(output, "Conversation history cleared.") {
		t.Error("Expected clear confirmation message")
	}
}

func TestChatLoopTemplateDetection(t *testing.T) {
	input := "exit\n"
	r := strings.NewReader(input)
	var w bytes.Buffer

	config := internal.REPLConfig{
		ChatMode:  true,
		ModelPath: "/models/Meta-Llama-3-8B-Instruct.gguf",
	}

	err := internal.RunChatLoopWithConfig(r, &w, nil, config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := w.String()
	if !strings.Contains(output, "template: llama3") {
		t.Errorf("Expected llama3 template detection, got: %s", output)
	}
}
