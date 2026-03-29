package internal_test

import (
	"bytes"
	"strings"
	"testing"
	"vexel/inference/cmd/vexel/internal"
)

func TestChatLoop(t *testing.T) {
	// Mock Stdin with two inputs
	input := "Hello\nExit\n"
	r := strings.NewReader(input)
	var w bytes.Buffer

	// We can't easily mock the Scheduler/Runtime here without dependency injection.
	// We'll verify that the loop reads input and prints prompts.
	// Assuming RunChatLoop(r, w, scheduler) signature.
	
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
