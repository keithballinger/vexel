package runtime_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/runtime"
)

func TestBlockRuntime(t *testing.T) {
	// Create backend and IR
	b := cpu.NewBackend()
	// ir := ir.NewBlockIR() // IR not needed for constructor anymore

	// Initialize BlockRuntime
	rt := runtime.NewBlockRuntime(b)
	
	if rt == nil {
		t.Error("Failed to create BlockRuntime")
	}
}
