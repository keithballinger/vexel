//go:build metal

package metal_test

import (
	"testing"
	"vexel/inference/backend/metal"
	"vexel/inference/ir"
)

func TestCompileBlockGraph(t *testing.T) {
	// Create a dummy IR
	graph := ir.NewBlockIR()

	// Create Metal backend
	b, _ := metal.NewBackend(0)

	// Check for CompileBlockGraph method
	compiler, ok := b.(interface {
		CompileBlockGraph(graph *ir.BlockIR) (interface{}, error)
	})

	if !ok {
		t.Fatal("Backend does not implement CompileBlockGraph")
	}

	compiled, err := compiler.CompileBlockGraph(graph)
	if err != nil {
		t.Fatalf("CompileBlockGraph failed: %v", err)
	}

	if compiled == nil {
		t.Error("Compiled graph should not be nil")
	}
}
