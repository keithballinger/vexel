//go:build cuda

package cuda_test

import (
	"testing"
	"vexel/inference/backend/cuda"
	"vexel/inference/ir"
)

func TestCompileBlockGraph(t *testing.T) {
	// Create a dummy IR
	graph := ir.NewBlockIR()
	// Add dummy nodes if needed for a more realistic test
	
	// Create CUDA backend
	b, _ := cuda.NewBackend(0)
	
	// Attempt compilation
	// Note: We need to cast 'b' to the concrete type or interface that supports CompileBlockGraph
	// if it's not part of the standard Backend interface yet.
	// Assuming we might extend the interface or expose a specific method.
	// For this phase, let's assume we call a function in the cuda package or a method on the concrete type.
	
	// Since NewBackend returns the interface, we type assert to access the specific method
	// or we define the method on the struct and call it directly in this test.
	cb, ok := b.(interface {
		CompileBlockGraph(graph *ir.BlockIR) (interface{}, error)
	})
	
	if !ok {
		t.Skip("Backend does not support CompileBlockGraph yet")
	}

	compiled, err := cb.CompileBlockGraph(graph)
	if err != nil {
		t.Fatalf("CompileBlockGraph failed: %v", err)
	}
	
	if compiled == nil {
		t.Error("Compiled graph should not be nil")
	}
}
