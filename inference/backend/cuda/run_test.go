//go:build cuda

package cuda_test

import (
	"testing"
	"vexel/inference/backend/cuda"
	"vexel/inference/tensor"
)

func TestRunGraph(t *testing.T) {
	b, _ := cuda.NewBackend(0)

	// Check if backend implements the RunGraph method
	runner, ok := b.(interface {
		RunGraph(graphExec interface{}, inputs []tensor.Tensor, stream interface{}) error
	})

	if !ok {
		t.Fatal("Backend does not implement RunGraph")
	}

	// Create dummy inputs
	inputs := []tensor.Tensor{} // Mock inputs
	
	// Create dummy stream
	stream, _ := b.CreateStream()

	// Create dummy graphExec (mocking result of CompileBlockGraph)
	// In a real scenario, we'd call CompileBlockGraph, but for now we manually create a compatible mock
	// matching what CompileBlockGraph currently returns (struct{ name string })
	graphExec := struct{ name string }{"CUDAGraphExec"}

	err := runner.RunGraph(graphExec, inputs, stream)
	if err != nil {
		t.Fatalf("RunGraph failed: %v", err)
	}
}
