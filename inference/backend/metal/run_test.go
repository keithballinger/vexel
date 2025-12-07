//go:build metal

package metal_test

import (
	"testing"
	"vexel/inference/backend/metal"
	"vexel/inference/tensor"
)

func TestRunGraph(t *testing.T) {
	b, _ := metal.NewBackend(0)

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
	graphExec := struct{ name string }{"MetalPipeline"}

	err := runner.RunGraph(graphExec, inputs, stream)
	if err != nil {
		t.Fatalf("RunGraph failed: %v", err)
	}
}
