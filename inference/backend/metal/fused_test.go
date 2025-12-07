//go:build metal

package metal_test

import (
	"testing"
	"vexel/inference/backend/metal"
	"vexel/inference/ir"
)

func TestFusedKernelCompilation(t *testing.T) {
	// Create a graph with a fused node
	graph := ir.NewBlockIR()
	in := ir.TensorID(1)
	weight := ir.TensorID(2)
	out := ir.TensorID(3)
	
	// Add OpMatmulSiLU node directly (simulating post-fusion state)
	graph.AddInput(in)
	graph.AddNode(ir.NewOpNode(ir.OpMatmulSiLU, []ir.TensorID{in, weight}, []ir.TensorID{out}))
	graph.AddOutput(out)

	b, _ := metal.NewBackend(0)
	
	compiler, ok := b.(interface {
		CompileBlockGraph(graph *ir.BlockIR) (interface{}, error)
	})
	if !ok {
		t.Fatal("Backend does not implement CompileBlockGraph")
	}

	_, err := compiler.CompileBlockGraph(graph)
	if err != nil {
		t.Fatalf("CompileBlockGraph failed for Fused Op: %v", err)
	}
}
