//go:build cuda

package cuda_test

import (
	"testing"
	"vexel/inference/backend/cuda"
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

	b, _ := cuda.NewBackend(0)

	// We need to verify that CompileBlockGraph handles OpMatmulSiLU specifically.
	// Since we can't easily inspect the opaque compiled object in this test without internals,
	// we will rely on it NOT returning an error (implying support).
	// Ideally, we'd mock the internal compiler or inspect logs, but for now, "no error" = "handled".

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
