package ir_test

import (
	"testing"
	"vexel/inference/ir"
)

func TestFusionPass(t *testing.T) {
	// Create a graph with: Input -> Matmul -> Add -> Output
	graph := ir.NewBlockIR()
	in := ir.TensorID(1)
	weight := ir.TensorID(2)
	bias := ir.TensorID(3)
	mid := ir.TensorID(4)
	out := ir.TensorID(5)

	graph.AddInput(in)
	
	// Matmul: in, weight -> mid
	graph.AddNode(ir.NewOpNode(ir.OpMatmul, []ir.TensorID{in, weight}, []ir.TensorID{mid}))
	// Add: mid, bias -> out
	graph.AddNode(ir.NewOpNode(ir.OpAdd, []ir.TensorID{mid, bias}, []ir.TensorID{out}))
	
	graph.AddOutput(out)

	// Apply fusion pass
	// We expect the generic "FuseOps" function to scan the graph.
	// Even if it does nothing yet, we need to verify the mechanism exists.
	// Let's assume we implement a "LinearFusion" pass that looks for Matmul+Add
	// and conceptually marks them (or we just test the Pass interface).
	
	pass := ir.NewFusionPass()
	optimizedGraph := pass.Run(graph)

	// For now, since we haven't implemented the logic, we just expect a valid graph back.
	if len(optimizedGraph.Nodes()) == 0 {
		t.Error("Optimized graph should not be empty")
	}
}
