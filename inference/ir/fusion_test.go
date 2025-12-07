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

func TestFusionMatmulSiLU(t *testing.T) {
	// Pattern: Input -> Matmul -> SiLU -> Output
	graph := ir.NewBlockIR()
	in := ir.TensorID(1)
	weight := ir.TensorID(2)
	mid := ir.TensorID(3)
	out := ir.TensorID(4)

	graph.AddInput(in)
	
	// Matmul: in, weight -> mid
	graph.AddNode(ir.NewOpNode(ir.OpMatmul, []ir.TensorID{in, weight}, []ir.TensorID{mid}))
	// SiLU: mid -> out
	graph.AddNode(ir.NewOpNode(ir.OpSiLU, []ir.TensorID{mid}, []ir.TensorID{out}))
	
	graph.AddOutput(out)

	// Run Pass
	pass := ir.NewFusionPass()
	optimizedGraph := pass.Run(graph)

	// Expectation:
	// Nodes should be fused into a single OpMatmulSiLU node.
	// Original nodes (Matmul, SiLU) should be removed (or replaced).
	// For this test, we check that we have exactly 1 node of type OpMatmulSiLU.

	nodes := optimizedGraph.Nodes()
	if len(nodes) != 1 {
		t.Fatalf("Expected 1 node after fusion, got %d", len(nodes))
	}

	if nodes[0].Kind() != ir.OpMatmulSiLU {
		t.Errorf("Expected node kind %v, got %v", ir.OpMatmulSiLU, nodes[0].Kind())
	}
}
