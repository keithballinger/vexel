package ir_test

import (
	"testing"
	"vexel/inference/ir"
)

func TestBlockIR(t *testing.T) {
	// Define inputs/outputs for the whole block
	inputID := ir.TensorID(1)
	outputID := ir.TensorID(2)

	// Create nodes
	node := ir.NewOpNode(ir.OpMatmul, []ir.TensorID{inputID}, []ir.TensorID{outputID})

	// Create BlockIR
	graph := ir.NewBlockIR()
	graph.AddInput(inputID)
	graph.AddOutput(outputID)
	graph.AddNode(node)

	// Verify state
	if len(graph.Inputs()) != 1 || graph.Inputs()[0] != inputID {
		t.Error("BlockIR inputs mismatch")
	}
	if len(graph.Outputs()) != 1 || graph.Outputs()[0] != outputID {
		t.Error("BlockIR outputs mismatch")
	}
	if len(graph.Nodes()) != 1 {
		t.Error("BlockIR nodes mismatch")
	}
}
