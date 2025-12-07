package ir_test

import (
	"testing"
	"vexel/inference/ir"
)

func TestOpKind(t *testing.T) {
	tests := []struct {
		name string
		kind ir.OpKind
	}{
		{"Matmul", ir.OpMatmul},
		{"Add", ir.OpAdd},
		{"SiLU", ir.OpSiLU},
		{"RoPE", ir.OpRoPE},
		{"RMSNorm", ir.OpRMSNorm},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.kind.String() == "" {
				t.Error("OpKind.String() returned empty string")
			}
		})
	}
}

func TestOpNode(t *testing.T) {
	// Setup
	id1 := ir.TensorID(1)
	id2 := ir.TensorID(2)
	outID := ir.TensorID(3)
	
	// Create a mock OpNode
	op := ir.NewOpNode(ir.OpMatmul, []ir.TensorID{id1, id2}, []ir.TensorID{outID})

	if op.Kind() != ir.OpMatmul {
		t.Errorf("OpNode.Kind() = %v, want %v", op.Kind(), ir.OpMatmul)
	}
	
	inputs := op.Inputs()
	if len(inputs) != 2 || inputs[0] != id1 || inputs[1] != id2 {
		t.Error("OpNode.Inputs() mismatch")
	}

	outputs := op.Outputs()
	if len(outputs) != 1 || outputs[0] != outID {
		t.Error("OpNode.Outputs() mismatch")
	}
}
