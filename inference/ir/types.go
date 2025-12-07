package ir

// TensorID uniquely identifies a tensor within a BlockIR.
type TensorID int

// OpKind represents the type of operation.
type OpKind int

const (
	OpMatmul OpKind = iota
	OpAdd
	OpSiLU
	OpRoPE
	OpRMSNorm
	OpMatmulSiLU // Fused Matmul + SiLU
)

func (k OpKind) String() string {
	switch k {
	case OpMatmul:
		return "Matmul"
	case OpAdd:
		return "Add"
	case OpSiLU:
		return "SiLU"
	case OpRoPE:
		return "RoPE"
	case OpRMSNorm:
		return "RMSNorm"
	case OpMatmulSiLU:
		return "MatmulSiLU"
	default:
		return "Unknown"
	}
}

// OpNode represents a single operation in the compute graph.
type OpNode struct {
	kind    OpKind
	inputs  []TensorID
	outputs []TensorID
}

// NewOpNode creates a new operation node.
func NewOpNode(kind OpKind, inputs, outputs []TensorID) OpNode {
	return OpNode{
		kind:    kind,
		inputs:  inputs,
		outputs: outputs,
	}
}

// Kind returns the operation type.
func (n OpNode) Kind() OpKind {
	return n.kind
}

// Inputs returns the input tensor IDs.
func (n OpNode) Inputs() []TensorID {
	return n.inputs
}

// Outputs returns the output tensor IDs.
func (n OpNode) Outputs() []TensorID {
	return n.outputs
}
