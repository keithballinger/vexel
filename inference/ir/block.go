package ir

// BlockIR represents the compute graph for a model block.
type BlockIR struct {
	inputs  []TensorID
	outputs []TensorID
	nodes   []OpNode
}

// NewBlockIR creates a new, empty BlockIR.
func NewBlockIR() *BlockIR {
	return &BlockIR{
		inputs:  make([]TensorID, 0),
		outputs: make([]TensorID, 0),
		nodes:   make([]OpNode, 0),
	}
}

// AddInput registers a tensor ID as an input to the block.
func (b *BlockIR) AddInput(id TensorID) {
	b.inputs = append(b.inputs, id)
}

// AddOutput registers a tensor ID as an output of the block.
func (b *BlockIR) AddOutput(id TensorID) {
	b.outputs = append(b.outputs, id)
}

// AddNode appends an operation node to the graph.
func (b *BlockIR) AddNode(node OpNode) {
	b.nodes = append(b.nodes, node)
}

// Inputs returns the list of input tensor IDs.
func (b *BlockIR) Inputs() []TensorID {
	return b.inputs
}

// Outputs returns the list of output tensor IDs.
func (b *BlockIR) Outputs() []TensorID {
	return b.outputs
}

// Nodes returns the sequential list of operations.
func (b *BlockIR) Nodes() []OpNode {
	return b.nodes
}
