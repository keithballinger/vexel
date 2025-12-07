package ir

// FusionPass represents an optimization pass that fuses operations.
type FusionPass struct{}

// NewFusionPass creates a new fusion pass.
func NewFusionPass() *FusionPass {
	return &FusionPass{}
}

// Run executes the fusion pass on the given graph.
// Currently, this is a pass-through (identity) operation.
// In the future, it will detect patterns like Matmul+Add and replace them with fused nodes.
func (p *FusionPass) Run(graph *BlockIR) *BlockIR {
	// TODO: Implement pattern matching and graph rewriting.
	// For now, we return the graph as-is, or a shallow copy if needed.
	// Since we are not modifying it yet, returning the pointer is fine.
	return graph
}
