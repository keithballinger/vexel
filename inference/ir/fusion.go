package ir

// FusionPass represents an optimization pass that fuses operations.
type FusionPass struct{}

// NewFusionPass creates a new fusion pass.
func NewFusionPass() *FusionPass {
	return &FusionPass{}
}

// Run executes the fusion pass on the given graph.
func (p *FusionPass) Run(graph *BlockIR) *BlockIR {
	// Simple greedy fusion
	// In a real implementation, we might use a worklist or DAG traversal.
	// Here we iterate and build a new list of nodes.

	nodes := graph.Nodes()
	fusedNodes := make([]OpNode, 0, len(nodes))
	
	// Track which nodes have been consumed by fusion
	consumed := make(map[int]bool)

	for i := 0; i < len(nodes); i++ {
		if consumed[i] {
			continue
		}

		node := nodes[i]

		// Pattern: Matmul -> SiLU
		if node.Kind() == OpMatmul {
			// Look ahead for SiLU
			// Limitation: This assumes topological sort where SiLU follows immediately or we search.
			// BlockIR nodes are strictly ordered list.
			
			// We need to check if any subsequent node is a SiLU that consumes *only* this Matmul's output.
			
			foundFusion := false
			matmulOut := node.Outputs()[0] // Matmul has 1 output

			for j := i + 1; j < len(nodes); j++ {
				if consumed[j] {
					continue
				}
				candidate := nodes[j]
				
				// Check if candidate is SiLU
				if candidate.Kind() == OpSiLU {
					// Check if SiLU input matches Matmul output
					if len(candidate.Inputs()) == 1 && candidate.Inputs()[0] == matmulOut {
						// Found match!
						
						// Create new fused node
						// Inputs: Matmul inputs
						// Outputs: SiLU outputs (since Matmul output is internal now)
						fusedNode := NewOpNode(OpMatmulSiLU, node.Inputs(), candidate.Outputs())
						fusedNodes = append(fusedNodes, fusedNode)
						
						consumed[j] = true // Mark SiLU as consumed
						foundFusion = true
						break
					}
				} else {
					// If we encounter another node that uses Matmul output, we can't fuse strictly?
					// Or if we encounter a write to the Matmul output?
					// For this simple pass, we assume if we find the SiLU consumer, we merge.
					// Real compiler needs use-def chains.
				}
			}

			if !foundFusion {
				fusedNodes = append(fusedNodes, node)
			}
		} else {
			fusedNodes = append(fusedNodes, node)
		}
	}

	// Return a new graph with fused nodes
	newGraph := NewBlockIR()
	newGraph.inputs = graph.inputs
	newGraph.outputs = graph.outputs
	for _, n := range fusedNodes {
		newGraph.AddNode(n)
	}

	return newGraph
}