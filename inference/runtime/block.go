package runtime

import (
	"vexel/inference/backend/cpu"
	"vexel/inference/ir"
)

// BlockRuntime represents a compiled block ready for execution.
type BlockRuntime struct {
	backend cpu.Backend
	graph   *ir.BlockIR
}

// NewBlockRuntime compiles the IR for the given backend.
func NewBlockRuntime(backend cpu.Backend, graph *ir.BlockIR) (*BlockRuntime, error) {
	// TODO: Perform JIT compilation or graph preparation here.
	// For now, we just store the graph.
	return &BlockRuntime{
		backend: backend,
		graph:   graph,
	}, nil
}
