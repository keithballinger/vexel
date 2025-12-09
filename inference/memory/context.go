package memory

import "vexel/inference/tensor"

// InferenceContext holds the memory state for a specific device.
type InferenceContext struct {
	loc    tensor.Location
	arenas map[ArenaKind]*Arena
}

// NewInferenceContext creates a new context for the given location.
func NewInferenceContext(loc tensor.Location) *InferenceContext {
	return &InferenceContext{
		loc:    loc,
		arenas: make(map[ArenaKind]*Arena),
	}
}

// AddArena creates and adds a new arena to the context.
func (c *InferenceContext) AddArena(kind ArenaKind, size int) {
	c.arenas[kind] = NewArena(c.loc, size, kind)
}

// AddArenaWithBackend creates and adds a new arena using a backend allocator.
// This should be used for GPU arenas to ensure proper GPU memory allocation.
func (c *InferenceContext) AddArenaWithBackend(kind ArenaKind, size int, alloc AllocFunc) {
	c.arenas[kind] = NewArenaWithBackend(c.loc, size, kind, alloc)
}

// GetArena retrieves an arena by kind. Returns nil if not found.
func (c *InferenceContext) GetArena(kind ArenaKind) *Arena {
	return c.arenas[kind]
}

// ResetScratch resets the scratch arena, if it exists.
func (c *InferenceContext) ResetScratch() {
	if arena, ok := c.arenas[Scratch]; ok {
		arena.Reset()
	}
}
