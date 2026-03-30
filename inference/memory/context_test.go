package memory_test

import (
	"testing"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

func TestInferenceContext(t *testing.T) {
	loc := tensor.CPU

	// Create context
	ctx := memory.NewInferenceContext(loc)

	// Add arenas
	weightsSize := 1024
	scratchSize := 512
	ctx.AddArena(memory.Weights, weightsSize)
	ctx.AddArena(memory.Scratch, scratchSize)

	// Get arenas
	weightsArena := ctx.GetArena(memory.Weights)
	if weightsArena == nil {
		t.Fatal("Failed to retrieve Weights arena")
	}
	if weightsArena.TotalBytes() != weightsSize {
		t.Errorf("Weights arena size mismatch: got %d, want %d", weightsArena.TotalBytes(), weightsSize)
	}

	scratchArena := ctx.GetArena(memory.Scratch)
	if scratchArena == nil {
		t.Fatal("Failed to retrieve Scratch arena")
	}
	if scratchArena.TotalBytes() != scratchSize {
		t.Errorf("Scratch arena size mismatch: got %d, want %d", scratchArena.TotalBytes(), scratchSize)
	}

	// Test missing arena
	if ctx.GetArena(memory.KV) != nil {
		t.Error("GetArena(KV) should return nil when not added")
	}

	// Test Reset
	// Allocate something in scratch
	_, _ = scratchArena.Alloc(100)
	if scratchArena.UsedBytes() == 0 {
		t.Fatal("Scratch arena allocation failed setup")
	}

	ctx.ResetScratch()
	if scratchArena.UsedBytes() != 0 {
		t.Error("ResetScratch should have reset the scratch arena")
	}
}
