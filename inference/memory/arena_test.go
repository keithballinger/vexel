package memory_test

import (
	"testing"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

func TestArena(t *testing.T) {
	// Mock a device location (CPU for simplicity)
	loc := tensor.CPU
	size := 1024 // 1KB arena

	// Create a new scratch arena
	arena := memory.NewArena(loc, size, memory.Scratch)

	// Test initial state
	if arena.UsedBytes() != 0 {
		t.Errorf("New arena should have 0 used bytes, got %d", arena.UsedBytes())
	}
	if arena.TotalBytes() != size {
		t.Errorf("New arena should have %d total bytes, got %d", size, arena.TotalBytes())
	}

	// Test allocation
	allocSize := 128
	ptr, err := arena.Alloc(allocSize)
	if err != nil {
		t.Fatalf("Arena.Alloc(%d) failed: %v", allocSize, err)
	}
	if ptr.IsNil() {
		t.Error("Allocated pointer should not be nil")
	}
	if ptr.Location() != loc {
		t.Errorf("Allocated pointer location mismatch: got %v, want %v", ptr.Location(), loc)
	}

	if arena.UsedBytes() != allocSize {
		t.Errorf("Arena used bytes mismatch: got %d, want %d", arena.UsedBytes(), allocSize)
	}

	// Test Reset
	arena.Reset()
	if arena.UsedBytes() != 0 {
		t.Error("Arena should have 0 used bytes after Reset")
	}
}

func TestArenaOOM(t *testing.T) {
	arena := memory.NewArena(tensor.CPU, 100, memory.Scratch)

	// Allocate more than capacity
	_, err := arena.Alloc(200)
	if err == nil {
		t.Error("Expected error when allocating more than capacity, got nil")
	}
}
