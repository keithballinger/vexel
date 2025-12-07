//go:build metal

package metal_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/backend/metal"
	"vexel/inference/tensor"
)

func TestMetalBackend(t *testing.T) {
	// Create Metal backend
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Skipf("Skipping Metal backend test: %v", err)
	}

	// Verify Device
	dev := b.Device()
	if dev.Location != tensor.Metal {
		t.Errorf("Backend.Device().Location = %v, want %v", dev.Location, tensor.Metal)
	}
	if dev.Index != 0 {
		t.Errorf("Backend.Device().Index = %v, want 0", dev.Index)
	}

	// Verify Interface Satisfaction
	var _ cpu.Backend = b

	// Test Stream Creation
	stream, err := b.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}
	if stream == nil {
		t.Error("CreateStream returned nil")
	}

	// Test Event Logic (Optional interface check for now)
	// We check if the backend supports explicit synchronization methods if we add them to the main interface or a sub-interface.
	// For now, we assume standard Backend interface.
}
