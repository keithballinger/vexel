//go:build cuda

package cuda_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/backend/cuda"
	"vexel/inference/tensor"
)

func TestCUDABackend(t *testing.T) {
	// Create CUDA backend
	b, err := cuda.NewBackend(0)
	if err != nil {
		// If CUDA is not available, skip test or fail depending on CI
		// For now, we assume we might be running in an environment without GPU
		// but we still want to test the struct initialization if possible.
		// However, NewBackend usually initializes context.
		t.Skipf("Skipping CUDA backend test: %v", err)
	}

	// Verify Device
	dev := b.Device()
	if dev.Location != tensor.CUDA {
		t.Errorf("Backend.Device().Location = %v, want %v", dev.Location, tensor.CUDA)
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
}
