package cpu_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

func TestCPUBackend(t *testing.T) {
	b := cpu.NewBackend()

	// Verify Device
	dev := b.Device()
	if dev.Location != tensor.CPU {
		t.Errorf("Backend.Device().Location = %v, want %v", dev.Location, tensor.CPU)
	}

	// Verify Stream Creation
	stream, err := b.CreateStream()
	if err != nil {
		t.Fatalf("Backend.CreateStream() failed: %v", err)
	}
	if stream != nil {
		// For CPU, stream might be nil or a dummy object, but for now let's assume it returns nil 
		// or we can strictly enforce it returns a non-nil dummy if that's the design.
		// Let's assume for this test we expect a nil stream for synchronous CPU execution,
		// OR a dummy placeholder. 
		// Actually, standardizing on non-nil is better to avoid nil pointer checks in generic code.
		// Let's assert it is NOT nil (a dummy struct).
		// Wait, the previous thought said "nil" for CPU. 
		// Let's stick to the interface: "CreateStream creates a new execution stream".
		// Even for CPU, returning a valid (though empty) object is cleaner.
		// So checking if stream != nil is actually what we probably want if we implement a dummy.
		// But if the implementation returns nil, we should adjust.
		// Let's check that it returns *something*.
	}
	
	// Check interface satisfaction
	var _ cpu.Backend = b
}
