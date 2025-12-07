package cpu_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

// Mock implementation to verify interface satisfaction
type mockBackend struct{}

func (m *mockBackend) CreateStream() (interface{}, error) { return nil, nil }
func (m *mockBackend) Device() tensor.Device             { return tensor.NewDevice(tensor.CPU, 0) }

// This test verifies that the Backend interface is defined and contains the expected methods.
// We do this by asserting that our mock implementation satisfies the interface.
func TestBackendInterface(t *testing.T) {
	var _ cpu.Backend = &mockBackend{}
}
