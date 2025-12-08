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
func (m *mockBackend) Matmul(a, b, out []float32, row, col, k int) {}
func (m *mockBackend) RMSNorm(x, w, out []float32, r, c int, e float32) {}
func (m *mockBackend) RoPE(q, k []float32, h, s, p int, t float32) {}
func (m *mockBackend) SiLU(x, out []float32, n int) {}
func (m *mockBackend) Embedding(ids []int, table, out []float32, dim int) {}

// This test verifies that the Backend interface is defined and contains the expected methods.
// We do this by asserting that our mock implementation satisfies the interface.
func TestBackendInterface(t *testing.T) {
	var _ cpu.Backend = &mockBackend{}
}