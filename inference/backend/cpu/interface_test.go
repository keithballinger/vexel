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
func (m *mockBackend) MatmulTransposeB(a, b, out []float32, row, col, k int) {}
func (m *mockBackend) RMSNorm(x, w, out []float32, r, c int, e float32) {}
func (m *mockBackend) RoPE(q, k []float32, headDim, numHeads, seqLen, startPos int, t float32) {}
func (m *mockBackend) SiLU(x, out []float32, n int) {}
func (m *mockBackend) Embedding(ids []int, table, out []float32, dim int) {}
func (m *mockBackend) Softmax(x, out []float32, rows, cols int) {}
func (m *mockBackend) SDPA(q, k, v, out []float32, kvLen, numQHeads, numKVHeads, headDim int, scale float32) {}
func (m *mockBackend) RoPEShift(k []float32, headDim, numKVHeads, numTokens, shift int, theta float32) {}

// This test verifies that the Backend interface is defined and contains the expected methods.
// We do this by asserting that our mock implementation satisfies the interface.
func TestBackendInterface(t *testing.T) {
	var _ cpu.Backend = &mockBackend{}
}