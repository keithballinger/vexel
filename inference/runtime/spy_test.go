package runtime_test

import (
	"testing"
	"unsafe"
	"vexel/inference/backend/cpu"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// SpyBackend wraps cpuBackend to record calls
type SpyBackend struct {
	cpu.Backend
	MatmulCalls int
	RMSNormCalls int
	RoPECalls int
	SiLUCalls int
	SoftmaxCalls int
	EmbeddingCalls int
}

func (s *SpyBackend) Matmul(a, b, out []float32, m, n, k int) {
	s.MatmulCalls++
}

func (s *SpyBackend) RMSNorm(x, weight, out []float32, rows, cols int, eps float32) {
	s.RMSNormCalls++
}

func (s *SpyBackend) RoPE(q, k []float32, headDim, seqLen, startPos int, theta float32) {
	s.RoPECalls++
}

func (s *SpyBackend) SiLU(x, out []float32, n int) {
	s.SiLUCalls++
}

func (s *SpyBackend) Softmax(x, out []float32, rows, cols int) {
	s.SoftmaxCalls++
}

func (s *SpyBackend) Embedding(ids []int, table, out []float32, dim int) {
	s.EmbeddingCalls++
}

func (s *SpyBackend) CreateStream() (interface{}, error) { return nil, nil }
func (s *SpyBackend) Device() tensor.Device { return tensor.NewDevice(tensor.CPU, 0) }

func TestBlockExecution(t *testing.T) {
	spy := &SpyBackend{Backend: cpu.NewBackend()}
	block := runtime.NewBlockRuntime(spy)

	// Mock input (Real memory)
	data := make([]float32, 4)
	addr := uintptr(unsafe.Pointer(&data[0]))
	
	inputShape := tensor.NewShape(1, 4)
	input := tensor.NewTensor(inputShape, tensor.Float32, tensor.NewDevicePtr(tensor.CPU, addr))
	
	scratchData := make([]float32, 100) // Big enough
	scratchAddr := uintptr(unsafe.Pointer(&scratchData[0]))
	// Fix shape to match allocation
	scratchShape := tensor.NewShape(100)
	scratch := tensor.NewTensor(scratchShape, tensor.Float32, tensor.NewDevicePtr(tensor.CPU, scratchAddr))
	
	// Execute
	_, err := block.Execute(input, scratch)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Verify calls
	// Llama Block:
	// 1. RMSNorm (Attention Norm)
	// 2. Attention:
	//    - Matmul Q, K, V
	//    - RoPE
	//    - Matmul Output
	// 3. RMSNorm (FFN Norm)
	// 4. MLP:
	//    - Matmul Gate, Up
	//    - SiLU
	//    - Matmul Down
	
	if spy.RMSNormCalls < 2 {
		t.Errorf("Expected at least 2 RMSNorm calls, got %d", spy.RMSNormCalls)
	}
	
	if spy.MatmulCalls < 7 { // Q, K, V, O, Gate, Up, Down
		t.Errorf("Expected at least 7 Matmul calls, got %d", spy.MatmulCalls)
	}
	
	if spy.SiLUCalls < 1 {
		t.Errorf("Expected at least 1 SiLU call, got %d", spy.SiLUCalls)
	}
	
	if spy.RoPECalls < 1 {
		t.Errorf("Expected at least 1 RoPE call, got %d", spy.RoPECalls)
	}
}
