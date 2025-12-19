package runtime_test

import (
	"testing"
	"unsafe"
	"vexel/inference/backend"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// SpyBackend implements backend.Backend and records calls
type SpyBackend struct {
	MatmulCalls    int
	RMSNormCalls   int
	RoPECalls      int
	SiLUCalls      int
	SoftmaxCalls   int
	EmbeddingCalls int
	SDPACalls      int
	AddCalls       int
	MulCalls       int
}

// Verify SpyBackend implements Backend
var _ backend.Backend = (*SpyBackend)(nil)

func (s *SpyBackend) Device() tensor.Device { return tensor.NewDevice(tensor.CPU, 0) }

// Memory management
func (s *SpyBackend) Alloc(bytes int) tensor.DevicePtr {
	if bytes <= 0 {
		return tensor.DevicePtr{}
	}
	data := make([]byte, bytes)
	return tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&data[0])))
}
func (s *SpyBackend) Free(ptr tensor.DevicePtr)                 {}
func (s *SpyBackend) ToDevice(dst tensor.DevicePtr, src []byte) {}
func (s *SpyBackend) ToHost(dst []byte, src tensor.DevicePtr)   {}
func (s *SpyBackend) Sync()                                     {}

// Compute kernels
func (s *SpyBackend) MatMul(a, b, out tensor.DevicePtr, m, n, k int) {
	s.MatmulCalls++
}
func (s *SpyBackend) MatMulTransposed(a, b, out tensor.DevicePtr, m, n, k int) {
	s.MatmulCalls++
}
func (s *SpyBackend) RMSNorm(x, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	s.RMSNormCalls++
}
func (s *SpyBackend) RoPE(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float32, ropeNeox bool) {
	s.RoPECalls++
}
func (s *SpyBackend) SiLU(x, out tensor.DevicePtr, n int) {
	s.SiLUCalls++
}
func (s *SpyBackend) SiLUMul(gate, up, out tensor.DevicePtr, n int) {
	s.SiLUCalls++
}
func (s *SpyBackend) Softmax(x, out tensor.DevicePtr, rows, cols int) {
	s.SoftmaxCalls++
}
func (s *SpyBackend) Embedding(ids tensor.DevicePtr, numTokens int, table, out tensor.DevicePtr, vocabSize, dim int) {
	s.EmbeddingCalls++
}
func (s *SpyBackend) SDPA(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int) {
	s.SDPACalls++
}
func (s *SpyBackend) SDPAPrefill(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	s.SDPACalls++
}
func (s *SpyBackend) Add(a, b, out tensor.DevicePtr, n int) {
	s.AddCalls++
}
func (s *SpyBackend) Mul(a, b, out tensor.DevicePtr, n int) {
	s.MulCalls++
}
func (s *SpyBackend) AddRMSNorm(x, residual, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	s.RMSNormCalls++
	s.AddCalls++
}
func (s *SpyBackend) MatMulQ4_0_FusedRMSNorm(x, normWeight, wMat, out tensor.DevicePtr, m, n, k int, eps float32) {
	s.MatmulCalls++
	s.RMSNormCalls++
}

func TestBlockExecution(t *testing.T) {
	spy := &SpyBackend{}
	cfg := testConfig()
	block := runtime.NewBlockRuntime(spy, cfg)

	// Initialize weights to trigger execution paths
	dummyPtr := tensor.NewDevicePtr(tensor.CPU, 123)
	hiddenSize := cfg.HiddenSize
	intermediateSize := cfg.IntermediateSize
	numHeads := cfg.NumAttentionHeads
	numKVHeads := cfg.NumKeyValueHeads
	headDim := hiddenSize / numHeads

	// Q: [numHeads*headDim, hiddenSize]
	// K,V: [numKVHeads*headDim, hiddenSize] for GQA
	block.Wq = tensor.NewTensor(tensor.NewShape(numHeads*headDim, hiddenSize), tensor.Float32, dummyPtr)
	block.Wk = tensor.NewTensor(tensor.NewShape(numKVHeads*headDim, hiddenSize), tensor.Float32, dummyPtr)
	block.Wv = tensor.NewTensor(tensor.NewShape(numKVHeads*headDim, hiddenSize), tensor.Float32, dummyPtr)
	block.Wo = tensor.NewTensor(tensor.NewShape(hiddenSize, numHeads*headDim), tensor.Float32, dummyPtr)
	block.W1 = tensor.NewTensor(tensor.NewShape(intermediateSize, hiddenSize), tensor.Float32, dummyPtr)
	block.W2 = tensor.NewTensor(tensor.NewShape(hiddenSize, intermediateSize), tensor.Float32, dummyPtr)
	block.W3 = tensor.NewTensor(tensor.NewShape(intermediateSize, hiddenSize), tensor.Float32, dummyPtr)
	block.AttnNorm = tensor.NewTensor(tensor.NewShape(hiddenSize), tensor.Float32, dummyPtr)
	block.FFNNorm = tensor.NewTensor(tensor.NewShape(hiddenSize), tensor.Float32, dummyPtr)

	// Mock Input with real memory
	inputData := make([]float32, hiddenSize)
	inputAddr := uintptr(unsafe.Pointer(&inputData[0]))
	inputShape := tensor.NewShape(1, hiddenSize)
	input := tensor.NewTensor(inputShape, tensor.Float32, tensor.NewDevicePtr(tensor.CPU, inputAddr))

	// Allocate larger scratch for GQA buffers
	scratchData := make([]float32, 8192)
	scratchAddr := uintptr(unsafe.Pointer(&scratchData[0]))
	scratchShape := tensor.NewShape(8192)
	scratch := tensor.NewTensor(scratchShape, tensor.Float32, tensor.NewDevicePtr(tensor.CPU, scratchAddr))

	// Execute
	_, err := block.Execute(input, scratch, nil, 0, 0)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Verify calls
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
