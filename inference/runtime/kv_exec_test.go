package runtime_test

import (
	"testing"
	"unsafe"

	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestBlockExecutionWithKV(t *testing.T) {
	spy := &SpyBackend{}
	cfg := testConfig()
	block := runtime.NewBlockRuntime(spy, cfg)

	// Setup weight tensors with proper GQA dimensions
	hiddenSize := cfg.HiddenSize
	intermediateSize := cfg.IntermediateSize
	numHeads := cfg.NumAttentionHeads
	numKVHeads := cfg.NumKeyValueHeads
	headDim := hiddenSize / numHeads
	dummyPtr := tensor.NewDevicePtr(tensor.CPU, 123) // Dummy for spy backend

	block.Wq = tensor.NewTensor(tensor.NewShape(numHeads*headDim, hiddenSize), tensor.Float32, dummyPtr)
	block.Wk = tensor.NewTensor(tensor.NewShape(numKVHeads*headDim, hiddenSize), tensor.Float32, dummyPtr)
	block.Wv = tensor.NewTensor(tensor.NewShape(numKVHeads*headDim, hiddenSize), tensor.Float32, dummyPtr)
	block.Wo = tensor.NewTensor(tensor.NewShape(hiddenSize, numHeads*headDim), tensor.Float32, dummyPtr)
	block.W1 = tensor.NewTensor(tensor.NewShape(intermediateSize, hiddenSize), tensor.Float32, dummyPtr)
	block.W2 = tensor.NewTensor(tensor.NewShape(hiddenSize, intermediateSize), tensor.Float32, dummyPtr)
	block.W3 = tensor.NewTensor(tensor.NewShape(intermediateSize, hiddenSize), tensor.Float32, dummyPtr)
	block.AttnNorm = tensor.NewTensor(tensor.NewShape(hiddenSize), tensor.Float32, dummyPtr)
	block.FFNNorm = tensor.NewTensor(tensor.NewShape(hiddenSize), tensor.Float32, dummyPtr)

	// Allocate real memory for input and scratch
	inputData := make([]float32, hiddenSize)
	inputAddr := uintptr(unsafe.Pointer(&inputData[0]))
	inputShape := tensor.NewShape(1, hiddenSize)
	input := tensor.NewTensor(inputShape, tensor.Float32, tensor.NewDevicePtr(tensor.CPU, inputAddr))

	scratchData := make([]float32, 8192)
	scratchAddr := uintptr(unsafe.Pointer(&scratchData[0]))
	scratch := tensor.NewTensor(tensor.NewShape(8192), tensor.Float32, tensor.NewDevicePtr(tensor.CPU, scratchAddr))

	// Execute with KV params (kvCache=nil for test, layer=0, pos=10)
	block.Execute(input, scratch, nil, 0, 10)
}
