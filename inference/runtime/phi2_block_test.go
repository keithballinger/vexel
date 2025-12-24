//go:build metal && darwin && cgo

package runtime

import (
	"testing"
	"vexel/inference/backend/metal"
	"vexel/inference/tensor"
)

func TestPhi2BlockLogic(t *testing.T) {
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	hiddenSize := 256
	intermediateSize := 1024
	numHeads := 4
	numKVHeads := 4
	config := ModelConfig{
		HiddenSize:        hiddenSize,
		IntermediateSize:  intermediateSize,
		NumAttentionHeads: numHeads,
		NumKeyValueHeads:  numKVHeads,
		NormType:          NormLayerNorm,
		MLPType:           MLPGELU,
		HasBias:           true,
		ParallelResidual:  true,
		RMSNormEPS:        1e-5,
	}

	block := NewBlockRuntime(b, config)
	
	// Initialize weights as identity or simple values
	// We want to verify that bias is added and parallel residual works
	
	// Mock weights
	wShape := tensor.NewShape(hiddenSize, hiddenSize)
	ones := make([]float32, hiddenSize*hiddenSize)
	for i := range ones { ones[i] = 0 }
	for i := 0; i < hiddenSize; i++ { ones[i*hiddenSize+i] = 1.0 } // Identity
	
	initWeight := func(name string) tensor.Tensor {
		ptr := b.Alloc(len(ones) * 4)
		b.ToDevice(ptr, float32ToBytes(ones))
		return tensor.NewTensor(wShape, tensor.Float32, ptr)
	}
	
	initBias := func(name string, val float32) tensor.Tensor {
		data := make([]float32, hiddenSize)
		for i := range data { data[i] = val }
		ptr := b.Alloc(len(data) * 4)
		b.ToDevice(ptr, float32ToBytes(data))
		return tensor.NewTensor(tensor.NewShape(hiddenSize), tensor.Float32, ptr)
	}

	block.AttnNorm = initWeight("AttnNorm")
	block.AttnNormBias = initBias("AttnNormBias", 0)
	block.Wq = initWeight("Wq")
	block.WqBias = initBias("WqBias", 0.1) // Add 0.1 bias
	block.Wk = initWeight("Wk")
	block.Wv = initWeight("Wv")
	block.Wo = initWeight("Wo")
	block.WoBias = initBias("WoBias", 0.2) // Add 0.2 bias
	
	block.W1 = initWeight("W1")
	block.W1Bias = initBias("W1Bias", 0.3) // Add 0.3 bias
	block.W2 = initWeight("W2")
	block.W2Bias = initBias("W2Bias", 0.4) // Add 0.4 bias

	// Input: all zeros
	inputData := make([]float32, hiddenSize)
	inputPtr := b.Alloc(len(inputData) * 4)
	b.ToDevice(inputPtr, float32ToBytes(inputData))
	input := tensor.NewTensor(tensor.NewShape(1, hiddenSize), tensor.Float32, inputPtr)
	
	scratchPtr := b.Alloc(1024 * 1024 * 4)
	scratch := tensor.NewTensor(tensor.NewShape(1024*1024), tensor.Float32, scratchPtr)

	// Execute
	out, err := block.Execute(input, scratch, nil, 0, 0)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	b.Sync()

	resultBytes := make([]byte, hiddenSize*4)
	b.ToHost(resultBytes, out.DevicePtr())
	result := bytesToFloat32(resultBytes)

	// Check result
	// Input 0 -> Norm 0 -> WqBias 0.1, ...
	// This is a complex chain, but we just want to ensure it's NOT zero
	// and that biases were applied.
	if result[0] == 0 {
		t.Errorf("Result is zero, expected non-zero due to biases")
	}
	t.Logf("Result[0]: %f", result[0])
}
