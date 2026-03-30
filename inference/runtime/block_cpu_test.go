package runtime

import (
	"math"
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

func TestPhi2BlockLogicCPU(t *testing.T) {
	b := cpu.NewCPUBackend()

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
		RoPETheta:         10000.0,
		RoPENeox:          true,
	}

	block := NewBlockRuntime(b, config)

	// Initialize weights as identity or simple values
	initWeight := func(name string, rows, cols int) tensor.Tensor {
		data := make([]float32, rows*cols)
		// Identity-ish: 1.0 on diagonal
		for i := 0; i < rows && i < cols; i++ {
			data[i*cols+i] = 1.0
		}
		ptr := b.Alloc(len(data) * 4)
		b.ToDevice(ptr, float32ToBytes(data))
		return tensor.NewTensor(tensor.NewShape(rows, cols), tensor.Float32, ptr)
	}

	initBias := func(name string, size int, val float32) tensor.Tensor {
		data := make([]float32, size)
		for i := range data {
			data[i] = val
		}
		ptr := b.Alloc(len(data) * 4)
		b.ToDevice(ptr, float32ToBytes(data))
		return tensor.NewTensor(tensor.NewShape(size), tensor.Float32, ptr)
	}

	block.AttnNorm = initWeight("AttnNorm", hiddenSize, hiddenSize)
	block.AttnNormBias = initBias("AttnNormBias", hiddenSize, 0)
	block.Wq = initWeight("Wq", hiddenSize, hiddenSize)
	block.WqBias = initBias("WqBias", hiddenSize, 0.1)
	block.Wk = initWeight("Wk", hiddenSize, hiddenSize)
	block.Wv = initWeight("Wv", hiddenSize, hiddenSize)
	block.Wo = initWeight("Wo", hiddenSize, hiddenSize)
	block.WoBias = initBias("WoBias", hiddenSize, 0.2)

	block.W1 = initWeight("W1", intermediateSize, hiddenSize)
	block.W1Bias = initBias("W1Bias", intermediateSize, 0.3)
	block.W2 = initWeight("W2", hiddenSize, intermediateSize)
	block.W2Bias = initBias("W2Bias", hiddenSize, 0.4)

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

	resultBytes := make([]byte, hiddenSize*4)
	b.ToHost(resultBytes, out.DevicePtr())
	result := bytesToFloat32(resultBytes)

	// Check result
	if result[0] == 0 {
		t.Errorf("Result is zero, expected non-zero due to biases")
	}
	if math.IsNaN(float64(result[0])) {
		t.Errorf("Result is NaN")
	}
	t.Logf("Result[0]: %f", result[0])
}
