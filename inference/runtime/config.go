package runtime

import "vexel/inference/tensor"

// ModelConfig defines the hyperparameters for the model architecture.
type ModelConfig struct {
	HiddenSize  int
	NumLayers   int
	NumHeads    int
	NumKVHeads  int
	VocabSize   int
	MaxSeqLen   int
	RopeBase    float64
	NormEpsilon float64
	DType       tensor.DType
}

// Llama3_8B returns the configuration for the Llama 3 8B model.
func Llama3_8B() ModelConfig {
	return ModelConfig{
		HiddenSize:  4096,
		NumLayers:   32,
		NumHeads:    32,
		NumKVHeads:  8,
		VocabSize:   128256,
		MaxSeqLen:   8192,
		RopeBase:    500000.0,
		NormEpsilon: 1e-5,
		DType:       tensor.BFloat16,
	}
}
