package runtime

import "vexel/inference/tensor"

// ModelConfig defines the hyperparameters for the model architecture.
type ModelConfig struct {
	HiddenSize       int
	IntermediateSize int
	NumHiddenLayers  int
	NumAttentionHeads int
	NumKeyValueHeads  int
	VocabSize        int
	MaxSeqLen        int
	RoPETheta        float64
	RMSNormEPS       float64
	DType            tensor.DType
}

// Llama3_8B returns the configuration for the Llama 3 8B model.
func Llama3_8B() ModelConfig {
	return ModelConfig{
		HiddenSize:        4096,
		IntermediateSize:  14336,
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		VocabSize:         128256,
		MaxSeqLen:         8192,
		RoPETheta:         500000.0,
		RMSNormEPS:        1e-5,
		DType:             tensor.BFloat16,
	}
}
