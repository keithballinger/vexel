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

// ApproxParams estimates the total number of parameters in the model.
func (c ModelConfig) ApproxParams() int64 {
	// Embedding: Vocab * Hidden
	embedding := int64(c.VocabSize) * int64(c.HiddenSize)

	// Per Layer:
	// Attention:
	// Q, K, V, O projections
	// Q: Hidden * Hidden
	// K: Hidden * (Hidden * KV / Heads) -> Hidden * HeadDim * KV
	// V: Hidden * HeadDim * KV
	// O: Hidden * Hidden
	
	headDim := int64(c.HiddenSize) / int64(c.NumAttentionHeads)
	kvSize := headDim * int64(c.NumKeyValueHeads)
	
	attn := int64(c.HiddenSize)*int64(c.HiddenSize) + // Q
		int64(c.HiddenSize)*kvSize + // K
		int64(c.HiddenSize)*kvSize + // V
		int64(c.HiddenSize)*int64(c.HiddenSize)   // O

	// MLP:
	// Gate, Up, Down
	// Gate: Hidden * Intermediate
	// Up: Hidden * Intermediate
	// Down: Intermediate * Hidden
	mlp := int64(c.HiddenSize)*int64(c.IntermediateSize)*3

	// Norms (RMSNorm is usually just Hidden size per layer x 2 for Attn/MLP norm)
	// We ignore small params like norms for "Approx" usually, but let's add them for fun.
	norms := int64(c.HiddenSize) * 2

	layerParams := attn + mlp + norms

	// Output Head: Vocab * Hidden
	output := int64(c.VocabSize) * int64(c.HiddenSize)

	return embedding + (layerParams * int64(c.NumHiddenLayers)) + output
}