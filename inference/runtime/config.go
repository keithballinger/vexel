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

// WeightsBytes calculates the memory required for weights given a quantization profile.
func (c ModelConfig) WeightsBytes(profile tensor.QuantProfile) int64 {
	params := c.ApproxParams()
	
	switch profile {
	case tensor.QuantNone:
		// Use DType size (default BF16/FP16 = 2 bytes)
		// Assuming DType SizeBytes method exists or we hardcode for now.
		// tensor.DType usually has SizeBytes().
		// BFloat16 is 2 bytes.
		return params * 2
	case tensor.Q8_0:
		// 8 bits = 1 byte per param + small overhead for blocks
		// Simplified: 1 byte
		return params
	case tensor.Q4_0:
		// 4 bits = 0.5 bytes per param + overhead
		return params / 2
	default:
		return params * 2 // Default to 16-bit
	}
}

// KVBytes calculates the memory required for the KV cache.
func (c ModelConfig) KVBytes(activeSequences int, contextLen int, profile tensor.QuantProfile) int64 {
	// Head Dim = Hidden / Heads
	headDim := int64(c.HiddenSize) / int64(c.NumAttentionHeads)
	
	// Elements per token = 2 (Key + Value) * Layers * KVHeads * HeadDim
	elementsPerToken := 2 * int64(c.NumHiddenLayers) * int64(c.NumKeyValueHeads) * headDim
	
	totalTokens := int64(activeSequences) * int64(contextLen)
	
	var bytesPerElem int64 = 2 // Default BF16
	
	return elementsPerToken * totalTokens * bytesPerElem
}

// ScratchBytes calculates the peak scratch memory required for a given batch size.
func (c ModelConfig) ScratchBytes(maxBatchSize int) int64 {
	// Peak usage scenarios:
	// 1. Logits calculation: [Batch, Vocab]
	logits := int64(maxBatchSize) * int64(c.VocabSize)
	
	// 2. MLP Expansion: [Batch, Intermediate]
	mlp := int64(maxBatchSize) * int64(c.IntermediateSize)
	
	// Bytes per element (assume high precision for activations)
	bytesPerElem := int64(2)
	
	// Return the larger of the two peaks
	if logits > mlp {
		return logits * bytesPerElem
	}
	return mlp * bytesPerElem
}
