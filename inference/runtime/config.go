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

// MemoryPlan holds the estimated memory usage breakdown.
type MemoryPlan struct {
	Weights int64
	KV      int64
	Scratch int64
	Total   int64
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
		return params * 2
	case tensor.Q8_0:
		return params
	case tensor.Q4_0:
		return params / 2
	default:
		return params * 2
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
	// Guard against uninitialized config
	if c.NumAttentionHeads == 0 {
		return 0
	}

	// Calculate GQA-aware sizes
	headDim := int64(c.HiddenSize) / int64(c.NumAttentionHeads)
	qSize := int64(maxBatchSize) * int64(c.NumAttentionHeads) * headDim     // Q buffer
	kvSize := int64(maxBatchSize) * int64(c.NumKeyValueHeads) * headDim     // K and V buffers (each)

	// Scratch layout for BlockRuntime.Execute:
	// [normOut][Q][K][V][attnOut][scores][gate][up]
	// All buffers are allocated at once (no reuse during execution)
	normOut := int64(maxBatchSize) * int64(c.HiddenSize)
	attnOut := qSize                                                         // Same size as Q
	scores := int64(maxBatchSize) * int64(maxBatchSize)                      // seqLen x seqLen per head (reused)
	gate := int64(maxBatchSize) * int64(c.IntermediateSize)
	up := int64(maxBatchSize) * int64(c.IntermediateSize)

	// Total scratch needed for BlockRuntime.Execute
	blockScratch := normOut + qSize + kvSize + kvSize + attnOut + scores + gate + up

	// Logits calculation needs separate buffer (in DecodeStep)
	logits := int64(maxBatchSize) * int64(c.VocabSize)

	// Take max (usually block scratch is larger for small batch sizes)
	peak := blockScratch
	if logits > peak {
		peak = logits
	}

	bytesPerElem := int64(4) // Float32
	return peak * bytesPerElem
}

// MemoryPlan aggregates memory usage estimates into a comprehensive plan.
func (c ModelConfig) MemoryPlan(batchSize int, contextLen int, weightsProfile tensor.QuantProfile) MemoryPlan {
	weights := c.WeightsBytes(weightsProfile)
	// For KV cache, assuming standard precision (BF16) for now as we don't pass separate profile yet
	kv := c.KVBytes(batchSize, contextLen, tensor.QuantNone)
	scratch := c.ScratchBytes(batchSize)
	
	return MemoryPlan{
		Weights: weights,
		KV:      kv,
		Scratch: scratch,
		Total:   weights + kv + scratch,
	}
}