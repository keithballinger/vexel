package runtime

import (
	"log"
	"math"

	"vexel/inference/pkg/gguf"
	"vexel/inference/tensor"
)

// NormType specifies the normalization layer type.
type NormType int

const (
	// NormRMSNorm uses Root Mean Square normalization (LLaMA, Mistral).
	NormRMSNorm NormType = iota
	// NormLayerNorm uses Layer Normalization with mean subtraction (Phi, GPT-2).
	NormLayerNorm
)

func (n NormType) String() string {
	switch n {
	case NormRMSNorm:
		return "RMSNorm"
	case NormLayerNorm:
		return "LayerNorm"
	default:
		return "Unknown"
	}
}

// MLPType specifies the feed-forward network structure.
type MLPType int

const (
	// MLPSwiGLU uses SiLU-gated linear unit with 3 projections (LLaMA, Mistral).
	// Structure: down(silu(gate(x)) * up(x))
	MLPSwiGLU MLPType = iota
	// MLPGELU uses GELU activation with 2 projections (Phi, GPT-2).
	// Structure: down(gelu(up(x)))
	MLPGELU
	// MLPGeGLU uses GELU-gated linear unit with 3 projections (Gemma).
	// Structure: down(gelu(gate(x)) * up(x))
	MLPGeGLU
	// MLPSwiGLUFused uses SiLU-gated linear unit with a pre-fused gate+up projection (Phi-3).
	// Structure: down(silu(gate_up[:half](x)) * gate_up[half:](x))
	// The GGUF stores gate and up as a single [hidden, 2*intermediate] tensor.
	MLPSwiGLUFused
	// MLPMoE uses a Mixture of Experts feed-forward network (DeepSeek).
	// A router/gate selects a subset of experts per token, each expert is a SwiGLU MLP.
	// Structure: router(x) -> top-k experts -> weighted sum of expert outputs
	MLPMoE
)

func (m MLPType) String() string {
	switch m {
	case MLPSwiGLU:
		return "SwiGLU"
	case MLPGELU:
		return "GELU"
	case MLPGeGLU:
		return "GeGLU"
	case MLPSwiGLUFused:
		return "SwiGLU(Fused)"
	case MLPMoE:
		return "MoE"
	default:
		return "Unknown"
	}
}

// AttentionWindowType specifies the attention window pattern for each layer.
type AttentionWindowType int

const (
	// WindowGlobal uses full context attention on every layer (default, e.g. LLaMA).
	WindowGlobal AttentionWindowType = iota
	// WindowSliding uses sliding window attention on every layer (e.g. Mistral).
	WindowSliding
	// WindowAlternating uses global attention on even layers and sliding window
	// on odd layers (e.g. Gemma 2).
	WindowAlternating
)

func (w AttentionWindowType) String() string {
	switch w {
	case WindowGlobal:
		return "Global"
	case WindowSliding:
		return "Sliding"
	case WindowAlternating:
		return "Alternating"
	default:
		return "Unknown"
	}
}

// ModelConfig defines the hyperparameters for the model architecture.
type ModelConfig struct {
	HiddenSize        int
	IntermediateSize  int
	NumHiddenLayers   int
	NumAttentionHeads int
	NumKeyValueHeads  int
	VocabSize         int
	MaxSeqLen         int
	RoPETheta         float64
	RMSNormEPS        float64
	DType             tensor.DType

	// Architecture-specific settings
	NormType         NormType // Normalization type (RMSNorm or LayerNorm)
	MLPType          MLPType  // MLP structure (SwiGLU or GELU)
	HasBias          bool     // Whether model has bias terms in all projections (Phi-2)
	HasQKVBias       bool     // Whether model has bias terms in Q/K/V projections only (Qwen2)
	ParallelResidual bool     // True for parallel residual (Phi): x + attn(norm(x)) + mlp(norm(x))
	// False for serial residual (LLaMA): x + attn(norm1(x)), then x + mlp(norm2(x))
	RoPEDim int // Dimensions to apply RoPE to (0 = full headDim for LLaMA-style)
	// Phi-2 uses partial RoPE where only first 32 dims of 80 are rotated
	RoPENeox            bool                // Use NEOX-style RoPE (split pairs: i, i+dim/2) vs LLaMA-style (interleaved: 2i, 2i+1)
	SlidingWindow       int                 // Sliding window size for attention (0 = infinite/full context)
	AttentionWindowType AttentionWindowType // Window pattern: Global (default), Sliding (all layers), Alternating (even=global, odd=sliding)

	// Head dimension override (0 = hiddenSize/numAttentionHeads).
	// Gemma 2 2B: hiddenSize=2304, numHeads=8, but headDim=256 (not 288).
	HeadDim int

	// Gemma 2-specific settings
	AttentionLogitSoftCap float32   // Logit soft-capping value (0 = disabled, 50.0 for Gemma 2 attention)
	FinalLogitSoftCap     float32   // Final logit soft-capping (0 = disabled, 30.0 for Gemma 2)
	HasPostNorms          bool      // Apply RMSNorm after attention and MLP (before residual). Gemma 2 only.
	RoPEFreqScales        []float32 // Per-dimension inverse frequency values for learned RoPE (nil = use theta-based)

	// Gemma embedding scaling
	EmbeddingScale float32 // Multiply embeddings by this factor after lookup (0 = disabled). Gemma uses sqrt(hiddenSize).

	// MoE (Mixture of Experts) settings
	NumMoEExperts  int // Total number of routed experts (e.g., 64 or 256 for DeepSeek)
	NumMoESelected int // Number of experts selected per token (e.g., 6 or 8 for DeepSeek)
}

// EffectiveHeadDim returns the head dimension, using the explicit HeadDim if set,
// otherwise falling back to HiddenSize / NumAttentionHeads.
func (c ModelConfig) EffectiveHeadDim() int {
	if c.HeadDim > 0 {
		return c.HeadDim
	}
	return c.HiddenSize / c.NumAttentionHeads
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
		NormType:          NormRMSNorm,
		MLPType:           MLPSwiGLU,
		HasBias:           false,
	}
}

// Phi2 returns the configuration for the Phi-2 model.
func Phi2() ModelConfig {
	return ModelConfig{
		HiddenSize:        2560,
		IntermediateSize:  10240, // 4x hidden
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  32, // Phi-2 uses MHA, not GQA
		VocabSize:         51200,
		MaxSeqLen:         2048,
		RoPETheta:         10000.0,
		RMSNormEPS:        1e-5, // Phi uses LayerNorm eps
		DType:             tensor.Float32,
		NormType:          NormLayerNorm,
		MLPType:           MLPGELU,
		HasBias:           true,
		ParallelResidual:  true, // Phi uses parallel residual: x + attn(norm(x)) + mlp(norm(x))
	}
}

// Gemma2B returns the configuration for the Gemma 2B model.
func Gemma2B() ModelConfig {
	return ModelConfig{
		HiddenSize:        2048,
		IntermediateSize:  16384,
		NumHiddenLayers:   18,
		NumAttentionHeads: 8,
		NumKeyValueHeads:  1, // Gemma 2B uses MQA
		VocabSize:         256128,
		MaxSeqLen:         8192,
		RoPETheta:         10000.0,
		RMSNormEPS:        1e-6,
		DType:             tensor.Float32,
		NormType:          NormRMSNorm,
		MLPType:           MLPGeGLU,
		HasBias:           false,
		EmbeddingScale:    float32(math.Sqrt(2048)), // sqrt(hiddenSize)
	}
}

// Gemma7B returns the configuration for the Gemma 7B model.
func Gemma7B() ModelConfig {
	return ModelConfig{
		HiddenSize:        3072,
		IntermediateSize:  24576,
		NumHiddenLayers:   28,
		NumAttentionHeads: 16,
		NumKeyValueHeads:  16, // Gemma 7B uses MHA
		VocabSize:         256128,
		MaxSeqLen:         8192,
		RoPETheta:         10000.0,
		RMSNormEPS:        1e-6,
		DType:             tensor.Float32,
		NormType:          NormRMSNorm,
		MLPType:           MLPGeGLU,
		HasBias:           false,
		EmbeddingScale:    float32(math.Sqrt(3072)), // sqrt(hiddenSize)
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

	headDim := int64(c.EffectiveHeadDim())
	kvSize := headDim * int64(c.NumKeyValueHeads)
	qSize2 := headDim * int64(c.NumAttentionHeads)

	attn := int64(c.HiddenSize)*qSize2 + // Q
		int64(c.HiddenSize)*kvSize + // K
		int64(c.HiddenSize)*kvSize + // V
		qSize2*int64(c.HiddenSize) // O

	// MLP:
	// Gate, Up, Down
	// Gate: Hidden * Intermediate
	// Up: Hidden * Intermediate
	// Down: Intermediate * Hidden
	mlp := int64(c.HiddenSize) * int64(c.IntermediateSize) * 3

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
	// Head Dim (may differ from hiddenSize/numHeads for Gemma 2)
	headDim := int64(c.EffectiveHeadDim())

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

	// Calculate GQA-aware sizes (headDim may differ from hiddenSize/numHeads for Gemma 2)
	headDim := int64(c.EffectiveHeadDim())
	qSize := int64(maxBatchSize) * int64(c.NumAttentionHeads) * headDim // Q buffer
	kvSize := int64(maxBatchSize) * int64(c.NumKeyValueHeads) * headDim // K and V buffers (each)

	// Scratch layout for BlockRuntime.Execute:
	// [normOut][Q][K][V][attnOut][scores][gate][up][fusedMLPTemp]
	// All buffers are allocated at once (no reuse during execution)
	normOut := int64(maxBatchSize) * int64(c.HiddenSize)
	attnOut := qSize                                    // Same size as Q
	scores := int64(maxBatchSize) * int64(maxBatchSize) // seqLen x seqLen per head (reused)
	gate := int64(maxBatchSize) * int64(c.IntermediateSize)
	up := int64(maxBatchSize) * int64(c.IntermediateSize)
	fusedMLPTemp := int64(maxBatchSize) * int64(c.IntermediateSize) * 2 // Fused W1W3 output before deinterleave

	// Total scratch needed for BlockRuntime.Execute
	blockScratch := normOut + qSize + kvSize + kvSize + attnOut + scores + gate + up + fusedMLPTemp

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

// TotalArenaBytes calculates the total scratch arena capacity needed for a forward pass
// with up to maxBatchSize tokens. This accounts for ALL allocations that decode functions
// (DecodeWithGPUKV, DecodeStep, PrefillWithPagedKV, etc.) make from the scratch arena:
//
//   - Token IDs buffer:     maxBatchSize * 4 bytes (int32)
//   - Hidden state buffer:  maxBatchSize * HiddenSize * 4 bytes (float32)
//   - Layer scratch buffer: ScratchBytes(maxBatchSize)
//   - Output logits buffer: VocabSize * 4 bytes (float32, last-token only)
//
// The maxBatchSize should be the maximum number of tokens in any single forward pass,
// including prefill (where batchSize = prompt length). Use max(maxPromptLen, maxGenBatch).
func (c ModelConfig) TotalArenaBytes(maxBatchSize int) int64 {
	tokenBytes := int64(maxBatchSize) * 4                       // int32 token IDs
	stateBytes := int64(maxBatchSize) * int64(c.HiddenSize) * 4 // [batch, hidden] float32
	scratchBytes := c.ScratchBytes(maxBatchSize)                // layer scratch (Q/K/V/scores/gate/up)
	logitsBytes := int64(c.VocabSize) * 4                       // [1, vocab] float32 output logits

	total := tokenBytes + stateBytes + scratchBytes + logitsBytes

	// Add 10% headroom for alignment and minor additional allocations
	total = total + total/10

	return total
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

// ModelConfigFromGGUF creates a ModelConfig from GGUF file configuration.
func ModelConfigFromGGUF(g gguf.ModelConfigValues) ModelConfig {
	// Use extracted values, with sensible defaults for missing data
	maxSeqLen := g.ContextLength
	if maxSeqLen == 0 {
		maxSeqLen = 2048 // Default context length
	}

	ropeTheta := float64(g.RoPETheta)
	if ropeTheta == 0 {
		ropeTheta = 10000.0 // Default RoPE theta
	}

	// Determine architecture-specific settings based on model architecture
	normType := NormRMSNorm
	mlpType := MLPSwiGLU
	hasBias := false
	parallelResidual := false
	ropeNeox := false              // Default to LLaMA-style (interleaved pairs)
	attnLogitSoftCap := float32(0) // 0 = disabled, typically 30.0 for Gemma 2
	attnWindowType := WindowGlobal // Default: full context on every layer
	hasPostNorms := false          // Default: no post-norms
	embeddingScale := float32(0)   // 0 = disabled, Gemma uses sqrt(hiddenSize)

	hasQKVBias := false

	switch g.Architecture {
	case "phi", "phi2":
		normType = NormLayerNorm
		mlpType = MLPGELU
		hasBias = true
		parallelResidual = true // Phi-2 uses parallel residual
		ropeNeox = true         // Phi uses NEOX-style RoPE (split pairs)
	case "phi3":
		normType = NormRMSNorm
		mlpType = MLPSwiGLUFused // Phi-3 uses SwiGLU with pre-fused gate+up tensor
		hasBias = false
		parallelResidual = false // Phi-3 uses serial residual
		ropeNeox = true          // Phi-3 uses NEOX-style RoPE (split pairs)
	case "gpt2", "gptneox":
		normType = NormLayerNorm
		mlpType = MLPGELU
		hasBias = true
		ropeNeox = true // GPT-NeoX uses NEOX-style RoPE (split pairs)
		// GPT-2/NeoX use serial residual
	case "gemma":
		normType = NormRMSNorm
		mlpType = MLPGeGLU
		hasBias = false
		embeddingScale = float32(math.Sqrt(float64(g.HiddenSize)))
		// Gemma 1 uses LLaMA-style RoPE (interleaved pairs)
	case "gemma2":
		normType = NormRMSNorm
		mlpType = MLPGeGLU
		hasBias = false
		embeddingScale = float32(math.Sqrt(float64(g.HiddenSize)))
		attnLogitSoftCap = g.AttnLogitSoftCap // Read from GGUF (typically 50.0 for Gemma 2)
		if attnLogitSoftCap == 0 {
			attnLogitSoftCap = 50.0 // Fallback default
		}
		attnWindowType = WindowAlternating // Even layers=global, odd layers=sliding window
		hasPostNorms = true                // Gemma 2 applies RMSNorm after attn and MLP
		// Gemma 2 uses LLaMA-style RoPE (interleaved pairs)
	case "deepseek", "deepseek2":
		normType = NormRMSNorm
		mlpType = MLPMoE // MoE layers; dense layers within the model also use SwiGLU
		hasBias = false
		// DeepSeek uses LLaMA-style RoPE (interleaved pairs)
	case "llama", "mistral":
		// Default LLaMA-family settings
		normType = NormRMSNorm
		mlpType = MLPSwiGLU
		hasBias = false
		// LLaMA uses LLaMA-style RoPE (interleaved pairs)
	case "qwen2":
		// Qwen2 is LLaMA-compatible but has QKV bias (no output/MLP bias)
		// and uses NEOX-style RoPE (split pairs: i, i+dim/2)
		normType = NormRMSNorm
		mlpType = MLPSwiGLU
		hasBias = false
		hasQKVBias = true
		ropeNeox = true
	}

	log.Printf("[CONFIG] Architecture=%s: NormType=%v, MLPType=%v, HasBias=%v, ParallelResidual=%v, RoPENeox=%v",
		g.Architecture, normType, mlpType, hasBias, parallelResidual, ropeNeox)
	if g.HeadDim > 0 {
		log.Printf("[CONFIG] HeadDim=%d (explicit, vs hiddenSize/numHeads=%d)", g.HeadDim, g.HiddenSize/g.NumHeads)
	}
	if attnLogitSoftCap > 0 {
		log.Printf("[CONFIG] AttentionLogitSoftCap=%.1f, FinalLogitSoftCap=%.1f", attnLogitSoftCap, g.FinalLogitSoftCap)
	}
	if hasPostNorms {
		log.Printf("[CONFIG] HasPostNorms=true (Gemma 2 post-attention and post-FFN norms)")
	}

	// Use GGUF-provided epsilon if available, otherwise default
	rmsNormEPS := float64(g.RMSNormEPS)
	if rmsNormEPS == 0 {
		rmsNormEPS = 1e-5 // Standard default
	}

	return ModelConfig{
		HiddenSize:            g.HiddenSize,
		IntermediateSize:      g.IntermediateSize,
		NumHiddenLayers:       g.NumLayers,
		NumAttentionHeads:     g.NumHeads,
		NumKeyValueHeads:      g.NumKVHeads,
		VocabSize:             g.VocabSize,
		MaxSeqLen:             maxSeqLen,
		RoPETheta:             ropeTheta,
		RMSNormEPS:            rmsNormEPS,
		DType:                 tensor.Float32, // We dequantize to F32 for CPU
		NormType:              normType,
		MLPType:               mlpType,
		HasBias:               hasBias,
		HasQKVBias:            hasQKVBias,
		ParallelResidual:      parallelResidual,
		RoPEDim:               g.RoPEDimCount, // 0 = full headDim, otherwise partial RoPE
		RoPENeox:              ropeNeox,       // NEOX-style (split) vs LLaMA-style (interleaved) RoPE
		SlidingWindow:         g.SlidingWindow,
		AttentionWindowType:   attnWindowType,
		HeadDim:               g.HeadDim, // 0 = derive from hiddenSize/numHeads
		AttentionLogitSoftCap: attnLogitSoftCap,
		FinalLogitSoftCap:     g.FinalLogitSoftCap,
		HasPostNorms:          hasPostNorms,
		EmbeddingScale:        embeddingScale,
		NumMoEExperts:         g.ExpertCount,
		NumMoESelected:        g.ExpertUsedCount,
	}
}
