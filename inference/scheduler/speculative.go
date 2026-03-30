package scheduler

import (
	"math"
	"time"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/pkg/sampler"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// SpeculativeConfig holds configuration for speculative decoding.
type SpeculativeConfig struct {
	// NumDraftTokens is how many tokens to speculatively generate (K).
	// Higher values increase potential speedup but may have more rejections.
	NumDraftTokens int

	// AcceptanceThreshold is the minimum probability ratio for acceptance.
	// If p_target(token) >= AcceptanceThreshold * p_draft(token), accept.
	// Set to 0 for standard speculative decoding (always accept if valid).
	AcceptanceThreshold float32
}

// DefaultSpeculativeConfig returns reasonable defaults.
func DefaultSpeculativeConfig() SpeculativeConfig {
	return SpeculativeConfig{
		NumDraftTokens:      4, // Generate 4 draft tokens per step
		AcceptanceThreshold: 0,
	}
}

// SpeculativeMetrics tracks speculative decoding performance.
type SpeculativeMetrics struct {
	DraftTokensGenerated int           // Total draft tokens generated
	DraftTokensAccepted  int           // How many were accepted
	VerificationSteps    int           // Number of verification calls
	DraftTime            time.Duration // Time spent generating drafts
	VerifyTime           time.Duration // Time spent on verification
}

// AcceptanceRate returns the ratio of accepted to generated tokens.
func (m SpeculativeMetrics) AcceptanceRate() float64 {
	if m.DraftTokensGenerated == 0 {
		return 0
	}
	return float64(m.DraftTokensAccepted) / float64(m.DraftTokensGenerated)
}

// Speedup returns the effective speedup from speculative decoding.
// Calculated as: tokens_accepted / verification_steps
func (m SpeculativeMetrics) Speedup() float64 {
	if m.VerificationSteps == 0 {
		return 1.0
	}
	// +1 because we always get at least 1 token from the target model
	return float64(m.DraftTokensAccepted+m.VerificationSteps) / float64(m.VerificationSteps)
}

// SpeculativeDecoder handles speculative decoding with a draft model.
type SpeculativeDecoder struct {
	targetModel *runtime.ModelRuntime
	draftModel  *runtime.ModelRuntime
	sampler     *sampler.Sampler
	config      SpeculativeConfig
	metrics     SpeculativeMetrics
}

// NewSpeculativeDecoder creates a new speculative decoder.
// draftModel should be a smaller/faster model than targetModel.
func NewSpeculativeDecoder(target, draft *runtime.ModelRuntime, s *sampler.Sampler, config SpeculativeConfig) *SpeculativeDecoder {
	return &SpeculativeDecoder{
		targetModel: target,
		draftModel:  draft,
		sampler:     s,
		config:      config,
	}
}

// Metrics returns the current speculative decoding metrics.
func (sd *SpeculativeDecoder) Metrics() SpeculativeMetrics {
	return sd.metrics
}

// GenerateDraftTokens generates K tokens using the draft model.
// Returns the draft tokens and their probabilities.
// Deprecated: Use GenerateDraftTokensFrom which accepts the initial input token.
func (sd *SpeculativeDecoder) GenerateDraftTokens(startPos int) ([]int, []float32, error) {
	return nil, nil, nil // Use GenerateDraftTokensFrom instead
}

// GenerateDraftTokensFrom generates K tokens using the draft model,
// starting from the given input token at the given position.
// Returns the draft tokens and their probabilities.
func (sd *SpeculativeDecoder) GenerateDraftTokensFrom(inputToken int, startPos int) ([]int, []float32, error) {
	draftTokens := make([]int, 0, sd.config.NumDraftTokens)
	draftProbs := make([]float32, 0, sd.config.NumDraftTokens)

	startTime := time.Now()

	// Get the current cache state to prepare for draft generation
	cache := sd.draftModel.GPUKVCache()
	if cache == nil {
		return nil, nil, nil // No GPU cache, skip speculation
	}

	currentToken := inputToken
	currentPos := startPos

	for i := 0; i < sd.config.NumDraftTokens; i++ {
		// Run draft model decode
		logits, err := sd.draftModel.DecodeWithGPUKV([]int{currentToken}, currentPos)
		if err != nil {
			return draftTokens, draftProbs, err
		}

		// Get logits on CPU
		logitsData := getLogitsSlice(logits, sd.draftModel.Backend(), sd.draftModel.Config().VocabSize)
		if logitsData == nil {
			break
		}

		// Sample token
		tokenID := sd.sampler.Sample(logitsData)
		tokenProb := getTokenProbability(logitsData, tokenID, sd.sampler)

		draftTokens = append(draftTokens, tokenID)
		draftProbs = append(draftProbs, tokenProb)

		currentToken = tokenID
		currentPos++
	}

	sd.metrics.DraftTime += time.Since(startTime)
	sd.metrics.DraftTokensGenerated += len(draftTokens)

	return draftTokens, draftProbs, nil
}

// VerifyDraftTokens runs the target model on draft tokens and returns accepted count.
// It processes all draft tokens in a single forward pass and verifies each.
// Returns:
// - numAccepted: how many draft tokens were accepted (0 to K)
// - finalToken: the correct token at the rejection point (or after all accepted)
// - finalLogits: logits from the target model for the final position
func (sd *SpeculativeDecoder) VerifyDraftTokens(
	startPos int,
	inputToken int,
	draftTokens []int,
	draftProbs []float32,
) (numAccepted int, finalToken int, finalLogits []float32, err error) {
	if len(draftTokens) == 0 {
		return 0, 0, nil, nil
	}

	startTime := time.Now()
	sd.metrics.VerificationSteps++

	// Build verification input: [input, draft_0, draft_1, ..., draft_K-1]
	// This allows the target model to produce logits for positions:
	// [pos_input, pos_draft_0, pos_draft_1, ..., pos_draft_K-1]
	verifyTokens := make([]int, 1+len(draftTokens))
	verifyTokens[0] = inputToken
	copy(verifyTokens[1:], draftTokens)

	// Run target model on all tokens at once
	// No reset needed - we append to the existing cache context

	logits, err := sd.targetModel.DecodeWithGPUKV(verifyTokens, startPos)
	if err != nil {
		return 0, 0, nil, err
	}

	sd.metrics.VerifyTime += time.Since(startTime)

	// Get all logits
	vocabSize := sd.targetModel.Config().VocabSize
	allLogits := getLogitsSlice(logits, sd.targetModel.Backend(), len(verifyTokens)*vocabSize)
	if allLogits == nil {
		return 0, 0, nil, nil
	}

	// Verify each draft token
	// Target logits[i] predicts the token at position i+1
	// So target logits[0] predicts draftTokens[0], etc.
	for i, draftToken := range draftTokens {
		targetLogits := allLogits[i*vocabSize : (i+1)*vocabSize]

		// Get target model's probability for the draft token
		targetProb := getTokenProbability(targetLogits, draftToken, sd.sampler)

		// Get target model's preferred token
		targetToken := sd.sampler.Sample(targetLogits)

		// Accept if target agrees with draft or if probability is high enough
		// Standard speculative decoding: accept if target would have sampled this token
		if draftToken == targetToken || targetProb >= draftProbs[i] {
			numAccepted++
			sd.metrics.DraftTokensAccepted++
		} else {
			// Reject: return the correct token from target model
			return numAccepted, targetToken, targetLogits, nil
		}
	}

	// All draft tokens accepted!
	// Sample one more token from the last position's logits
	finalLogits = allLogits[len(draftTokens)*vocabSize:]
	finalToken = sd.sampler.Sample(finalLogits)

	return numAccepted, finalToken, finalLogits, nil
}

// verifyDraftAgainstLogits is the core verification algorithm extracted for testing.
// It checks draft tokens against pre-computed target model logits and returns
// the number of accepted tokens, the final token (correction or bonus), and
// the final position's logits.
//
// allLogits: flat []float32 with shape [(numDraft+1) * vocabSize] or [numDraft * vocabSize]
// draftTokens: proposed tokens from the draft model
// draftProbs: probability of each draft token from the draft model
// vocabSize: vocabulary size for slicing logits
// s: sampler for token selection (argmax at temp=0)
func verifyDraftAgainstLogits(
	allLogits []float32,
	draftTokens []int,
	draftProbs []float32,
	vocabSize int,
	s *sampler.Sampler,
) (numAccepted int, finalToken int, finalLogits []float32) {
	if len(draftTokens) == 0 || len(allLogits) == 0 {
		return 0, 0, nil
	}

	for i, draftToken := range draftTokens {
		if (i+1)*vocabSize > len(allLogits) {
			break
		}
		targetLogits := allLogits[i*vocabSize : (i+1)*vocabSize]

		// Get target model's probability for the draft token
		targetProb := getTokenProbability(targetLogits, draftToken, s)

		// Get target model's preferred token
		targetToken := s.Sample(targetLogits)

		// Accept if target agrees with draft or if probability is high enough
		if draftToken == targetToken || targetProb >= draftProbs[i] {
			numAccepted++
		} else {
			// Reject: return the correct token from target model
			return numAccepted, targetToken, targetLogits
		}
	}

	// All draft tokens accepted — sample bonus token from next position if available
	bonusStart := len(draftTokens) * vocabSize
	if bonusStart+vocabSize <= len(allLogits) {
		finalLogits = allLogits[bonusStart : bonusStart+vocabSize]
		finalToken = s.Sample(finalLogits)
	}

	return numAccepted, finalToken, finalLogits
}

// getLogitsSlice extracts logits from a tensor as []float32.
// numElements is the total number of float32 values to extract (vocabSize for single-token,
// seqLen*vocabSize for multi-token verification).
// If the tensor is on CPU, a direct slice is returned. If on GPU, data is copied to host.
func getLogitsSlice(logits tensor.Tensor, b backend.Backend, numElements int) []float32 {
	ptr := logits.DevicePtr()
	if ptr.IsNil() {
		return nil
	}

	// Fast path: tensor already on CPU — return direct slice
	if ptr.Location() == tensor.CPU {
		return tensor.ToFloat32Slice(logits)[:numElements]
	}

	// GPU path: copy to host
	result := make([]float32, numElements)
	hostBytes := unsafe.Slice((*byte)(unsafe.Pointer(&result[0])), numElements*4)
	b.ToHost(hostBytes, ptr)

	return result
}

// getTokenProbability computes the probability of a token given logits.
func getTokenProbability(logits []float32, tokenID int, s *sampler.Sampler) float32 {
	if logits == nil || tokenID < 0 || tokenID >= len(logits) {
		return 0
	}

	// Apply softmax to get probabilities
	// First find max for numerical stability
	maxLogit := logits[0]
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	// Compute exp(logit - max) and sum
	var sum float32
	for _, l := range logits {
		sum += float32(math.Exp(float64(l - maxLogit)))
	}

	// Probability of our token
	return float32(math.Exp(float64(logits[tokenID]-maxLogit))) / sum
}

// SelfSpeculativeConfig configures self-speculative decoding where
// we use the same model but with reduced layers for drafting.
type SelfSpeculativeConfig struct {
	// DraftLayers is how many layers to use for draft model (early exit).
	// Must be less than the full model's layer count.
	DraftLayers int

	// NumDraftTokens is how many tokens to speculatively generate.
	NumDraftTokens int
}

// DefaultSelfSpeculativeConfig returns reasonable defaults for self-speculation.
func DefaultSelfSpeculativeConfig() SelfSpeculativeConfig {
	return SelfSpeculativeConfig{
		DraftLayers:    8, // Use first 8 layers for draft
		NumDraftTokens: 4, // Generate 4 draft tokens
	}
}
