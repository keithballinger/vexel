package scheduler

import (
	"time"

	"vexel/inference/pkg/sampler"
	"vexel/inference/runtime"
)

// SelfSpeculativeDecoder uses the same model for both draft and target.
// Draft generation runs the model with early exit (only first N layers),
// while verification runs all layers. This avoids needing a separate draft model.
type SelfSpeculativeDecoder struct {
	model   *runtime.ModelRuntime
	sampler *sampler.Sampler
	config  SelfSpeculativeConfig
	metrics SpeculativeMetrics
}

// NewSelfSpeculativeDecoder creates a self-speculative decoder.
// The model is used for both draft (early exit) and target (full) inference.
func NewSelfSpeculativeDecoder(model *runtime.ModelRuntime, s *sampler.Sampler, config SelfSpeculativeConfig) *SelfSpeculativeDecoder {
	return &SelfSpeculativeDecoder{
		model:   model,
		sampler: s,
		config:  config,
	}
}

// Metrics returns the current speculative decoding metrics.
func (sd *SelfSpeculativeDecoder) Metrics() SpeculativeMetrics {
	return sd.metrics
}

// GenerateDraftTokens generates K draft tokens using early-exit decoding
// (running only the first DraftLayers layers). Returns tokens and their probabilities.
func (sd *SelfSpeculativeDecoder) GenerateDraftTokens(inputToken int, startPos int) ([]int, []float32, error) {
	draftTokens := make([]int, 0, sd.config.NumDraftTokens)
	draftProbs := make([]float32, 0, sd.config.NumDraftTokens)

	startTime := time.Now()

	cache := sd.model.GPUKVCache()
	if cache == nil {
		return nil, nil, nil
	}

	currentToken := inputToken
	currentPos := startPos

	for i := 0; i < sd.config.NumDraftTokens; i++ {
		// Run model with early exit (only DraftLayers layers)
		logits, err := sd.model.DecodeEarlyExit([]int{currentToken}, currentPos, sd.config.DraftLayers)
		if err != nil {
			return draftTokens, draftProbs, err
		}

		logitsData := getLogitsSlice(logits, sd.model.Backend(), sd.model.Config().VocabSize)
		if logitsData == nil {
			break
		}

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

// VerifyDraftTokens runs the full model (all layers) on draft tokens and returns accepted count.
// This resets the KV cache to startPos first to overwrite the early-exit entries,
// then runs the full forward pass for verification.
func (sd *SelfSpeculativeDecoder) VerifyDraftTokens(
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

	// Truncate KV cache to startPos to overwrite draft entries
	// The early-exit draft wrote partial KV entries (only first N layers).
	// We need to re-run from startPos with all layers.
	if cache := sd.model.GPUKVCache(); cache != nil {
		cache.Truncate(startPos)
	}

	// Build verification input: [input, draft_0, ..., draft_K-1]
	verifyTokens := make([]int, 1+len(draftTokens))
	verifyTokens[0] = inputToken
	copy(verifyTokens[1:], draftTokens)

	// Run full model (all layers)
	logits, err := sd.model.DecodeWithGPUKV(verifyTokens, startPos)
	if err != nil {
		return 0, 0, nil, err
	}

	sd.metrics.VerifyTime += time.Since(startTime)

	vocabSize := sd.model.Config().VocabSize
	allLogits := getLogitsSlice(logits, sd.model.Backend(), len(verifyTokens)*vocabSize)
	if allLogits == nil {
		return 0, 0, nil, nil
	}

	// Verify each draft token
	for i, draftToken := range draftTokens {
		targetLogits := allLogits[i*vocabSize : (i+1)*vocabSize]
		targetProb := getTokenProbability(targetLogits, draftToken, sd.sampler)
		targetToken := sd.sampler.Sample(targetLogits)

		if draftToken == targetToken || targetProb >= draftProbs[i] {
			numAccepted++
			sd.metrics.DraftTokensAccepted++
		} else {
			return numAccepted, targetToken, targetLogits, nil
		}
	}

	// All accepted — sample one more from the last position
	finalLogits = allLogits[len(draftTokens)*vocabSize:]
	finalToken = sd.sampler.Sample(finalLogits)

	return numAccepted, finalToken, finalLogits, nil
}
