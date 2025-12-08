package sampler

import (
	"math"
	"math/rand"
	"sort"
)

// Config holds sampling parameters.
type Config struct {
	Temperature float32 // Temperature for softmax scaling (0 = greedy)
	TopK        int     // Keep top K tokens (0 = disabled)
	TopP        float32 // Keep tokens with cumulative prob <= P (0 = disabled)
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
	}
}

// GreedyConfig returns config for deterministic greedy decoding.
func GreedyConfig() Config {
	return Config{
		Temperature: 0,
		TopK:        0,
		TopP:        0,
	}
}

// Sampler handles token sampling with various strategies.
type Sampler struct {
	config Config
	rng    *rand.Rand
}

// New creates a new Sampler with the given config.
func New(config Config, seed int64) *Sampler {
	return &Sampler{
		config: config,
		rng:    rand.New(rand.NewSource(seed)),
	}
}

// Sample selects a token from the logits distribution.
func (s *Sampler) Sample(logits []float32) int {
	// Temperature 0 = greedy
	if s.config.Temperature == 0 {
		return Argmax(logits)
	}

	// Apply temperature
	scaled := make([]float32, len(logits))
	for i, v := range logits {
		scaled[i] = v / s.config.Temperature
	}

	// Convert to probabilities
	probs := softmax(scaled)

	// Apply top-k
	if s.config.TopK > 0 && s.config.TopK < len(probs) {
		probs = topK(probs, s.config.TopK)
	}

	// Apply top-p (nucleus)
	if s.config.TopP > 0 && s.config.TopP < 1.0 {
		probs = topP(probs, s.config.TopP)
	}

	// Sample from distribution
	return sampleFromProbs(probs, s.rng)
}

// Argmax returns the index of the maximum value in the slice.
func Argmax(logits []float32) int {
	maxVal := float32(-math.MaxFloat32)
	maxIdx := 0

	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// softmax converts logits to probabilities.
func softmax(logits []float32) []float32 {
	// Find max for numerical stability
	maxVal := float32(-math.MaxFloat32)
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp(x - max) and sum
	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

// topK zeros out all but the top K probabilities, then renormalizes.
func topK(probs []float32, k int) []float32 {
	// Find the k-th largest value
	sorted := make([]float32, len(probs))
	copy(sorted, probs)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
	threshold := sorted[k-1]

	// Zero out values below threshold
	var sum float32
	for i, p := range probs {
		if p < threshold {
			probs[i] = 0
		} else {
			sum += p
		}
	}

	// Renormalize
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// topP keeps the smallest set of tokens with cumulative prob > p.
func topP(probs []float32, p float32) []float32 {
	// Create index-probability pairs and sort by probability descending
	type idxProb struct {
		idx  int
		prob float32
	}
	pairs := make([]idxProb, len(probs))
	for i, prob := range probs {
		pairs[i] = idxProb{i, prob}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].prob > pairs[j].prob })

	// Find cutoff where cumulative probability exceeds p
	var cumsum float32
	cutoffIdx := len(pairs)
	for i, pair := range pairs {
		cumsum += pair.prob
		if cumsum > p {
			cutoffIdx = i + 1
			break
		}
	}

	// Create mask of tokens to keep
	keep := make(map[int]bool)
	for i := 0; i < cutoffIdx; i++ {
		keep[pairs[i].idx] = true
	}

	// Zero out tokens not in nucleus, renormalize
	var sum float32
	for i := range probs {
		if !keep[i] {
			probs[i] = 0
		} else {
			sum += probs[i]
		}
	}

	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// sampleFromProbs samples an index from a probability distribution.
func sampleFromProbs(probs []float32, rng *rand.Rand) int {
	r := rng.Float32()
	var cumsum float32
	for i, p := range probs {
		cumsum += p
		if r < cumsum {
			return i
		}
	}
	// Fallback to last token (shouldn't happen with proper normalization)
	return len(probs) - 1
}
