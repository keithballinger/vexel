package sampler_test

import (
	"testing"
	"vexel/inference/pkg/sampler"
)

func TestArgmax(t *testing.T) {
	// Logits: [0.1, 0.5, 0.2, 0.9, 0.3]
	logits := []float32{0.1, 0.5, 0.2, 0.9, 0.3}

	// Expected index: 3 (0.9 is max)
	tokenID := sampler.Argmax(logits)

	if tokenID != 3 {
		t.Errorf("Expected token 3, got %d", tokenID)
	}
}

func TestGreedySampling(t *testing.T) {
	// Temperature 0 should behave like argmax
	cfg := sampler.GreedyConfig()
	s := sampler.New(cfg, 42)

	logits := []float32{0.1, 0.5, 0.2, 0.9, 0.3}
	tokenID := s.Sample(logits)

	if tokenID != 3 {
		t.Errorf("Expected token 3 (greedy), got %d", tokenID)
	}
}

func TestTemperatureSampling(t *testing.T) {
	// With temperature, we should get some variation
	cfg := sampler.Config{
		Temperature: 1.0,
		TopK:        0,
		TopP:        0,
	}
	s := sampler.New(cfg, 42)

	// Make logits nearly uniform to increase variation
	logits := []float32{1.0, 1.0, 1.0, 1.1, 1.0}

	// Sample multiple times - should get different results
	results := make(map[int]int)
	for i := 0; i < 100; i++ {
		results[s.Sample(logits)]++
	}

	// Should have sampled at least 2 different tokens
	if len(results) < 2 {
		t.Errorf("Expected variation in sampling, got only %d unique tokens", len(results))
	}
}

func TestTopKSampling(t *testing.T) {
	cfg := sampler.Config{
		Temperature: 1.0,
		TopK:        2, // Only consider top 2
		TopP:        0,
	}
	s := sampler.New(cfg, 42)

	// Token 3 and 1 are top 2
	logits := []float32{0.1, 0.5, 0.2, 0.9, 0.3}

	// Sample many times - should only get tokens 1 or 3
	for i := 0; i < 50; i++ {
		tokenID := s.Sample(logits)
		if tokenID != 1 && tokenID != 3 {
			t.Errorf("TopK=2 should only return tokens 1 or 3, got %d", tokenID)
		}
	}
}

func TestTopPSampling(t *testing.T) {
	cfg := sampler.Config{
		Temperature: 1.0,
		TopK:        0,
		TopP:        0.5, // Only keep tokens until 50% cumulative prob
	}
	s := sampler.New(cfg, 42)

	// After softmax, token 3 (0.9) will dominate
	// TopP=0.5 should mostly select token 3
	logits := []float32{0.1, 0.5, 0.2, 0.9, 0.3}

	token3Count := 0
	for i := 0; i < 50; i++ {
		if s.Sample(logits) == 3 {
			token3Count++
		}
	}

	// Token 3 should be selected most of the time
	if token3Count < 20 {
		t.Errorf("TopP=0.5 should favor token 3, but only got %d/50", token3Count)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := sampler.DefaultConfig()

	if cfg.Temperature != 0.7 {
		t.Errorf("Expected default temp 0.7, got %f", cfg.Temperature)
	}
	if cfg.TopK != 40 {
		t.Errorf("Expected default top-k 40, got %d", cfg.TopK)
	}
	if cfg.TopP != 0.9 {
		t.Errorf("Expected default top-p 0.9, got %f", cfg.TopP)
	}
}
