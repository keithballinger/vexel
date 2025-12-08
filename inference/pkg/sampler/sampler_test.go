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
