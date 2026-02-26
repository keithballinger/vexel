package scheduler

import (
	"testing"

	"vexel/inference/pkg/sampler"
)

func TestSelfSpeculativeDecoderCreation(t *testing.T) {
	s := sampler.New(sampler.Config{Temperature: 0}, 42)
	config := DefaultSelfSpeculativeConfig()

	sd := NewSelfSpeculativeDecoder(nil, s, config)
	if sd == nil {
		t.Fatal("expected non-nil SelfSpeculativeDecoder")
	}

	if sd.config.DraftLayers != 8 {
		t.Errorf("expected 8 draft layers, got %d", sd.config.DraftLayers)
	}
	if sd.config.NumDraftTokens != 4 {
		t.Errorf("expected 4 draft tokens, got %d", sd.config.NumDraftTokens)
	}
}

func TestSelfSpeculativeMetrics(t *testing.T) {
	s := sampler.New(sampler.Config{Temperature: 0}, 42)
	config := DefaultSelfSpeculativeConfig()

	sd := NewSelfSpeculativeDecoder(nil, s, config)
	m := sd.Metrics()

	if m.DraftTokensGenerated != 0 {
		t.Errorf("expected 0 draft tokens generated, got %d", m.DraftTokensGenerated)
	}
	if m.AcceptanceRate() != 0 {
		t.Errorf("expected 0 acceptance rate, got %f", m.AcceptanceRate())
	}
	if m.Speedup() != 1.0 {
		t.Errorf("expected 1.0 speedup, got %f", m.Speedup())
	}
}

func TestSelfSpeculativeNilCache(t *testing.T) {
	s := sampler.New(sampler.Config{Temperature: 0}, 42)
	config := DefaultSelfSpeculativeConfig()

	sd := NewSelfSpeculativeDecoder(nil, s, config)

	// With nil model, GenerateDraftTokens should not panic (model is nil, so GPUKVCache returns nil)
	// We can't call this directly since model is nil — it would panic.
	// Instead verify the empty-tokens path of VerifyDraftTokens
	numAccepted, finalToken, finalLogits, err := sd.VerifyDraftTokens(0, 1, nil, nil)
	if err != nil {
		t.Errorf("expected nil error for empty draft tokens, got %v", err)
	}
	if numAccepted != 0 || finalToken != 0 || finalLogits != nil {
		t.Errorf("expected zero values for empty draft, got accepted=%d token=%d logits=%v",
			numAccepted, finalToken, finalLogits)
	}
}

func TestSelfSpeculativeConfigDefaults(t *testing.T) {
	config := DefaultSelfSpeculativeConfig()

	if config.DraftLayers != 8 {
		t.Errorf("expected 8 draft layers, got %d", config.DraftLayers)
	}
	if config.NumDraftTokens != 4 {
		t.Errorf("expected 4 draft tokens, got %d", config.NumDraftTokens)
	}
}
