package scheduler

import (
	"testing"
)

func TestAdaptiveDraftLengthDefaults(t *testing.T) {
	ad := NewAdaptiveDraftLength(DefaultAdaptiveConfig())

	if ad.NumDraftTokens() != 4 {
		t.Errorf("initial NumDraftTokens = %d, want 4", ad.NumDraftTokens())
	}
}

func TestAdaptiveDraftLengthIncreaseOnHighAcceptance(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         5,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Record a window of high acceptance: 100% over 5 steps
	for i := 0; i < 5; i++ {
		ad.RecordStep(4, 4) // accepted=4, drafted=4 → 100%
	}

	got := ad.NumDraftTokens()
	if got != 5 {
		t.Errorf("after high acceptance, NumDraftTokens = %d, want 5", got)
	}
}

func TestAdaptiveDraftLengthDecreaseOnLowAcceptance(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         5,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Record a window of low acceptance: 0% over 5 steps
	for i := 0; i < 5; i++ {
		ad.RecordStep(0, 4) // accepted=0, drafted=4 → 0%
	}

	got := ad.NumDraftTokens()
	if got != 3 {
		t.Errorf("after low acceptance, NumDraftTokens = %d, want 3", got)
	}
}

func TestAdaptiveDraftLengthNoChangeInMiddle(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         5,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Record 60% acceptance — between thresholds, no change
	for i := 0; i < 5; i++ {
		ad.RecordStep(3, 5) // 60%
	}

	got := ad.NumDraftTokens()
	if got != 4 {
		t.Errorf("after middle acceptance, NumDraftTokens = %d, want 4", got)
	}
}

func TestAdaptiveDraftLengthClampMax(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 7,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         3,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Push acceptance high to increase from 7 to 8
	for i := 0; i < 3; i++ {
		ad.RecordStep(7, 7) // 100%
	}
	if ad.NumDraftTokens() != 8 {
		t.Errorf("expected increase to 8, got %d", ad.NumDraftTokens())
	}

	// Push again — should stay clamped at 8
	for i := 0; i < 3; i++ {
		ad.RecordStep(8, 8)
	}
	if ad.NumDraftTokens() != 8 {
		t.Errorf("expected clamped at 8, got %d", ad.NumDraftTokens())
	}
}

func TestAdaptiveDraftLengthClampMin(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 2,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         3,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Push acceptance low to decrease from 2 to 1
	for i := 0; i < 3; i++ {
		ad.RecordStep(0, 2) // 0%
	}
	if ad.NumDraftTokens() != 1 {
		t.Errorf("expected decrease to 1, got %d", ad.NumDraftTokens())
	}

	// Push again — should stay clamped at 1
	for i := 0; i < 3; i++ {
		ad.RecordStep(0, 1)
	}
	if ad.NumDraftTokens() != 1 {
		t.Errorf("expected clamped at 1, got %d", ad.NumDraftTokens())
	}
}

func TestAdaptiveDraftLengthRollingWindow(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         4,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Fill window with high acceptance → increase to 5
	for i := 0; i < 4; i++ {
		ad.RecordStep(4, 4) // 100%
	}
	if ad.NumDraftTokens() != 5 {
		t.Errorf("expected increase to 5, got %d", ad.NumDraftTokens())
	}

	// Slide in low acceptance entries one at a time and verify rolling behavior.
	// Step 5: window = [0%, 100%, 100%, 100%] → rate = 0.75, no change (current=5)
	ad.RecordStep(0, 5)
	if ad.NumDraftTokens() != 5 {
		t.Errorf("after 1 low entry, expected 5, got %d", ad.NumDraftTokens())
	}

	// Step 6: window = [0%, 0%, 100%, 100%] → rate = 0.50, no change (current=5)
	ad.RecordStep(0, 5)
	if ad.NumDraftTokens() != 5 {
		t.Errorf("after 2 low entries, expected 5, got %d", ad.NumDraftTokens())
	}

	// Step 7: window = [0%, 0%, 0%, 100%] → rate = 0.25 < 0.40 → decrease (5→4)
	ad.RecordStep(0, 5)
	if ad.NumDraftTokens() != 4 {
		t.Errorf("after 3 low entries, expected decrease to 4, got %d", ad.NumDraftTokens())
	}

	// Step 8: window = [0%, 0%, 0%, 0%] → rate = 0.0 < 0.40 → decrease (4→3)
	ad.RecordStep(0, 5)
	if ad.NumDraftTokens() != 3 {
		t.Errorf("after 4 low entries, expected decrease to 3, got %d", ad.NumDraftTokens())
	}
}

func TestAdaptiveDraftLengthPartialWindow(t *testing.T) {
	// Before the window is full, no adjustment should occur
	cfg := AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         5,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Record only 2 steps (window not full) — should not change
	ad.RecordStep(4, 4) // 100%
	ad.RecordStep(4, 4) // 100%

	if ad.NumDraftTokens() != 4 {
		t.Errorf("expected no change with partial window, got %d", ad.NumDraftTokens())
	}
}

func TestAdaptiveDraftLengthZeroDrafted(t *testing.T) {
	// RecordStep with drafted=0 should be ignored (avoid division by zero)
	cfg := DefaultAdaptiveConfig()
	ad := NewAdaptiveDraftLength(cfg)

	ad.RecordStep(0, 0) // Should not panic or add to window

	if ad.NumDraftTokens() != 4 {
		t.Errorf("expected no change after zero-drafted step, got %d", ad.NumDraftTokens())
	}
}

func TestAdaptiveDraftLengthWindowRate(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         4,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Mixed window: 2/4 = 50%, 3/4 = 75%, 4/4 = 100%, 1/4 = 25%
	// Average = (0.5 + 0.75 + 1.0 + 0.25) / 4 = 0.625
	ad.RecordStep(2, 4)
	ad.RecordStep(3, 4)
	ad.RecordStep(4, 4)
	ad.RecordStep(1, 4)

	rate := ad.WindowAcceptanceRate()
	if rate < 0.62 || rate > 0.63 {
		t.Errorf("WindowAcceptanceRate = %f, want ~0.625", rate)
	}
}

func TestDefaultAdaptiveConfig(t *testing.T) {
	cfg := DefaultAdaptiveConfig()

	if cfg.InitialDraftTokens != 4 {
		t.Errorf("InitialDraftTokens = %d, want 4", cfg.InitialDraftTokens)
	}
	if cfg.MinDraftTokens != 1 {
		t.Errorf("MinDraftTokens = %d, want 1", cfg.MinDraftTokens)
	}
	if cfg.MaxDraftTokens != 8 {
		t.Errorf("MaxDraftTokens = %d, want 8", cfg.MaxDraftTokens)
	}
	if cfg.WindowSize != 10 {
		t.Errorf("WindowSize = %d, want 10", cfg.WindowSize)
	}
	if cfg.IncreaseThreshold != 0.80 {
		t.Errorf("IncreaseThreshold = %f, want 0.80", cfg.IncreaseThreshold)
	}
	if cfg.DecreaseThreshold != 0.40 {
		t.Errorf("DecreaseThreshold = %f, want 0.40", cfg.DecreaseThreshold)
	}
}
