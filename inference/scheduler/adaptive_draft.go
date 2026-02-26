package scheduler

// AdaptiveConfig configures adaptive draft length adjustment.
type AdaptiveConfig struct {
	// InitialDraftTokens is the starting number of draft tokens.
	InitialDraftTokens int

	// MinDraftTokens is the minimum allowed draft length.
	MinDraftTokens int

	// MaxDraftTokens is the maximum allowed draft length.
	MaxDraftTokens int

	// WindowSize is the number of recent steps to consider when computing
	// the rolling acceptance rate. Adjustment only occurs once the window
	// is fully populated.
	WindowSize int

	// IncreaseThreshold: if rolling acceptance rate > this, increase NumDraftTokens by 1.
	IncreaseThreshold float64

	// DecreaseThreshold: if rolling acceptance rate < this, decrease NumDraftTokens by 1.
	DecreaseThreshold float64
}

// DefaultAdaptiveConfig returns sensible defaults for adaptive draft length.
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         10,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
}

// acceptanceEntry records the result of a single speculative step.
type acceptanceEntry struct {
	accepted int
	drafted  int
}

// AdaptiveDraftLength tracks acceptance rates over a rolling window
// and adjusts the number of draft tokens accordingly.
type AdaptiveDraftLength struct {
	config  AdaptiveConfig
	current int // current NumDraftTokens

	window []acceptanceEntry // circular buffer
	head   int               // next write position
	count  int               // number of entries written (capped at WindowSize)
}

// NewAdaptiveDraftLength creates a new adaptive draft length tracker.
func NewAdaptiveDraftLength(config AdaptiveConfig) *AdaptiveDraftLength {
	return &AdaptiveDraftLength{
		config:  config,
		current: config.InitialDraftTokens,
		window:  make([]acceptanceEntry, config.WindowSize),
	}
}

// NumDraftTokens returns the current recommended number of draft tokens.
func (ad *AdaptiveDraftLength) NumDraftTokens() int {
	return ad.current
}

// RecordStep records the result of a speculative decoding step.
// accepted is how many draft tokens were accepted, drafted is how many were generated.
// This updates the rolling window and may adjust NumDraftTokens.
func (ad *AdaptiveDraftLength) RecordStep(accepted, drafted int) {
	if drafted <= 0 {
		return // Ignore zero-draft steps to avoid division by zero
	}

	// Write to circular buffer
	ad.window[ad.head] = acceptanceEntry{accepted: accepted, drafted: drafted}
	ad.head = (ad.head + 1) % ad.config.WindowSize
	if ad.count < ad.config.WindowSize {
		ad.count++
	}

	// Only adjust when window is fully populated
	if ad.count < ad.config.WindowSize {
		return
	}

	rate := ad.windowRate()

	if rate > ad.config.IncreaseThreshold {
		ad.current++
	} else if rate < ad.config.DecreaseThreshold {
		ad.current--
	}

	// Clamp to [min, max]
	if ad.current < ad.config.MinDraftTokens {
		ad.current = ad.config.MinDraftTokens
	}
	if ad.current > ad.config.MaxDraftTokens {
		ad.current = ad.config.MaxDraftTokens
	}
}

// WindowAcceptanceRate returns the average acceptance rate over the current window.
// Returns 0 if the window is empty.
func (ad *AdaptiveDraftLength) WindowAcceptanceRate() float64 {
	return ad.windowRate()
}

// windowRate computes the average acceptance rate across all entries in the window.
func (ad *AdaptiveDraftLength) windowRate() float64 {
	if ad.count == 0 {
		return 0
	}

	var sum float64
	for i := 0; i < ad.count; i++ {
		e := ad.window[i]
		if e.drafted > 0 {
			sum += float64(e.accepted) / float64(e.drafted)
		}
	}
	return sum / float64(ad.count)
}
