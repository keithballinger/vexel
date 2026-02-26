package scheduler

import (
	"context"
	"sync"
	"testing"
	"time"

	"vexel/inference/medusa"
)

// mockTrainer is a test double for medusa.Trainer that records AddSample calls.
type mockTrainer struct {
	mu      sync.Mutex
	samples []mockSample
	phase   medusa.TrainingPhase
	hot     bool
	heads   medusa.HeadsInterface
	metrics medusa.TrainingMetrics
}

type mockSample struct {
	hidden       []float32
	futureTokens []int
	pos          int
}

func newMockTrainer(numHeads, hiddenSize, vocabSize int) *mockTrainer {
	return &mockTrainer{
		phase: medusa.PhaseCold,
		heads: medusa.NewHeads(numHeads, hiddenSize, vocabSize),
		metrics: medusa.TrainingMetrics{
			HeadAccuracies: make([]float32, numHeads),
		},
	}
}

func (m *mockTrainer) Start(ctx context.Context) {}
func (m *mockTrainer) Stop()                     {}
func (m *mockTrainer) AddSample(hidden []float32, futureTokens []int, pos int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.samples = append(m.samples, mockSample{
		hidden:       append([]float32(nil), hidden...),
		futureTokens: append([]int(nil), futureTokens...),
		pos:          pos,
	})
}
func (m *mockTrainer) Phase() medusa.TrainingPhase {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.phase
}
func (m *mockTrainer) IsHot() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.hot
}
func (m *mockTrainer) ForceHot() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.hot = true
	m.phase = medusa.PhaseHot
}
func (m *mockTrainer) Heads() medusa.HeadsInterface {
	return m.heads
}
func (m *mockTrainer) Metrics() medusa.TrainingMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := m.metrics
	result.Phase = m.phase // match real impl: Phase() is the source of truth
	return result
}
func (m *mockTrainer) SaveHeads(path string) error {
	return nil
}

func (m *mockTrainer) getSamples() []mockSample {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]mockSample, len(m.samples))
	copy(cp, m.samples)
	return cp
}

// TestCollectTrainingSample verifies the sliding window correctly pairs hidden
// states with future tokens and feeds them to the trainer.
func TestCollectTrainingSample(t *testing.T) {
	numHeads := 4
	hiddenSize := 8
	vocabSize := 32
	mt := newMockTrainer(numHeads, hiddenSize, vocabSize)

	ms := &MedusaScheduler{
		trainer:       mt,
		medusaConfig:  MedusaConfig{NumHeads: numHeads},
		recentTokens:  make([]int, 0, numHeads+1),
		recentHiddens: make([][]float32, 0, numHeads+1),
		maxRecent:     numHeads + 1,
	}

	// Feed tokens one at a time.
	// collectTrainingSample is called after each decode step with
	// (hidden_state, token_id).
	// After numHeads calls, the sliding window should emit its first sample.
	makeHidden := func(id float32) []float32 {
		h := make([]float32, hiddenSize)
		for i := range h {
			h[i] = id + float32(i)*0.01
		}
		return h
	}

	// First 3 calls: not enough history yet for 4-head sample
	for i := 0; i < numHeads-1; i++ {
		ms.collectTrainingSample(makeHidden(float32(i)), i+100)
	}

	samples := mt.getSamples()
	if len(samples) != 0 {
		t.Fatalf("expected 0 samples after %d calls, got %d", numHeads-1, len(samples))
	}

	// 4th call: now we have [t0, t1, t2, t3] -> emit sample with hidden[0]
	ms.collectTrainingSample(makeHidden(float32(numHeads-1)), numHeads-1+100)

	samples = mt.getSamples()
	if len(samples) != 1 {
		t.Fatalf("expected 1 sample after %d calls, got %d", numHeads, len(samples))
	}

	// The first sample should use hidden[0] with futureTokens [100, 101, 102, 103]
	s := samples[0]
	if len(s.futureTokens) != numHeads {
		t.Errorf("expected %d future tokens, got %d", numHeads, len(s.futureTokens))
	}
	for i := 0; i < numHeads; i++ {
		expected := i + 100
		if s.futureTokens[i] != expected {
			t.Errorf("futureTokens[%d] = %d, want %d", i, s.futureTokens[i], expected)
		}
	}

	// Verify hidden state is from the first call (id=0.0)
	if s.hidden[0] != 0.0 {
		t.Errorf("hidden[0] = %f, want 0.0 (first call)", s.hidden[0])
	}

	// 5th call: should emit second sample with hidden[1]
	ms.collectTrainingSample(makeHidden(float32(numHeads)), numHeads+100)

	samples = mt.getSamples()
	if len(samples) != 2 {
		t.Fatalf("expected 2 samples after %d calls, got %d", numHeads+1, len(samples))
	}

	s2 := samples[1]
	// Second sample should use hidden[1] with tokens [101, 102, 103, 104]
	if s2.futureTokens[0] != 101 {
		t.Errorf("second sample futureTokens[0] = %d, want 101", s2.futureTokens[0])
	}
	if s2.hidden[0] != 1.0 {
		t.Errorf("second sample hidden[0] = %f, want 1.0", s2.hidden[0])
	}
}

// TestCollectTrainingSampleHiddenCopied verifies hidden states are copied,
// not shared by reference, preventing races with GPU buffer reuse.
func TestCollectTrainingSampleHiddenCopied(t *testing.T) {
	numHeads := 2
	hiddenSize := 4
	vocabSize := 16
	mt := newMockTrainer(numHeads, hiddenSize, vocabSize)

	ms := &MedusaScheduler{
		trainer:       mt,
		medusaConfig:  MedusaConfig{NumHeads: numHeads},
		recentTokens:  make([]int, 0, numHeads+1),
		recentHiddens: make([][]float32, 0, numHeads+1),
		maxRecent:     numHeads + 1,
	}

	// Reuse the same buffer (simulating GPU buffer reuse)
	buf := make([]float32, hiddenSize)

	buf[0] = 1.0
	ms.collectTrainingSample(buf, 10)

	buf[0] = 2.0
	ms.collectTrainingSample(buf, 20)

	// After 2 calls with numHeads=2, one sample should exist
	samples := mt.getSamples()
	if len(samples) != 1 {
		t.Fatalf("expected 1 sample, got %d", len(samples))
	}

	// The sample's hidden should be from the FIRST call (buf[0]=1.0),
	// not corrupted by the second call's buf mutation.
	if samples[0].hidden[0] != 1.0 {
		t.Errorf("hidden[0] = %f, want 1.0 (should be a copy)", samples[0].hidden[0])
	}
}

// TestMedusaSchedulerPhaseTransition verifies that the MedusaScheduler
// correctly selects between standard decode and speculation based on trainer phase.
func TestMedusaSchedulerPhaseTransition(t *testing.T) {
	numHeads := 4
	hiddenSize := 8
	vocabSize := 32
	mt := newMockTrainer(numHeads, hiddenSize, vocabSize)

	ms := &MedusaScheduler{
		trainer:      mt,
		medusaConfig: MedusaConfig{NumHeads: numHeads},
	}

	// Initially cold: should not use speculation
	useMedusa := ms.trainer != nil && ms.trainer.IsHot()
	if useMedusa {
		t.Error("expected useMedusa=false when trainer is Cold")
	}

	// Force hot: should use speculation
	mt.ForceHot()
	useMedusa = ms.trainer != nil && ms.trainer.IsHot()
	if !useMedusa {
		t.Error("expected useMedusa=true when trainer is Hot")
	}
}

// TestMedusaMetricsReporting verifies MedusaMetrics aggregates trainer
// and speculative metrics correctly.
func TestMedusaMetricsReporting(t *testing.T) {
	numHeads := 2
	hiddenSize := 8
	vocabSize := 32
	mt := newMockTrainer(numHeads, hiddenSize, vocabSize)

	ms := &MedusaScheduler{
		trainer:      mt,
		medusaConfig: MedusaConfig{NumHeads: numHeads},
		specMetrics: SpeculativeMetrics{
			DraftTokensGenerated: 100,
			DraftTokensAccepted:  75,
			VerificationSteps:    25,
		},
	}

	metrics := ms.MedusaMetrics()

	if metrics.DraftTokensGenerated != 100 {
		t.Errorf("DraftTokensGenerated = %d, want 100", metrics.DraftTokensGenerated)
	}
	if metrics.DraftTokensAccepted != 75 {
		t.Errorf("DraftTokensAccepted = %d, want 75", metrics.DraftTokensAccepted)
	}
	if metrics.Phase != "cold" {
		t.Errorf("Phase = %q, want cold", metrics.Phase)
	}

	// After forcing hot, phase should update
	mt.ForceHot()
	metrics = ms.MedusaMetrics()
	if metrics.Phase != "hot" {
		t.Errorf("Phase = %q, want hot after ForceHot", metrics.Phase)
	}
}

// TestMedusaSchedulerNoTrainer verifies graceful behavior when trainer is nil.
func TestMedusaSchedulerNoTrainer(t *testing.T) {
	ms := &MedusaScheduler{
		medusaConfig: MedusaConfig{NumHeads: 4},
	}

	// Metrics should report "disabled"
	metrics := ms.MedusaMetrics()
	if metrics.Phase != "disabled" {
		t.Errorf("Phase = %q, want disabled", metrics.Phase)
	}

	// ForceHot should not panic
	ms.ForceHot()

	// SaveHeads should return error
	err := ms.SaveHeads("/tmp/test_heads.bin")
	if err == nil {
		t.Error("expected error from SaveHeads with nil trainer")
	}
}

// TestMedusaSchedulerSlidingWindowSize verifies the sliding window only
// keeps numHeads entries at most.
func TestMedusaSchedulerSlidingWindowSize(t *testing.T) {
	numHeads := 3
	hiddenSize := 4
	vocabSize := 16
	mt := newMockTrainer(numHeads, hiddenSize, vocabSize)

	ms := &MedusaScheduler{
		trainer:       mt,
		medusaConfig:  MedusaConfig{NumHeads: numHeads},
		recentTokens:  make([]int, 0, numHeads+1),
		recentHiddens: make([][]float32, 0, numHeads+1),
		maxRecent:     numHeads + 1,
	}

	hidden := make([]float32, hiddenSize)

	// Add 10 samples
	for i := 0; i < 10; i++ {
		ms.collectTrainingSample(hidden, i)
	}

	// The sliding window should keep at most numHeads-1 entries in recentTokens
	// because each time we reach numHeads we emit a sample and slide off the oldest
	if len(ms.recentTokens) >= numHeads+1 {
		t.Errorf("recentTokens size = %d, should be bounded (< %d)",
			len(ms.recentTokens), numHeads+1)
	}

	// We should have emitted (10 - numHeads + 1) = 8 samples
	expectedSamples := 10 - numHeads + 1
	samples := mt.getSamples()
	if len(samples) != expectedSamples {
		t.Errorf("expected %d samples after 10 calls with numHeads=%d, got %d",
			expectedSamples, numHeads, len(samples))
	}
}

// TestMedusaSchedulerDefaultConfig verifies DefaultMedusaConfig returns sane values.
func TestMedusaSchedulerDefaultConfig(t *testing.T) {
	cfg := DefaultMedusaConfig()

	if cfg.NumHeads != 4 {
		t.Errorf("NumHeads = %d, want 4", cfg.NumHeads)
	}
	if !cfg.EnableOnlineTraining {
		t.Error("EnableOnlineTraining should default to true")
	}
	if cfg.TrainingConfig.BufferCapacity != 50000 {
		t.Errorf("BufferCapacity = %d, want 50000", cfg.TrainingConfig.BufferCapacity)
	}
}

// TestMedusaArgmaxFloat32 verifies the package-level argmax helper.
func TestMedusaArgmaxFloat32(t *testing.T) {
	tests := []struct {
		name   string
		values []float32
		want   int
	}{
		{"single", []float32{1.0}, 0},
		{"max_first", []float32{5.0, 2.0, 3.0}, 0},
		{"max_last", []float32{1.0, 2.0, 9.0}, 2},
		{"max_middle", []float32{1.0, 9.0, 3.0}, 1},
		{"negative", []float32{-3.0, -1.0, -2.0}, 1},
		{"empty", []float32{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := argmaxFloat32(tt.values)
			if got != tt.want {
				t.Errorf("argmaxFloat32(%v) = %d, want %d", tt.values, got, tt.want)
			}
		})
	}
}

// TestMedusaSchedulerDefaultConfigTree verifies tree-related config defaults.
func TestMedusaSchedulerDefaultConfigTree(t *testing.T) {
	cfg := DefaultMedusaConfig()

	if !cfg.UseTreeVerification {
		t.Error("UseTreeVerification should default to true")
	}
	if cfg.TreeTopK != 3 {
		t.Errorf("TreeTopK = %d, want 3", cfg.TreeTopK)
	}
	if cfg.TreeMaxNodes != 64 {
		t.Errorf("TreeMaxNodes = %d, want 64", cfg.TreeMaxNodes)
	}
}

// TestSelectBestTreePathFullAccept verifies path selection when the best path
// is fully accepted by the target model.
func TestSelectBestTreePathFullAccept(t *testing.T) {
	vocabSize := 8

	// Create 3 candidate paths sorted by confidence
	paths := []medusa.CandidatePath{
		{Tokens: []int{3, 1}, Confidence: 9.0},  // best
		{Tokens: []int{3, 5}, Confidence: 7.0},  // shares prefix
		{Tokens: []int{7, 1}, Confidence: 5.0},  // different start
	}

	// Verification logits: target confirms [3, 1] + bonus token 6
	// Position 0 logits confirm token 3, position 1 confirms token 1,
	// position 2 (bonus) predicts token 6.
	verifyLogits := make([]float32, 3*vocabSize)
	verifyLogits[0*vocabSize+3] = 10.0 // position 0 -> token 3
	verifyLogits[1*vocabSize+1] = 10.0 // position 1 -> token 1
	verifyLogits[2*vocabSize+6] = 10.0 // position 2 -> token 6 (bonus)

	bestIdx, accepted, finalToken := selectBestTreePath(paths, verifyLogits, vocabSize)

	if bestIdx != 0 {
		t.Errorf("bestIdx = %d, want 0", bestIdx)
	}
	if accepted != 2 {
		t.Errorf("accepted = %d, want 2", accepted)
	}
	if finalToken != 6 {
		t.Errorf("finalToken = %d, want 6 (bonus)", finalToken)
	}
}

// TestSelectBestTreePathFallback verifies that when the best path is rejected,
// an alternate path with the target's preferred token is selected.
func TestSelectBestTreePathFallback(t *testing.T) {
	vocabSize := 8

	// Best path is [3, 1], but target wants [7, ...] at position 0
	paths := []medusa.CandidatePath{
		{Tokens: []int{3, 1}, Confidence: 9.0}, // best confidence
		{Tokens: []int{7, 1}, Confidence: 5.0}, // alternate with token 7
		{Tokens: []int{7, 5}, Confidence: 4.0}, // another alternate with token 7
	}

	// Target prefers token 7 at position 0 (rejects 3)
	verifyLogits := make([]float32, 3*vocabSize)
	verifyLogits[0*vocabSize+7] = 10.0 // position 0 -> token 7 (not 3!)
	verifyLogits[1*vocabSize+1] = 10.0 // position 1 -> token 1
	verifyLogits[2*vocabSize+4] = 10.0 // bonus

	bestIdx, accepted, finalToken := selectBestTreePath(paths, verifyLogits, vocabSize)

	// Best path [3, 1] gets 0 accepted. Alternate [7, 1] should match position 0.
	// Note: verifyLogits at position 1 are conditioned on token 3 (not 7),
	// so we can only guarantee acceptance at the divergence point.
	// selectBestTreePath should pick the path with most accepted tokens.
	if accepted < 1 {
		t.Errorf("accepted = %d, want >= 1 (alternate should match at position 0)", accepted)
	}
	if bestIdx == 0 {
		t.Error("should not select path 0 (it was rejected at position 0)")
	}
	_ = finalToken
}

// TestSelectBestTreePathAllRejected verifies behavior when no path matches
// the target at position 0.
func TestSelectBestTreePathAllRejected(t *testing.T) {
	vocabSize := 8

	paths := []medusa.CandidatePath{
		{Tokens: []int{3, 1}, Confidence: 9.0},
		{Tokens: []int{7, 5}, Confidence: 5.0},
	}

	// Target wants token 2 at position 0 - no path has this
	verifyLogits := make([]float32, 2*vocabSize)
	verifyLogits[0*vocabSize+2] = 10.0 // token 2 at position 0

	bestIdx, accepted, finalToken := selectBestTreePath(paths, verifyLogits, vocabSize)

	if accepted != 0 {
		t.Errorf("accepted = %d, want 0 (no path matches)", accepted)
	}
	// When no tokens accepted, finalToken should be the target's preferred token
	if finalToken != 2 {
		t.Errorf("finalToken = %d, want 2 (target correction)", finalToken)
	}
	_ = bestIdx
}

// TestSelectBestTreePathEmpty verifies edge case with no paths.
func TestSelectBestTreePathEmpty(t *testing.T) {
	bestIdx, accepted, finalToken := selectBestTreePath(nil, nil, 8)
	if accepted != 0 {
		t.Errorf("accepted = %d, want 0", accepted)
	}
	if bestIdx != 0 {
		t.Errorf("bestIdx = %d, want 0", bestIdx)
	}
	_ = finalToken
}

// TestSelectBestTreePathPartialAccept verifies that when the best path is
// partially accepted and an alternate shares the accepted prefix but diverges
// later, the path with the most accepted tokens wins.
func TestSelectBestTreePathPartialAccept(t *testing.T) {
	vocabSize := 8

	// 3-token paths
	paths := []medusa.CandidatePath{
		{Tokens: []int{3, 1, 5}, Confidence: 12.0}, // best
		{Tokens: []int{3, 1, 2}, Confidence: 10.0}, // same prefix, different last
		{Tokens: []int{3, 4, 2}, Confidence: 8.0},  // diverges at position 1
	}

	// Target accepts [3, 1] but rejects 5, prefers 2 at position 2
	verifyLogits := make([]float32, 4*vocabSize)
	verifyLogits[0*vocabSize+3] = 10.0 // position 0 -> 3
	verifyLogits[1*vocabSize+1] = 10.0 // position 1 -> 1
	verifyLogits[2*vocabSize+2] = 10.0 // position 2 -> 2 (rejects 5)
	verifyLogits[3*vocabSize+0] = 10.0 // bonus

	bestIdx, accepted, finalToken := selectBestTreePath(paths, verifyLogits, vocabSize)

	// Path 1 [3, 1, 2] should be fully accepted (3 tokens)
	// since it shares prefix [3, 1] and has the target's preferred token 2
	if bestIdx != 1 {
		t.Errorf("bestIdx = %d, want 1 (path with token 2 at position 2)", bestIdx)
	}
	if accepted != 3 {
		t.Errorf("accepted = %d, want 3 (full path accepted)", accepted)
	}
	if finalToken != 0 {
		t.Errorf("finalToken = %d, want 0 (bonus token)", finalToken)
	}
}

// TestOnlineTrainerWithMedusaConfig verifies that the OnlineTrainer
// properly progresses through training phases.
func TestOnlineTrainerWithMedusaConfig(t *testing.T) {
	hiddenSize := 16
	vocabSize := 32

	config := medusa.OnlineConfig{
		NumHeads:       2,
		BufferCapacity: 50,
		WarmupSamples:  5,
		MinAccuracy:    0.1,
		BatchSize:      4,
		LearningRate:   0.01,
		TrainInterval:  5 * time.Millisecond,
		EvalInterval:   20 * time.Millisecond,
	}

	trainer := medusa.NewOnlineTrainer(hiddenSize, vocabSize, config)

	// Phase starts Cold
	if trainer.Phase() != medusa.PhaseCold {
		t.Fatalf("initial phase = %s, want cold", trainer.Phase())
	}

	// Add samples to trigger Warming
	for i := 0; i < 10; i++ {
		hidden := make([]float32, hiddenSize)
		for j := range hidden {
			hidden[j] = float32(i * j)
		}
		trainer.AddSample(hidden, []int{i % vocabSize, (i + 1) % vocabSize}, i)
	}

	if trainer.Phase() != medusa.PhaseWarming {
		t.Errorf("phase after 10 samples = %s, want warming", trainer.Phase())
	}

	// Start background training
	ctx, cancel := context.WithCancel(context.Background())
	trainer.Start(ctx)

	// Let it train
	time.Sleep(80 * time.Millisecond)

	metrics := trainer.Metrics()
	if metrics.TrainingSteps == 0 {
		t.Error("expected training steps > 0")
	}
	if metrics.SamplesCollected != 10 {
		t.Errorf("samples collected = %d, want 10", metrics.SamplesCollected)
	}

	cancel()
	trainer.Stop()
}
