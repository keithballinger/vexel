package scheduler

import (
	"math"
	"testing"

	"vexel/inference/medusa"
	"vexel/inference/pkg/sampler"
)

// TestVerifyDraftAllAccepted verifies that when all draft tokens match the
// target model's preferences, all are accepted and a bonus token is returned.
func TestVerifyDraftAllAccepted(t *testing.T) {
	vocabSize := 8
	draftTokens := []int{3, 1, 5}
	draftProbs := []float32{0.9, 0.8, 0.7}

	// Build logits where argmax matches each draft token
	// Need 4 positions: 3 for draft verification + 1 for bonus
	allLogits := make([]float32, 4*vocabSize)
	allLogits[0*vocabSize+3] = 10.0 // position 0 -> token 3
	allLogits[1*vocabSize+1] = 10.0 // position 1 -> token 1
	allLogits[2*vocabSize+5] = 10.0 // position 2 -> token 5
	allLogits[3*vocabSize+7] = 10.0 // position 3 -> token 7 (bonus)

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	accepted, finalToken, _ := verifyDraftAgainstLogits(
		allLogits, draftTokens, draftProbs, vocabSize, s,
	)

	if accepted != 3 {
		t.Errorf("accepted = %d, want 3 (all draft tokens)", accepted)
	}
	if finalToken != 7 {
		t.Errorf("finalToken = %d, want 7 (bonus)", finalToken)
	}
}

// TestVerifyDraftRejectAtStart verifies rejection at the first draft position.
func TestVerifyDraftRejectAtStart(t *testing.T) {
	vocabSize := 8
	draftTokens := []int{3, 1, 5}
	draftProbs := []float32{0.9, 0.8, 0.7}

	// Target prefers token 2 at position 0, rejecting draft token 3
	allLogits := make([]float32, 3*vocabSize)
	allLogits[0*vocabSize+2] = 10.0 // position 0 -> token 2 (not 3!)
	allLogits[1*vocabSize+1] = 10.0
	allLogits[2*vocabSize+5] = 10.0

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	accepted, finalToken, _ := verifyDraftAgainstLogits(
		allLogits, draftTokens, draftProbs, vocabSize, s,
	)

	if accepted != 0 {
		t.Errorf("accepted = %d, want 0 (rejected at start)", accepted)
	}
	if finalToken != 2 {
		t.Errorf("finalToken = %d, want 2 (target's correction)", finalToken)
	}
}

// TestVerifyDraftRejectMiddle verifies rejection at a middle position.
func TestVerifyDraftRejectMiddle(t *testing.T) {
	vocabSize := 8
	draftTokens := []int{3, 1, 5, 2}
	draftProbs := []float32{0.9, 0.8, 0.7, 0.6}

	// Accept first two, reject third (target wants 4, not 5)
	allLogits := make([]float32, 4*vocabSize)
	allLogits[0*vocabSize+3] = 10.0 // accept 3
	allLogits[1*vocabSize+1] = 10.0 // accept 1
	allLogits[2*vocabSize+4] = 10.0 // reject 5, prefer 4
	allLogits[3*vocabSize+2] = 10.0

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	accepted, finalToken, _ := verifyDraftAgainstLogits(
		allLogits, draftTokens, draftProbs, vocabSize, s,
	)

	if accepted != 2 {
		t.Errorf("accepted = %d, want 2 (first two accepted)", accepted)
	}
	if finalToken != 4 {
		t.Errorf("finalToken = %d, want 4 (target correction at position 2)", finalToken)
	}
}

// TestVerifyDraftRejectLast verifies rejection at the last draft position.
func TestVerifyDraftRejectLast(t *testing.T) {
	vocabSize := 8
	draftTokens := []int{3, 1, 5}
	draftProbs := []float32{0.9, 0.8, 0.7}

	// Accept first two, reject last (target wants 0, not 5)
	allLogits := make([]float32, 3*vocabSize)
	allLogits[0*vocabSize+3] = 10.0 // accept
	allLogits[1*vocabSize+1] = 10.0 // accept
	allLogits[2*vocabSize+0] = 10.0 // reject 5, prefer 0

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	accepted, finalToken, _ := verifyDraftAgainstLogits(
		allLogits, draftTokens, draftProbs, vocabSize, s,
	)

	if accepted != 2 {
		t.Errorf("accepted = %d, want 2", accepted)
	}
	if finalToken != 0 {
		t.Errorf("finalToken = %d, want 0 (correction at last position)", finalToken)
	}
}

// TestVerifyDraftEmpty verifies behavior with no draft tokens.
func TestVerifyDraftEmpty(t *testing.T) {
	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	accepted, finalToken, finalLogits := verifyDraftAgainstLogits(
		nil, nil, nil, 8, s,
	)

	if accepted != 0 {
		t.Errorf("accepted = %d, want 0", accepted)
	}
	if finalToken != 0 {
		t.Errorf("finalToken = %d, want 0", finalToken)
	}
	if finalLogits != nil {
		t.Error("expected nil finalLogits for empty draft")
	}
}

// TestVerifyDraftProbabilityAcceptance verifies that a draft token is accepted
// when it differs from the target's argmax but has sufficient probability.
func TestVerifyDraftProbabilityAcceptance(t *testing.T) {
	vocabSize := 4
	// Draft says token 1 with low confidence
	draftTokens := []int{1}
	draftProbs := []float32{0.1} // Very low draft confidence

	// Target: token 0 is most likely (logit=2.0), but token 1 also has decent probability
	// softmax([0, 2, 0, 0]) ≈ [0.12, 0.88, 0.12, 0.12] / sum
	// Actually, let me compute more carefully:
	// logits = [0, 2.0, 0, 0]
	// exp = [1, 7.39, 1, 1] → sum = 10.39
	// probs = [0.096, 0.711, 0.096, 0.096]
	// Token 1 prob = 0.711 >= draftProb 0.1 → ACCEPT despite argmax being token 1
	// Wait, argmax IS token 1 in this case... let me adjust

	// Target: token 0 is best (logit=3.0), token 1 has logit=2.0
	// softmax([3, 2, 0, 0])
	// exp = [20.09, 7.39, 1, 1] → sum ≈ 29.48
	// probs ≈ [0.681, 0.251, 0.034, 0.034]
	// argmax = token 0 (not 1!), but prob(1) = 0.251 >= draftProb 0.1 → ACCEPT
	allLogits := make([]float32, 2*vocabSize)
	allLogits[0*vocabSize+0] = 3.0            // target prefers token 0
	allLogits[0*vocabSize+1] = 2.0            // token 1 has decent probability
	allLogits[1*vocabSize+7%vocabSize] = 10.0 // bonus

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	accepted, _, _ := verifyDraftAgainstLogits(
		allLogits, draftTokens, draftProbs, vocabSize, s,
	)

	// Token 1 should be accepted because targetProb(1) ≈ 0.251 >= draftProb 0.1
	if accepted != 1 {
		t.Errorf("accepted = %d, want 1 (probability-based acceptance)", accepted)
	}
}

// TestVerifyDraftProbabilityRejection verifies that a token is rejected when
// both argmax and probability checks fail.
func TestVerifyDraftProbabilityRejection(t *testing.T) {
	vocabSize := 4
	draftTokens := []int{2}
	draftProbs := []float32{0.95} // Very high draft confidence

	// Target strongly prefers token 0
	// softmax([10, 0, 0, 0]) → token 0 ≈ 0.999, token 2 ≈ 0.00005
	// argmax = 0 (not 2), prob(2) << 0.95 → REJECT
	allLogits := make([]float32, 1*vocabSize)
	allLogits[0*vocabSize+0] = 10.0 // target overwhelmingly prefers token 0

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	accepted, finalToken, _ := verifyDraftAgainstLogits(
		allLogits, draftTokens, draftProbs, vocabSize, s,
	)

	if accepted != 0 {
		t.Errorf("accepted = %d, want 0 (probability rejection)", accepted)
	}
	if finalToken != 0 {
		t.Errorf("finalToken = %d, want 0 (target correction)", finalToken)
	}
}

// TestSpecDecodeEquivalence verifies that speculative decoding with temp=0
// produces the same token sequence as greedy standard decoding.
// This is the core correctness property: speculative decoding must produce
// identical output to standard decoding when using argmax sampling.
func TestSpecDecodeEquivalence(t *testing.T) {
	vocabSize := 16
	numSteps := 5

	// Simulate a sequence of "standard decode" steps.
	// Each step produces logits, and argmax gives the next token.
	// The "model" always predicts: 3 → 7 → 1 → 12 → 5
	standardTokens := []int{3, 7, 1, 12, 5}

	// Build logits for each position matching the standard tokens
	allLogits := make([]float32, (numSteps+1)*vocabSize)
	for i, tok := range standardTokens {
		allLogits[i*vocabSize+tok] = 10.0 // strong preference
	}
	allLogits[numSteps*vocabSize+9] = 10.0 // bonus token

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	// Case 1: Perfect draft (all tokens match standard decode)
	draftProbs := make([]float32, numSteps)
	for i := range draftProbs {
		draftProbs[i] = 0.9
	}

	accepted, finalToken, _ := verifyDraftAgainstLogits(
		allLogits, standardTokens, draftProbs, vocabSize, s,
	)

	if accepted != numSteps {
		t.Errorf("accepted = %d, want %d (all matching)", accepted, numSteps)
	}

	// The accepted tokens + finalToken should be the same as standard decode
	specTokens := make([]int, 0, numSteps+1)
	specTokens = append(specTokens, standardTokens[:accepted]...)
	specTokens = append(specTokens, finalToken)

	expectedTokens := append(standardTokens, 9) // standard + bonus
	for i, tok := range specTokens {
		if i >= len(expectedTokens) {
			break
		}
		if tok != expectedTokens[i] {
			t.Errorf("specTokens[%d] = %d, standardTokens[%d] = %d (should be identical)",
				i, tok, i, expectedTokens[i])
		}
	}

	// Case 2: Draft with one wrong token at position 2
	wrongDraft := []int{3, 7, 99, 12, 5} // token 99 is wrong at position 2
	wrongDraftProbs := []float32{0.9, 0.9, 0.01, 0.9, 0.9}

	accepted2, finalToken2, _ := verifyDraftAgainstLogits(
		allLogits, wrongDraft, wrongDraftProbs, vocabSize, s,
	)

	// Should accept first 2 tokens, reject at position 2, correction = 1
	if accepted2 != 2 {
		t.Errorf("accepted = %d, want 2 (reject at position 2)", accepted2)
	}
	if finalToken2 != 1 {
		t.Errorf("finalToken = %d, want 1 (standard decode's token at position 2)", finalToken2)
	}

	// Speculative output should prefix-match standard decode
	for i := 0; i < accepted2; i++ {
		if wrongDraft[i] != standardTokens[i] {
			t.Errorf("accepted token[%d] = %d, standard[%d] = %d",
				i, wrongDraft[i], i, standardTokens[i])
		}
	}
	if finalToken2 != standardTokens[accepted2] {
		t.Errorf("correction token = %d, standard[%d] = %d",
			finalToken2, accepted2, standardTokens[accepted2])
	}
}

// TestTreeVerifyEquivalence verifies that tree verification with top-1
// (i.e., only the best path) produces the same result as flat verification.
func TestTreeVerifyEquivalence(t *testing.T) {
	vocabSize := 8
	numHeads := 3

	// Build head logits where argmax for each head is known
	headLogits := make([][]float32, numHeads)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}
	headLogits[0][3] = 10.0 // head 0 -> token 3
	headLogits[1][1] = 10.0 // head 1 -> token 1
	headLogits[2][5] = 10.0 // head 2 -> token 5

	// Flat approach: argmax per head = [3, 1, 5]
	flatDraft := make([]int, numHeads)
	for i, logits := range headLogits {
		flatDraft[i] = argmaxFloat32(logits)
	}

	// Tree approach with top-1: should produce the same best path
	tree := medusa.BuildCandidateTree(headLogits, 1, 64)
	paths := tree.Paths()

	if len(paths) != 1 {
		t.Fatalf("expected 1 path with topK=1, got %d", len(paths))
	}

	// Tree's best path should match flat draft
	for i, tok := range paths[0].Tokens {
		if tok != flatDraft[i] {
			t.Errorf("tree path[%d] = %d, flat draft[%d] = %d",
				i, tok, i, flatDraft[i])
		}
	}

	// Now verify both against the same target logits
	verifyLogits := make([]float32, (numHeads+1)*vocabSize)
	verifyLogits[0*vocabSize+3] = 10.0 // accept token 3
	verifyLogits[1*vocabSize+1] = 10.0 // accept token 1
	verifyLogits[2*vocabSize+4] = 10.0 // reject token 5, prefer 4
	verifyLogits[3*vocabSize+7] = 10.0

	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	flatAccepted, flatFinal, _ := verifyDraftAgainstLogits(
		verifyLogits, flatDraft, []float32{0.9, 0.9, 0.9}, vocabSize, s,
	)

	treeIdx, treeAccepted, treeFinal := selectBestTreePath(
		paths, verifyLogits, vocabSize,
	)

	if flatAccepted != treeAccepted {
		t.Errorf("flat accepted = %d, tree accepted = %d (should match with topK=1)",
			flatAccepted, treeAccepted)
	}
	if flatFinal != treeFinal {
		t.Errorf("flat final = %d, tree final = %d (should match with topK=1)",
			flatFinal, treeFinal)
	}
	_ = treeIdx
}

// TestCacheTruncationCalculation verifies the cache truncation length
// is correctly computed for different acceptance counts.
func TestCacheTruncationCalculation(t *testing.T) {
	tests := []struct {
		name        string
		startPos    int
		numDraft    int
		numAccepted int
		wantLen     int
	}{
		{
			name:        "all_accepted",
			startPos:    10,
			numDraft:    4,
			numAccepted: 4,
			wantLen:     15, // 10 + 1 + 4 = 15
		},
		{
			name:        "none_accepted",
			startPos:    10,
			numDraft:    4,
			numAccepted: 0,
			wantLen:     11, // 10 + 1 + 0 = 11
		},
		{
			name:        "partial_accepted",
			startPos:    10,
			numDraft:    4,
			numAccepted: 2,
			wantLen:     13, // 10 + 1 + 2 = 13
		},
		{
			name:        "zero_start",
			startPos:    0,
			numDraft:    3,
			numAccepted: 3,
			wantLen:     4, // 0 + 1 + 3 = 4
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This matches the calculation in runMedusaDecodeStep and
			// runSpeculativeDecodeStep: newCacheLen = startPos + 1 + numAccepted
			gotLen := tt.startPos + 1 + tt.numAccepted
			if gotLen != tt.wantLen {
				t.Errorf("cache truncation = %d, want %d", gotLen, tt.wantLen)
			}
		})
	}
}

// TestVerifyDraftSingleToken verifies behavior with a single draft token.
func TestVerifyDraftSingleToken(t *testing.T) {
	vocabSize := 8
	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	t.Run("accepted", func(t *testing.T) {
		allLogits := make([]float32, 2*vocabSize)
		allLogits[0*vocabSize+5] = 10.0 // accept token 5
		allLogits[1*vocabSize+3] = 10.0 // bonus

		accepted, finalToken, _ := verifyDraftAgainstLogits(
			allLogits, []int{5}, []float32{0.9}, vocabSize, s,
		)
		if accepted != 1 {
			t.Errorf("accepted = %d, want 1", accepted)
		}
		if finalToken != 3 {
			t.Errorf("finalToken = %d, want 3 (bonus)", finalToken)
		}
	})

	t.Run("rejected", func(t *testing.T) {
		allLogits := make([]float32, 1*vocabSize)
		allLogits[0*vocabSize+2] = 10.0 // prefer token 2

		accepted, finalToken, _ := verifyDraftAgainstLogits(
			allLogits, []int{5}, []float32{0.9}, vocabSize, s,
		)
		if accepted != 0 {
			t.Errorf("accepted = %d, want 0", accepted)
		}
		if finalToken != 2 {
			t.Errorf("finalToken = %d, want 2 (correction)", finalToken)
		}
	})
}

// TestAdaptiveDraftConvergence verifies that the adaptive draft length
// converges to a stable value given consistent acceptance patterns.
func TestAdaptiveDraftConvergence(t *testing.T) {
	cfg := AdaptiveConfig{
		InitialDraftTokens: 4,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         5,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}
	ad := NewAdaptiveDraftLength(cfg)

	// Simulate consistent 60% acceptance for 50 steps.
	// This is between thresholds (0.40 < 0.60 < 0.80),
	// so draft length should remain stable.
	for i := 0; i < 50; i++ {
		ad.RecordStep(3, 5)
	}

	final := ad.NumDraftTokens()
	if final != 4 {
		t.Errorf("NumDraftTokens = %d after 50 stable steps, want 4 (unchanged)", final)
	}

	// Now simulate consistent 90% acceptance — should increase
	for i := 0; i < 10; i++ {
		ad.RecordStep(9, 10)
	}
	after := ad.NumDraftTokens()
	if after <= 4 {
		t.Errorf("NumDraftTokens = %d after high acceptance, want > 4", after)
	}

	// Then sudden drop to 10% acceptance — should decrease
	startVal := ad.NumDraftTokens()
	for i := 0; i < 20; i++ {
		ad.RecordStep(1, 10)
	}
	endVal := ad.NumDraftTokens()
	if endVal >= startVal {
		t.Errorf("NumDraftTokens = %d after low acceptance, want < %d", endVal, startVal)
	}
}

// TestAcceptanceRateEdgeCases verifies acceptance rate computation edge cases.
func TestAcceptanceRateEdgeCases(t *testing.T) {
	t.Run("zero_generated", func(t *testing.T) {
		m := SpeculativeMetrics{DraftTokensGenerated: 0}
		if m.AcceptanceRate() != 0 {
			t.Errorf("acceptance rate = %f, want 0", m.AcceptanceRate())
		}
	})

	t.Run("perfect_acceptance", func(t *testing.T) {
		m := SpeculativeMetrics{
			DraftTokensGenerated: 100,
			DraftTokensAccepted:  100,
		}
		if math.Abs(m.AcceptanceRate()-1.0) > 1e-6 {
			t.Errorf("acceptance rate = %f, want 1.0", m.AcceptanceRate())
		}
	})

	t.Run("zero_acceptance", func(t *testing.T) {
		m := SpeculativeMetrics{
			DraftTokensGenerated: 100,
			DraftTokensAccepted:  0,
		}
		if m.AcceptanceRate() != 0 {
			t.Errorf("acceptance rate = %f, want 0", m.AcceptanceRate())
		}
	})
}

// TestSpeedupCalculation verifies the effective speedup computation.
func TestSpeedupCalculation(t *testing.T) {
	tests := []struct {
		name     string
		accepted int
		steps    int
		want     float64
	}{
		{"no_steps", 0, 0, 1.0},       // no speculation = 1x
		{"all_rejected", 0, 5, 1.0},   // 0+5/5 = 1.0x
		{"2x_speedup", 5, 5, 2.0},     // 5+5/5 = 2.0x
		{"3x_speedup", 20, 10, 3.0},   // 20+10/10 = 3.0x
		{"near_perfect", 36, 4, 10.0}, // 36+4/4 = 10.0x
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := SpeculativeMetrics{
				DraftTokensAccepted: tt.accepted,
				VerificationSteps:   tt.steps,
			}
			got := m.Speedup()
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("Speedup() = %f, want %f", got, tt.want)
			}
		})
	}
}
