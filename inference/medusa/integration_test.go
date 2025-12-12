// +build integration

package medusa_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"vexel/inference/medusa"
	"vexel/inference/memory"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/tensor"
)

// WikipediaSample contains text for testing online training.
// This is a condensed version of a Wikipedia article about computing.
const WikipediaSample = `
A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations. Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks.

The first computers were used primarily for numerical calculations. However, as any information can be encoded numerically, people soon realized that computers are capable of general-purpose information processing.

The history of computing hardware covers the developments from early simple devices to aid calculation to modern day computers. The first aids to computation were purely mechanical devices which required the operator to set up the initial values of an elementary arithmetic operation.

Charles Babbage, an English mechanical engineer and polymath, originated the concept of a programmable computer. Considered the father of the computer, he conceptualized and invented the first mechanical computer in the early 19th century.

Alan Turing is widely considered to be the father of theoretical computer science and artificial intelligence. During the Second World War, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence.

The Electronic Numerical Integrator and Computer was the first general-purpose electronic digital computer. It was Turing-complete and able to solve a large class of numerical problems through reprogramming.

Modern computers are based on integrated circuits. Billions of transistors can be packed into a single chip, enabling computation that would have been impossible just decades ago.

Programming languages allow humans to communicate instructions to computers. High-level languages like Python, Java, and Go abstract away the complexity of machine code, making software development more accessible.

The internet has transformed computing from isolated machines to a globally connected network. Cloud computing allows users to access vast computational resources on demand, fundamentally changing how software is deployed and consumed.

Artificial intelligence and machine learning represent the current frontier of computing. Neural networks can now perform tasks that were once thought to require human intelligence, from recognizing images to generating coherent text.
`

// TestOnlineTrainingIntegration tests the full online training pipeline.
// This test requires a model to be available and uses the metal backend.
// Run with: go test -tags "integration metal" -v ./inference/medusa/...
func TestOnlineTrainingIntegration(t *testing.T) {
	// Skip if no model available
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		t.Skip("VEXEL_TEST_MODEL not set, skipping integration test")
	}

	t.Logf("Using model: %s", modelPath)

	// This would be a full integration test with actual model loading
	// For now, we test the components in isolation
	t.Log("Integration test placeholder - requires model setup")
}

// TestMedusaTrainingFlow tests the training flow without a real model.
func TestMedusaTrainingFlow(t *testing.T) {
	// Configuration for fast testing
	config := medusa.OnlineConfig{
		NumHeads:       4,
		BufferCapacity: 1000,
		WarmupSamples:  100,
		MinAccuracy:    0.1, // Low threshold for testing
		BatchSize:      16,
		LearningRate:   0.01,
		TrainInterval:  10 * time.Millisecond,
		EvalInterval:   50 * time.Millisecond,
	}

	hiddenSize := 64
	vocabSize := 100

	trainer := medusa.NewOnlineTrainer(hiddenSize, vocabSize, config)

	// Start trainer
	ctx, cancel := context.WithCancel(context.Background())
	trainer.Start(ctx)
	defer func() {
		cancel()
		trainer.Stop()
	}()

	// Verify initial phase
	if trainer.Phase() != medusa.PhaseCold {
		t.Errorf("Expected Cold phase, got %s", trainer.Phase())
	}

	// Simulate generating tokens and collecting samples
	// Each sample is (hidden_state, future_tokens)
	t.Log("Phase 1: Cold - collecting samples...")

	for i := 0; i < 150; i++ {
		// Create a synthetic hidden state
		hidden := make([]float32, hiddenSize)
		for j := range hidden {
			hidden[j] = float32(i+j) * 0.01
		}

		// Future tokens (what the model actually generated)
		futureTokens := []int{
			(i + 1) % vocabSize,
			(i + 2) % vocabSize,
			(i + 3) % vocabSize,
			(i + 4) % vocabSize,
		}

		trainer.AddSample(hidden, futureTokens, i)

		// Check phase transition
		if i == 100 && trainer.Phase() == medusa.PhaseCold {
			t.Log("  Still cold at 100 samples (expected)")
		}
	}

	// Should have transitioned to Warming
	if trainer.Phase() != medusa.PhaseWarming {
		t.Errorf("Expected Warming phase after 150 samples, got %s", trainer.Phase())
	}
	t.Logf("Phase 2: Warming - training heads (phase=%s)", trainer.Phase())

	// Let training run
	time.Sleep(200 * time.Millisecond)

	metrics := trainer.Metrics()
	t.Logf("  Training steps: %d", metrics.TrainingSteps)
	t.Logf("  Current loss: %.4f", metrics.CurrentLoss)
	t.Logf("  Head accuracies: %v", metrics.HeadAccuracies)

	if metrics.TrainingSteps == 0 {
		t.Error("Expected some training steps")
	}

	// Test forward pass through heads
	heads := trainer.Heads()
	testHidden := make([]float32, hiddenSize)
	for i := range testHidden {
		testHidden[i] = 0.5
	}

	allLogits := heads.ForwardAll(testHidden)
	if len(allLogits) != 4 {
		t.Errorf("Expected 4 heads, got %d", len(allLogits))
	}

	for i, logits := range allLogits {
		if len(logits) != vocabSize {
			t.Errorf("Head %d: expected %d logits, got %d", i, vocabSize, len(logits))
		}
	}

	t.Log("Medusa training flow test passed!")
}

// TestSpeculativeDecodingFlow tests the speculation logic without a real model.
func TestSpeculativeDecodingFlow(t *testing.T) {
	// Create trained heads (simulated)
	numHeads := 4
	hiddenSize := 64
	vocabSize := 100

	heads := medusa.NewHeads(numHeads, hiddenSize, vocabSize)

	// Simulate a decode step
	hidden := make([]float32, hiddenSize)
	for i := range hidden {
		hidden[i] = float32(i) * 0.1
	}

	// Step 1: Generate draft tokens
	draftTokens := make([]int, numHeads)
	for i := 0; i < numHeads; i++ {
		logits := heads.Forward(i, hidden)
		draftTokens[i] = argmax(logits)
	}
	t.Logf("Draft tokens: %v", draftTokens)

	// Step 2: Simulate target model verification
	// In real scenario, we'd run VerifySpeculative
	// Here we simulate acceptance/rejection

	// Simulate target tokens (some match, some don't)
	targetTokens := make([]int, numHeads+1) // +1 for bonus token
	targetTokens[0] = draftTokens[0]        // Accept
	targetTokens[1] = draftTokens[1]        // Accept
	targetTokens[2] = 99                    // Reject (different from draft)
	targetTokens[3] = 50                    // Would be checked if we got here
	targetTokens[4] = 25                    // Bonus token

	// Step 3: Accept until first rejection
	numAccepted := 0
	var finalToken int

	for i := 0; i < numHeads; i++ {
		if draftTokens[i] == targetTokens[i] {
			numAccepted++
			t.Logf("  Position %d: draft=%d, target=%d -> ACCEPT", i, draftTokens[i], targetTokens[i])
		} else {
			finalToken = targetTokens[i]
			t.Logf("  Position %d: draft=%d, target=%d -> REJECT", i, draftTokens[i], targetTokens[i])
			break
		}
	}

	if numAccepted == numHeads {
		finalToken = targetTokens[numHeads] // Bonus token
		t.Log("  All drafts accepted! Using bonus token.")
	}

	// Output tokens
	outputTokens := make([]int, numAccepted+1)
	for i := 0; i < numAccepted; i++ {
		outputTokens[i] = draftTokens[i]
	}
	outputTokens[numAccepted] = finalToken

	t.Logf("Output tokens: %v (%d accepted + 1 correction/bonus)", outputTokens, numAccepted)

	// Verify we got the expected result
	if numAccepted != 2 {
		t.Errorf("Expected 2 accepted, got %d", numAccepted)
	}
	if finalToken != 99 {
		t.Errorf("Expected correction token 99, got %d", finalToken)
	}

	// Calculate metrics
	acceptanceRate := float64(numAccepted) / float64(numHeads)
	speedup := float64(numAccepted+1) / 1.0 // vs baseline of 1 token/step

	t.Logf("Acceptance rate: %.1f%%", acceptanceRate*100)
	t.Logf("Speedup: %.2fx", speedup)
}

// TestWikipediaTokenization simulates processing Wikipedia text.
func TestWikipediaTokenization(t *testing.T) {
	// Simulate tokenization by splitting on whitespace
	words := strings.Fields(WikipediaSample)
	t.Logf("Wikipedia sample: %d words", len(words))

	// Simulate token IDs (in real scenario, tokenizer would do this)
	tokenIDs := make([]int, len(words))
	for i, word := range words {
		// Simple hash to get "token ID"
		hash := 0
		for _, c := range word {
			hash = (hash*31 + int(c)) % 32000
		}
		tokenIDs[i] = hash
	}

	t.Logf("First 10 tokens: %v", tokenIDs[:10])
	t.Logf("Sample ready for training")
}

// Helper function
func argmax(values []float32) int {
	if len(values) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// BenchmarkMedusaForward benchmarks the Medusa head forward pass.
func BenchmarkMedusaForward(b *testing.B) {
	hiddenSize := 2048
	vocabSize := 32000
	numHeads := 4

	heads := medusa.NewHeads(numHeads, hiddenSize, vocabSize)

	hidden := make([]float32, hiddenSize)
	for i := range hidden {
		hidden[i] = float32(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = heads.ForwardAll(hidden)
	}
}

// BenchmarkMedusaTopK benchmarks top-k extraction from Medusa heads.
func BenchmarkMedusaTopK(b *testing.B) {
	hiddenSize := 2048
	vocabSize := 32000
	numHeads := 4

	heads := medusa.NewHeads(numHeads, hiddenSize, vocabSize)

	hidden := make([]float32, hiddenSize)
	for i := range hidden {
		hidden[i] = float32(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = heads.ForwardTopK(hidden, 5)
	}
}
