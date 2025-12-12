package medusa

import (
	"bytes"
	"context"
	"math/rand"
	"testing"
	"time"
)

func TestHeadsForward(t *testing.T) {
	numHeads := 4
	hiddenSize := 64
	vocabSize := 100

	heads := NewHeads(numHeads, hiddenSize, vocabSize)

	// Create random hidden state
	hidden := make([]float32, hiddenSize)
	for i := range hidden {
		hidden[i] = rand.Float32()*2 - 1
	}

	// Test single head forward
	logits := heads.Forward(0, hidden)
	if logits == nil {
		t.Fatal("Forward returned nil")
	}
	if len(logits) != vocabSize {
		t.Errorf("Expected %d logits, got %d", vocabSize, len(logits))
	}

	// Test ForwardAll
	allLogits := heads.ForwardAll(hidden)
	if len(allLogits) != numHeads {
		t.Errorf("Expected %d heads, got %d", numHeads, len(allLogits))
	}
	for i, l := range allLogits {
		if len(l) != vocabSize {
			t.Errorf("Head %d: expected %d logits, got %d", i, vocabSize, len(l))
		}
	}

	// Test ForwardTopK
	topK := heads.ForwardTopK(hidden, 5)
	if len(topK) != numHeads {
		t.Errorf("Expected %d heads in topK, got %d", numHeads, len(topK))
	}
	for i, tokens := range topK {
		if len(tokens) != 5 {
			t.Errorf("Head %d: expected 5 top tokens, got %d", i, len(tokens))
		}
	}
}

func TestHeadsSaveLoad(t *testing.T) {
	numHeads := 2
	hiddenSize := 32
	vocabSize := 50

	original := NewHeads(numHeads, hiddenSize, vocabSize)

	// Create random hidden state for testing
	hidden := make([]float32, hiddenSize)
	for i := range hidden {
		hidden[i] = rand.Float32()
	}

	// Get original predictions
	origLogits := original.ForwardAll(hidden)

	// Save to buffer
	var buf bytes.Buffer
	if err := original.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	// Load from buffer
	loaded, err := ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}

	// Verify dimensions
	if loaded.NumHeads != original.NumHeads {
		t.Errorf("NumHeads mismatch: %d vs %d", loaded.NumHeads, original.NumHeads)
	}
	if loaded.HiddenSize != original.HiddenSize {
		t.Errorf("HiddenSize mismatch: %d vs %d", loaded.HiddenSize, original.HiddenSize)
	}
	if loaded.VocabSize != original.VocabSize {
		t.Errorf("VocabSize mismatch: %d vs %d", loaded.VocabSize, original.VocabSize)
	}

	// Get loaded predictions
	loadedLogits := loaded.ForwardAll(hidden)

	// Verify predictions match
	for h := 0; h < numHeads; h++ {
		for i := 0; i < vocabSize; i++ {
			if origLogits[h][i] != loadedLogits[h][i] {
				t.Errorf("Head %d logit %d mismatch: %.6f vs %.6f",
					h, i, origLogits[h][i], loadedLogits[h][i])
				break
			}
		}
	}
}

func TestRingBuffer(t *testing.T) {
	capacity := 10
	rb := NewRingBuffer(capacity)

	// Add samples
	for i := 0; i < 5; i++ {
		rb.Add(TrainingSample{
			HiddenState:  []float32{float32(i)},
			FutureTokens: []int{i + 1, i + 2},
			Position:     i,
		})
	}

	if rb.Count() != 5 {
		t.Errorf("Expected count 5, got %d", rb.Count())
	}

	// Fill and overflow
	for i := 5; i < 15; i++ {
		rb.Add(TrainingSample{
			HiddenState:  []float32{float32(i)},
			FutureTokens: []int{i + 1},
			Position:     i,
		})
	}

	if rb.Count() != capacity {
		t.Errorf("Expected count %d after overflow, got %d", capacity, rb.Count())
	}

	if !rb.IsFull() {
		t.Error("Buffer should be full")
	}

	// Sample from buffer
	rng := rand.New(rand.NewSource(42))
	batch := rb.Sample(3, rng.Intn)
	if len(batch) != 3 {
		t.Errorf("Expected batch size 3, got %d", len(batch))
	}
}

func TestOnlineTrainer(t *testing.T) {
	hiddenSize := 32
	vocabSize := 50

	config := OnlineConfig{
		NumHeads:       2,
		BufferCapacity: 100,
		WarmupSamples:  10,
		MinAccuracy:    0.1, // Low threshold for testing
		BatchSize:      4,
		LearningRate:   0.01,
		TrainInterval:  10 * time.Millisecond,
		EvalInterval:   50 * time.Millisecond,
	}

	trainer := NewOnlineTrainer(hiddenSize, vocabSize, config)

	// Verify initial phase
	if trainer.Phase() != PhaseCold {
		t.Errorf("Expected phase Cold, got %s", trainer.Phase())
	}

	// Add samples
	for i := 0; i < 15; i++ {
		hidden := make([]float32, hiddenSize)
		for j := range hidden {
			hidden[j] = rand.Float32()
		}
		trainer.AddSample(hidden, []int{i % vocabSize, (i + 1) % vocabSize}, i)
	}

	// Should transition to Warming after warmup samples
	if trainer.Phase() != PhaseWarming {
		t.Errorf("Expected phase Warming after %d samples, got %s",
			config.WarmupSamples+5, trainer.Phase())
	}

	// Start trainer
	ctx, cancel := context.WithCancel(context.Background())
	trainer.Start(ctx)

	// Let it train for a bit
	time.Sleep(100 * time.Millisecond)

	// Check metrics
	metrics := trainer.Metrics()
	if metrics.TrainingSteps == 0 {
		t.Error("Expected some training steps")
	}

	// Stop trainer
	cancel()
	trainer.Stop()
}

func TestHeadsUpdateWeights(t *testing.T) {
	numHeads := 1
	hiddenSize := 4
	vocabSize := 8

	heads := NewHeads(numHeads, hiddenSize, vocabSize)

	// Get original FC1 value
	origFC1_0 := heads.heads[0].FC1[0]

	// Create gradient
	grad1 := make([]float32, hiddenSize*hiddenSize)
	grad2 := make([]float32, hiddenSize*vocabSize)
	grad1[0] = 1.0 // Non-zero gradient

	// Update weights
	lr := float32(0.1)
	heads.UpdateWeights(0, grad1, grad2, lr)

	// Verify weight changed
	newFC1_0 := heads.heads[0].FC1[0]
	expected := origFC1_0 - lr*grad1[0]
	if newFC1_0 != expected {
		t.Errorf("Expected FC1[0] = %.4f, got %.4f", expected, newFC1_0)
	}
}

func TestHeadsClone(t *testing.T) {
	numHeads := 2
	hiddenSize := 16
	vocabSize := 32

	original := NewHeads(numHeads, hiddenSize, vocabSize)
	clone := original.Clone()

	// Verify clone is independent
	original.heads[0].FC1[0] = 999.0

	if clone.heads[0].FC1[0] == 999.0 {
		t.Error("Clone should be independent of original")
	}
}

func TestMemorySize(t *testing.T) {
	numHeads := 4
	hiddenSize := 2048
	vocabSize := 32000

	heads := NewHeads(numHeads, hiddenSize, vocabSize)
	memSize := heads.MemorySize()

	// Expected: numHeads * (hiddenSize*hiddenSize + hiddenSize*vocabSize) * 4 bytes
	expected := int64(numHeads) * int64(hiddenSize*hiddenSize+hiddenSize*vocabSize) * 4
	if memSize != expected {
		t.Errorf("Expected memory size %d, got %d", expected, memSize)
	}

	// For reference: 4 heads with 2048 hidden, 32000 vocab ≈ 1GB
	t.Logf("Memory size for 4 heads (2048x32000): %.1f MB", float64(memSize)/(1024*1024))
}
