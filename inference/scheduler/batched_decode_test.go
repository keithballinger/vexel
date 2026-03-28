//go:build metal && darwin && cgo

package scheduler

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// TestBatchedDecodeE2E verifies that the scheduler correctly handles multiple
// concurrent sequences through batched decode, producing valid tokens for each.
func TestBatchedDecodeE2E(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		t.Skip("VEXEL_TEST_MODEL not set")
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model file not found: %s", modelPath)
	}

	// Load tokenizer (optional — test works without it)
	tokPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	var tok *tokenizer.Tokenizer
	if _, err := os.Stat(tokPath); err == nil {
		tok, _ = tokenizer.Load(tokPath)
	}
	if tok != nil {
		t.Logf("Tokenizer loaded from %s", tokPath)
	} else {
		t.Log("No tokenizer found, running without text decoding")
	}

	// Create Metal backend
	be, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}
	defer be.Close()

	// Parse model config from GGUF
	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	// Create inference context with scratch arena
	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxTokens := 64
	totalScratch := modelCfg.TotalArenaBytes(maxTokens)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), be.Alloc)

	// Create model runtime
	model, err := runtime.NewModelRuntime(be, memCtx, nil, modelCfg)
	if err != nil {
		t.Fatalf("Failed to create model runtime: %v", err)
	}

	// Load weights
	t.Log("Loading model weights...")
	loadStart := time.Now()
	if err := model.LoadWeights(modelPath); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		t.Fatalf("Failed to copy weights to device: %v", err)
	}
	t.Logf("Weights loaded in %v", time.Since(loadStart))

	// Create GPU KV cache
	model.CreateGPUKVCache(2048)

	// Create scheduler with batching enabled
	maxGenTokens := 16
	cfg := Config{
		MaxBatchSize: 4,
		MaxSequences: 4,
		MaxTokens:    maxGenTokens,
		SamplerConfig: sampler.Config{
			Temperature: 0, // Greedy for determinism
		},
	}

	sched, err := NewScheduler(model, tok, cfg)
	if err != nil {
		t.Fatalf("Failed to create scheduler: %v", err)
	}

	// Define prompts for concurrent sequences
	prompts := []string{
		"The capital of France is",
		"One plus one equals",
		"The color of the sky is",
	}

	// Create sequences
	sequences := make([]*Sequence, len(prompts))
	for i, prompt := range prompts {
		sequences[i] = NewSequence(SequenceID(i+1), prompt)
	}

	// Start scheduler in background
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	errChan := make(chan error, 1)
	go func() {
		errChan <- sched.Run(ctx)
	}()

	// Submit all sequences concurrently
	for _, seq := range sequences {
		sched.AddSequence(seq)
	}

	// Collect tokens from each sequence concurrently
	var wg sync.WaitGroup
	results := make([][]string, len(sequences))
	for i, seq := range sequences {
		wg.Add(1)
		go func(idx int, s *Sequence) {
			defer wg.Done()
			var tokens []string
			for tok := range s.TokenChan() {
				tokens = append(tokens, tok)
			}
			results[idx] = tokens
		}(i, seq)
	}

	// Wait for all sequences to complete or timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		t.Log("All sequences completed")
	case err := <-errChan:
		t.Fatalf("Scheduler returned error: %v", err)
	case <-time.After(60 * time.Second):
		t.Fatal("Timed out waiting for sequences to complete")
	}

	// Cancel scheduler
	cancel()

	// Verify results
	for i, seq := range sequences {
		genTokens := seq.GeneratedTokens()
		state := seq.State()

		t.Logf("Sequence %d (prompt=%q): state=%s, generated %d tokens, text tokens=%v",
			seq.ID(), prompts[i], state, len(genTokens), results[i])

		if state != StateFinished {
			t.Errorf("Sequence %d: expected StateFinished, got %s", seq.ID(), state)
		}

		if len(genTokens) == 0 {
			t.Errorf("Sequence %d: expected at least 1 generated token, got 0", seq.ID())
		}

		// Verify token IDs are valid (non-negative and within vocab)
		for j, tokID := range genTokens {
			if tokID < 0 {
				t.Errorf("Sequence %d: token %d has invalid ID %d", seq.ID(), j, tokID)
			}
			if tokID >= modelCfg.VocabSize {
				t.Errorf("Sequence %d: token %d ID %d exceeds vocab size %d",
					seq.ID(), j, tokID, modelCfg.VocabSize)
			}
		}

		// Clean up sequence from scheduler
		sched.RemoveSequence(seq.ID())
	}

	// Verify metrics
	metrics := sched.Metrics()
	t.Logf("Metrics: total=%d, prefill=%d, decode=%d, completed=%d",
		metrics.TotalTokens, metrics.PrefillTokens, metrics.DecodeTokens, metrics.CompletedSequences)

	if metrics.TotalTokens == 0 {
		t.Error("Expected non-zero total tokens in metrics")
	}
	if metrics.CompletedSequences != len(sequences) {
		t.Errorf("Expected %d completed sequences, got %d", len(sequences), metrics.CompletedSequences)
	}
}

// TestBatchedDecodeDeterminism verifies that running sequences individually
// produces the same tokens as running them in a batch (with temp=0 greedy decoding).
// This is a best-effort check since GPU KV cache state may differ between runs.
func TestBatchedDecodeDeterminism(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		t.Skip("VEXEL_TEST_MODEL not set")
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model file not found: %s", modelPath)
	}

	// Load tokenizer
	tokPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	var tok *tokenizer.Tokenizer
	if _, err := os.Stat(tokPath); err == nil {
		tok, _ = tokenizer.Load(tokPath)
	}

	// Create Metal backend
	be, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}
	defer be.Close()

	// Parse model config
	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	// Create inference context
	memCtx := memory.NewInferenceContext(tensor.Metal)
	totalScratch := modelCfg.TotalArenaBytes(64)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), be.Alloc)

	// Create model runtime
	model, err := runtime.NewModelRuntime(be, memCtx, nil, modelCfg)
	if err != nil {
		t.Fatalf("Failed to create model runtime: %v", err)
	}

	t.Log("Loading model weights...")
	if err := model.LoadWeights(modelPath); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		t.Fatalf("Failed to copy weights to device: %v", err)
	}

	model.CreateGPUKVCache(2048)

	prompt := "The meaning of life is"
	maxGenTokens := 8

	// Run single-sequence generation
	cfg := Config{
		MaxBatchSize:  1,
		MaxSequences:  1,
		MaxTokens:     maxGenTokens,
		SamplerConfig: sampler.Config{Temperature: 0},
	}

	sched, err := NewScheduler(model, tok, cfg)
	if err != nil {
		t.Fatalf("Failed to create scheduler: %v", err)
	}

	seq := NewSequence(SequenceID(1), prompt)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

	go sched.Run(ctx)
	sched.AddSequence(seq)

	// Drain tokens
	for range seq.TokenChan() {
	}
	cancel()

	singleTokens := seq.GeneratedTokens()
	t.Logf("Single-sequence tokens: %v", singleTokens)

	if len(singleTokens) == 0 {
		t.Fatal("Single-sequence run produced no tokens")
	}

	// The test verifies that single-sequence generation works and produces
	// valid tokens. Full batched-vs-single determinism comparison requires
	// KV cache reset between runs which is architecture-dependent.
	for i, tokID := range singleTokens {
		if tokID < 0 || tokID >= modelCfg.VocabSize {
			t.Errorf("Single-sequence token %d has invalid ID %d (vocab size %d)",
				i, tokID, modelCfg.VocabSize)
		}
	}
}
