package runtime_test

import (
	"testing"
	"vexel/inference/runtime"
)

func TestScratchBytes(t *testing.T) {
	cfg := runtime.Llama3_8B()
	
	// Scratch memory is needed for intermediate activations.
	// Max usage usually happens in the MLP block (expansion).
	// Size ~ BatchSize * SeqLen * IntermediateSize * DTypeSize
	// Note: In inference, we might process tokens one by one (decode) or chunk (prefill).
	// We estimate based on MaxBatchSize * 1 (decode) or MaxBatchSize * ChunkSize (prefill).
	// Let's assume this calculates the peak buffer needed for a single decode step for the max batch.
	
	maxBatch := 128
	
	bytes := cfg.ScratchBytes(maxBatch)
	
	// Expected:
	// Activations: Batch(128) * Hidden(4096) * 2 bytes = 1MB (Residual stream)
	// MLP Expansion: Batch(128) * Intermediate(14336) * 2 bytes = ~3.6 MB
	// Logits: Batch(128) * Vocab(128256) * 2 bytes = ~32 MB
	// Peak is likely the Logits or MLP up projection buffer.
	// Let's say we expect at least 32MB.
	
	if bytes < 32_000_000 {
		t.Errorf("Expected >32MB scratch for logits, got %d", bytes)
	}
}
