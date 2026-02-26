package runtime_test

import (
	"testing"
	"vexel/inference/runtime"
)

// llama2_7B returns a ModelConfig matching LLaMA 2 7B dimensions.
func llama2_7B() runtime.ModelConfig {
	return runtime.ModelConfig{
		HiddenSize:        4096,
		IntermediateSize:  11008,
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  32, // LLaMA 2 7B uses MHA
		VocabSize:         32000,
		MaxSeqLen:         4096,
	}
}

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

// TestTotalArenaBytes verifies that TotalArenaBytes accounts for all allocations
// that DecodeWithGPUKV makes from the arena during a forward pass.
func TestTotalArenaBytes(t *testing.T) {
	cfg := llama2_7B()

	// TotalArenaBytes must be large enough for the full arena allocation pattern
	// in DecodeWithGPUKV:
	//   1. tokenBytes:  batchSize * 4           (int32 token IDs)
	//   2. statePtr:    batchSize * hiddenSize * 4  (float32 hidden state)
	//   3. scratchPtr:  ScratchBytes(batchSize)     (layer scratch)
	//   4. logitsPtr:   vocabSize * 4               (float32 logits for last token)

	for _, batchSize := range []int{1, 20, 64, 128, 256, 512, 1024, 2048} {
		t.Run("", func(t *testing.T) {
			arenaSize := cfg.TotalArenaBytes(batchSize)

			// Simulate the allocation pattern from DecodeWithGPUKV
			tokenBytes := int64(batchSize) * 4
			stateBytes := int64(batchSize) * int64(cfg.HiddenSize) * 4
			scratchBytes := cfg.ScratchBytes(batchSize)
			logitsBytes := int64(cfg.VocabSize) * 4

			totalNeeded := tokenBytes + stateBytes + scratchBytes + logitsBytes

			if arenaSize < totalNeeded {
				t.Errorf("TotalArenaBytes(%d) = %d, but DecodeWithGPUKV needs %d "+
					"(token=%d state=%d scratch=%d logits=%d)",
					batchSize, arenaSize, totalNeeded,
					tokenBytes, stateBytes, scratchBytes, logitsBytes)
			}
		})
	}
}

// TestArenaOOM_PrefillRegression verifies that the arena sizing bug that caused
// OOM on long prompts is fixed.
//
// Bug: The old arena formula `ScratchBytes(maxTokens) + 2*logitsSize + attnSize`
// did not account for tokenBytes and statePtr allocations. During prefill,
// DecodeWithGPUKV allocates statePtr = promptLen * hiddenSize * 4 from the arena,
// which is unbudgeted. When promptLen ≈ maxTokens, the deficit exceeds the surplus
// padding and triggers OOM. Example: maxTokens=64, promptLen=64 → 0.9 MB deficit;
// maxTokens=256, promptLen=256 → 3.8 MB deficit.
func TestArenaOOM_PrefillRegression(t *testing.T) {
	cfg := llama2_7B()

	cases := []struct {
		name      string
		maxTokens int
		promptLen int
	}{
		{"generate-default-long-prompt", 64, 64},
		{"serve-default-long-prompt", 256, 256},
		{"serve-default-medium-prompt", 256, 200},
		{"large-context-prefill", 512, 512},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Simulate the OLD arena sizing formula from commands.go
			oldScratchSize := cfg.ScratchBytes(tc.maxTokens)
			oldLogitsSize := int64(cfg.VocabSize) * 4
			oldAttnSize := int64(tc.maxTokens * tc.maxTokens * 4)
			oldArenaSize := oldScratchSize + oldLogitsSize*2 + oldAttnSize

			// Simulate the DecodeWithGPUKV allocation pattern
			tokenBytes := int64(tc.promptLen) * 4
			stateBytes := int64(tc.promptLen) * int64(cfg.HiddenSize) * 4
			scratchBytes := cfg.ScratchBytes(tc.promptLen)
			logitsBytes := int64(cfg.VocabSize) * 4
			totalNeeded := tokenBytes + stateBytes + scratchBytes + logitsBytes

			// Verify the OLD formula fails for this case
			if oldArenaSize >= totalNeeded {
				t.Skipf("Old formula sufficient for this case: arena=%d >= needed=%d",
					oldArenaSize, totalNeeded)
			}
			deficit := totalNeeded - oldArenaSize
			t.Logf("Confirmed bug: old arena=%d < needed=%d (deficit=%d bytes = %.1f MB)",
				oldArenaSize, totalNeeded, deficit, float64(deficit)/1024/1024)

			// The NEW formula (TotalArenaBytes) must be sufficient
			maxBatch := max(tc.maxTokens, tc.promptLen)
			newArenaSize := cfg.TotalArenaBytes(maxBatch)
			if newArenaSize < totalNeeded {
				t.Errorf("TotalArenaBytes(%d) = %d, still too small for %d-token prefill "+
					"(need %d)", maxBatch, newArenaSize, tc.promptLen, totalNeeded)
			}
			surplus := newArenaSize - totalNeeded
			t.Logf("Fixed: new arena=%d >= needed=%d (surplus=%d bytes = %.1f MB)",
				newArenaSize, totalNeeded, surplus, float64(surplus)/1024/1024)
		})
	}
}

// TestArenaOOM_SmallMaxTokens verifies the OOM for the generate default (maxTokens=64).
func TestArenaOOM_SmallMaxTokens(t *testing.T) {
	cfg := llama2_7B()
	maxTokens := 64 // generate default

	// With maxTokens=64, even a 10-token prompt should trigger the old bug
	for _, promptLen := range []int{10, 32, 64, 128} {
		t.Run("", func(t *testing.T) {
			tokenBytes := int64(promptLen) * 4
			stateBytes := int64(promptLen) * int64(cfg.HiddenSize) * 4
			scratchBytes := cfg.ScratchBytes(promptLen)
			logitsBytes := int64(cfg.VocabSize) * 4
			totalNeeded := tokenBytes + stateBytes + scratchBytes + logitsBytes

			// New formula should always work
			maxBatch := max(maxTokens, promptLen)
			arenaSize := cfg.TotalArenaBytes(maxBatch)
			if arenaSize < totalNeeded {
				t.Errorf("TotalArenaBytes(%d) = %d < needed %d for %d-token prefill",
					maxBatch, arenaSize, totalNeeded, promptLen)
			}
		})
	}
}
