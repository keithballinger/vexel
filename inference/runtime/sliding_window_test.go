package runtime

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
)

func TestSlidingWindow(t *testing.T) {
	// Setup
	b := cpu.NewCPUBackend()
	config := ModelConfig{
		HiddenSize:        64,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  4,
		SlidingWindow:     4,
	}

	br := NewBlockRuntime(b, config)

	// Create Paged Cache
	pagedConfig := kv.PagedKVConfig{
		NumLayers:  1,
		NumKVHeads: 4,
		HeadDim:    16,
		BlockSize:  2,
		MaxBlocks:  10,
	}
	cache := kv.NewPagedKVCache(pagedConfig)
	seqID := cache.CreateSequence()

	// Fill cache with 8 tokens (positions 0-7)
	// Window is 4, so for pos 8 (next), we should attend to [5, 6, 7, 8].
	// Or rather, if we are AT pos 7, we attend to [4, 5, 6, 7].

	// Let's store tokens 0-7
	kvSize := 4 * 16

	// Proper batch store: [8 * kvSize]
	kBatch := make([]float32, 8*kvSize)
	vBatch := make([]float32, 8*kvSize)
	// Mark data to identify positions
	for i := 0; i < 8; i++ {
		for j := 0; j < kvSize; j++ {
			kBatch[i*kvSize+j] = float32(i) // K value = position index
		}
	}
	cache.StoreKVBatch(seqID, 0, 0, kBatch, vBatch, 8)

	// Manually invoke logic used in ExecuteWithPagedKV
	currentPos := 7 // We are at position 7
	attnStartPos := 0
	if br.SlidingWindow > 0 && currentPos >= br.SlidingWindow {
		attnStartPos = currentPos - br.SlidingWindow + 1
	}
	// Window 4. 7 - 4 + 1 = 4. Range [4, 7].

	if attnStartPos != 4 {
		t.Errorf("Expected start pos 4, got %d", attnStartPos)
	}

	fullK, _ := cache.GetKVSlice(seqID, 0, attnStartPos, currentPos)

	// Check length: should be 4 tokens
	expectedLen := 4 * kvSize
	if len(fullK) != expectedLen {
		t.Errorf("Expected %d elements (4 tokens), got %d", expectedLen, len(fullK))
	}

	// Check content: first token should be from position 4
	if fullK[0] != 4.0 {
		t.Errorf("Expected first token value 4.0, got %f", fullK[0])
	}
}
