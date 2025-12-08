package runtime_test

import (
	"math"
	"testing"

	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
)

// TestPagedKVCacheWithSDPA tests that the paged KV cache integrates correctly
// with SDPA for autoregressive generation.
func TestPagedKVCacheWithSDPA(t *testing.T) {
	backend := cpu.NewBackend()
	ops := backend.(interface {
		SDPA(q, k, v, out []float32, kvLen, numQHeads, numKVHeads, headDim int, scale float32)
	})

	config := kv.PagedKVConfig{
		NumLayers:  2,
		NumKVHeads: 2,
		HeadDim:    4,
		BlockSize:  4,
		MaxBlocks:  20,
	}

	cache := kv.NewPagedKVCache(config)
	seqID := cache.CreateSequence()

	numQHeads := 4 // GQA: 4 Q heads, 2 KV heads
	numKVHeads := config.NumKVHeads
	headDim := config.HeadDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	t.Run("position_dependent_attention", func(t *testing.T) {
		// Simulate autoregressive generation: process tokens one at a time
		// Store K,V at positions 0, 1, 2
		// Query at position 2 should attend to all three

		layer := 0

		// Position 0: Store K,V
		k0 := []float32{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0} // [2 KV heads, 4 headDim]
		v0 := []float32{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}
		err := cache.StoreKV(seqID, layer, 0, k0, v0)
		if err != nil {
			t.Fatalf("failed to store KV at pos 0: %v", err)
		}

		// Position 1: Store K,V
		k1 := []float32{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}
		v1 := []float32{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}
		err = cache.StoreKV(seqID, layer, 1, k1, v1)
		if err != nil {
			t.Fatalf("failed to store KV at pos 1: %v", err)
		}

		// Position 2: Store K,V
		k2 := []float32{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}
		v2 := []float32{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}
		err = cache.StoreKV(seqID, layer, 2, k2, v2)
		if err != nil {
			t.Fatalf("failed to store KV at pos 2: %v", err)
		}

		// Now query at position 2
		// Q: [4 Q heads, 4 headDim]
		// This Q is designed to match K at all positions somewhat
		q := []float32{
			1.0, 1.0, 1.0, 0.0, // Q head 0 -> KV head 0
			0.5, 0.5, 0.5, 0.5, // Q head 1 -> KV head 0
			1.0, 1.0, 1.0, 0.0, // Q head 2 -> KV head 1
			0.5, 0.5, 0.5, 0.5, // Q head 3 -> KV head 1
		}

		// Get cached K,V for positions [0, 2]
		cachedK, cachedV := cache.GetKVSlice(seqID, layer, 2)
		kvLen := 3

		// Verify we got all positions
		expectedKVSize := kvLen * numKVHeads * headDim
		if len(cachedK) != expectedKVSize {
			t.Fatalf("expected cachedK size %d, got %d", expectedKVSize, len(cachedK))
		}

		// Run SDPA
		out := make([]float32, numQHeads*headDim)
		ops.SDPA(q, cachedK, cachedV, out, kvLen, numQHeads, numKVHeads, headDim, scale)

		// Output should be a weighted combination of V values
		// Since Q has nonzero components matching all K vectors,
		// output should be nonzero in dimensions 0, 1, 2
		hasNonzero := false
		for _, v := range out {
			if v != 0 {
				hasNonzero = true
				break
			}
		}
		if !hasNonzero {
			t.Error("expected nonzero output from attention")
		}

		t.Logf("Attention output: %v", out)
	})

	t.Run("incremental_generation", func(t *testing.T) {
		// Simulate typical autoregressive: generate positions 0, 1, 2
		// and verify that attention at each step works correctly

		cache2 := kv.NewPagedKVCache(config)
		seqID2 := cache2.CreateSequence()
		layer := 0

		kvSize := numKVHeads * headDim

		// Generate position 0
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)
		for i := range k {
			k[i] = 0.1 * float32(i+1)
			v[i] = 0.2 * float32(i+1)
		}
		cache2.StoreKV(seqID2, layer, 0, k, v)

		// Query at position 0 (only attends to itself)
		q := make([]float32, numQHeads*headDim)
		for i := range q {
			q[i] = 0.1
		}
		cachedK, cachedV := cache2.GetKVSlice(seqID2, layer, 0)
		out := make([]float32, numQHeads*headDim)
		ops.SDPA(q, cachedK, cachedV, out, 1, numQHeads, numKVHeads, headDim, scale)

		// With single position, output = V (softmax of single element = 1)
		// Check that output matches V pattern for the corresponding KV head
		for h := 0; h < numQHeads; h++ {
			kvHead := h / 2 // GQA mapping
			for d := 0; d < headDim; d++ {
				expected := v[kvHead*headDim+d]
				got := out[h*headDim+d]
				if absf(got-expected) > 1e-5 {
					t.Errorf("pos 0, head %d, dim %d: expected %f, got %f", h, d, expected, got)
				}
			}
		}

		// Generate position 1
		for i := range k {
			k[i] = 0.3 * float32(i+1)
			v[i] = 0.4 * float32(i+1)
		}
		cache2.StoreKV(seqID2, layer, 1, k, v)

		// Query at position 1 (attends to positions 0 and 1)
		cachedK, cachedV = cache2.GetKVSlice(seqID2, layer, 1)
		ops.SDPA(q, cachedK, cachedV, out, 2, numQHeads, numKVHeads, headDim, scale)

		// Output should be different from just V at position 1
		// (it's a weighted combination of both positions)
		t.Logf("Attention output at pos 1: %v", out)
	})

	t.Run("cross_block_boundary", func(t *testing.T) {
		// Test that attention works correctly when KV spans multiple blocks
		// Block size is 4, so we need 5+ positions

		cache3 := kv.NewPagedKVCache(config)
		seqID3 := cache3.CreateSequence()
		layer := 0

		kvSize := numKVHeads * headDim

		// Store 5 positions (spans 2 blocks)
		for pos := 0; pos < 5; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for i := range k {
				k[i] = float32(pos+1) * 0.1 * float32(i+1)
				v[i] = float32(pos+1) * 0.2 * float32(i+1)
			}
			err := cache3.StoreKV(seqID3, layer, pos, k, v)
			if err != nil {
				t.Fatalf("failed to store KV at pos %d: %v", pos, err)
			}
		}

		// Query at position 4 (attends to all 5 positions across 2 blocks)
		q := make([]float32, numQHeads*headDim)
		for i := range q {
			q[i] = 1.0 // Uniform query
		}

		cachedK, cachedV := cache3.GetKVSlice(seqID3, layer, 4)
		expectedSize := 5 * numKVHeads * headDim
		if len(cachedK) != expectedSize {
			t.Errorf("expected cachedK size %d for 5 positions, got %d", expectedSize, len(cachedK))
		}

		out := make([]float32, numQHeads*headDim)
		ops.SDPA(q, cachedK, cachedV, out, 5, numQHeads, numKVHeads, headDim, scale)

		// Output should be nonzero
		hasNonzero := false
		for _, val := range out {
			if val != 0 {
				hasNonzero = true
				break
			}
		}
		if !hasNonzero {
			t.Error("expected nonzero output from cross-block attention")
		}

		t.Logf("Cross-block attention output: %v", out)
	})
}

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
