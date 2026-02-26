package kv

import (
	"testing"
)

func TestBlockAllocator(t *testing.T) {
	config := PagedKVConfig{
		NumLayers:  2,
		NumKVHeads: 4,
		HeadDim:    64,
		BlockSize:  16,
		MaxBlocks:  10,
	}

	alloc := NewBlockAllocator(config)

	t.Run("initial state", func(t *testing.T) {
		for layer := 0; layer < config.NumLayers; layer++ {
			if got := alloc.FreeCount(layer); got != config.MaxBlocks {
				t.Errorf("layer %d: expected %d free blocks, got %d", layer, config.MaxBlocks, got)
			}
		}
	})

	t.Run("allocate and free", func(t *testing.T) {
		block, err := alloc.Allocate(0)
		if err != nil {
			t.Fatalf("failed to allocate: %v", err)
		}
		if block == nil {
			t.Fatal("expected block, got nil")
		}
		if block.LayerIdx != 0 {
			t.Errorf("expected layer 0, got %d", block.LayerIdx)
		}
		if alloc.FreeCount(0) != config.MaxBlocks-1 {
			t.Errorf("expected %d free blocks, got %d", config.MaxBlocks-1, alloc.FreeCount(0))
		}

		// Free the block
		alloc.Free(0, block.ID)
		if alloc.FreeCount(0) != config.MaxBlocks {
			t.Errorf("expected %d free blocks after free, got %d", config.MaxBlocks, alloc.FreeCount(0))
		}
	})

	t.Run("allocate all blocks", func(t *testing.T) {
		blocks := make([]*Block, config.MaxBlocks)
		for i := 0; i < config.MaxBlocks; i++ {
			block, err := alloc.Allocate(1)
			if err != nil {
				t.Fatalf("failed to allocate block %d: %v", i, err)
			}
			blocks[i] = block
		}

		// Should fail now
		_, err := alloc.Allocate(1)
		if err == nil {
			t.Error("expected error when allocating from empty pool")
		}

		// Free all
		for _, block := range blocks {
			alloc.Free(1, block.ID)
		}
		if alloc.FreeCount(1) != config.MaxBlocks {
			t.Errorf("expected %d free blocks after freeing all, got %d", config.MaxBlocks, alloc.FreeCount(1))
		}
	})

	t.Run("reference counting", func(t *testing.T) {
		block, _ := alloc.Allocate(0)
		initialFree := alloc.FreeCount(0)

		// Add references
		alloc.AddRef(block.ID)
		alloc.AddRef(block.ID)

		// First free - should not return to pool (ref count = 2)
		alloc.Free(0, block.ID)
		if alloc.FreeCount(0) != initialFree {
			t.Error("block should not be freed with ref count > 0")
		}

		// Second free - still has refs
		alloc.Free(0, block.ID)
		if alloc.FreeCount(0) != initialFree {
			t.Error("block should not be freed with ref count > 0")
		}

		// Third free - should return to pool
		alloc.Free(0, block.ID)
		if alloc.FreeCount(0) != initialFree+1 {
			t.Errorf("block should be freed when ref count = 0, got free count %d", alloc.FreeCount(0))
		}
	})

	t.Run("block data size", func(t *testing.T) {
		block, _ := alloc.Allocate(0)
		expectedSize := config.BlockSizeFloats()
		if len(block.Data) != expectedSize {
			t.Errorf("expected block data size %d, got %d", expectedSize, len(block.Data))
		}
		alloc.Free(0, block.ID)
	})
}

func TestBlockTable(t *testing.T) {
	numLayers := 3
	table := NewBlockTable(numLayers)

	t.Run("initial state", func(t *testing.T) {
		if table.SeqLen() != 0 {
			t.Errorf("expected seqLen 0, got %d", table.SeqLen())
		}
		for layer := 0; layer < numLayers; layer++ {
			if table.NumBlocks(layer) != 0 {
				t.Errorf("layer %d: expected 0 blocks, got %d", layer, table.NumBlocks(layer))
			}
		}
	})

	t.Run("append and get blocks", func(t *testing.T) {
		table.AppendBlock(0, BlockID(5))
		table.AppendBlock(0, BlockID(10))
		table.AppendBlock(1, BlockID(3))

		if table.NumBlocks(0) != 2 {
			t.Errorf("layer 0: expected 2 blocks, got %d", table.NumBlocks(0))
		}
		if table.NumBlocks(1) != 1 {
			t.Errorf("layer 1: expected 1 block, got %d", table.NumBlocks(1))
		}

		if got := table.GetBlockID(0, 0); got != BlockID(5) {
			t.Errorf("expected block ID 5, got %d", got)
		}
		if got := table.GetBlockID(0, 1); got != BlockID(10) {
			t.Errorf("expected block ID 10, got %d", got)
		}
		if got := table.GetBlockID(1, 0); got != BlockID(3) {
			t.Errorf("expected block ID 3, got %d", got)
		}

		// Invalid indices should return InvalidBlockID
		if got := table.GetBlockID(0, 5); got != InvalidBlockID {
			t.Errorf("expected InvalidBlockID for out of bounds, got %d", got)
		}
		if got := table.GetBlockID(10, 0); got != InvalidBlockID {
			t.Errorf("expected InvalidBlockID for invalid layer, got %d", got)
		}
	})

	t.Run("sequence length", func(t *testing.T) {
		table.SetSeqLen(100)
		if table.SeqLen() != 100 {
			t.Errorf("expected seqLen 100, got %d", table.SeqLen())
		}
	})
}

func TestPagedKVCache(t *testing.T) {
	config := PagedKVConfig{
		NumLayers:  2,
		NumKVHeads: 2,
		HeadDim:    4,
		BlockSize:  4, // Small for testing
		MaxBlocks:  10,
	}

	t.Run("create and delete sequence", func(t *testing.T) {
		cache := NewPagedKVCache(config)

		seqID := cache.CreateSequence()
		if seqID != 1 {
			t.Errorf("expected first seqID to be 1, got %d", seqID)
		}

		table := cache.GetSequence(seqID)
		if table == nil {
			t.Fatal("expected sequence to exist")
		}

		cache.DeleteSequence(seqID)
		if cache.GetSequence(seqID) != nil {
			t.Error("expected sequence to be deleted")
		}
	})

	t.Run("store and retrieve KV", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()

		kvSize := config.NumKVHeads * config.HeadDim // 2 * 4 = 8
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)

		// Fill with test data
		for i := range k {
			k[i] = float32(i)
			v[i] = float32(i + 100)
		}

		// Store at position 0, layer 0
		err := cache.StoreKV(seqID, 0, 0, k, v)
		if err != nil {
			t.Fatalf("failed to store KV: %v", err)
		}

		// Retrieve
		gotK, gotV := cache.GetKVSlice(seqID, 0, 0, 0)
		if len(gotK) != kvSize || len(gotV) != kvSize {
			t.Errorf("expected k,v size %d, got k=%d v=%d", kvSize, len(gotK), len(gotV))
		}

		for i := range k {
			if gotK[i] != k[i] {
				t.Errorf("k[%d]: expected %f, got %f", i, k[i], gotK[i])
			}
			if gotV[i] != v[i] {
				t.Errorf("v[%d]: expected %f, got %f", i, v[i], gotV[i])
			}
		}
	})

	t.Run("store across block boundary", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()

		kvSize := config.NumKVHeads * config.HeadDim

		// Store 5 positions (block size is 4, so spans 2 blocks)
		for pos := 0; pos < 5; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for i := range k {
				k[i] = float32(pos*10 + i)
				v[i] = float32(pos*10 + i + 100)
			}
			err := cache.StoreKV(seqID, 0, pos, k, v)
			if err != nil {
				t.Fatalf("failed to store KV at pos %d: %v", pos, err)
			}
		}

		// Verify block allocation
		table := cache.GetSequence(seqID)
		if table.NumBlocks(0) != 2 {
			t.Errorf("expected 2 blocks for 5 tokens with block size 4, got %d", table.NumBlocks(0))
		}

		// Retrieve all positions
		gotK, gotV := cache.GetKVSlice(seqID, 0, 0, 4) // endPos=4 means positions 0-4
		expectedSize := 5 * kvSize
		if len(gotK) != expectedSize {
			t.Errorf("expected k size %d, got %d", expectedSize, len(gotK))
		}

		// Verify data integrity
		for pos := 0; pos < 5; pos++ {
			for i := 0; i < kvSize; i++ {
				expectedK := float32(pos*10 + i)
				expectedV := float32(pos*10 + i + 100)
				gotKVal := gotK[pos*kvSize+i]
				gotVVal := gotV[pos*kvSize+i]
				if gotKVal != expectedK {
					t.Errorf("pos %d, i %d: expected k=%f, got %f", pos, i, expectedK, gotKVal)
				}
				if gotVVal != expectedV {
					t.Errorf("pos %d, i %d: expected v=%f, got %f", pos, i, expectedV, gotVVal)
				}
			}
		}
	})

	t.Run("free blocks returns count", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		initialFree := cache.FreeBlocks()

		seqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim

		// Store one position (allocates 1 block per layer)
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)
		cache.StoreKV(seqID, 0, 0, k, v)

		if cache.FreeBlocks() != initialFree-1 {
			t.Errorf("expected %d free blocks, got %d", initialFree-1, cache.FreeBlocks())
		}

		cache.DeleteSequence(seqID)
		if cache.FreeBlocks() != initialFree {
			t.Errorf("expected %d free blocks after delete, got %d", initialFree, cache.FreeBlocks())
		}
	})
}

func TestTruncateSequence(t *testing.T) {
	config := PagedKVConfig{
		NumLayers:  2,
		NumKVHeads: 2,
		HeadDim:    4,
		BlockSize:  4, // 4 tokens per block
		MaxBlocks:  20,
	}

	t.Run("truncate_within_block", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim

		// Store 3 tokens (all in first block)
		for pos := 0; pos < 3; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for i := range k {
				k[i] = float32(pos)
			}
			for layer := 0; layer < config.NumLayers; layer++ {
				cache.StoreKV(seqID, layer, pos, k, v)
			}
		}

		// Truncate to 2 tokens (stays in same block)
		cache.TruncateSequence(seqID, 2)

		table := cache.GetSequence(seqID)
		if table.SeqLen() != 2 {
			t.Errorf("expected seqLen 2, got %d", table.SeqLen())
		}
		// Block should still be allocated (partially used)
		if table.NumBlocks(0) != 1 {
			t.Errorf("expected 1 block, got %d", table.NumBlocks(0))
		}
	})

	t.Run("truncate_frees_excess_blocks", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim

		initialFree := cache.FreeBlocks()

		// Store 10 tokens (3 blocks per layer: 4+4+2)
		for pos := 0; pos < 10; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for layer := 0; layer < config.NumLayers; layer++ {
				cache.StoreKV(seqID, layer, pos, k, v)
			}
		}

		table := cache.GetSequence(seqID)
		if table.NumBlocks(0) != 3 {
			t.Fatalf("expected 3 blocks for 10 tokens, got %d", table.NumBlocks(0))
		}

		// Used 3 blocks per layer * 2 layers = 6 blocks
		freeAfterStore := cache.FreeBlocks()
		if freeAfterStore != initialFree-3 {
			t.Errorf("expected %d free blocks after store, got %d", initialFree-3, freeAfterStore)
		}

		// Truncate to 3 tokens (only need 1 block per layer)
		cache.TruncateSequence(seqID, 3)

		if table.SeqLen() != 3 {
			t.Errorf("expected seqLen 3, got %d", table.SeqLen())
		}
		if table.NumBlocks(0) != 1 {
			t.Errorf("expected 1 block after truncate, got %d", table.NumBlocks(0))
		}

		// Should have freed 2 blocks per layer
		freeAfterTruncate := cache.FreeBlocks()
		if freeAfterTruncate != initialFree-1 {
			t.Errorf("expected %d free blocks after truncate, got %d", initialFree-1, freeAfterTruncate)
		}
	})

	t.Run("truncate_to_zero", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim

		initialFree := cache.FreeBlocks()

		// Store 5 tokens
		for pos := 0; pos < 5; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for layer := 0; layer < config.NumLayers; layer++ {
				cache.StoreKV(seqID, layer, pos, k, v)
			}
		}

		// Truncate to 0
		cache.TruncateSequence(seqID, 0)

		table := cache.GetSequence(seqID)
		if table.SeqLen() != 0 {
			t.Errorf("expected seqLen 0, got %d", table.SeqLen())
		}
		if table.NumBlocks(0) != 0 {
			t.Errorf("expected 0 blocks, got %d", table.NumBlocks(0))
		}

		// All blocks should be freed
		if cache.FreeBlocks() != initialFree {
			t.Errorf("expected %d free blocks, got %d", initialFree, cache.FreeBlocks())
		}
	})

	t.Run("truncate_nonexistent_sequence", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		// Should not panic
		cache.TruncateSequence(999, 5)
	})

	t.Run("truncate_beyond_current_length_is_noop", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim

		// Store 3 tokens
		for pos := 0; pos < 3; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for layer := 0; layer < config.NumLayers; layer++ {
				cache.StoreKV(seqID, layer, pos, k, v)
			}
		}

		// Truncate to 10 (larger than current length) — should be a no-op
		cache.TruncateSequence(seqID, 10)

		table := cache.GetSequence(seqID)
		if table.SeqLen() != 3 {
			t.Errorf("expected seqLen 3 (unchanged), got %d", table.SeqLen())
		}
	})
}

func TestFragmentCache(t *testing.T) {
	config := PagedKVConfig{
		NumLayers:  2,
		NumKVHeads: 2,
		HeadDim:    4,
		BlockSize:  4,
		MaxBlocks:  20,
	}

	t.Run("cache and retrieve fragment", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()

		kvSize := config.NumKVHeads * config.HeadDim

		// Store some KV data
		for pos := 0; pos < 8; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for i := range k {
				k[i] = float32(pos*10 + i)
				v[i] = float32(pos*10 + i + 100)
			}
			for layer := 0; layer < config.NumLayers; layer++ {
				cache.StoreKV(seqID, layer, pos, k, v)
			}
		}

		// Cache as fragment
		fragment, err := cache.CacheContent("system_prompt", seqID)
		if err != nil {
			t.Fatalf("failed to cache content: %v", err)
		}
		if fragment == nil {
			t.Fatal("expected fragment, got nil")
		}
		if fragment.NumTokens != 8 {
			t.Errorf("expected 8 tokens, got %d", fragment.NumTokens)
		}

		// Verify fragment can be retrieved
		got, ok := cache.Fragments().Get("system_prompt")
		if !ok {
			t.Error("expected to find fragment")
		}
		if got.NumTokens != 8 {
			t.Errorf("expected 8 tokens, got %d", got.NumTokens)
		}
	})

	t.Run("insert fragment", func(t *testing.T) {
		cache := NewPagedKVCache(config)

		// Create and populate a sequence to cache
		srcSeqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim
		for pos := 0; pos < 4; pos++ {
			k := make([]float32, kvSize)
			v := make([]float32, kvSize)
			for i := range k {
				k[i] = float32(pos + 1) // 1, 2, 3, 4
				v[i] = float32(pos + 1 + 100)
			}
			for layer := 0; layer < config.NumLayers; layer++ {
				cache.StoreKV(srcSeqID, layer, pos, k, v)
			}
		}

		// Cache it
		cache.CacheContent("prefix", srcSeqID)

		// Create new sequence and insert the fragment
		dstSeqID := cache.CreateSequence()
		tokensInserted, err := cache.InsertFragment(dstSeqID, "prefix", 0)
		if err != nil {
			t.Fatalf("failed to insert fragment: %v", err)
		}
		if tokensInserted != 4 {
			t.Errorf("expected 4 tokens inserted, got %d", tokensInserted)
		}

		// Verify the sequence has the blocks
		table := cache.GetSequence(dstSeqID)
		if table.SeqLen() != 4 {
			t.Errorf("expected seqLen 4, got %d", table.SeqLen())
		}
	})

	t.Run("fragment not found", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()

		tokensInserted, err := cache.InsertFragment(seqID, "nonexistent", 0)
		if err != nil {
			t.Errorf("expected no error for missing fragment, got %v", err)
		}
		if tokensInserted != 0 {
			t.Errorf("expected 0 tokens inserted for missing fragment, got %d", tokensInserted)
		}
	})

	t.Run("list fragments", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()

		kvSize := config.NumKVHeads * config.HeadDim
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)
		cache.StoreKV(seqID, 0, 0, k, v)
		cache.StoreKV(seqID, 1, 0, k, v)

		cache.CacheContent("frag1", seqID)
		cache.CacheContent("frag2", seqID)

		names := cache.Fragments().List()
		if len(names) != 2 {
			t.Errorf("expected 2 fragments, got %d", len(names))
		}
	})

	t.Run("delete fragment", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()

		kvSize := config.NumKVHeads * config.HeadDim
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)
		cache.StoreKV(seqID, 0, 0, k, v)
		cache.StoreKV(seqID, 1, 0, k, v)

		cache.CacheContent("to_delete", seqID)

		cache.Fragments().Delete("to_delete")

		_, ok := cache.Fragments().Get("to_delete")
		if ok {
			t.Error("expected fragment to be deleted")
		}
	})
}

func TestFragmentRoPEShift(t *testing.T) {
	config := PagedKVConfig{
		NumLayers:  1,
		NumKVHeads: 2,
		HeadDim:    4,
		BlockSize:  4,
		MaxBlocks:  20,
	}

	// Mock RoPEShift function that just records calls
	type shiftCall struct {
		numTokens int
		shift     int
	}
	var calls []shiftCall
	mockShift := func(k []float32, headDim, numKVHeads, numTokens, shift int, theta float32) {
		calls = append(calls, shiftCall{numTokens, shift})
	}

	t.Run("no_fragment_no_shift", func(t *testing.T) {
		cache := NewPagedKVCache(config)
		seqID := cache.CreateSequence()
		calls = nil

		// Store regular KV data
		kvSize := config.NumKVHeads * config.HeadDim
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)
		cache.StoreKV(seqID, 0, 0, k, v)
		cache.StoreKV(seqID, 0, 1, k, v)

		// Retrieve with shift function
		_, _ = cache.GetKVSliceForAttention(seqID, 0, 1, 10000.0, mockShift)

		if len(calls) != 0 {
			t.Errorf("expected no shift calls for regular KV, got %d", len(calls))
		}
	})

	t.Run("fragment_at_position_0", func(t *testing.T) {
		cache := NewPagedKVCache(config)

		// Create source sequence and cache as fragment
		srcSeqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)
		cache.StoreKV(srcSeqID, 0, 0, k, v)
		cache.StoreKV(srcSeqID, 0, 1, k, v)
		cache.CacheContent("sys", srcSeqID)

		// Insert fragment at position 0
		dstSeqID := cache.CreateSequence()
		cache.InsertFragment(dstSeqID, "sys", 0)

		calls = nil
		_, _ = cache.GetKVSliceForAttention(dstSeqID, 0, 1, 10000.0, mockShift)

		// Insert at position 0 means shift by 0 - no shift needed
		if len(calls) != 0 {
			t.Errorf("expected no shift for fragment at position 0, got %d calls", len(calls))
		}
	})

	t.Run("fragment_at_position_5", func(t *testing.T) {
		cache := NewPagedKVCache(config)

		// Create source sequence and cache as fragment
		srcSeqID := cache.CreateSequence()
		kvSize := config.NumKVHeads * config.HeadDim
		k := make([]float32, kvSize)
		v := make([]float32, kvSize)
		cache.StoreKV(srcSeqID, 0, 0, k, v)
		cache.StoreKV(srcSeqID, 0, 1, k, v)
		cache.StoreKV(srcSeqID, 0, 2, k, v)
		cache.CacheContent("prompt", srcSeqID)

		// Create dest sequence with some data, then insert fragment
		dstSeqID := cache.CreateSequence()

		// Pre-fill positions 0-4 with regular data
		for pos := 0; pos < 5; pos++ {
			cache.StoreKV(dstSeqID, 0, pos, k, v)
		}

		// Insert fragment at position 5
		cache.InsertFragment(dstSeqID, "prompt", 5)

		calls = nil
		// Query up to position 7 (includes fragment positions 5,6,7)
		_, _ = cache.GetKVSliceForAttention(dstSeqID, 0, 7, 10000.0, mockShift)

		// Should have one shift call for the fragment range
		if len(calls) != 1 {
			t.Fatalf("expected 1 shift call, got %d", len(calls))
		}
		if calls[0].shift != 5 {
			t.Errorf("expected shift=5, got %d", calls[0].shift)
		}
		if calls[0].numTokens != 3 {
			t.Errorf("expected numTokens=3, got %d", calls[0].numTokens)
		}
	})
}

func TestParseInserts(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		wantClean      string
		wantInsertName string
		wantErr        bool
	}{
		{
			name:           "simple insert",
			input:          `Hello <insert name="system" /> world`,
			wantClean:      "Hello  world",
			wantInsertName: "system",
		},
		{
			name:           "no inserts",
			input:          "Hello world",
			wantClean:      "Hello world",
			wantInsertName: "",
		},
		{
			name:      "unclosed tag",
			input:     `Hello <insert name="test" world`,
			wantErr:   true,
		},
		{
			name:      "missing name",
			input:     `Hello <insert /> world`,
			wantErr:   true,
		},
		{
			name:           "insert at start",
			input:          `<insert name="prefix" />What is the weather?`,
			wantClean:      "What is the weather?",
			wantInsertName: "prefix",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clean, inserts, err := ParseInserts(tt.input)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if clean != tt.wantClean {
				t.Errorf("clean: expected %q, got %q", tt.wantClean, clean)
			}

			if tt.wantInsertName == "" {
				if len(inserts) != 0 {
					t.Errorf("expected no inserts, got %d", len(inserts))
				}
			} else {
				if len(inserts) != 1 {
					t.Fatalf("expected 1 insert, got %d", len(inserts))
				}
				if inserts[0].FragmentName != tt.wantInsertName {
					t.Errorf("insert name: expected %q, got %q", tt.wantInsertName, inserts[0].FragmentName)
				}
			}
		})
	}
}
