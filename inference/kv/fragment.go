package kv

import (
	"fmt"
	"sync"
)

// CachedFragment represents a pre-computed KV cache for a named content.
type CachedFragment struct {
	Name      string
	NumTokens int
	// blocks[layer] = list of BlockIDs holding this fragment's KV
	Blocks [][]BlockID
	// Original positions (0, 1, 2, ..., NumTokens-1)
	// When inserted at position N, we apply RoPE shift of N
}

// FragmentCache stores named cached fragments.
type FragmentCache struct {
	mu        sync.RWMutex
	fragments map[string]*CachedFragment
	allocator *BlockAllocator
}

// NewFragmentCache creates a new fragment cache.
func NewFragmentCache(allocator *BlockAllocator) *FragmentCache {
	return &FragmentCache{
		fragments: make(map[string]*CachedFragment),
		allocator: allocator,
	}
}

// CacheContent pre-computes and stores KV cache for the given tokens.
// This is called after running forward pass to capture the KV values.
// Returns the fragment for further use.
func (fc *FragmentCache) CacheContent(name string, numTokens int, blocks [][]BlockID) (*CachedFragment, error) {
	fc.mu.Lock()
	defer fc.mu.Unlock()

	// If fragment with this name exists, release old blocks first
	if old, exists := fc.fragments[name]; exists {
		fc.releaseFragmentBlocks(old)
	}

	// Add references to all blocks
	for layer := range blocks {
		for _, blockID := range blocks[layer] {
			fc.allocator.AddRef(blockID)
		}
	}

	fragment := &CachedFragment{
		Name:      name,
		NumTokens: numTokens,
		Blocks:    blocks,
	}

	fc.fragments[name] = fragment
	return fragment, nil
}

// Get retrieves a cached fragment by name.
func (fc *FragmentCache) Get(name string) (*CachedFragment, bool) {
	fc.mu.RLock()
	defer fc.mu.RUnlock()
	f, ok := fc.fragments[name]
	return f, ok
}

// Delete removes a cached fragment and releases its blocks.
func (fc *FragmentCache) Delete(name string) {
	fc.mu.Lock()
	defer fc.mu.Unlock()

	if fragment, exists := fc.fragments[name]; exists {
		fc.releaseFragmentBlocks(fragment)
		delete(fc.fragments, name)
	}
}

// releaseFragmentBlocks decrements ref counts for all blocks in a fragment.
// Must be called with lock held.
func (fc *FragmentCache) releaseFragmentBlocks(fragment *CachedFragment) {
	for layer, blockIDs := range fragment.Blocks {
		for _, blockID := range blockIDs {
			fc.allocator.Free(layer, blockID)
		}
	}
}

// List returns all cached fragment names.
func (fc *FragmentCache) List() []string {
	fc.mu.RLock()
	defer fc.mu.RUnlock()

	names := make([]string, 0, len(fc.fragments))
	for name := range fc.fragments {
		names = append(names, name)
	}
	return names
}

// InsertSpec describes how to insert a cached fragment into a sequence.
type InsertSpec struct {
	FragmentName string
	InsertPos    int // Position in the sequence where fragment starts
}

// ParseInserts parses a prompt for <insert name="..." /> tags.
// Returns the cleaned prompt (with inserts removed) and list of InsertSpecs.
// The positions are relative to the cleaned prompt's token positions.
func ParseInserts(prompt string) (cleanedPrompt string, inserts []InsertSpec, err error) {
	// Simple parser for <insert name="..." />
	// In production, use a proper XML/template parser

	result := ""
	inserts = make([]InsertSpec, 0)
	i := 0
	tokenPos := 0 // Track logical position for inserts

	for i < len(prompt) {
		// Look for <insert
		if i+7 < len(prompt) && prompt[i:i+7] == "<insert" {
			// Find the closing />
			end := i + 7
			for end < len(prompt) && !(prompt[end-1] == '/' && prompt[end] == '>') {
				end++
			}
			if end >= len(prompt) {
				return "", nil, fmt.Errorf("unclosed <insert> tag at position %d", i)
			}
			end++ // Include the '>'

			// Extract name attribute
			tag := prompt[i:end]
			name, err := extractNameAttr(tag)
			if err != nil {
				return "", nil, err
			}

			// Record insert position (will be adjusted after tokenization)
			inserts = append(inserts, InsertSpec{
				FragmentName: name,
				InsertPos:    tokenPos, // Placeholder - caller must adjust
			})

			i = end
			continue
		}

		result += string(prompt[i])
		i++
	}

	return result, inserts, nil
}

// extractNameAttr extracts the name="..." value from an insert tag.
func extractNameAttr(tag string) (string, error) {
	// Look for name="..."
	start := 0
	for start < len(tag) {
		if start+5 < len(tag) && tag[start:start+5] == "name=" {
			start += 5
			if start < len(tag) && tag[start] == '"' {
				start++
				end := start
				for end < len(tag) && tag[end] != '"' {
					end++
				}
				if end < len(tag) {
					return tag[start:end], nil
				}
			}
		}
		start++
	}
	return "", fmt.Errorf("no name attribute found in tag: %s", tag)
}
