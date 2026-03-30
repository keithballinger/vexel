package kv_test

import (
	"testing"
	"vexel/inference/kv"
)

func TestHandle(t *testing.T) {
	// PageIndex is just a type alias for int, but we test its usage
	var p kv.PageIndex = 5
	if int(p) != 5 {
		t.Error("PageIndex should be an int")
	}

	// Test SeqKVHandle
	// Ideally this handle is created by the cache manager
	// For now we test the struct directly
	pages := []kv.PageIndex{1, 2, 3}
	handle := kv.NewSeqKVHandle(pages)

	if handle.NumPages() != 3 {
		t.Errorf("Handle should have 3 pages, got %d", handle.NumPages())
	}

	if handle.Pages()[1] != 2 {
		t.Errorf("Handle page mismatch")
	}

	// Test appending a page
	handle.AddPage(4)
	if handle.NumPages() != 4 {
		t.Errorf("Handle should have 4 pages after add, got %d", handle.NumPages())
	}
}
