package kv

// PageIndex represents the index of a physical block in the KV cache.
type PageIndex int

// SeqKVHandle maps a logical sequence to a list of physical pages (blocks).
// This is effectively the "Block Table" for a sequence.
type SeqKVHandle struct {
	pages []PageIndex
}

// NewSeqKVHandle creates a new handle with an initial set of pages.
func NewSeqKVHandle(pages []PageIndex) *SeqKVHandle {
	return &SeqKVHandle{
		pages: pages,
	}
}

// AddPage appends a new page index to the sequence's block table.
func (h *SeqKVHandle) AddPage(idx PageIndex) {
	h.pages = append(h.pages, idx)
}

// Pages returns the current list of page indices.
func (h *SeqKVHandle) Pages() []PageIndex {
	return h.pages
}

// NumPages returns the number of pages currently assigned.
func (h *SeqKVHandle) NumPages() int {
	return len(h.pages)
}
