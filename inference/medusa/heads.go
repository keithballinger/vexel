// Package medusa implements Medusa speculative decoding with online training.
// Medusa adds small prediction heads to a language model that predict multiple
// future tokens in parallel, enabling speculative decoding without a draft model.
package medusa

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sync"
	"unsafe"
)

// silu computes the SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
// Also known as the Swish function. SiLU is smooth and non-monotonic,
// which helps gradient flow for small prediction heads.
func silu(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

// siluDerivative computes the derivative of SiLU: sigmoid(x) * (1 + x*(1-sigmoid(x)))
func siluDerivative(x float32) float32 {
	sig := float32(1.0 / (1.0 + math.Exp(float64(-x))))
	return sig * (1.0 + x*(1.0-sig))
}

// Head represents a single Medusa prediction head.
// Architecture: hidden -> fc1 -> SiLU -> fc2 -> logits
// Each head predicts a different future position (t+1, t+2, etc.)
type Head struct {
	// FC1: [hiddenSize, hiddenSize] - first linear layer
	FC1 []float32
	// FC2: [hiddenSize, vocabSize] - second linear layer (output)
	FC2 []float32
}

// Heads manages multiple Medusa prediction heads.
type Heads struct {
	mu sync.RWMutex

	NumHeads   int // typically 4 (predicts t+1, t+2, t+3, t+4)
	HiddenSize int // matches model hidden size
	VocabSize  int // matches model vocab size

	heads []Head
}

// NewHeads creates a new set of Medusa heads with random initialization.
func NewHeads(numHeads, hiddenSize, vocabSize int) *Heads {
	h := &Heads{
		NumHeads:   numHeads,
		HiddenSize: hiddenSize,
		VocabSize:  vocabSize,
		heads:      make([]Head, numHeads),
	}

	// Xavier initialization scale
	scale1 := float32(math.Sqrt(2.0 / float64(hiddenSize)))
	scale2 := float32(math.Sqrt(2.0 / float64(hiddenSize)))

	for i := 0; i < numHeads; i++ {
		h.heads[i] = Head{
			FC1: make([]float32, hiddenSize*hiddenSize),
			FC2: make([]float32, hiddenSize*vocabSize),
		}

		// Random initialization
		for j := range h.heads[i].FC1 {
			h.heads[i].FC1[j] = (rand.Float32()*2 - 1) * scale1
		}
		for j := range h.heads[i].FC2 {
			h.heads[i].FC2[j] = (rand.Float32()*2 - 1) * scale2
		}
	}

	return h
}

// Forward computes logits for a single head given a hidden state.
// hidden: [hiddenSize]
// Returns: logits [vocabSize]
func (h *Heads) Forward(headIdx int, hidden []float32) []float32 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if headIdx < 0 || headIdx >= h.NumHeads {
		return nil
	}

	head := &h.heads[headIdx]

	// FC1: intermediate = SiLU(hidden @ FC1)
	intermediate := make([]float32, h.HiddenSize)
	for i := 0; i < h.HiddenSize; i++ {
		var sum float32
		for j := 0; j < h.HiddenSize; j++ {
			sum += hidden[j] * head.FC1[j*h.HiddenSize+i]
		}
		intermediate[i] = silu(sum)
	}

	// FC2: intermediate @ FC2
	logits := make([]float32, h.VocabSize)
	for i := 0; i < h.VocabSize; i++ {
		var sum float32
		for j := 0; j < h.HiddenSize; j++ {
			sum += intermediate[j] * head.FC2[j*h.VocabSize+i]
		}
		logits[i] = sum
	}

	return logits
}

// ForwardAll computes logits for all heads given a hidden state.
// Returns: [][]float32 where [i] is logits for head i
func (h *Heads) ForwardAll(hidden []float32) [][]float32 {
	results := make([][]float32, h.NumHeads)
	for i := 0; i < h.NumHeads; i++ {
		results[i] = h.Forward(i, hidden)
	}
	return results
}

// GetNumHeads returns the number of prediction heads.
func (h *Heads) GetNumHeads() int {
	return h.NumHeads
}

// ForwardBatch computes top-k token predictions for all heads.
// Returns: [][]int where [i] contains top-k token IDs for head i
func (h *Heads) ForwardTopK(hidden []float32, k int) [][]int {
	allLogits := h.ForwardAll(hidden)
	results := make([][]int, h.NumHeads)

	for i, logits := range allLogits {
		results[i] = topK(logits, k)
	}
	return results
}

// topK returns the indices of the k largest values.
func topK(values []float32, k int) []int {
	if k > len(values) {
		k = len(values)
	}

	// Simple O(n*k) selection - fine for small k
	indices := make([]int, k)
	used := make([]bool, len(values))

	for i := 0; i < k; i++ {
		bestIdx := -1
		bestVal := float32(-math.MaxFloat32)
		for j, v := range values {
			if !used[j] && v > bestVal {
				bestVal = v
				bestIdx = j
			}
		}
		indices[i] = bestIdx
		if bestIdx >= 0 {
			used[bestIdx] = true
		}
	}

	return indices
}

// Clone creates a deep copy of the heads.
func (h *Heads) Clone() *Heads {
	h.mu.RLock()
	defer h.mu.RUnlock()

	clone := &Heads{
		NumHeads:   h.NumHeads,
		HiddenSize: h.HiddenSize,
		VocabSize:  h.VocabSize,
		heads:      make([]Head, h.NumHeads),
	}

	for i := 0; i < h.NumHeads; i++ {
		clone.heads[i] = Head{
			FC1: make([]float32, len(h.heads[i].FC1)),
			FC2: make([]float32, len(h.heads[i].FC2)),
		}
		copy(clone.heads[i].FC1, h.heads[i].FC1)
		copy(clone.heads[i].FC2, h.heads[i].FC2)
	}

	return clone
}

// UpdateWeights applies a gradient update to a specific head.
// grad1: gradient for FC1 [hiddenSize * hiddenSize]
// grad2: gradient for FC2 [hiddenSize * vocabSize]
func (h *Heads) UpdateWeights(headIdx int, grad1, grad2 []float32, lr float32) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if headIdx < 0 || headIdx >= h.NumHeads {
		return
	}

	head := &h.heads[headIdx]

	for i := range head.FC1 {
		head.FC1[i] -= lr * grad1[i]
	}
	for i := range head.FC2 {
		head.FC2[i] -= lr * grad2[i]
	}
}

// Save writes the Medusa heads to a file.
// Format: [magic][version][numHeads][hiddenSize][vocabSize][weights...]
func (h *Heads) Save(path string) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	return h.WriteTo(f)
}

// WriteTo writes the Medusa heads to a writer.
func (h *Heads) WriteTo(w io.Writer) error {
	// Header
	header := make([]byte, 20)
	copy(header[0:4], []byte("MDSA"))             // magic
	binary.LittleEndian.PutUint32(header[4:8], 1) // version
	binary.LittleEndian.PutUint32(header[8:12], uint32(h.NumHeads))
	binary.LittleEndian.PutUint32(header[12:16], uint32(h.HiddenSize))
	binary.LittleEndian.PutUint32(header[16:20], uint32(h.VocabSize))

	if _, err := w.Write(header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// Weights — bulk write via unsafe cast to avoid per-element encoding
	for i := 0; i < h.NumHeads; i++ {
		fc1Bytes := unsafe.Slice((*byte)(unsafe.Pointer(&h.heads[i].FC1[0])), len(h.heads[i].FC1)*4)
		if _, err := w.Write(fc1Bytes); err != nil {
			return fmt.Errorf("write FC1: %w", err)
		}
		fc2Bytes := unsafe.Slice((*byte)(unsafe.Pointer(&h.heads[i].FC2[0])), len(h.heads[i].FC2)*4)
		if _, err := w.Write(fc2Bytes); err != nil {
			return fmt.Errorf("write FC2: %w", err)
		}
	}

	return nil
}

// Load reads Medusa heads from a file.
func Load(path string) (*Heads, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	return ReadFrom(f)
}

// ReadFrom reads Medusa heads from a reader.
func ReadFrom(r io.Reader) (*Heads, error) {
	// Header
	header := make([]byte, 20)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	if string(header[0:4]) != "MDSA" {
		return nil, fmt.Errorf("invalid magic: %s", string(header[0:4]))
	}

	version := binary.LittleEndian.Uint32(header[4:8])
	if version != 1 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	numHeads := int(binary.LittleEndian.Uint32(header[8:12]))
	hiddenSize := int(binary.LittleEndian.Uint32(header[12:16]))
	vocabSize := int(binary.LittleEndian.Uint32(header[16:20]))

	h := &Heads{
		NumHeads:   numHeads,
		HiddenSize: hiddenSize,
		VocabSize:  vocabSize,
		heads:      make([]Head, numHeads),
	}

	// Weights — bulk read via unsafe cast
	for i := 0; i < numHeads; i++ {
		h.heads[i] = Head{
			FC1: make([]float32, hiddenSize*hiddenSize),
			FC2: make([]float32, hiddenSize*vocabSize),
		}

		fc1Bytes := unsafe.Slice((*byte)(unsafe.Pointer(&h.heads[i].FC1[0])), len(h.heads[i].FC1)*4)
		if _, err := io.ReadFull(r, fc1Bytes); err != nil {
			return nil, fmt.Errorf("read FC1: %w", err)
		}
		fc2Bytes := unsafe.Slice((*byte)(unsafe.Pointer(&h.heads[i].FC2[0])), len(h.heads[i].FC2)*4)
		if _, err := io.ReadFull(r, fc2Bytes); err != nil {
			return nil, fmt.Errorf("read FC2: %w", err)
		}
	}

	return h, nil
}

// MemorySize returns the approximate memory usage in bytes.
func (h *Heads) MemorySize() int64 {
	perHead := int64(h.HiddenSize*h.HiddenSize + h.HiddenSize*h.VocabSize)
	return int64(h.NumHeads) * perHead * 4 // float32 = 4 bytes
}
