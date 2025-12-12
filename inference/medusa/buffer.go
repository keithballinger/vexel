// Package medusa implements Medusa speculative decoding with online training.
package medusa

import (
	"sync"
)

// TrainingSample holds a single training example for Medusa heads.
// Each sample consists of a hidden state and the ground truth future tokens.
type TrainingSample struct {
	// HiddenState is the model's hidden state at position t [hiddenSize]
	HiddenState []float32

	// FutureTokens contains ground truth tokens at t+1, t+2, ..., t+K
	// where K is the number of Medusa heads
	FutureTokens []int

	// Position in the sequence (for debugging/analysis)
	Position int
}

// RingBuffer is a thread-safe circular buffer for training samples.
// It maintains a fixed capacity and overwrites oldest samples when full.
type RingBuffer struct {
	mu sync.RWMutex

	samples  []TrainingSample
	capacity int
	head     int // Next write position
	count    int // Current number of valid samples
}

// NewRingBuffer creates a new ring buffer with the specified capacity.
func NewRingBuffer(capacity int) *RingBuffer {
	return &RingBuffer{
		samples:  make([]TrainingSample, capacity),
		capacity: capacity,
	}
}

// Add adds a new sample to the buffer, overwriting the oldest if full.
func (rb *RingBuffer) Add(sample TrainingSample) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.samples[rb.head] = sample
	rb.head = (rb.head + 1) % rb.capacity

	if rb.count < rb.capacity {
		rb.count++
	}
}

// AddBatch adds multiple samples efficiently.
func (rb *RingBuffer) AddBatch(samples []TrainingSample) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	for _, sample := range samples {
		rb.samples[rb.head] = sample
		rb.head = (rb.head + 1) % rb.capacity

		if rb.count < rb.capacity {
			rb.count++
		}
	}
}

// Count returns the current number of samples in the buffer.
func (rb *RingBuffer) Count() int {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	return rb.count
}

// IsFull returns true if the buffer has reached capacity.
func (rb *RingBuffer) IsFull() bool {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	return rb.count >= rb.capacity
}

// Capacity returns the buffer's maximum capacity.
func (rb *RingBuffer) Capacity() int {
	return rb.capacity
}

// Sample returns a random batch of samples for training.
// Returns up to batchSize samples, may return fewer if buffer has less.
func (rb *RingBuffer) Sample(batchSize int, rng func(int) int) []TrainingSample {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	if rb.count == 0 {
		return nil
	}

	if batchSize > rb.count {
		batchSize = rb.count
	}

	// Reservoir sampling for uniform random batch
	batch := make([]TrainingSample, batchSize)
	indices := make(map[int]bool)

	for i := 0; i < batchSize; i++ {
		// Find unique random index
		for {
			idx := rng(rb.count)
			if !indices[idx] {
				indices[idx] = true
				// Map logical index to physical position
				physicalIdx := (rb.head - rb.count + idx + rb.capacity) % rb.capacity
				batch[i] = rb.samples[physicalIdx]
				break
			}
		}
	}

	return batch
}

// GetAll returns all current samples (for debugging/analysis).
// The returned slice is a copy and safe to modify.
func (rb *RingBuffer) GetAll() []TrainingSample {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	if rb.count == 0 {
		return nil
	}

	result := make([]TrainingSample, rb.count)
	for i := 0; i < rb.count; i++ {
		physicalIdx := (rb.head - rb.count + i + rb.capacity) % rb.capacity
		result[i] = rb.samples[physicalIdx]
	}

	return result
}

// Clear removes all samples from the buffer.
func (rb *RingBuffer) Clear() {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.head = 0
	rb.count = 0
}

// Stats returns buffer statistics.
type BufferStats struct {
	Count       int
	Capacity    int
	Utilization float64
}

func (rb *RingBuffer) Stats() BufferStats {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	return BufferStats{
		Count:       rb.count,
		Capacity:    rb.capacity,
		Utilization: float64(rb.count) / float64(rb.capacity),
	}
}
