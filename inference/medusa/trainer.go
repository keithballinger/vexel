// Package medusa implements Medusa speculative decoding with online training.
package medusa

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Trainer is the interface for Medusa online training.
type Trainer interface {
	// Start begins background training.
	Start(ctx context.Context)
	// Stop stops background training.
	Stop()
	// AddSample adds a training sample.
	AddSample(hidden []float32, futureTokens []int, pos int)
	// Phase returns current training phase.
	Phase() TrainingPhase
	// IsHot returns true if heads are ready for speculation.
	IsHot() bool
	// ForceHot forces hot phase (for testing).
	ForceHot()
	// Heads returns prediction heads for speculation.
	Heads() HeadsInterface
	// Metrics returns training metrics.
	Metrics() TrainingMetrics
	// SaveHeads saves heads to file.
	SaveHeads(path string) error
	// GPULock/GPUUnlock serialize GPU access between training and inference.
	// The inference scheduler must hold this lock during GPU-intensive decode
	// operations to prevent concurrent Metal command encoding with the
	// background training goroutine.
	GPULock()
	GPUUnlock()
}

// HeadsInterface is the interface for Medusa prediction heads.
type HeadsInterface interface {
	Forward(headIdx int, hidden []float32) []float32
	ForwardAll(hidden []float32) [][]float32
	GetNumHeads() int
}

// TrainingPhase represents the current state of online training.
type TrainingPhase int

const (
	// PhaseCold means we're collecting samples but heads aren't trained yet.
	PhaseCold TrainingPhase = iota

	// PhaseWarming means we're actively training but not using heads for speculation.
	PhaseWarming

	// PhaseHot means heads are trained and active for speculation.
	PhaseHot
)

func (p TrainingPhase) String() string {
	switch p {
	case PhaseCold:
		return "cold"
	case PhaseWarming:
		return "warming"
	case PhaseHot:
		return "hot"
	default:
		return "unknown"
	}
}

// OnlineConfig configures online Medusa training.
type OnlineConfig struct {
	// NumHeads is how many prediction heads (typically 4).
	NumHeads int

	// BufferCapacity is how many samples to keep in the ring buffer.
	BufferCapacity int

	// WarmupSamples is how many samples to collect before starting training.
	WarmupSamples int

	// MinAccuracy is the minimum accuracy required to enter Hot phase.
	MinAccuracy float32

	// BatchSize for training updates.
	BatchSize int

	// LearningRate for SGD updates.
	LearningRate float32

	// TrainInterval is how often to run a training step.
	TrainInterval time.Duration

	// EvalInterval is how often to evaluate accuracy.
	EvalInterval time.Duration
}

// DefaultOnlineConfig returns reasonable defaults for online training.
func DefaultOnlineConfig() OnlineConfig {
	return OnlineConfig{
		NumHeads:       4,
		BufferCapacity: 50000,                  // 50K samples
		WarmupSamples:  200,                    // Start training after 200 samples
		MinAccuracy:    0.3,                    // 30% top-1 accuracy to go hot
		BatchSize:      64,                     // Train on 64 samples at a time
		LearningRate:   0.001,                  // Moderate LR; BypassFC1 protects early, FC1 divergence after step 100
		TrainInterval:  500 * time.Millisecond, // Train every 500ms; TryLock skips when inference active
		EvalInterval:   1 * time.Second,
	}
}

// TrainingMetrics tracks online training progress.
type TrainingMetrics struct {
	SamplesCollected int64
	TrainingSteps    int64
	CurrentLoss      float32
	HeadAccuracies   []float32 // Per-head top-1 accuracy
	Phase            TrainingPhase
	LastEvalTime     time.Time
}

// OnlineTrainer manages online training of Medusa heads.
type OnlineTrainer struct {
	mu sync.RWMutex

	heads   *Heads
	buffer  *RingBuffer
	config  OnlineConfig
	metrics TrainingMetrics
	rng     *rand.Rand

	phase     atomic.Int32
	isRunning atomic.Bool

	// Channels for coordination
	stopCh chan struct{}
	doneCh chan struct{}
}

// NewOnlineTrainer creates a new online trainer.
func NewOnlineTrainer(hiddenSize, vocabSize int, config OnlineConfig) *OnlineTrainer {
	t := &OnlineTrainer{
		heads:  NewHeads(config.NumHeads, hiddenSize, vocabSize),
		buffer: NewRingBuffer(config.BufferCapacity),
		config: config,
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
		metrics: TrainingMetrics{
			HeadAccuracies: make([]float32, config.NumHeads),
		},
	}
	t.phase.Store(int32(PhaseCold))
	return t
}

// NewOnlineTrainerWithHeads creates a trainer with pre-trained heads.
func NewOnlineTrainerWithHeads(heads *Heads, config OnlineConfig) *OnlineTrainer {
	t := &OnlineTrainer{
		heads:  heads,
		buffer: NewRingBuffer(config.BufferCapacity),
		config: config,
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
		metrics: TrainingMetrics{
			HeadAccuracies: make([]float32, config.NumHeads),
		},
	}
	// Start hot if we have pre-trained heads
	t.phase.Store(int32(PhaseHot))
	return t
}

// Start begins the background training loop.
func (t *OnlineTrainer) Start(ctx context.Context) {
	if t.isRunning.Swap(true) {
		return // Already running
	}

	go t.trainingLoop(ctx)
}

// Stop stops the background training loop.
func (t *OnlineTrainer) Stop() {
	if !t.isRunning.Load() {
		return
	}

	close(t.stopCh)
	<-t.doneCh
	t.isRunning.Store(false)
}

// AddSample adds a training sample from inference.
func (t *OnlineTrainer) AddSample(hidden []float32, futureTokens []int, pos int) {
	// Make copies to avoid race conditions
	hiddenCopy := make([]float32, len(hidden))
	copy(hiddenCopy, hidden)

	tokensCopy := make([]int, len(futureTokens))
	copy(tokensCopy, futureTokens)

	t.buffer.Add(TrainingSample{
		HiddenState:  hiddenCopy,
		FutureTokens: tokensCopy,
		Position:     pos,
	})

	atomic.AddInt64(&t.metrics.SamplesCollected, 1)

	// Check for phase transition from Cold to Warming
	if t.Phase() == PhaseCold && t.buffer.Count() >= t.config.WarmupSamples {
		t.phase.Store(int32(PhaseWarming))
	}
}

// Phase returns the current training phase.
func (t *OnlineTrainer) Phase() TrainingPhase {
	return TrainingPhase(t.phase.Load())
}

// IsHot returns true if heads are ready for speculation.
func (t *OnlineTrainer) IsHot() bool {
	return t.Phase() == PhaseHot
}

// ForceHot forces the trainer into Hot phase (for testing speculation without training).
func (t *OnlineTrainer) ForceHot() {
	t.phase.Store(int32(PhaseHot))
}

// Heads returns the current Medusa heads (thread-safe read).
func (t *OnlineTrainer) Heads() HeadsInterface {
	return t.heads
}

// Metrics returns current training metrics.
func (t *OnlineTrainer) Metrics() TrainingMetrics {
	t.mu.RLock()
	defer t.mu.RUnlock()

	m := t.metrics
	m.SamplesCollected = atomic.LoadInt64(&t.metrics.SamplesCollected)
	m.TrainingSteps = atomic.LoadInt64(&t.metrics.TrainingSteps)
	m.Phase = t.Phase()

	// Copy accuracies
	m.HeadAccuracies = make([]float32, len(t.metrics.HeadAccuracies))
	copy(m.HeadAccuracies, t.metrics.HeadAccuracies)

	return m
}

// trainingLoop runs in the background, performing training updates.
func (t *OnlineTrainer) trainingLoop(ctx context.Context) {
	defer close(t.doneCh)

	trainTicker := time.NewTicker(t.config.TrainInterval)
	evalTicker := time.NewTicker(t.config.EvalInterval)
	defer trainTicker.Stop()
	defer evalTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-t.stopCh:
			return

		case <-trainTicker.C:
			if t.Phase() >= PhaseWarming {
				t.trainStep()
			}

		case <-evalTicker.C:
			if t.Phase() >= PhaseWarming {
				t.evaluate()
			}
		}
	}
}

// trainStep performs one training iteration.
func (t *OnlineTrainer) trainStep() {
	// Sample a batch
	batch := t.buffer.Sample(t.config.BatchSize, t.rng.Intn)
	if len(batch) == 0 {
		return
	}

	// Train each head
	totalLoss := float32(0)
	for headIdx := 0; headIdx < t.config.NumHeads; headIdx++ {
		loss := t.trainHead(headIdx, batch)
		totalLoss += loss
	}

	t.mu.Lock()
	t.metrics.CurrentLoss = totalLoss / float32(t.config.NumHeads)
	t.mu.Unlock()

	atomic.AddInt64(&t.metrics.TrainingSteps, 1)
}

// trainHead trains a single head on the batch.
func (t *OnlineTrainer) trainHead(headIdx int, batch []TrainingSample) float32 {
	hiddenSize := t.heads.HiddenSize
	vocabSize := t.heads.VocabSize

	// Accumulate gradients
	grad1 := make([]float32, hiddenSize*hiddenSize)
	grad2 := make([]float32, hiddenSize*vocabSize)

	totalLoss := float32(0)
	validSamples := 0

	for _, sample := range batch {
		// Skip if we don't have the target token for this head
		if headIdx >= len(sample.FutureTokens) {
			continue
		}
		targetToken := sample.FutureTokens[headIdx]
		if targetToken < 0 || targetToken >= vocabSize {
			continue
		}

		// Forward pass
		logits := t.heads.Forward(headIdx, sample.HiddenState)
		if logits == nil {
			continue
		}

		// Compute softmax and loss
		probs := softmax(logits)
		loss := -float32(math.Log(float64(probs[targetToken]) + 1e-10))
		totalLoss += loss

		// Backward pass: dL/dlogits = probs - one_hot(target)
		dLogits := make([]float32, vocabSize)
		copy(dLogits, probs)
		dLogits[targetToken] -= 1.0

		// Get intermediate activations for backprop
		// Need to recompute forward pass to get intermediate values
		head := &t.heads.heads[headIdx]

		// FC1 forward: pre-activation and SiLU(hidden @ FC1)
		preAct := make([]float32, hiddenSize)
		intermediate := make([]float32, hiddenSize)
		for i := 0; i < hiddenSize; i++ {
			var sum float32
			for j := 0; j < hiddenSize; j++ {
				sum += sample.HiddenState[j] * head.FC1[j*hiddenSize+i]
			}
			preAct[i] = sum
			intermediate[i] = silu(sum)
		}

		// Gradient for FC2: dFC2 = intermediate^T @ dLogits
		for i := 0; i < hiddenSize; i++ {
			for j := 0; j < vocabSize; j++ {
				grad2[i*vocabSize+j] += intermediate[i] * dLogits[j]
			}
		}

		// Backprop through FC2: dIntermediate = dLogits @ FC2^T
		dIntermediate := make([]float32, hiddenSize)
		for i := 0; i < hiddenSize; i++ {
			for j := 0; j < vocabSize; j++ {
				dIntermediate[i] += dLogits[j] * head.FC2[i*vocabSize+j]
			}
		}

		// Backprop through SiLU: dSiLU/dx = sigmoid(x) * (1 + x*(1-sigmoid(x)))
		for i := 0; i < hiddenSize; i++ {
			dIntermediate[i] *= siluDerivative(preAct[i])
		}

		// Gradient for FC1: dFC1 = hidden^T @ dIntermediate
		for i := 0; i < hiddenSize; i++ {
			for j := 0; j < hiddenSize; j++ {
				grad1[i*hiddenSize+j] += sample.HiddenState[i] * dIntermediate[j]
			}
		}

		validSamples++
	}

	if validSamples == 0 {
		return 0
	}

	// Average gradients
	scale := 1.0 / float32(validSamples)
	for i := range grad1 {
		grad1[i] *= scale
	}
	for i := range grad2 {
		grad2[i] *= scale
	}

	// Apply gradients
	t.heads.UpdateWeights(headIdx, grad1, grad2, t.config.LearningRate)

	return totalLoss / float32(validSamples)
}

// evaluate computes accuracy on a held-out set.
func (t *OnlineTrainer) evaluate() {
	// Sample evaluation batch (larger than training batch)
	evalBatch := t.buffer.Sample(t.config.BatchSize*4, t.rng.Intn)
	if len(evalBatch) == 0 {
		return
	}

	accuracies := make([]float32, t.config.NumHeads)

	for headIdx := 0; headIdx < t.config.NumHeads; headIdx++ {
		correct := 0
		total := 0

		for _, sample := range evalBatch {
			if headIdx >= len(sample.FutureTokens) {
				continue
			}
			targetToken := sample.FutureTokens[headIdx]

			logits := t.heads.Forward(headIdx, sample.HiddenState)
			if logits == nil {
				continue
			}

			// Check if top-1 prediction matches
			predToken := argmax(logits)
			if predToken == targetToken {
				correct++
			}
			total++
		}

		if total > 0 {
			accuracies[headIdx] = float32(correct) / float32(total)
		}
	}

	t.mu.Lock()
	copy(t.metrics.HeadAccuracies, accuracies)
	t.metrics.LastEvalTime = time.Now()
	t.mu.Unlock()

	// Check for phase transition to Hot
	if t.Phase() == PhaseWarming {
		// Require head 0 (t+1 prediction) to meet minimum accuracy
		if accuracies[0] >= t.config.MinAccuracy {
			t.phase.Store(int32(PhaseHot))
		}
	}
}

// softmax computes softmax over logits.
func softmax(logits []float32) []float32 {
	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp and sum
	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

// argmax returns the index of the maximum value.
func argmax(values []float32) int {
	maxIdx := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// SaveHeads saves the current heads to a file.
func (t *OnlineTrainer) SaveHeads(path string) error {
	return t.heads.Save(path)
}

// GPULock is a no-op for the CPU trainer (no GPU contention).
func (t *OnlineTrainer) GPULock() {}

// GPUUnlock is a no-op for the CPU trainer.
func (t *OnlineTrainer) GPUUnlock() {}
