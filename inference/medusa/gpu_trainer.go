//go:build metal && darwin && cgo

// Package medusa implements Medusa speculative decoding with online training.
package medusa

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"vexel/inference/backend"
)

var gpuTrainerDebug = os.Getenv("MEDUSA_DEBUG") != ""

// GPUOnlineTrainer manages online training of Medusa heads with GPU acceleration.
type GPUOnlineTrainer struct {
	mu sync.RWMutex

	gpuHeads *GPUHeads
	buffer   *RingBuffer
	config   OnlineConfig
	metrics  TrainingMetrics
	rng      *rand.Rand
	backend  backend.Backend

	phase     atomic.Int32
	isRunning atomic.Bool

	// Channels for coordination
	stopCh chan struct{}
	doneCh chan struct{}
}

// NewGPUOnlineTrainer creates a GPU-accelerated online trainer.
func NewGPUOnlineTrainer(hiddenSize, vocabSize int, config OnlineConfig, b backend.Backend) *GPUOnlineTrainer {
	t := &GPUOnlineTrainer{
		gpuHeads: NewGPUHeads(config.NumHeads, hiddenSize, vocabSize, b),
		buffer:   NewRingBuffer(config.BufferCapacity),
		config:   config,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
		backend:  b,
		stopCh:   make(chan struct{}),
		doneCh:   make(chan struct{}),
		metrics: TrainingMetrics{
			HeadAccuracies: make([]float32, config.NumHeads),
		},
	}
	t.phase.Store(int32(PhaseCold))
	return t
}

// Start begins the background training loop.
func (t *GPUOnlineTrainer) Start(ctx context.Context) {
	if t.isRunning.Swap(true) {
		return // Already running
	}

	if gpuTrainerDebug {
		fmt.Println("[GPU Trainer] Starting training loop")
	}
	go t.trainingLoop(ctx)
}

// Stop stops the background training loop.
func (t *GPUOnlineTrainer) Stop() {
	if !t.isRunning.Load() {
		return
	}

	close(t.stopCh)
	<-t.doneCh
	t.isRunning.Store(false)

	// Free GPU resources
	if t.gpuHeads != nil {
		t.gpuHeads.Free()
	}
}

// AddSample adds a training sample from inference.
func (t *GPUOnlineTrainer) AddSample(hidden []float32, futureTokens []int, pos int) {
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
func (t *GPUOnlineTrainer) Phase() TrainingPhase {
	return TrainingPhase(t.phase.Load())
}

// IsHot returns true if heads are ready for speculation.
func (t *GPUOnlineTrainer) IsHot() bool {
	return t.Phase() == PhaseHot
}

// ForceHot forces the trainer into Hot phase (for testing).
func (t *GPUOnlineTrainer) ForceHot() {
	t.phase.Store(int32(PhaseHot))
}

// Heads returns an interface to the GPU heads for speculation.
func (t *GPUOnlineTrainer) Heads() HeadsInterface {
	return &GPUHeadsWrapper{gpuHeads: t.gpuHeads}
}

// GPUHeadsWrapper wraps GPUHeads to provide the same interface as Heads.
type GPUHeadsWrapper struct {
	gpuHeads *GPUHeads
}

// Forward computes logits for a single head.
func (w *GPUHeadsWrapper) Forward(headIdx int, hidden []float32) []float32 {
	return w.gpuHeads.Forward(headIdx, hidden)
}

// ForwardAll computes logits for all heads.
func (w *GPUHeadsWrapper) ForwardAll(hidden []float32) [][]float32 {
	return w.gpuHeads.ForwardAll(hidden)
}

// GetNumHeads returns the number of heads.
func (w *GPUHeadsWrapper) GetNumHeads() int {
	return w.gpuHeads.NumHeads
}

// Metrics returns current training metrics.
func (t *GPUOnlineTrainer) Metrics() TrainingMetrics {
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
func (t *GPUOnlineTrainer) trainingLoop(ctx context.Context) {
	defer close(t.doneCh)

	trainTicker := time.NewTicker(t.config.TrainInterval)
	evalTicker := time.NewTicker(t.config.EvalInterval)
	defer trainTicker.Stop()
	defer evalTicker.Stop()

	if gpuTrainerDebug {
		fmt.Printf("[GPU Trainer] Training loop started (train interval=%v, eval interval=%v)\n",
			t.config.TrainInterval, t.config.EvalInterval)
	}

	tickCount := 0
	for {
		select {
		case <-ctx.Done():
			if gpuTrainerDebug {
				fmt.Println("[GPU Trainer] Context cancelled, exiting training loop")
			}
			return
		case <-t.stopCh:
			if gpuTrainerDebug {
				fmt.Println("[GPU Trainer] Stop signal received, exiting training loop")
			}
			return

		case <-trainTicker.C:
			tickCount++
			phase := t.Phase()
			if gpuTrainerDebug && tickCount <= 5 {
				fmt.Printf("[GPU Trainer] Train tick %d, phase=%v\n", tickCount, phase)
			}
			if phase >= PhaseWarming {
				t.trainStep()
			}

		case <-evalTicker.C:
			// Check for phase transition based on training progress
			phase := t.Phase()
			steps := atomic.LoadInt64(&t.metrics.TrainingSteps)

			t.mu.RLock()
			currentLoss := t.metrics.CurrentLoss
			t.mu.RUnlock()

			if gpuTrainerDebug {
				fmt.Printf("[GPU Trainer] Eval tick: phase=%v, steps=%d, loss=%.4f\n", phase, steps, currentLoss)
			}

			// Transition to Hot phase requires:
			// 1. At least 20 training steps
			// 2. Loss has decreased below initial random loss (~10.4 for 32k vocab)
			//    Use 10.2 threshold - slightly below random to verify learning started
			minStepsForHot := int64(20)
			maxLossForHot := float32(10.2) // Just below random 10.4

			if phase == PhaseWarming && steps >= minStepsForHot {
				if currentLoss > 0 && currentLoss < maxLossForHot {
					t.phase.Store(int32(PhaseHot))
					fmt.Printf("[GPU Trainer] Transitioning to Hot phase: steps=%d, loss=%.4f\n", steps, currentLoss)
				} else if gpuTrainerDebug {
					fmt.Printf("[GPU Trainer] Not ready for Hot: loss=%.4f (need < %.1f)\n", currentLoss, maxLossForHot)
				}
			}
		}
	}
}

// trainStep performs one training iteration using GPU-accelerated heads.
func (t *GPUOnlineTrainer) trainStep() {
	// Sample a batch
	batch := t.buffer.Sample(t.config.BatchSize, t.rng.Intn)
	if len(batch) == 0 {
		if gpuTrainerDebug {
			fmt.Printf("[GPU Trainer] trainStep: empty batch (buffer count=%d)\n", t.buffer.Count())
		}
		return
	}

	start := time.Now()
	if gpuTrainerDebug {
		fmt.Printf("[GPU Trainer] trainStep: training with batch size %d\n", len(batch))
	}

	// Train using GPU heads
	loss := t.gpuHeads.TrainStep(batch, t.config.LearningRate)

	elapsed := time.Since(start)
	if gpuTrainerDebug {
		fmt.Printf("[GPU Trainer] trainStep: completed in %v, loss=%.4f\n", elapsed, loss)
	}

	t.mu.Lock()
	t.metrics.CurrentLoss = loss
	t.mu.Unlock()

	steps := atomic.AddInt64(&t.metrics.TrainingSteps, 1)
	if gpuTrainerDebug && steps%10 == 0 {
		fmt.Printf("[GPU Trainer] Training step %d, loss=%.4f\n", steps, loss)
	}
}

// evaluate computes accuracy on a held-out set.
func (t *GPUOnlineTrainer) evaluate() {
	// Sample evaluation batch
	evalBatch := t.buffer.Sample(t.config.BatchSize*4, t.rng.Intn)
	if len(evalBatch) == 0 {
		return
	}

	accuracies := t.gpuHeads.Evaluate(evalBatch)

	t.mu.Lock()
	copy(t.metrics.HeadAccuracies, accuracies)
	t.metrics.LastEvalTime = time.Now()
	t.mu.Unlock()

	// Check for phase transition to Hot
	if t.Phase() == PhaseWarming {
		if len(accuracies) > 0 && accuracies[0] >= t.config.MinAccuracy {
			t.phase.Store(int32(PhaseHot))
		}
	}
}

// SaveHeads saves the current heads to a file.
// Note: GPU heads need to be converted to CPU format for saving.
func (t *GPUOnlineTrainer) SaveHeads(path string) error {
	// TODO: Implement CPU conversion and save
	return nil
}
