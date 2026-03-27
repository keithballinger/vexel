//go:build metal && darwin && cgo

package medusa

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"vexel/inference/backend/metal"
)

func TestGPUTrainerSaveHeadsNilHeads(t *testing.T) {
	trainer := &GPUOnlineTrainer{
		gpuHeads: nil,
		config: OnlineConfig{
			NumHeads: 2,
		},
		metrics: TrainingMetrics{
			HeadAccuracies: make([]float32, 2),
		},
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}

	err := trainer.SaveHeads("/tmp/test_nil_heads.bin")
	if err == nil {
		t.Fatal("expected error when gpuHeads is nil, got nil")
	}
	if err.Error() != "gpu heads not initialized" {
		t.Fatalf("unexpected error message: %s", err.Error())
	}
}

func TestGPUTrainerSaveHeadsRoundTrip(t *testing.T) {
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	numHeads := 2
	hiddenSize := 16
	vocabSize := 32

	gpuHeads := NewGPUHeads(numHeads, hiddenSize, vocabSize, b)
	defer gpuHeads.Free()

	trainer := &GPUOnlineTrainer{
		gpuHeads: gpuHeads,
		backend:  b,
		config: OnlineConfig{
			NumHeads:       numHeads,
			TrainInterval:  time.Second,
			EvalInterval:   time.Second,
			BatchSize:      4,
			LearningRate:   0.001,
			BufferCapacity: 100,
			WarmupSamples:  10,
			MinAccuracy:    0.1,
		},
		metrics: TrainingMetrics{
			HeadAccuracies: make([]float32, numHeads),
		},
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}

	// Save to temp file
	dir := t.TempDir()
	path := filepath.Join(dir, "heads.bin")

	err = trainer.SaveHeads(path)
	if err != nil {
		t.Fatalf("SaveHeads failed: %v", err)
	}

	// Verify file exists and has content
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat saved file: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	// Load back and verify dimensions
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if loaded.NumHeads != numHeads {
		t.Errorf("NumHeads mismatch: got %d, want %d", loaded.NumHeads, numHeads)
	}
	if loaded.HiddenSize != hiddenSize {
		t.Errorf("HiddenSize mismatch: got %d, want %d", loaded.HiddenSize, hiddenSize)
	}
	if loaded.VocabSize != vocabSize {
		t.Errorf("VocabSize mismatch: got %d, want %d", loaded.VocabSize, vocabSize)
	}

	// Verify weight data matches what's on GPU
	cpuHeads := gpuHeads.ToCPUHeads()
	for i := 0; i < numHeads; i++ {
		for j := 0; j < hiddenSize*hiddenSize; j++ {
			if loaded.heads[i].FC1[j] != cpuHeads.heads[i].FC1[j] {
				t.Errorf("head %d FC1[%d] mismatch: got %f, want %f",
					i, j, loaded.heads[i].FC1[j], cpuHeads.heads[i].FC1[j])
				break
			}
		}
		for j := 0; j < hiddenSize*vocabSize; j++ {
			if loaded.heads[i].FC2[j] != cpuHeads.heads[i].FC2[j] {
				t.Errorf("head %d FC2[%d] mismatch: got %f, want %f",
					i, j, loaded.heads[i].FC2[j], cpuHeads.heads[i].FC2[j])
				break
			}
		}
	}
}

func TestGPUHeadsToCPUHeads(t *testing.T) {
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	numHeads := 3
	hiddenSize := 8
	vocabSize := 16

	gpuHeads := NewGPUHeads(numHeads, hiddenSize, vocabSize, b)
	defer gpuHeads.Free()

	cpuHeads := gpuHeads.ToCPUHeads()

	if cpuHeads.NumHeads != numHeads {
		t.Errorf("NumHeads: got %d, want %d", cpuHeads.NumHeads, numHeads)
	}
	if cpuHeads.HiddenSize != hiddenSize {
		t.Errorf("HiddenSize: got %d, want %d", cpuHeads.HiddenSize, hiddenSize)
	}
	if cpuHeads.VocabSize != vocabSize {
		t.Errorf("VocabSize: got %d, want %d", cpuHeads.VocabSize, vocabSize)
	}
	if len(cpuHeads.heads) != numHeads {
		t.Fatalf("heads count: got %d, want %d", len(cpuHeads.heads), numHeads)
	}

	for i := 0; i < numHeads; i++ {
		if len(cpuHeads.heads[i].FC1) != hiddenSize*hiddenSize {
			t.Errorf("head %d FC1 length: got %d, want %d", i, len(cpuHeads.heads[i].FC1), hiddenSize*hiddenSize)
		}
		if len(cpuHeads.heads[i].FC2) != hiddenSize*vocabSize {
			t.Errorf("head %d FC2 length: got %d, want %d", i, len(cpuHeads.heads[i].FC2), hiddenSize*vocabSize)
		}

		// Verify weights match the CPU copies that were uploaded to GPU during init
		for j, v := range cpuHeads.heads[i].FC1 {
			if v != gpuHeads.heads[i].FC1CPU[j] {
				t.Errorf("head %d FC1[%d]: got %f, want %f", i, j, v, gpuHeads.heads[i].FC1CPU[j])
				break
			}
		}
		for j, v := range cpuHeads.heads[i].FC2 {
			if v != gpuHeads.heads[i].FC2CPU[j] {
				t.Errorf("head %d FC2[%d]: got %f, want %f", i, j, v, gpuHeads.heads[i].FC2CPU[j])
				break
			}
		}
	}
}
