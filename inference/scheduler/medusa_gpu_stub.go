//go:build !metal || !darwin || !cgo

package scheduler

import (
	"vexel/inference/backend"
	"vexel/inference/medusa"
)

// createGPUTrainer returns nil when GPU training is not available.
func createGPUTrainer(hiddenSize, vocabSize int, config medusa.OnlineConfig, b backend.Backend) medusa.Trainer {
	return nil
}

// gpuTrainingAvailable returns false when GPU training is not available.
func gpuTrainingAvailable() bool {
	return false
}
