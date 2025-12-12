//go:build metal && darwin && cgo

package scheduler

import (
	"vexel/inference/backend"
	"vexel/inference/medusa"
)

// createGPUTrainer creates a GPU-accelerated trainer if available.
func createGPUTrainer(hiddenSize, vocabSize int, config medusa.OnlineConfig, b backend.Backend) medusa.Trainer {
	return medusa.NewGPUOnlineTrainer(hiddenSize, vocabSize, config, b)
}

// gpuTrainingAvailable returns true if GPU training is available.
func gpuTrainingAvailable() bool {
	return true
}
