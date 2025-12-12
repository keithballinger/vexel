//go:build metal && darwin && cgo

package scheduler

import (
	"vexel/inference/backend"
	"vexel/inference/medusa"
)

// createGPUTrainer creates a GPU-accelerated trainer if available.
// lmHeadWeights should be the base model's output projection (lm_head) weights
// in [vocab_size, hidden_size] layout for initializing Medusa heads.
func createGPUTrainer(hiddenSize, vocabSize int, config medusa.OnlineConfig, b backend.Backend, lmHeadWeights []float32) medusa.Trainer {
	return medusa.NewGPUOnlineTrainerWithInit(hiddenSize, vocabSize, config, b, lmHeadWeights)
}

// gpuTrainingAvailable returns true if GPU training is available.
func gpuTrainingAvailable() bool {
	return true
}
