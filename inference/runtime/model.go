package runtime

import (
	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

// ModelRuntime manages the execution of a model.
type ModelRuntime struct {
	backend cpu.Backend
	ctx     *memory.InferenceContext
	cache   *kv.KVCache
	config  ModelConfig
	layers  []*BlockRuntime

	// Global weights
	Embedding  tensor.Tensor
	FinalNorm  tensor.Tensor
	OutputHead tensor.Tensor

	// Keep mapped file alive
	mappedFile interface{ Close() error }

	// Keep converted weight slices alive to prevent GC
	keepAlive [][]float32
}

// NewModelRuntime initializes a new model runtime.
func NewModelRuntime(backend cpu.Backend, ctx *memory.InferenceContext, cache *kv.KVCache, config ModelConfig) (*ModelRuntime, error) {
	// Initialize layers with config for GQA support
	layers := make([]*BlockRuntime, config.NumHiddenLayers)
	for i := range layers {
		layers[i] = NewBlockRuntime(backend, config)
	}
	
	return &ModelRuntime{
		backend: backend,
		ctx:     ctx,
		cache:   cache,
		config:  config,
		layers:  layers,
	}, nil
}

// Config returns the model configuration.
func (m *ModelRuntime) Config() ModelConfig {
	return m.config
}

// Layer returns the block runtime at the given index.
func (m *ModelRuntime) Layer(i int) *BlockRuntime {
	if i < 0 || i >= len(m.layers) {
		return nil
	}
	return m.layers[i]
}
