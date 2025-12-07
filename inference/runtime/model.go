package runtime

import (
	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
	"vexel/inference/memory"
)

// ModelRuntime manages the execution of a model.
type ModelRuntime struct {
	backend cpu.Backend
	ctx     *memory.InferenceContext
	cache   *kv.KVCache
	config  ModelConfig
	layers  []*BlockRuntime
}

// NewModelRuntime initializes a new model runtime.
func NewModelRuntime(backend cpu.Backend, ctx *memory.InferenceContext, cache *kv.KVCache, config ModelConfig) (*ModelRuntime, error) {
	// Initialize layers
	layers := make([]*BlockRuntime, config.NumHiddenLayers)
	for i := range layers {
		layers[i] = NewBlockRuntime(backend)
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
