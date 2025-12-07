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
	// layers []BlockRuntime // To be added later
}

// NewModelRuntime initializes a new model runtime.
func NewModelRuntime(backend cpu.Backend, ctx *memory.InferenceContext, cache *kv.KVCache, config ModelConfig) (*ModelRuntime, error) {
	// TODO: Load weights and initialize layers here.
	
	return &ModelRuntime{
		backend: backend,
		ctx:     ctx,
		cache:   cache,
		config:  config,
	}, nil
}

// LoadWeights loads model weights from a file.
func (m *ModelRuntime) LoadWeights(path string) error {
	// TODO: Implement safetensors loading
	return nil
}

// Config returns the model configuration.
func (m *ModelRuntime) Config() ModelConfig {
	return m.config
}
