package runtime

import (
	"fmt"
	"os"
	"vexel/inference/pkg/safetensors"
)

// LoadWeights loads model weights from a safetensors file.
func (m *ModelRuntime) LoadWeights(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open weights file: %w", err)
	}
	defer f.Close()

	// Parse header
	header, err := safetensors.Load(f)
	if err != nil {
		return fmt.Errorf("failed to parse safetensors header: %w", err)
	}

	// TODO: Iterate over header entries and load tensors into m.ctx
	// For now, we just validate that we found expected keys.
	
	// Example check
	if _, ok := header["model.embed_tokens.weight"]; !ok {
		// Try alternative naming if needed, or log warning.
		// TinyLlama uses "model.embed_tokens.weight"
	}

	return nil
}
