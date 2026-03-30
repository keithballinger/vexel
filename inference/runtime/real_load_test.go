package runtime_test

import (
	"os"
	"path/filepath"
	"testing"
	"vexel/inference/runtime"
)

func TestRealModelLoading(t *testing.T) {
	// Check if model exists
	cwd, _ := os.Getwd()
	// Navigate up to root from inference/runtime
	root := filepath.Join(cwd, "../..")
	modelPath := filepath.Join(root, "models", "tiny_model.safetensors")

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("TinyLlama model not found, skipping integration test")
	}

	// Initialize runtime
	cfg := runtime.Llama3_8B() // TinyLlama is Llama-2 based, architecture is compatible
	rt, err := runtime.NewModelRuntime(nil, nil, nil, cfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	// Attempt to load weights
	if err := rt.LoadWeights(modelPath); err != nil {
		t.Fatalf("Failed to load real weights: %v", err)
	}

	t.Log("Successfully parsed TinyLlama safetensors header!")
}
