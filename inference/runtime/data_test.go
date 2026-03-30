package runtime_test

import (
	"os"
	"path/filepath"
	"testing"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestRealWeightValues(t *testing.T) {
	// 1. Load TinyLlama
	cwd, _ := os.Getwd()
	root := filepath.Join(cwd, "../..")
	modelPath := filepath.Join(root, "models", "tiny_model.safetensors")

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("TinyLlama model not found")
	}

	cfg := runtime.Llama3_8B() // Structure matches
	// Reduce layers to save memory for test, although mmap handles it.
	// But we load FULL model file, so we should keep config consistent or LoadWeights might complain/mismatch.
	// Let's use correct TinyLlama config params roughly.
	cfg.NumHiddenLayers = 22
	cfg.DType = tensor.Float32 // Force conversion to F32 so we can read values on CPU

	rt, err := runtime.NewModelRuntime(nil, nil, nil, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if err := rt.LoadWeights(modelPath); err != nil {
		t.Fatal(err)
	}

	// 2. Access a known weight tensor
	// E.g. Block 0, Attention Output Weight (wo) or RMSNorm weight.
	// We need access to layers. Layers are private?
	// NewModelRuntime returns *ModelRuntime.
	// Layers are private. I need to expose them for testing or use a "GetWeight" helper.
	// Or check public `Config()`? No.

	// I'll add a helper `GetLayer(i)` to ModelRuntime or similar for introspection?
	// Or just inspect via reflection/unsafe in test?
	// Better: Add `Layer(i)` method.

	layer0 := rt.Layer(0)
	if layer0 == nil {
		t.Fatal("Layer 0 not initialized")
	}

	// Check if tensor has non-nil pointer
	if layer0.AttnNorm.DevicePtr().IsNil() {
		t.Error("Layer 0 AttnNorm weights are nil (not loaded)")
	}

	t.Logf("Layer 0 AttnNorm DType: %v", layer0.AttnNorm.DType())

	// Check if data is non-zero (safetensors weights shouldn't be all zero)
	// WARNING: If DType is BF16/F16, ToFloat32Slice will misinterpret stride.
	// We need to check DType before casting.

	if layer0.AttnNorm.DType() == tensor.BFloat16 || layer0.AttnNorm.DType() == tensor.Float16 {
		// Just check first few bytes
		// We need a ToByteSlice helper or unsafe access
		// For now skipping value check if not F32 to avoid crash/garbage
		t.Log("Skipping value check for non-F32 weights")
		return
	}

	slice := tensor.ToFloat32Slice(layer0.AttnNorm)
	sum := float32(0)
	for _, v := range slice {
		sum += v
	}

	if sum == 0 {
		t.Error("Layer 0 AttnNorm weights are all zero")
	}
}
