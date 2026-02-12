package runtime

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

func TestArchitectureCompatibility(t *testing.T) {
	// Setup mock runtime environment
	b := cpu.NewCPUBackend()
	ctx := memory.NewInferenceContext(tensor.CPU)
	
	// Helper to create a dummy tensor
	newDummy := func() tensor.Tensor {
		return tensor.NewTensor(tensor.NewShape(1), tensor.Float32, tensor.DevicePtr{})
	}

	t.Run("Phi2_Mapping", func(t *testing.T) {
		cfg := Phi2()
		kvConfig := kv.NewKVConfig(tensor.Float32, 128, 16)
		cache, _ := kv.NewKVCache(ctx, kvConfig, 10)
		m, _ := NewModelRuntime(b, ctx, cache, cfg)

		// Test mapping
		m.mapTensor("model.layers.0.self_attn.qkv_proj.weight", newDummy())
		if m.layers[0].Wqkv.NumElements() == 0 {
			t.Error("Failed to map Phi-2 qkv_proj")
		}
		
		m.mapTensor("model.layers.0.mlp.fc1.weight", newDummy())
		if m.layers[0].W1.NumElements() == 0 {
			t.Error("Failed to map Phi-2 fc1")
		}
	})

	t.Run("Llama3_Mapping", func(t *testing.T) {
		cfg := Llama3_8B()
		kvConfig := kv.NewKVConfig(tensor.BFloat16, 128, 16)
		cache, _ := kv.NewKVCache(ctx, kvConfig, 10)
		m, _ := NewModelRuntime(b, ctx, cache, cfg)

		// LLaMA uses separate Q, K, V
		m.mapTensor("model.layers.5.self_attn.q_proj.weight", newDummy())
		if m.layers[5].Wq.NumElements() == 0 {
			t.Error("Failed to map LLaMA q_proj")
		}
		
		// LLaMA uses SwiGLU (Gate, Up, Down)
		m.mapTensor("model.layers.5.mlp.gate_proj.weight", newDummy())
		if m.layers[5].W1.NumElements() == 0 {
			t.Error("Failed to map LLaMA gate_proj")
		}
		
		m.mapTensor("model.layers.5.mlp.up_proj.weight", newDummy())
		if m.layers[5].W3.NumElements() == 0 {
			t.Error("Failed to map LLaMA up_proj")
		}
	})

	t.Run("Mistral_Mapping", func(t *testing.T) {
		// Mistral config (similar to LLaMA but usually with sliding window)
		cfg := Llama3_8B() // Reuse LLaMA base config
		cfg.SlidingWindow = 4096
		
		kvConfig := kv.NewKVConfig(tensor.Float16, 128, 16)
		cache, _ := kv.NewKVCache(ctx, kvConfig, 10)
		m, _ := NewModelRuntime(b, ctx, cache, cfg)

		// Verify sliding window config is preserved
		if m.layers[0].SlidingWindow != 4096 {
			t.Errorf("Expected sliding window 4096, got %d", m.layers[0].SlidingWindow)
		}

		// Test mapping (standard LLaMA naming)
		m.mapTensor("model.layers.0.self_attn.k_proj.weight", newDummy())
		if m.layers[0].Wk.NumElements() == 0 {
			t.Error("Failed to map Mistral k_proj")
		}
	})

	t.Run("ExecutionStructure", func(t *testing.T) {
		// Verify that LLaMA structure initializes correctly for execution
		cfg := Llama3_8B()
		// Use small dimensions for test
		cfg.HiddenSize = 64
		cfg.IntermediateSize = 128
		cfg.NumHiddenLayers = 1
		cfg.NumAttentionHeads = 4
		cfg.NumKeyValueHeads = 4
		
		kvConfig := kv.NewKVConfig(tensor.Float32, 16, 16)
		cache, _ := kv.NewKVCache(ctx, kvConfig, 10)
		m, _ := NewModelRuntime(b, ctx, cache, cfg)
		
		
		// We can't actually run Execute on CPU backend without full implementation,
		// but we can check if the plan/layers are set up.
		
		if len(m.layers) != 1 {
			t.Errorf("Expected 1 layer, got %d", len(m.layers))
		}
		
		// Check layer type
		if m.layers[0].MLPType != MLPSwiGLU {
			t.Errorf("Expected SwiGLU MLP for LLaMA, got %v", m.layers[0].MLPType)
		}
	})
}
