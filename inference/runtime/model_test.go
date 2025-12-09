package runtime_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestModelRuntime(t *testing.T) {
	// Setup dependencies
	backend := cpu.NewCPUBackend()
	loc := tensor.CPU
	ctx := memory.NewInferenceContext(loc)
	
	// Need KV arena for cache
	ctx.AddArena(memory.KV, 1024*1024)
	
	// Configs
	kvCfg := kv.NewKVConfig(tensor.Float16, 64, 16)
	modelCfg := runtime.Llama3_8B()
	
	cache, err := kv.NewKVCache(ctx, kvCfg, 100)
	if err != nil {
		t.Fatalf("Failed to create KV cache: %v", err)
	}

	// Create ModelRuntime
	rt, err := runtime.NewModelRuntime(backend, ctx, cache, modelCfg)
	if err != nil {
		t.Fatalf("NewModelRuntime failed: %v", err)
	}

	if rt == nil {
		t.Fatal("ModelRuntime should not be nil")
	}

	// Verify internal state matches config
	if rt.Config().HiddenSize != modelCfg.HiddenSize {
		t.Error("ModelRuntime config mismatch")
	}
}
