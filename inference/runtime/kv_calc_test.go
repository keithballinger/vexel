package runtime_test

import (
	"testing"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestKVBytes(t *testing.T) {
	// Llama-3 8B
	// 32 layers
	// 8 KV heads
	// Hidden 4096 / 32 heads = 128 head dim
	// 2 bytes per element (KV cache usually FP16/BF16)
	
	cfg := runtime.Llama3_8B()
	
	// Check for a single token, single sequence
	// KV size per token = 2 (K+V) * NumLayers * NumKVHeads * HeadDim * DTypeSize
	// = 2 * 32 * 8 * 128 * 2
	// = 131,072 bytes per token (~128 KB)
	
	// Test case: 1 sequence, 1024 context length
	activeSeqs := 1
	contextLen := 1024
	
	// We might need to pass KV config if it supports quantization (e.g. FP8 KV cache).
	// For now assuming default precision matching model DType.
	
	bytes := cfg.KVBytes(activeSeqs, contextLen, tensor.QuantNone)
	
	expected := int64(131072 * 1024) // ~134 MB
	
	if bytes != expected {
		t.Errorf("Expected %d bytes for KV cache, got %d", expected, bytes)
	}
}
