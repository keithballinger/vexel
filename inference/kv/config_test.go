package kv_test

import (
	"testing"
	"vexel/inference/kv"
	"vexel/inference/tensor"
)

func TestKVConfig(t *testing.T) {
	// Define parameters
	dtype := tensor.Float16
	headDim := 128
	blockLen := 16 // Paged attention block size

	cfg := kv.NewKVConfig(dtype, headDim, blockLen)

	if cfg.DType != dtype {
		t.Errorf("KVConfig.DType = %v, want %v", cfg.DType, dtype)
	}
	if cfg.HeadDim != headDim {
		t.Errorf("KVConfig.HeadDim = %v, want %v", cfg.HeadDim, headDim)
	}
	if cfg.BlockLen != blockLen {
		t.Errorf("KVConfig.BlockLen = %v, want %v", cfg.BlockLen, blockLen)
	}
	
	// Test BlockBytes calculation
	// 2 for FP16 * 128 head dim * 16 block len * 2 (K and V) = 8192 bytes
	expectedBytes := 2 * headDim * blockLen * 2
	if got := cfg.BlockBytes(); got != expectedBytes {
		t.Errorf("KVConfig.BlockBytes() = %v, want %v", got, expectedBytes)
	}
}
