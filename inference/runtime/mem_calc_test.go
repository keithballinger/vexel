package runtime_test

import (
	"testing"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestWeightsBytes(t *testing.T) {
	cfg := runtime.Llama3_8B()
	// params := cfg.ApproxParams()

	// Case 1: BF16 (2 bytes per param)
	// We expect ~8B params * 2 = ~16 GB
	bytesBF16 := cfg.WeightsBytes(tensor.QuantNone)

	// Check rough range (14GB - 17GB)
	if bytesBF16 < 14_000_000_000 || bytesBF16 > 17_000_000_000 {
		t.Errorf("Expected ~16GB for BF16, got %d", bytesBF16)
	}

	// Case 2: Q4_0 (4 bits per param = 0.5 bytes)
	// We expect ~8B params * 0.5 = ~4 GB
	bytesQ4 := cfg.WeightsBytes(tensor.Q4_0)

	if bytesQ4 < 3_500_000_000 || bytesQ4 > 4_500_000_000 {
		t.Errorf("Expected ~4GB for Q4_0, got %d", bytesQ4)
	}

	// Sanity: Q4 should be ~1/4 of BF16
	ratio := float64(bytesBF16) / float64(bytesQ4)
	if ratio < 3.8 || ratio > 4.2 {
		t.Errorf("Expected 4x compression, got %.2fx", ratio)
	}
}
