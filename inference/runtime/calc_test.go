package runtime_test

import (
	"testing"
	"vexel/inference/runtime"
)

func TestParameterCalculation(t *testing.T) {
	// Llama-3-8B config
	cfg := runtime.Llama3_8B()

	// Approximate param count check
	// 8B model should have ~8 billion params.
	// Allowing some margin for error in estimation.
	params := cfg.ApproxParams()
	
	if params < 7_500_000_000 || params > 8_500_000_000 {
		t.Errorf("Expected ~8B params, got %d", params)
	}
}
