//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"
)

// cpuGELU computes GELU activation (fast approximation).
func cpuGELU(x float32) float32 {
	const sqrtTwoPi = 0.7978845608 // sqrt(2/π)
	const coeff = 0.044715

	if x > 10 {
		return x
	}
	if x < -10 {
		return 0
	}
	x3 := x * x * x
	tanhArg := sqrtTwoPi * (x + coeff*x3)
	tanhVal := float32(math.Tanh(float64(tanhArg)))
	return 0.5 * x * (1.0 + tanhVal)
}

// TestGELUMul verifies the fused GELU-gated multiply kernel (GeGLU for Gemma).
//
// Track 6: Gemma Architecture, Phase 1 Task 2.
func TestGELUMul(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	n := 4096
	gate := make([]float32, n)
	up := make([]float32, n)

	// Fill with varied values including negative, zero, and large values
	for i := 0; i < n; i++ {
		gate[i] = float32(i-n/2) * 0.01 // Range: -20.48 to 20.47
		up[i] = float32(i) * 0.005       // Range: 0 to 20.475
	}

	// Compute CPU reference
	expected := make([]float32, n)
	for i := 0; i < n; i++ {
		expected[i] = cpuGELU(gate[i]) * up[i]
	}

	// Run on GPU
	gateBuf := be.Alloc(n * 4)
	upBuf := be.Alloc(n * 4)
	outBuf := be.Alloc(n * 4)

	be.ToDevice(gateBuf, float32ToBytes(gate))
	be.ToDevice(upBuf, float32ToBytes(up))
	be.Sync()

	be.GELUMul(gateBuf, upBuf, outBuf, n)
	be.Sync()

	outBytes := make([]byte, n*4)
	be.ToHost(outBytes, outBuf)
	result := bytesToFloat32(outBytes)

	// Compare
	maxDiff := float32(0)
	for i := 0; i < n; i++ {
		diff := float32(math.Abs(float64(result[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		// Check for NaN
		if math.IsNaN(float64(result[i])) {
			t.Fatalf("NaN at index %d (gate=%f, up=%f)", i, gate[i], up[i])
		}
	}

	// Allow small floating-point tolerance
	if maxDiff > 1e-4 {
		t.Errorf("Max diff: %e (exceeds tolerance 1e-4)", maxDiff)
	}

	t.Logf("GELUMul verified: n=%d, maxDiff=%e", n, maxDiff)

	be.Free(gateBuf)
	be.Free(upBuf)
	be.Free(outBuf)
}

// TestGELUMulEdgeCases verifies GELUMul handles edge cases correctly.
func TestGELUMulEdgeCases(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	tests := []struct {
		name       string
		gate, up   float32
		wantApprox float32
	}{
		{"zero_gate", 0.0, 1.0, 0.0},           // GELU(0) * 1 = 0
		{"zero_up", 1.0, 0.0, 0.0},               // GELU(1) * 0 = 0
		{"positive", 1.0, 2.0, cpuGELU(1.0) * 2.0},
		{"negative_gate", -1.0, 2.0, cpuGELU(-1.0) * 2.0},
		{"large_positive", 15.0, 1.0, 15.0},     // GELU(15) ≈ 15
		{"large_negative", -15.0, 1.0, 0.0},     // GELU(-15) ≈ 0
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gateBuf := be.Alloc(4)
			upBuf := be.Alloc(4)
			outBuf := be.Alloc(4)

			be.ToDevice(gateBuf, float32ToBytes([]float32{tt.gate}))
			be.ToDevice(upBuf, float32ToBytes([]float32{tt.up}))
			be.Sync()

			be.GELUMul(gateBuf, upBuf, outBuf, 1)
			be.Sync()

			outBytes := make([]byte, 4)
			be.ToHost(outBytes, outBuf)
			result := bytesToFloat32(outBytes)[0]

			diff := float32(math.Abs(float64(result - tt.wantApprox)))
			if diff > 1e-4 {
				t.Errorf("gate=%f, up=%f: got %f, want ~%f (diff=%e)",
					tt.gate, tt.up, result, tt.wantApprox, diff)
			}

			be.Free(gateBuf)
			be.Free(upBuf)
			be.Free(outBuf)
		})
	}
}

// TestGELUMulProductionSize tests GELUMul at Gemma 2B dimensions.
func TestGELUMulProductionSize(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	// Gemma 2B intermediate size = 16384
	n := 16384
	gate := make([]float32, n)
	up := make([]float32, n)
	for i := 0; i < n; i++ {
		gate[i] = (float32(i%1000) - 500) * 0.01
		up[i] = (float32(i%777) - 388) * 0.01
	}

	expected := make([]float32, n)
	for i := 0; i < n; i++ {
		expected[i] = cpuGELU(gate[i]) * up[i]
	}

	gateBuf := be.Alloc(n * 4)
	upBuf := be.Alloc(n * 4)
	outBuf := be.Alloc(n * 4)

	be.ToDevice(gateBuf, float32ToBytes(gate))
	be.ToDevice(upBuf, float32ToBytes(up))
	be.Sync()

	be.GELUMul(gateBuf, upBuf, outBuf, n)
	be.Sync()

	outBytes := make([]byte, n*4)
	be.ToHost(outBytes, outBuf)
	result := bytesToFloat32(outBytes)

	maxDiff := float32(0)
	for i := 0; i < n; i++ {
		diff := float32(math.Abs(float64(result[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > 1e-4 {
		t.Errorf("Production size: maxDiff=%e (exceeds tolerance)", maxDiff)
	}

	t.Logf("GELUMul at Gemma 2B size: n=%d, maxDiff=%e", n, maxDiff)

	be.Free(gateBuf)
	be.Free(upBuf)
	be.Free(outBuf)
}
