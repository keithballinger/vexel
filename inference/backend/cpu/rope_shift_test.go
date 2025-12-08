package cpu_test

import (
	"math"
	"testing"

	"vexel/inference/backend/cpu"
)

func TestRoPEShift(t *testing.T) {
	backend := cpu.NewBackend()
	ops := backend.(interface {
		RoPE(q, k []float32, headDim, numHeads, seqLen, startPos int, theta float32)
		RoPEShift(k []float32, headDim, numKVHeads, numTokens, shift int, theta float32)
	})

	theta := float32(10000.0)
	headDim := 4
	numKVHeads := 2

	t.Run("shift_equivalence", func(t *testing.T) {
		// Test that RoPE(pos) followed by RoPEShift(shift) equals RoPE(pos+shift)
		// Create two identical K vectors
		k1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0} // [1 token, 2 heads, 4 dim]
		k2 := make([]float32, len(k1))
		copy(k2, k1)

		// Method 1: Apply RoPE at position 5 directly
		ops.RoPE(nil, k1, headDim, numKVHeads, 1, 5, theta)

		// Method 2: Apply RoPE at position 2, then shift by 3
		ops.RoPE(nil, k2, headDim, numKVHeads, 1, 2, theta)
		ops.RoPEShift(k2, headDim, numKVHeads, 1, 3, theta)

		// Results should be identical (RoPE is multiplicative in angle space)
		for i := range k1 {
			if absFloat(k1[i]-k2[i]) > 1e-5 {
				t.Errorf("position %d: RoPE(5) = %f, RoPE(2)+Shift(3) = %f", i, k1[i], k2[i])
			}
		}
	})

	t.Run("zero_shift_identity", func(t *testing.T) {
		// Shift of 0 should not change anything
		k := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
		kOrig := make([]float32, len(k))
		copy(kOrig, k)

		ops.RoPEShift(k, headDim, numKVHeads, 1, 0, theta)

		for i := range k {
			if k[i] != kOrig[i] {
				t.Errorf("shift(0) changed value at %d: %f -> %f", i, kOrig[i], k[i])
			}
		}
	})

	t.Run("multiple_tokens", func(t *testing.T) {
		// Test shifting multiple tokens at once
		// Create K for 3 tokens
		k := make([]float32, 3*numKVHeads*headDim)
		for i := range k {
			k[i] = float32(i + 1)
		}
		kBefore := make([]float32, len(k))
		copy(kBefore, k)

		// Apply shift
		ops.RoPEShift(k, headDim, numKVHeads, 3, 10, theta)

		// All tokens should be modified
		changed := 0
		for i := range k {
			if k[i] != kBefore[i] {
				changed++
			}
		}
		if changed == 0 {
			t.Error("expected some values to change after shift")
		}
	})

	t.Run("fragment_caching_scenario", func(t *testing.T) {
		// Simulate fragment caching scenario:
		// 1. Cache system prompt at positions 0,1,2 with RoPE(0,1,2)
		// 2. User message comes in at position 3
		// 3. When using cached KV, shift system prompt K by 0 (no shift needed if inserted at 0)
		// 4. But if system prompt is inserted at position N, shift by N

		// Create "cached" K with RoPE at positions 0,1,2
		cachedK := make([]float32, 3*numKVHeads*headDim)
		for i := range cachedK {
			cachedK[i] = float32(i%headDim + 1)
		}
		// Apply RoPE at positions 0,1,2
		ops.RoPE(nil, cachedK, headDim, numKVHeads, 3, 0, theta)

		// Now create "fresh" K with RoPE starting at position 5,6,7
		freshK := make([]float32, 3*numKVHeads*headDim)
		for i := range freshK {
			freshK[i] = float32(i%headDim + 1)
		}
		ops.RoPE(nil, freshK, headDim, numKVHeads, 3, 5, theta)

		// Shift cached K by 5 to match positions 5,6,7
		ops.RoPEShift(cachedK, headDim, numKVHeads, 3, 5, theta)

		// Now cachedK should equal freshK
		for i := range cachedK {
			if absFloat(cachedK[i]-freshK[i]) > 1e-5 {
				t.Errorf("position %d: shifted cached = %f, fresh = %f", i, cachedK[i], freshK[i])
			}
		}
	})
}

func absFloat(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func TestRoPEShiftMathematicalProperties(t *testing.T) {
	backend := cpu.NewBackend()
	ops := backend.(interface {
		RoPEShift(k []float32, headDim, numKVHeads, numTokens, shift int, theta float32)
	})

	theta := float32(10000.0)
	headDim := 4
	numKVHeads := 1

	t.Run("rotation_preserves_magnitude", func(t *testing.T) {
		// RoPE is a rotation, so magnitude should be preserved
		// RoPE rotates pairs: (k[j], k[j+halfDim]) for j in [0, halfDim)
		// For headDim=4, halfDim=2: pairs are (k[0],k[2]) and (k[1],k[3])
		k := []float32{3.0, 0.0, 4.0, 0.0} // pair (3,4) has magnitude 5

		// Magnitude of pair (k[0], k[2])
		magBefore := float32(math.Sqrt(float64(k[0]*k[0] + k[2]*k[2])))

		ops.RoPEShift(k, headDim, numKVHeads, 1, 7, theta)

		magAfter := float32(math.Sqrt(float64(k[0]*k[0] + k[2]*k[2])))

		if absFloat(magBefore-magAfter) > 1e-5 {
			t.Errorf("magnitude changed: %f -> %f", magBefore, magAfter)
		}
	})

	t.Run("shift_is_additive", func(t *testing.T) {
		// RoPEShift(a) followed by RoPEShift(b) should equal RoPEShift(a+b)
		k1 := []float32{1.0, 2.0, 3.0, 4.0}
		k2 := make([]float32, len(k1))
		copy(k2, k1)

		// Method 1: Shift by 3, then by 5
		ops.RoPEShift(k1, headDim, numKVHeads, 1, 3, theta)
		ops.RoPEShift(k1, headDim, numKVHeads, 1, 5, theta)

		// Method 2: Shift by 8
		ops.RoPEShift(k2, headDim, numKVHeads, 1, 8, theta)

		for i := range k1 {
			if absFloat(k1[i]-k2[i]) > 1e-5 {
				t.Errorf("position %d: shift(3)+shift(5) = %f, shift(8) = %f", i, k1[i], k2[i])
			}
		}
	})
}
