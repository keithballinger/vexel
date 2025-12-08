package cpu_test

import (
	"math"
	"testing"

	"vexel/inference/backend/cpu"
)

func TestSDPA(t *testing.T) {
	b := cpu.NewBackend()
	ops, ok := b.(interface {
		SDPA(q, k, v, out []float32, kvLen, numQHeads, numKVHeads, headDim int, scale float32)
	})
	if !ok {
		t.Fatal("Backend does not expose SDPA")
	}

	t.Run("single_head_single_position", func(t *testing.T) {
		// Q: [1, 4] - single head, headDim=4
		// K: [1, 1, 4] - single position, single KV head
		// V: [1, 1, 4]
		// With only one position, attention weight = 1.0, output = V

		headDim := 4
		numQHeads := 1
		numKVHeads := 1
		kvLen := 1

		q := []float32{1.0, 0.0, 0.0, 0.0}
		k := []float32{1.0, 0.0, 0.0, 0.0}
		v := []float32{0.5, 0.6, 0.7, 0.8}
		out := make([]float32, headDim)

		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		ops.SDPA(q, k, v, out, kvLen, numQHeads, numKVHeads, headDim, scale)

		// With single position, softmax = 1.0, output = V
		for i := 0; i < headDim; i++ {
			if absf(out[i]-v[i]) > 1e-5 {
				t.Errorf("out[%d]: expected %f, got %f", i, v[i], out[i])
			}
		}
	})

	t.Run("single_head_two_positions", func(t *testing.T) {
		// Q: [1, 4]
		// K: [2, 1, 4] - two positions
		// V: [2, 1, 4]

		headDim := 4
		numQHeads := 1
		numKVHeads := 1
		kvLen := 2

		// Q dot K[0] should be higher, so more weight on V[0]
		q := []float32{1.0, 0.0, 0.0, 0.0}
		k := []float32{
			1.0, 0.0, 0.0, 0.0, // pos 0 - matches Q perfectly
			0.0, 1.0, 0.0, 0.0, // pos 1 - orthogonal to Q
		}
		v := []float32{
			1.0, 1.0, 1.0, 1.0, // pos 0
			0.0, 0.0, 0.0, 0.0, // pos 1
		}
		out := make([]float32, headDim)

		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		ops.SDPA(q, k, v, out, kvLen, numQHeads, numKVHeads, headDim, scale)

		// Q dot K[0] = 1.0 * scale = 0.5
		// Q dot K[1] = 0.0 * scale = 0.0
		// After softmax: weight[0] > weight[1]
		// Since V[1] = 0, output should be mostly V[0] * weight[0]

		// Verify output is close to V[0] (weighted heavily)
		// Softmax([0.5, 0]) = [exp(0.5)/(exp(0.5)+1), 1/(exp(0.5)+1)]
		//                   ≈ [0.622, 0.378]
		// Output = 0.622 * V[0] + 0.378 * V[1] = 0.622 * [1,1,1,1] + 0.378 * [0,0,0,0]
		//        = [0.622, 0.622, 0.622, 0.622]
		expected := float32(math.Exp(0.5) / (math.Exp(0.5) + 1.0))
		for i := 0; i < headDim; i++ {
			if absf(out[i]-expected) > 1e-4 {
				t.Errorf("out[%d]: expected ~%f, got %f", i, expected, out[i])
			}
		}
	})

	t.Run("gqa_two_q_heads_one_kv_head", func(t *testing.T) {
		// GQA: 2 Q heads share 1 KV head
		headDim := 4
		numQHeads := 2
		numKVHeads := 1
		kvLen := 1

		q := []float32{
			1.0, 0.0, 0.0, 0.0, // Q head 0
			0.0, 1.0, 0.0, 0.0, // Q head 1
		}
		k := []float32{
			1.0, 1.0, 0.0, 0.0, // single KV head
		}
		v := []float32{
			0.1, 0.2, 0.3, 0.4, // single KV head
		}
		out := make([]float32, numQHeads*headDim)

		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		ops.SDPA(q, k, v, out, kvLen, numQHeads, numKVHeads, headDim, scale)

		// Both Q heads should output V (since softmax of single element = 1)
		for h := 0; h < numQHeads; h++ {
			for d := 0; d < headDim; d++ {
				expected := v[d]
				got := out[h*headDim+d]
				if absf(got-expected) > 1e-5 {
					t.Errorf("head %d, dim %d: expected %f, got %f", h, d, expected, got)
				}
			}
		}
	})

	t.Run("multi_position_causal", func(t *testing.T) {
		// Verify correct attention computation with multiple KV positions
		headDim := 2
		numQHeads := 1
		numKVHeads := 1
		kvLen := 3

		// Simple uniform query
		q := []float32{1.0, 1.0}
		k := []float32{
			1.0, 0.0, // pos 0
			0.0, 1.0, // pos 1
			1.0, 1.0, // pos 2 - matches Q best
		}
		v := []float32{
			1.0, 0.0, // pos 0
			0.0, 1.0, // pos 1
			0.5, 0.5, // pos 2
		}
		out := make([]float32, headDim)

		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		ops.SDPA(q, k, v, out, kvLen, numQHeads, numKVHeads, headDim, scale)

		// Q dot K[0] = 1.0, Q dot K[1] = 1.0, Q dot K[2] = 2.0
		// Scaled: 0.707, 0.707, 1.414
		// Position 2 should have highest weight
		// Output should be weighted average, biased towards V[2]

		// Just verify output is reasonable (between V values)
		for d := 0; d < headDim; d++ {
			if out[d] < 0.0 || out[d] > 1.0 {
				t.Errorf("out[%d] = %f out of expected range [0,1]", d, out[d])
			}
		}
	})
}

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
