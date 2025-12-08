package cpu_test

import (
	"math"
	"testing"
	"vexel/inference/backend/cpu"
)

func TestRoPE(t *testing.T) {
	// Input: 1 token, 1 head, head_dim=2
	// Position 0
	// Value = [1, 0]
	// Theta = 1.0 (for simplicity)
	// Rotation matrix for pos 0:
	// cos(0) -sin(0)  ->  1  0
	// sin(0)  cos(0)      0  1
	// Result: [1, 0] (No change at pos 0)

	b := cpu.NewBackend()
	ops, ok := b.(interface {
		RoPE(q, k []float32, headDim, numHeads, seqLen, startPos int, theta float32)
	})
	if !ok {
		t.Fatal("Backend does not expose RoPE")
	}

	// Test Pos 0
	// We pass input as both q and k for simplicity, or nil for k.
	// Function likely rotates in-place or out-of-place.
	// Ideally RoPE rotates Q and K in place.

	q := []float32{1, 0}
	// headDim=2, numHeads=1, seqLen=1, startPos=0
	ops.RoPE(q, nil, 2, 1, 1, 0, 10000.0)

	if q[0] != 1 || q[1] != 0 {
		t.Errorf("Pos 0 should not change. Got %v", q)
	}

	// Test Pos 1, Theta=1.0 (Freq = 1.0 for dim 0?)
	// RoPE formula:
	// theta_i = 10000 ^ (-2(i-1)/d) ... usually
	// freq = 1 / (theta ^ (i / dim))
	// For dim 0 (i=0): freq = 1 / (10000^0) = 1
	// Rotation angle = pos * freq = 1 * 1 = 1 radian.
	// Input [1, 0]
	// Rotated:
	// x' = x*cos(1) - y*sin(1) = 1*0.54 - 0 = 0.54
	// y' = x*sin(1) + y*cos(1) = 1*0.84 + 0 = 0.84

	q2 := []float32{1, 0}
	// headDim=2, numHeads=1, seqLen=1, startPos=1
	ops.RoPE(q2, nil, 2, 1, 1, 1, 1.0) // theta base 1.0 means freq is always 1.0?
	// Wait, theta parameter usually refers to the base (e.g. 10000 or 500000).
	// If base=1.0, then freq = 1.0^(-...) = 1.0.

	expectedX := float32(math.Cos(1.0))
	expectedY := float32(math.Sin(1.0))

	if abs(q2[0]-expectedX) > 1e-4 || abs(q2[1]-expectedY) > 1e-4 {
		t.Errorf("Pos 1 rotation failed. Expected [%f, %f], got [%f, %f]", expectedX, expectedY, q2[0], q2[1])
	}
}
