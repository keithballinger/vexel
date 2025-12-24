//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"
)

func TestLayerNorm(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	rows, cols := 2, 4
	eps := float32(1e-5)

	input := []float32{
		1, 2, 3, 4,
		-1, 0, 1, 2,
	}
	weight := []float32{1, 1, 1, 1}
	bias := []float32{0, 0, 0, 0}

	xBuf := b.Alloc(len(input) * 4)
	wBuf := b.Alloc(len(weight) * 4)
	bBuf := b.Alloc(len(bias) * 4)
	outBuf := b.Alloc(len(input) * 4)
	defer b.Free(xBuf)
	defer b.Free(wBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(xBuf, float32ToBytes(input))
	b.ToDevice(wBuf, float32ToBytes(weight))
	b.ToDevice(bBuf, float32ToBytes(bias))

	b.LayerNorm(xBuf, wBuf, bBuf, outBuf, rows, cols, eps)
	b.Sync()

	resultBytes := make([]byte, len(input)*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	// Reference check for first row
	// Mean = (1+2+3+4)/4 = 2.5
	// Var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
	// Out[0] = (1 - 2.5) / sqrt(1.25 + 1e-5) = -1.5 / 1.118 = -1.3416
	expected0 := float32((1.0 - 2.5) / math.Sqrt(1.25+1e-5))
	if math.Abs(float64(result[0]-expected0)) > 1e-3 {
		t.Errorf("LayerNorm row 0 mismatch: got %f, want %f", result[0], expected0)
	}
}

func TestGELU(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	input := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	n := len(input)

	xBuf := b.Alloc(n * 4)
	outBuf := b.Alloc(n * 4)
	defer b.Free(xBuf)
	defer b.Free(outBuf)

	b.ToDevice(xBuf, float32ToBytes(input))
	b.GELU(xBuf, outBuf, n)
	b.Sync()

	resultBytes := make([]byte, n*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	// GELU(0) = 0
	if math.Abs(float64(result[2])) > 1e-5 {
		t.Errorf("GELU(0) should be 0, got %f", result[2])
	}
	// GELU(2) is approx 1.96
	if result[4] < 1.9 {
		t.Errorf("GELU(2) should be ~1.96, got %f", result[4])
	}
}

func TestRoPEPhi2(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Phi-2 style: headDim=80, ropeDim=32
	headDim := 80
	ropeDim := 32
	numHeads := 1
	numKVHeads := 1
	seqLen := 1
	startPos := 1
	theta := float32(10000.0)

	q := make([]float32, headDim)
	k := make([]float32, headDim)
	for i := range q {
		q[i] = 1.0
		k[i] = 1.0
	}

	qBuf := b.Alloc(len(q) * 4)
	kBuf := b.Alloc(len(k) * 4)
	defer b.Free(qBuf)
	defer b.Free(kBuf)

	b.ToDevice(qBuf, float32ToBytes(q))
	b.ToDevice(kBuf, float32ToBytes(k))

	// ropeNeox = true for Phi-2
	b.RoPE(qBuf, kBuf, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim, theta, true)
	b.Sync()

	resultQBytes := make([]byte, len(q)*4)
	b.ToHost(resultQBytes, qBuf)
	resultQ := bytesToFloat32(resultQBytes)

	// Verify that dimensions beyond ropeDim are UNCHANGED
	for i := ropeDim; i < headDim; i++ {
		if resultQ[i] != 1.0 {
			t.Errorf("Dimension %d changed (beyond ropeDim=%d): got %f, want 1.0", i, ropeDim, resultQ[i])
		}
	}

	// Verify that dimensions within ropeDim ARE changed
	// (Unless theta/pos make rotation zero, which they don't here)
	if resultQ[0] == 1.0 && resultQ[1] == 1.0 {
		t.Errorf("Dimensions within ropeDim were not rotated")
	}
}
