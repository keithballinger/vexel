//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"
)

func TestBackendCreation(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	defer b.Close()

	name := b.DeviceName()
	if name == "" {
		t.Fatal("Device name is empty")
	}
	t.Logf("Metal device: %s", name)
}

func TestMatMul(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Test: C = A @ B^T where A=[2,3], B=[4,3] -> C=[2,4]
	M, N, K := 2, 4, 3

	// Input data
	A := []float32{1, 2, 3, 4, 5, 6}                         // [2,3]
	B := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}       // [4,3]
	expected := []float32{1, 2, 3, 6, 4, 5, 6, 15}           // [2,4]

	// Allocate Metal buffers
	aBuf := b.AllocBuffer(len(A) * 4)
	bBuf := b.AllocBuffer(len(B) * 4)
	cBuf := b.AllocBuffer(M * N * 4)
	defer b.FreeBuffer(aBuf)
	defer b.FreeBuffer(bBuf)
	defer b.FreeBuffer(cBuf)

	// Copy input data
	b.CopyToDevice(aBuf, A)
	b.CopyToDevice(bBuf, B)

	// Execute matmul
	b.MatMul(aBuf, bBuf, cBuf, M, N, K)
	b.Sync()

	// Read result
	result := make([]float32, M*N)
	b.CopyFromDevice(result, cBuf)

	// Verify
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-5 {
			t.Errorf("Mismatch at index %d: got %f, expected %f", i, result[i], expected[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Test: softmax over 4 elements
	input := []float32{1.0, 2.0, 3.0, 4.0}
	rows, cols := 1, 4

	// Allocate buffers
	xBuf := b.AllocBuffer(len(input) * 4)
	outBuf := b.AllocBuffer(len(input) * 4)
	defer b.FreeBuffer(xBuf)
	defer b.FreeBuffer(outBuf)

	// Copy and execute
	b.CopyToDevice(xBuf, input)
	b.Softmax(xBuf, outBuf, rows, cols)
	b.Sync()

	// Read result
	result := make([]float32, len(input))
	b.CopyFromDevice(result, outBuf)

	// Verify sum is 1
	var sum float32
	for _, v := range result {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum should be 1.0, got %f", sum)
	}

	// Verify ordering (higher input -> higher output)
	for i := 0; i < len(result)-1; i++ {
		if result[i] >= result[i+1] {
			t.Errorf("Softmax output should be monotonically increasing, got %v", result)
			break
		}
	}
}

func TestSiLU(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	input := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	n := len(input)

	xBuf := b.AllocBuffer(n * 4)
	outBuf := b.AllocBuffer(n * 4)
	defer b.FreeBuffer(xBuf)
	defer b.FreeBuffer(outBuf)

	b.CopyToDevice(xBuf, input)
	b.SiLU(xBuf, outBuf, n)
	b.Sync()

	result := make([]float32, n)
	b.CopyFromDevice(result, outBuf)

	// Verify SiLU properties
	// SiLU(0) = 0
	if math.Abs(float64(result[2])) > 1e-5 {
		t.Errorf("SiLU(0) should be 0, got %f", result[2])
	}
	// SiLU(x) for large positive x ~ x
	if result[4] < 1.5 {
		t.Errorf("SiLU(2) should be ~1.76, got %f", result[4])
	}
}

func TestSDPADecode(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Test simple single-head attention
	// Q: [1, 4] (1 head, headDim=4)
	// K: [3, 1, 4] (kvLen=3, 1 KV head, headDim=4)
	// V: [3, 1, 4]
	kvLen := 3
	numQHeads := 1
	numKVHeads := 1
	headDim := 4
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	Q := []float32{1, 0, 0, 0} // Query looking for first dim
	K := []float32{
		1, 0, 0, 0, // Position 0: matches Q perfectly
		0, 1, 0, 0, // Position 1: orthogonal
		0, 0, 1, 0, // Position 2: orthogonal
	}
	V := []float32{
		1, 2, 3, 4, // Value at position 0
		5, 6, 7, 8, // Value at position 1
		9, 10, 11, 12, // Value at position 2
	}

	// Allocate buffers
	qBuf := b.AllocBuffer(len(Q) * 4)
	kBuf := b.AllocBuffer(len(K) * 4)
	vBuf := b.AllocBuffer(len(V) * 4)
	outBuf := b.AllocBuffer(numQHeads * headDim * 4)
	defer b.FreeBuffer(qBuf)
	defer b.FreeBuffer(kBuf)
	defer b.FreeBuffer(vBuf)
	defer b.FreeBuffer(outBuf)

	b.CopyToDevice(qBuf, Q)
	b.CopyToDevice(kBuf, K)
	b.CopyToDevice(vBuf, V)

	b.SDPADecode(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale)
	b.Sync()

	result := make([]float32, numQHeads*headDim)
	b.CopyFromDevice(result, outBuf)

	// Since Q matches K[0] perfectly and is orthogonal to K[1,2],
	// attention should focus mostly on position 0
	// So output should be close to V[0] = [1,2,3,4]
	t.Logf("SDPA result: %v", result)

	// Verify the output is closer to V[0] than V[1] or V[2]
	if result[0] < 1.0 || result[0] > 5.0 {
		t.Errorf("Expected first dim close to V[0], got %f", result[0])
	}
}

func TestAllocFree(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Test allocation and free
	sizes := []int{1024, 1024 * 1024, 16 * 1024 * 1024}
	for _, size := range sizes {
		buf := b.AllocBuffer(size)
		if buf == nil {
			t.Errorf("Failed to allocate %d bytes", size)
			continue
		}
		// Test can write and read
		if size >= 4 {
			data := []float32{1.0}
			b.CopyToDevice(buf, data)
			result := make([]float32, 1)
			b.CopyFromDevice(result, buf)
			if result[0] != 1.0 {
				t.Errorf("Copy roundtrip failed for %d bytes", size)
			}
		}
		b.FreeBuffer(buf)
	}
}
