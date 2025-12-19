//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"testing"
)

// float32ToBytes converts []float32 to []byte
func float32ToBytes(data []float32) []byte {
	bytes := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(v))
	}
	return bytes
}

// bytesToFloat32 converts []byte to []float32
func bytesToFloat32(data []byte) []float32 {
	result := make([]float32, len(data)/4)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result
}

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
	A := []float32{1, 2, 3, 4, 5, 6}                   // [2,3]
	B := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1} // [4,3]
	expected := []float32{1, 2, 3, 6, 4, 5, 6, 15}     // [2,4]

	// Allocate Metal buffers using new interface
	aBuf := b.Alloc(len(A) * 4)
	bBuf := b.Alloc(len(B) * 4)
	cBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(cBuf)

	// Copy input data
	b.ToDevice(aBuf, float32ToBytes(A))
	b.ToDevice(bBuf, float32ToBytes(B))

	// Execute matmul
	// B is provided as [N,K], so use transposed matmul to compute A @ B^T.
	b.MatMulTransposed(aBuf, bBuf, cBuf, M, N, K)
	b.Sync()

	// Read result
	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, cBuf)
	result := bytesToFloat32(resultBytes)

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
	xBuf := b.Alloc(len(input) * 4)
	outBuf := b.Alloc(len(input) * 4)
	defer b.Free(xBuf)
	defer b.Free(outBuf)

	// Copy and execute
	b.ToDevice(xBuf, float32ToBytes(input))
	b.Softmax(xBuf, outBuf, rows, cols)
	b.Sync()

	// Read result
	resultBytes := make([]byte, len(input)*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

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

	xBuf := b.Alloc(n * 4)
	outBuf := b.Alloc(n * 4)
	defer b.Free(xBuf)
	defer b.Free(outBuf)

	b.ToDevice(xBuf, float32ToBytes(input))
	b.SiLU(xBuf, outBuf, n)
	b.Sync()

	resultBytes := make([]byte, n*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

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

func TestSDPA(t *testing.T) {
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
		1, 2, 3, 4,    // Value at position 0
		5, 6, 7, 8,    // Value at position 1
		9, 10, 11, 12, // Value at position 2
	}

	// Allocate buffers
	qBuf := b.Alloc(len(Q) * 4)
	kBuf := b.Alloc(len(K) * 4)
	vBuf := b.Alloc(len(V) * 4)
	outBuf := b.Alloc(numQHeads * headDim * 4)
	defer b.Free(qBuf)
	defer b.Free(kBuf)
	defer b.Free(vBuf)
	defer b.Free(outBuf)

	b.ToDevice(qBuf, float32ToBytes(Q))
	b.ToDevice(kBuf, float32ToBytes(K))
	b.ToDevice(vBuf, float32ToBytes(V))

	// For head-major layout, stride = maxSeqLen * headDim
	// Since numKVHeads=1, layout is same as sequence-major
	kvHeadStride := kvLen * headDim
	b.SDPA(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
	b.Sync()

	resultBytes := make([]byte, numQHeads*headDim*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	// Since Q matches K[0] perfectly and is orthogonal to K[1,2],
	// attention should focus mostly on position 0
	// So output should be close to V[0] = [1,2,3,4]
	t.Logf("SDPA result: %v", result)

	// Verify the output is closer to V[0] than V[1] or V[2]
	if result[0] < 1.0 || result[0] > 5.0 {
		t.Errorf("Expected first dim close to V[0], got %f", result[0])
	}
}

func TestMatMulLargeK(t *testing.T) {
	// Test matmul with Phi-2 dimensions: M=2, N=2560, K=2560
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	M, N, K := 2, 2560, 2560

	// Create input data with distinct row values
	A := make([]float32, M*K)
	for i := 0; i < K; i++ {
		A[i] = float32(i) / float32(K)         // Row 0: 0, 1/K, 2/K, ...
		A[K+i] = -float32(i) / float32(K)      // Row 1: 0, -1/K, -2/K, ... (negative to be clearly different)
	}

	// Create identity-like weight matrix (first 4 columns: identity, rest zeros)
	// This way we can verify the output row 0 != row 1
	B := make([]float32, N*K)
	for i := 0; i < N && i < K; i++ {
		B[i*K+i] = 1.0 // Identity diagonal
	}

	aBuf := b.Alloc(len(A) * 4)
	bBuf := b.Alloc(len(B) * 4)
	cBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(cBuf)

	b.ToDevice(aBuf, float32ToBytes(A))
	b.ToDevice(bBuf, float32ToBytes(B))

	// C = A @ B^T
	b.MatMulTransposed(aBuf, bBuf, cBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, cBuf)
	result := bytesToFloat32(resultBytes)

	// Verify row 0 and row 1 are different
	// Row 0 should have positive values, Row 1 should have negative values
	row0Sum := float32(0)
	row1Sum := float32(0)
	for i := 0; i < N; i++ {
		row0Sum += result[i]
		row1Sum += result[N+i]
	}

	t.Logf("Row 0 first4: [%f, %f, %f, %f]", result[0], result[1], result[2], result[3])
	t.Logf("Row 1 first4: [%f, %f, %f, %f]", result[N], result[N+1], result[N+2], result[N+3])
	t.Logf("Row 0 sum: %f, Row 1 sum: %f", row0Sum, row1Sum)

	// The sums should have opposite signs (or at least be clearly different)
	if math.Abs(float64(row0Sum-row1Sum)) < 1e-5 {
		t.Errorf("Row 0 and Row 1 have nearly identical sums - matmul may be computing same value for both rows")
	}

	// Specifically, with our input, result[0] should be A[0,0] = 0, result[1] should be A[0,1] = 1/K
	// And result[N] should be A[1,0] = 0, result[N+1] should be A[1,1] = -1/K
	if math.Abs(float64(result[1]-result[N+1])) < 1e-5 {
		t.Errorf("result[1] = %f, result[N+1] = %f - expected different values", result[1], result[N+1])
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
		buf := b.Alloc(size)
		if buf.IsNil() {
			t.Errorf("Failed to allocate %d bytes", size)
			continue
		}
		// Test can write and read
		if size >= 4 {
			data := []float32{1.0}
			b.ToDevice(buf, float32ToBytes(data))
			resultBytes := make([]byte, 4)
			b.ToHost(resultBytes, buf)
			result := bytesToFloat32(resultBytes)
			if result[0] != 1.0 {
				t.Errorf("Copy roundtrip failed for %d bytes", size)
			}
		}
		b.Free(buf)
	}
}
