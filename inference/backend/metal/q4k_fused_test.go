//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// generateQ4KWeights creates N rows of Q4_K weights for K-dimensional input.
// Returns packed Q4_K binary data ready for GPU upload.
func generateQ4KWeights(n, k int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	numBlocksPerRow := (k + Q4KBlockSize - 1) / Q4KBlockSize
	bytesPerRow := numBlocksPerRow * Q4KBytesPerBlock
	data := make([]byte, n*bytesPerRow)

	for row := 0; row < n; row++ {
		for blk := 0; blk < numBlocksPerRow; blk++ {
			d := rng.Float32()*0.8 + 0.1   // 0.1..0.9
			dmin := rng.Float32()*0.3 + 0.05 // 0.05..0.35
			var scales, mins [8]uint8
			for i := range scales {
				scales[i] = uint8(rng.Intn(32) + 1) // 1..32
				mins[i] = uint8(rng.Intn(16))        // 0..15
			}
			var values [256]uint8
			for i := range values {
				values[i] = uint8(rng.Intn(16))
			}
			block := createQ4_KBlock(d, dmin, scales, mins, values)
			copy(data[row*bytesPerRow+blk*Q4KBytesPerBlock:], block)
		}
	}
	return data
}

// halfBytesToFloat32 converts raw FP16 bytes to float32 slice.
func halfBytesToFloat32(data []byte) []float32 {
	n := len(data) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		h := binary.LittleEndian.Uint16(data[i*2:])
		out[i] = float16ToFloat32CPU(h)
	}
	return out
}

// checkCombinedTolerance verifies GPU results against CPU reference using
// combined absolute/relative tolerance: max(absTol, relTol * |expected|).
// This accounts for both FP16 quantization noise (absolute) and float32
// accumulation order differences (relative).
func checkCombinedTolerance(t *testing.T, name string, expected, got []float32, absTol, relTol float64) {
	t.Helper()
	maxDiff := 0.0
	maxDiffIdx := 0
	for i := range expected {
		diff := math.Abs(float64(got[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
	}
	tol := math.Max(absTol, relTol*math.Abs(float64(expected[maxDiffIdx])))
	t.Logf("%s: max_diff=%.6f at idx=%d (tol=%.3f, expected=%.4f, got=%.4f)",
		name, maxDiff, maxDiffIdx, tol, expected[maxDiffIdx], got[maxDiffIdx])
	if maxDiff > tol {
		t.Errorf("%s: max_diff=%.6f > tol=%.3f at idx=%d (expected=%.6f, got=%.6f)",
			name, maxDiff, tol, maxDiffIdx, expected[maxDiffIdx], got[maxDiffIdx])
	}
}

// TestQ4KFusedRMSNormQKV_F16 tests the fused RMSNorm + Q/K/V projections kernel
// for Q4_K weights with FP16 output.
// Uses combined tolerance: max(absTol, 0.1% * |expected|), following the pattern
// from TestFusedRMSNormQKV_CPU in fused_kernel_test.go.
// The absolute floor accounts for FP16 intermediate storage noise (x is stored as
// half in shared memory), which adds ~sqrt(K)*0.001*avg_weight noise to outputs.
func TestQ4KFusedRMSNormQKV_F16(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4KFusedRMSNormQKVF16Pipeline == nil {
		t.Skip("Q4_K fused RMSNorm QKV F16 pipeline not available")
	}

	tests := []struct {
		name   string
		qDim   int
		kvDim  int
		K      int
		absTol float64 // absolute floor (FP16 intermediate noise)
		relTol float64 // relative tolerance (FP16 output + accumulation)
	}{
		// Pre-normalized FP16 activations: double half round-trip (x→half→half(x*rms*w))
		// adds ~0.05% extra quantization noise. abs floor ~2.0 for small K.
		{"small_256", 128, 64, 256, 2.0, 0.001},
		{"medium_512", 256, 128, 512, 2.0, 0.001},
		{"llama7b_4096", 4096, 1024, 4096, 3.0, 0.001},
		{"odd_dims", 100, 50, 256, 2.0, 0.001},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			// Input activations
			x := make([]float32, tc.K)
			for i := range x {
				x[i] = (rng.Float32() - 0.5) * 2.0
			}

			// RMSNorm weights
			normWeight := make([]float32, tc.K)
			for i := range normWeight {
				normWeight[i] = rng.Float32() + 0.5
			}

			eps := float32(1e-5)

			// Q4_K weight matrices
			wqData := generateQ4KWeights(tc.qDim, tc.K, 100)
			wkData := generateQ4KWeights(tc.kvDim, tc.K, 200)
			wvData := generateQ4KWeights(tc.kvDim, tc.K, 300)

			// CPU reference: RMSNorm → MatVec
			xNorm := cpuRMSNorm(x, normWeight, eps)
			expectedQ := cpuMatVecQ4_K(xNorm, wqData, 1, tc.qDim, tc.K)
			expectedK := cpuMatVecQ4_K(xNorm, wkData, 1, tc.kvDim, tc.K)
			expectedV := cpuMatVecQ4_K(xNorm, wvData, 1, tc.kvDim, tc.K)

			// GPU buffers
			xBuf := backend.Alloc(tc.K * 4)
			normBuf := backend.Alloc(tc.K * 4)
			wqBuf := backend.Alloc(len(wqData))
			wkBuf := backend.Alloc(len(wkData))
			wvBuf := backend.Alloc(len(wvData))
			outQBuf := backend.Alloc(tc.qDim * 2)  // FP16
			outKBuf := backend.Alloc(tc.kvDim * 2)  // FP16
			outVBuf := backend.Alloc(tc.kvDim * 2)  // FP16
			defer backend.Free(xBuf)
			defer backend.Free(normBuf)
			defer backend.Free(wqBuf)
			defer backend.Free(wkBuf)
			defer backend.Free(wvBuf)
			defer backend.Free(outQBuf)
			defer backend.Free(outKBuf)
			defer backend.Free(outVBuf)

			backend.ToDevice(xBuf, float32ToBytes(x))
			backend.ToDevice(normBuf, float32ToBytes(normWeight))
			backend.ToDevice(wqBuf, wqData)
			backend.ToDevice(wkBuf, wkData)
			backend.ToDevice(wvBuf, wvData)

			// Call fused kernel
			backend.MatMulQ4_K_FusedRMSNormQKV_F16(xBuf, normBuf,
				wqBuf, wkBuf, wvBuf,
				outQBuf, outKBuf, outVBuf,
				tc.qDim, tc.kvDim, tc.K, eps)
			backend.Sync()

			// Read back FP16 results and convert
			qBytes := make([]byte, tc.qDim*2)
			kBytes := make([]byte, tc.kvDim*2)
			vBytes := make([]byte, tc.kvDim*2)
			backend.ToHost(qBytes, outQBuf)
			backend.ToHost(kBytes, outKBuf)
			backend.ToHost(vBytes, outVBuf)

			resultQ := halfBytesToFloat32(qBytes)
			resultK := halfBytesToFloat32(kBytes)
			resultV := halfBytesToFloat32(vBytes)

			checkCombinedTolerance(t, "Q", expectedQ, resultQ, tc.absTol, tc.relTol)
			checkCombinedTolerance(t, "K", expectedK, resultK, tc.absTol, tc.relTol)
			checkCombinedTolerance(t, "V", expectedV, resultV, tc.absTol, tc.relTol)
		})
	}
}

// TestQ4KFusedMLP tests the fused MLP kernel: SiLU(x @ W1) * (x @ W3) for Q4_K.
func TestQ4KFusedMLP(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4KFusedMLPPipeline == nil {
		t.Skip("Q4_K fused MLP pipeline not available")
	}

	tests := []struct {
		name   string
		N      int // intermediate size
		K      int // hidden size
		absTol float64
		relTol float64
	}{
		// F32 output: float32 accumulation order + SiLU compounds error
		// abs floor 0.01, relative 1% (compounds two ~0.3% matvec errors via SiLU*up)
		{"small", 128, 256, 0.01, 0.01},
		{"medium", 512, 512, 0.1, 0.01},
		{"llama7b", 11008, 4096, 2.0, 0.01},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			x := make([]float32, tc.K)
			for i := range x {
				x[i] = (rng.Float32() - 0.5) * 2.0
			}

			w1Data := generateQ4KWeights(tc.N, tc.K, 100)
			w3Data := generateQ4KWeights(tc.N, tc.K, 200)

			// CPU reference: gate = x @ W1^T, up = x @ W3^T, out = SiLU(gate) * up
			gate := cpuMatVecQ4_K(x, w1Data, 1, tc.N, tc.K)
			up := cpuMatVecQ4_K(x, w3Data, 1, tc.N, tc.K)
			expected := make([]float32, tc.N)
			for i := 0; i < tc.N; i++ {
				sigmoid := float32(1.0 / (1.0 + math.Exp(-float64(gate[i]))))
				expected[i] = (gate[i] * sigmoid) * up[i]
			}

			// GPU
			xBuf := backend.Alloc(tc.K * 4)
			w1Buf := backend.Alloc(len(w1Data))
			w3Buf := backend.Alloc(len(w3Data))
			outBuf := backend.Alloc(tc.N * 4)
			defer backend.Free(xBuf)
			defer backend.Free(w1Buf)
			defer backend.Free(w3Buf)
			defer backend.Free(outBuf)

			backend.ToDevice(xBuf, float32ToBytes(x))
			backend.ToDevice(w1Buf, w1Data)
			backend.ToDevice(w3Buf, w3Data)

			backend.MatMulQ4_K_FusedMLP(xBuf, w1Buf, w3Buf, outBuf, 1, tc.N, tc.K)
			backend.Sync()

			resultBytes := make([]byte, tc.N*4)
			backend.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			checkCombinedTolerance(t, "MLP", expected, result, tc.absTol, tc.relTol)
		})
	}
}

// TestQ4KF16In tests Q4_K matvec with FP16 input (for Wo after SDPA).
func TestQ4KF16In(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4KF16InPipeline == nil {
		t.Skip("Q4_K F16 input pipeline not available")
	}

	tests := []struct {
		name   string
		N      int
		K      int
		absTol float64
		relTol float64
	}{
		// F32 output from FP16 input: accumulation order differences
		{"small", 128, 256, 0.01, 0.005},
		{"llama7b_wo", 4096, 4096, 0.1, 0.005},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			// FP32 activations (will be converted to FP16 for GPU)
			aF32 := make([]float32, tc.K)
			for i := range aF32 {
				aF32[i] = (rng.Float32() - 0.5) * 2.0
			}

			// Convert to FP16 bytes for GPU upload
			aF16Bytes := make([]byte, tc.K*2)
			for i, v := range aF32 {
				binary.LittleEndian.PutUint16(aF16Bytes[i*2:], float32ToFloat16(v))
			}

			// Use FP16 values (rounded) for CPU reference
			aRounded := make([]float32, tc.K)
			for i := range aF32 {
				h := float32ToFloat16(aF32[i])
				aRounded[i] = float16ToFloat32CPU(h)
			}

			wData := generateQ4KWeights(tc.N, tc.K, 100)

			// CPU reference with FP16-rounded inputs
			expected := cpuMatVecQ4_K(aRounded, wData, 1, tc.N, tc.K)

			// GPU
			aBuf := backend.Alloc(tc.K * 2) // FP16
			wBuf := backend.Alloc(len(wData))
			outBuf := backend.Alloc(tc.N * 4) // FP32
			defer backend.Free(aBuf)
			defer backend.Free(wBuf)
			defer backend.Free(outBuf)

			backend.ToDevice(aBuf, aF16Bytes)
			backend.ToDevice(wBuf, wData)

			backend.MatMulQ4_K_F16In(aBuf, wBuf, outBuf, tc.N, tc.K)
			backend.Sync()

			resultBytes := make([]byte, tc.N*4)
			backend.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			checkCombinedTolerance(t, "F16In", expected, result, tc.absTol, tc.relTol)
		})
	}
}

// TestQ4KAdd tests Q4_K matvec that adds to output (for W2 + residual).
func TestQ4KAdd(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4KAddPipeline == nil {
		t.Skip("Q4_K Add pipeline not available")
	}

	tests := []struct {
		name   string
		N      int
		K      int
		absTol float64
		relTol float64
	}{
		// F32 output: accumulation order differences
		{"small", 128, 256, 0.01, 0.005},
		{"llama7b_w2", 4096, 11008, 0.1, 0.005},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			a := make([]float32, tc.K)
			for i := range a {
				a[i] = (rng.Float32() - 0.5) * 2.0
			}

			// Pre-existing output (residual)
			residual := make([]float32, tc.N)
			for i := range residual {
				residual[i] = (rng.Float32() - 0.5) * 1.0
			}

			wData := generateQ4KWeights(tc.N, tc.K, 100)

			// CPU reference: out = residual + (a @ W^T)
			matvecResult := cpuMatVecQ4_K(a, wData, 1, tc.N, tc.K)
			expected := make([]float32, tc.N)
			for i := range expected {
				expected[i] = residual[i] + matvecResult[i]
			}

			// GPU
			aBuf := backend.Alloc(tc.K * 4)
			wBuf := backend.Alloc(len(wData))
			outBuf := backend.Alloc(tc.N * 4)
			defer backend.Free(aBuf)
			defer backend.Free(wBuf)
			defer backend.Free(outBuf)

			backend.ToDevice(aBuf, float32ToBytes(a))
			backend.ToDevice(wBuf, wData)

			backend.ToDevice(outBuf, float32ToBytes(residual)) // pre-fill with residual

			backend.MatMulQ4_K_Add(aBuf, wBuf, outBuf, tc.N, tc.K)
			backend.Sync()

			resultBytes := make([]byte, tc.N*4)
			backend.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			checkCombinedTolerance(t, "Add", expected, result, tc.absTol, tc.relTol)
		})
	}
}

// TestQ4KF16InAdd tests Q4_K matvec with FP16 input that ADDS to output (Wo+Add fusion).
func TestQ4KF16InAdd(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4KF16InAddPipeline == nil {
		t.Skip("Q4_K F16InAdd pipeline not available")
	}

	tests := []struct {
		name   string
		N      int
		K      int
		absTol float64
		relTol float64
	}{
		{"small", 128, 256, 0.01, 0.005},
		{"llama7b_wo", 4096, 4096, 0.1, 0.005},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			// FP32 activations → convert to FP16
			aF32 := make([]float32, tc.K)
			for i := range aF32 {
				aF32[i] = (rng.Float32() - 0.5) * 2.0
			}
			aF16Bytes := make([]byte, tc.K*2)
			for i, v := range aF32 {
				binary.LittleEndian.PutUint16(aF16Bytes[i*2:], float32ToFloat16(v))
			}
			aRounded := make([]float32, tc.K)
			for i := range aF32 {
				aRounded[i] = float16ToFloat32CPU(float32ToFloat16(aF32[i]))
			}

			// Pre-existing output (residual)
			residual := make([]float32, tc.N)
			for i := range residual {
				residual[i] = (rng.Float32() - 0.5) * 1.0
			}

			wData := generateQ4KWeights(tc.N, tc.K, 100)

			// CPU reference: out = residual + (aRounded @ W^T)
			matvecResult := cpuMatVecQ4_K(aRounded, wData, 1, tc.N, tc.K)
			expected := make([]float32, tc.N)
			for i := range expected {
				expected[i] = residual[i] + matvecResult[i]
			}

			// GPU
			aBuf := backend.Alloc(tc.K * 2) // FP16
			wBuf := backend.Alloc(len(wData))
			outBuf := backend.Alloc(tc.N * 4)
			defer backend.Free(aBuf)
			defer backend.Free(wBuf)
			defer backend.Free(outBuf)

			backend.ToDevice(aBuf, aF16Bytes)
			backend.ToDevice(wBuf, wData)
			backend.ToDevice(outBuf, float32ToBytes(residual)) // pre-fill with residual

			backend.MatMulQ4_K_F16InAdd(aBuf, wBuf, outBuf, tc.N, tc.K)
			backend.Sync()

			resultBytes := make([]byte, tc.N*4)
			backend.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			checkCombinedTolerance(t, "F16InAdd", expected, result, tc.absTol, tc.relTol)
		})
	}
}

// TestQ4KFusedRMSNormMLP tests fused RMSNorm + MLP: SiLU(RMSNorm(x)@W1) * (RMSNorm(x)@W3) for Q4_K.
// Used when Wo+Add is active — reads only x (residual already in x).
func TestQ4KFusedRMSNormMLP(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4KFusedRMSNormMLPPipeline == nil {
		t.Skip("Q4_K FusedRMSNormMLP pipeline not available")
	}

	tests := []struct {
		name   string
		N      int // intermediate size
		K      int // hidden size
		absTol float64
		relTol float64
	}{
		{"small", 128, 256, 0.01, 0.01},
		{"medium", 512, 512, 0.1, 0.01},
		{"llama7b", 11008, 4096, 2.0, 0.01},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			x := make([]float32, tc.K)
			for i := range x {
				x[i] = (rng.Float32() - 0.5) * 2.0
			}

			normWeight := make([]float32, tc.K)
			for i := range normWeight {
				normWeight[i] = rng.Float32() + 0.5
			}

			eps := float32(1e-5)

			w1Data := generateQ4KWeights(tc.N, tc.K, 100)
			w3Data := generateQ4KWeights(tc.N, tc.K, 200)

			// CPU reference (FP32): xNorm = RMSNorm(x), gate = xNorm @ W1^T, up = xNorm @ W3^T, out = SiLU(gate) * up
			// Note: GPU uses FP16 dequant + dot in fused kernel for 2× ALU throughput.
			// This adds ~0.1-1.5% relative error vs FP32 CPU reference, proportional to sqrt(K).
			xNorm := cpuRMSNorm(x, normWeight, eps)
			gate := cpuMatVecQ4_K(xNorm, w1Data, 1, tc.N, tc.K)
			up := cpuMatVecQ4_K(xNorm, w3Data, 1, tc.N, tc.K)
			expected := make([]float32, tc.N)
			for i := 0; i < tc.N; i++ {
				sigmoid := float32(1.0 / (1.0 + math.Exp(-float64(gate[i]))))
				expected[i] = (gate[i] * sigmoid) * up[i]
			}

			// GPU
			xBuf := backend.Alloc(tc.K * 4)
			normBuf := backend.Alloc(tc.K * 4)
			w1Buf := backend.Alloc(len(w1Data))
			w3Buf := backend.Alloc(len(w3Data))
			outBuf := backend.Alloc(tc.N * 4)
			defer backend.Free(xBuf)
			defer backend.Free(normBuf)
			defer backend.Free(w1Buf)
			defer backend.Free(w3Buf)
			defer backend.Free(outBuf)

			backend.ToDevice(xBuf, float32ToBytes(x))
			backend.ToDevice(normBuf, float32ToBytes(normWeight))
			backend.ToDevice(w1Buf, w1Data)
			backend.ToDevice(w3Buf, w3Data)

			backend.MatMulQ4_K_FusedRMSNormMLP(xBuf, normBuf, w1Buf, w3Buf, outBuf, tc.N, tc.K, eps)
			backend.Sync()

			resultBytes := make([]byte, tc.N*4)
			backend.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			checkCombinedTolerance(t, "FusedRMSNormMLP", expected, result, tc.absTol, tc.relTol)
		})
	}
}

// TestQ4KFusedRMSNormQKVRoPEScatter tests the fully fused QKV+RoPE+Scatter kernel for Q4_K.
// Verifies that Q output has RoPE applied, K values are scattered to cache with RoPE,
// and V values are scattered to cache without RoPE.
func TestQ4KFusedRMSNormQKVRoPEScatter(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4KFusedRMSNormQKVRoPEScatterF16Pipeline == nil {
		t.Skip("Q4_K FusedRMSNormQKVRoPEScatter pipeline not available")
	}

	// LLaMA 2 7B dimensions
	qDim := 4096
	kvDim := 1024    // GQA: 8 KV heads * 128 headDim
	K := 4096
	headDim := 128
	ropeDim := 128
	numKVHeads := 8
	maxSeqLen := 64
	seqPos := 5       // current sequence position
	startPos := seqPos // position for RoPE frequencies
	theta := float32(10000.0)
	eps := float32(1e-5)

	rng := rand.New(rand.NewSource(42))

	x := make([]float32, K)
	for i := range x {
		x[i] = (rng.Float32() - 0.5) * 2.0
	}
	normWeight := make([]float32, K)
	for i := range normWeight {
		normWeight[i] = rng.Float32() + 0.5
	}

	wqData := generateQ4KWeights(qDim, K, 100)
	wkData := generateQ4KWeights(kvDim, K, 200)
	wvData := generateQ4KWeights(kvDim, K, 300)

	// CPU reference: RMSNorm → MatVec → RoPE
	xNorm := cpuRMSNorm(x, normWeight, eps)
	rawQ := cpuMatVecQ4_K(xNorm, wqData, 1, qDim, K)
	rawK := cpuMatVecQ4_K(xNorm, wkData, 1, kvDim, K)
	rawV := cpuMatVecQ4_K(xNorm, wvData, 1, kvDim, K)

	// Apply RoPE to Q and K
	expectedQ := make([]float32, qDim)
	copy(expectedQ, rawQ)
	cpuRoPE(expectedQ, headDim, ropeDim, startPos, theta)

	expectedK := make([]float32, kvDim)
	copy(expectedK, rawK)
	cpuRoPE(expectedK, headDim, ropeDim, startPos, theta)

	// GPU buffers
	xBuf := backend.Alloc(K * 4)
	normBuf := backend.Alloc(K * 4)
	wqBuf := backend.Alloc(len(wqData))
	wkBuf := backend.Alloc(len(wkData))
	wvBuf := backend.Alloc(len(wvData))
	outQBuf := backend.Alloc(qDim * 2) // FP16
	kCacheSize := numKVHeads * maxSeqLen * headDim * 2
	vCacheSize := numKVHeads * maxSeqLen * headDim * 2
	kCacheBuf := backend.Alloc(kCacheSize) // FP16
	vCacheBuf := backend.Alloc(vCacheSize) // FP16
	defer backend.Free(xBuf)
	defer backend.Free(normBuf)
	defer backend.Free(wqBuf)
	defer backend.Free(wkBuf)
	defer backend.Free(wvBuf)
	defer backend.Free(outQBuf)
	defer backend.Free(kCacheBuf)
	defer backend.Free(vCacheBuf)

	// Zero cache
	backend.Zero(kCacheBuf, kCacheSize)
	backend.Zero(vCacheBuf, vCacheSize)

	backend.ToDevice(xBuf, float32ToBytes(x))
	backend.ToDevice(normBuf, float32ToBytes(normWeight))
	backend.ToDevice(wqBuf, wqData)
	backend.ToDevice(wkBuf, wkData)
	backend.ToDevice(wvBuf, wvData)

	backend.MatMulQ4_K_FusedRMSNormQKVRoPEScatter_F16(
		xBuf, normBuf,
		wqBuf, wkBuf, wvBuf,
		outQBuf, kCacheBuf, vCacheBuf,
		qDim, kvDim, K, eps,
		headDim, ropeDim, startPos, theta, maxSeqLen, seqPos)
	backend.Sync()

	// Read back Q (FP16)
	qBytes := make([]byte, qDim*2)
	backend.ToHost(qBytes, outQBuf)
	resultQ := halfBytesToFloat32(qBytes)

	// Read back K cache at seqPos for each head
	kCacheBytes := make([]byte, kCacheSize)
	backend.ToHost(kCacheBytes, kCacheBuf)
	resultK := make([]float32, kvDim)
	for head := 0; head < numKVHeads; head++ {
		for d := 0; d < headDim; d++ {
			offset := (head*maxSeqLen*headDim + seqPos*headDim + d) * 2
			h := binary.LittleEndian.Uint16(kCacheBytes[offset:])
			resultK[head*headDim+d] = float16ToFloat32CPU(h)
		}
	}

	// Read back V cache at seqPos for each head
	vCacheBytes := make([]byte, vCacheSize)
	backend.ToHost(vCacheBytes, vCacheBuf)
	resultV := make([]float32, kvDim)
	for head := 0; head < numKVHeads; head++ {
		for d := 0; d < headDim; d++ {
			offset := (head*maxSeqLen*headDim + seqPos*headDim + d) * 2
			h := binary.LittleEndian.Uint16(vCacheBytes[offset:])
			resultV[head*headDim+d] = float16ToFloat32CPU(h)
		}
	}

	// Q4_K + FP16 intermediate + RoPE compounding: wider tolerance
	checkCombinedTolerance(t, "Q(RoPE)", expectedQ, resultQ, 3.0, 0.002)
	checkCombinedTolerance(t, "K(RoPE+Scatter)", expectedK, resultK, 3.0, 0.002)
	checkCombinedTolerance(t, "V(Scatter)", rawV, resultV, 3.0, 0.001)
}

// cpuRoPE applies standard RoPE (non-neox) to a flat vector in-place.
// The vector is interpreted as [numHeads][headDim], and RoPE is applied
// pairwise to the first ropeDim elements of each head.
func cpuRoPE(data []float32, headDim, ropeDim, pos int, theta float32) {
	numHeads := len(data) / headDim
	for h := 0; h < numHeads; h++ {
		offset := h * headDim
		for j := 0; j < ropeDim/2; j++ {
			freq := float64(1.0) / math.Pow(float64(theta), float64(2*j)/float64(ropeDim))
			angle := float64(pos) * freq
			cosVal := float32(math.Cos(angle))
			sinVal := float32(math.Sin(angle))
			v0 := data[offset+2*j]
			v1 := data[offset+2*j+1]
			data[offset+2*j] = v0*cosVal - v1*sinVal
			data[offset+2*j+1] = v0*sinVal + v1*cosVal
		}
	}
}
