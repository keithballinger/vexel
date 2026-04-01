//go:build metal && darwin && cgo

package train

import (
	"math"
	"testing"

	"vexel/inference/backend/metal"
)

// TestMatMulBackward verifies that MatMul(dOut, W, dIn, M, N, K)
// correctly computes dIn = dOut @ W where:
//   Forward: out = in @ W^T  (MatMulTransposed)
//   Backward: dIn = dOut @ W  (MatMul, non-transposed)
//
// W is stored as [outDim, inDim] (row-major, transposed convention).
func TestMatMulBackward(t *testing.T) {
	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()

	seqLen := 3
	inDim := 4
	outDim := 5

	// W[outDim, inDim] — standard weight layout
	wData := make([]float32, outDim*inDim)
	for i := range wData {
		wData[i] = 0.1 * float32(i%7-3)
	}

	// Input
	inData := make([]float32, seqLen*inDim)
	for i := range inData {
		inData[i] = 0.2 * float32(i%5-2)
	}

	// Forward on CPU: out = in @ W^T → out[seqLen, outDim]
	cpuOut := make([]float32, seqLen*outDim)
	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			var sum float32
			for k := 0; k < inDim; k++ {
				// W^T[k, o] = W[o, k]
				sum += inData[s*inDim+k] * wData[o*inDim+k]
			}
			cpuOut[s*outDim+o] = sum
		}
	}

	// Forward on GPU via MatMulTransposed
	W := gpuBackend.AllocPermanent(len(wData) * 4)
	gpuBackend.ToDevice(W, float32SliceToBytes(wData))
	In := gpuBackend.AllocPermanent(len(inData) * 4)
	gpuBackend.ToDevice(In, float32SliceToBytes(inData))
	Out := gpuBackend.AllocPermanent(seqLen * outDim * 4)
	gpuBackend.MatMulTransposed(In, W, Out, seqLen, outDim, inDim)
	gpuBackend.Sync()
	gpuOut := downloadF32(gpuBackend, Out, seqLen*outDim)

	// Verify forward
	var fwdMaxErr float64
	for i := range cpuOut {
		diff := math.Abs(float64(cpuOut[i] - gpuOut[i]))
		if diff > fwdMaxErr {
			fwdMaxErr = diff
		}
	}
	t.Logf("Forward maxErr=%.8f", fwdMaxErr)

	// Backward: dIn = dOut @ W where W is [outDim, inDim]
	// MatMul(A, B, C, M, N, K): C[M,N] = A[M,K] @ B[K,N]
	// We want dIn[seqLen, inDim] = dOut[seqLen, outDim] @ W[outDim, inDim]
	// So M=seqLen, K=outDim, N=inDim
	dOutData := make([]float32, seqLen*outDim)
	for i := range dOutData {
		dOutData[i] = 0.05 * float32(i%9-4)
	}

	// CPU backward: dIn[s, k] = sum_o(dOut[s, o] * W[o, k])
	cpuDIn := make([]float32, seqLen*inDim)
	for s := 0; s < seqLen; s++ {
		for k := 0; k < inDim; k++ {
			var sum float32
			for o := 0; o < outDim; o++ {
				sum += dOutData[s*outDim+o] * wData[o*inDim+k]
			}
			cpuDIn[s*inDim+k] = sum
		}
	}

	// GPU backward via MatMul
	dOutGPU := gpuBackend.AllocPermanent(len(dOutData) * 4)
	gpuBackend.ToDevice(dOutGPU, float32SliceToBytes(dOutData))
	dInGPU := gpuBackend.AllocPermanent(seqLen * inDim * 4)
	gpuBackend.MatMul(dOutGPU, W, dInGPU, seqLen, inDim, outDim)
	gpuBackend.Sync()
	gpuDIn := downloadF32(gpuBackend, dInGPU, seqLen*inDim)

	// Compare
	var bwdMaxErr float64
	for i := range cpuDIn {
		diff := math.Abs(float64(cpuDIn[i] - gpuDIn[i]))
		if diff > bwdMaxErr {
			bwdMaxErr = diff
		}
	}
	t.Logf("Backward maxErr=%.8f", bwdMaxErr)

	if bwdMaxErr > 1e-5 {
		t.Errorf("MatMul backward error too large: %.8f", bwdMaxErr)
		for i := 0; i < min(10, len(cpuDIn)); i++ {
			t.Logf("  [%d] cpu=%.8f gpu=%.8f", i, cpuDIn[i], gpuDIn[i])
		}
	}
}
