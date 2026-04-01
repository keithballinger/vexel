//go:build metal && darwin && cgo

package train

import (
	"math"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
)

// TestRMSNormBackward verifies the RMSNormBackward kernel via numerical gradients.
func TestRMSNormBackward(t *testing.T) {
	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()

	training := backend.TrainingOps(gpuBackend)

	rows := 3
	cols := 8
	eps := float32(1e-5)

	inputData := make([]float32, rows*cols)
	weightData := make([]float32, cols)
	dOutData := make([]float32, rows*cols)

	for i := range inputData {
		inputData[i] = 0.5 * float32(i%7-3)
	}
	for i := range weightData {
		weightData[i] = 1.0 + 0.1*float32(i)
	}
	for i := range dOutData {
		dOutData[i] = 0.01 * float32(i%11-5)
	}

	// CPU RMSNorm forward
	cpuForward := func(input, weight []float32) []float32 {
		out := make([]float32, rows*cols)
		for r := 0; r < rows; r++ {
			var sumSq float64
			for c := 0; c < cols; c++ {
				x := float64(input[r*cols+c])
				sumSq += x * x
			}
			invRMS := 1.0 / math.Sqrt(sumSq/float64(cols)+float64(eps))
			for c := 0; c < cols; c++ {
				out[r*cols+c] = float32(float64(input[r*cols+c]) * invRMS * float64(weight[c]))
			}
		}
		return out
	}

	// CPU RMSNorm backward (numerical)
	cpuDInput := make([]float32, rows*cols)
	epsNum := float32(1e-3)
	for idx := 0; idx < rows*cols; idx++ {
		// f(x+eps) and f(x-eps)
		inputPlus := make([]float32, rows*cols)
		inputMinus := make([]float32, rows*cols)
		copy(inputPlus, inputData)
		copy(inputMinus, inputData)
		inputPlus[idx] += epsNum
		inputMinus[idx] -= epsNum

		outPlus := cpuForward(inputPlus, weightData)
		outMinus := cpuForward(inputMinus, weightData)

		// dL/dx[idx] = sum_j(dL/dy[j] * dy/dx[idx])
		// = sum_j(dOutData[j] * (outPlus[j] - outMinus[j]) / (2*eps))
		var grad float64
		for j := 0; j < rows*cols; j++ {
			grad += float64(dOutData[j]) * float64(outPlus[j]-outMinus[j]) / float64(2*epsNum)
		}
		cpuDInput[idx] = float32(grad)
	}

	// GPU RMSNormBackward
	input := gpuBackend.AllocPermanent(rows * cols * 4)
	weight := gpuBackend.AllocPermanent(cols * 4)
	dOut := gpuBackend.AllocPermanent(rows * cols * 4)
	dInput := gpuBackend.AllocPermanent(rows * cols * 4)

	gpuBackend.ToDevice(input, float32SliceToBytes(inputData))
	gpuBackend.ToDevice(weight, float32SliceToBytes(weightData))
	gpuBackend.ToDevice(dOut, float32SliceToBytes(dOutData))

	training.RMSNormBackward(dOut, input, weight, dInput, rows, cols, eps)
	gpuBackend.Sync()
	gpuDInput := downloadF32(gpuBackend, dInput, rows*cols)

	// Compare
	var maxErr, cpuNorm, gpuNorm float64
	for i := range cpuDInput {
		diff := math.Abs(float64(cpuDInput[i] - gpuDInput[i]))
		if diff > maxErr {
			maxErr = diff
		}
		cpuNorm += float64(cpuDInput[i]) * float64(cpuDInput[i])
		gpuNorm += float64(gpuDInput[i]) * float64(gpuDInput[i])
	}
	cpuNorm = math.Sqrt(cpuNorm)
	gpuNorm = math.Sqrt(gpuNorm)
	relErr := math.Abs(cpuNorm-gpuNorm) / (cpuNorm + 1e-8)

	t.Logf("RMSNormBackward: cpuNorm=%.8f gpuNorm=%.8f relErr=%.6f maxErr=%.8f",
		cpuNorm, gpuNorm, relErr, maxErr)

	if relErr > 0.01 {
		t.Errorf("RMSNormBackward relErr=%.4f > 0.01", relErr)
		for i := 0; i < min(24, rows*cols); i++ {
			t.Logf("  [%d] cpu=%.8f gpu=%.8f diff=%.8f", i, cpuDInput[i], gpuDInput[i], cpuDInput[i]-gpuDInput[i])
		}
	}
}
