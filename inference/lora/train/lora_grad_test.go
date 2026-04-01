//go:build metal && darwin && cgo

package train

import (
	"math"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
	"vexel/inference/lora"
)

// TestLoRAGradCPUvsGPU tests the loraGrads function against a CPU reference.
func TestLoRAGradCPUvsGPU(t *testing.T) {
	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()

	training := backend.TrainingOps(gpuBackend)

	seqLen := 3
	hiddenSize := 8
	qDim := 6
	vDim := 4
	rank := 2
	scale := float32(1.0)

	// Create test data
	normOutData := make([]float32, seqLen*hiddenSize)
	dQData := make([]float32, seqLen*qDim)
	dVData := make([]float32, seqLen*vDim)
	qaData := make([]float32, rank*hiddenSize)
	qbData := make([]float32, qDim*rank) // zeros (like init)
	vaData := make([]float32, rank*hiddenSize)
	vbData := make([]float32, vDim*rank) // zeros

	for i := range normOutData {
		normOutData[i] = 0.1 * float32(i%7-3)
	}
	for i := range dQData {
		dQData[i] = 0.01 * float32(i%11-5)
	}
	for i := range dVData {
		dVData[i] = 0.02 * float32(i%9-4)
	}
	for i := range qaData {
		qaData[i] = 0.05 * float32(i%5-2)
	}
	for i := range vaData {
		vaData[i] = 0.03 * float32(i%6-3)
	}

	// CPU reference: dQB = scale * dQ^T @ (normOut @ QA^T)
	// interQ = normOut @ QA^T  →  [seqLen, rank]
	cpuInterQ := make([]float32, seqLen*rank)
	for s := 0; s < seqLen; s++ {
		for r := 0; r < rank; r++ {
			var dot float32
			for h := 0; h < hiddenSize; h++ {
				// QA is [rank, hidden], QA^T is [hidden, rank]
				// normOut[s, h] * QA[r, h]
				dot += normOutData[s*hiddenSize+h] * qaData[r*hiddenSize+h]
			}
			cpuInterQ[s*rank+r] = dot
		}
	}

	// dQB[q, r] = scale * sum_s(dQ[s, q] * interQ[s, r])
	cpuDQB := make([]float32, qDim*rank)
	for q := 0; q < qDim; q++ {
		for r := 0; r < rank; r++ {
			var sum float32
			for s := 0; s < seqLen; s++ {
				sum += dQData[s*qDim+q] * cpuInterQ[s*rank+r]
			}
			cpuDQB[q*rank+r] = scale * sum
		}
	}

	// dQA[r, h] = scale * sum_s(dInterQ[s, r] * normOut[s, h])
	// where dInterQ[s, r] = sum_q(dQ[s, q] * QB[q, r])
	// Since QB = 0, dQA = 0. Expected.

	// dVB = scale * dV^T @ (normOut @ VA^T)
	cpuInterV := make([]float32, seqLen*rank)
	for s := 0; s < seqLen; s++ {
		for r := 0; r < rank; r++ {
			var dot float32
			for h := 0; h < hiddenSize; h++ {
				dot += normOutData[s*hiddenSize+h] * vaData[r*hiddenSize+h]
			}
			cpuInterV[s*rank+r] = dot
		}
	}
	cpuDVB := make([]float32, vDim*rank)
	for v := 0; v < vDim; v++ {
		for r := 0; r < rank; r++ {
			var sum float32
			for s := 0; s < seqLen; s++ {
				sum += dVData[s*vDim+v] * cpuInterV[s*rank+r]
			}
			cpuDVB[v*rank+r] = scale * sum
		}
	}

	// Upload to GPU and run loraGrads
	normOutGPU := gpuBackend.AllocPermanent(len(normOutData) * 4)
	gpuBackend.ToDevice(normOutGPU, float32SliceToBytes(normOutData))
	dQGPU := gpuBackend.AllocPermanent(len(dQData) * 4)
	gpuBackend.ToDevice(dQGPU, float32SliceToBytes(dQData))
	dVGPU := gpuBackend.AllocPermanent(len(dVData) * 4)
	gpuBackend.ToDevice(dVGPU, float32SliceToBytes(dVData))

	la := &lora.GPULayerAdapter{HasQ: true, HasV: true}
	la.QA = gpuBackend.AllocPermanent(len(qaData) * 4)
	gpuBackend.ToDevice(la.QA, float32SliceToBytes(qaData))
	la.QB = gpuBackend.AllocPermanent(len(qbData) * 4)
	gpuBackend.ToDevice(la.QB, float32SliceToBytes(qbData))
	la.VA = gpuBackend.AllocPermanent(len(vaData) * 4)
	gpuBackend.ToDevice(la.VA, float32SliceToBytes(vaData))
	la.VB = gpuBackend.AllocPermanent(len(vbData) * 4)
	gpuBackend.ToDevice(la.VB, float32SliceToBytes(vbData))

	numLayers := 1
	gpu := &lora.GPUAdapter{Scale: scale, Rank: rank, Layers: []lora.GPULayerAdapter{*la}}
	grads := AllocGradients(gpuBackend, gpu, numLayers, hiddenSize, qDim, vDim)
	ZeroGradients(training, grads, gpu, numLayers, hiddenSize, qDim, vDim)

	loraGrads(gpuBackend, training, &gpu.Layers[0], normOutGPU, dQGPU, dVGPU, grads, 0,
		seqLen, hiddenSize, qDim, vDim, rank, scale)
	gpuBackend.Sync()

	gpuDQB := downloadF32(gpuBackend, grads.DQB[0], qDim*rank)
	gpuDVB := downloadF32(gpuBackend, grads.DVB[0], vDim*rank)

	// Compare
	compareSlices := func(name string, cpu, gpu []float32) {
		var maxErr, cpuNorm float64
		for i := range cpu {
			diff := math.Abs(float64(cpu[i] - gpu[i]))
			if diff > maxErr {
				maxErr = diff
			}
			cpuNorm += float64(cpu[i]) * float64(cpu[i])
		}
		cpuNorm = math.Sqrt(cpuNorm)
		gpuNormVal := float64(0)
		for _, v := range gpu {
			gpuNormVal += float64(v) * float64(v)
		}
		gpuNormVal = math.Sqrt(gpuNormVal)
		relErr := math.Abs(cpuNorm-gpuNormVal) / (cpuNorm + 1e-8)
		t.Logf("%s: cpuNorm=%.8f gpuNorm=%.8f relErr=%.6f maxErr=%.8f",
			name, cpuNorm, gpuNormVal, relErr, maxErr)
		if relErr > 0.01 {
			for i := 0; i < min(12, len(cpu)); i++ {
				t.Logf("  [%d] cpu=%.8f gpu=%.8f", i, cpu[i], gpu[i])
			}
			t.Errorf("%s: relErr=%.4f > 0.01", name, relErr)
		}
	}

	compareSlices("dQB", cpuDQB, gpuDQB)
	compareSlices("dVB", cpuDVB, gpuDVB)
}
