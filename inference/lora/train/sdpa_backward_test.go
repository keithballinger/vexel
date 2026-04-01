//go:build metal && darwin && cgo

package train

import (
	"math"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
)

// TestSDPABackwardCPUvsGPU compares the Metal SDPA backward kernel against
// a pure CPU reference implementation with small, known inputs.
func TestSDPABackwardCPUvsGPU(t *testing.T) {
	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()

	training := backend.TrainingOps(gpuBackend)

	// Small dimensions for exact comparison
	seqLen := 4
	headDim := 8
	numHeads := 4
	numKVHeads := 2
	headsPerGroup := numHeads / numKVHeads

	// Generate deterministic test data
	qData := make([]float32, seqLen*numHeads*headDim)
	kData := make([]float32, seqLen*numKVHeads*headDim)
	vData := make([]float32, seqLen*numKVHeads*headDim)
	dOutData := make([]float32, seqLen*numHeads*headDim)

	for i := range qData {
		qData[i] = 0.1 * float32(i%7-3)
	}
	for i := range kData {
		kData[i] = 0.1 * float32(i%5-2)
	}
	for i := range vData {
		vData[i] = 0.1 * float32(i%9-4)
	}
	for i := range dOutData {
		dOutData[i] = 0.01 * float32(i%11-5)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// ---- CPU reference implementation ----
	// Compute attention weights: softmax(scale * Q @ K^T) with causal mask
	// Layout: Q[seqLen, numHeads, headDim], K[seqLen, numKVHeads, headDim]
	// attnWeights[numHeads, seqLen, seqLen]
	cpuAttnWeights := make([]float32, numHeads*seqLen*seqLen)

	for h := 0; h < numHeads; h++ {
		kvh := h / headsPerGroup
		for i := 0; i < seqLen; i++ {
			// Compute scores
			scores := make([]float32, seqLen)
			for j := 0; j < seqLen; j++ {
				if j > i {
					scores[j] = float32(math.Inf(-1))
					continue
				}
				var dot float32
				for d := 0; d < headDim; d++ {
					qIdx := i*numHeads*headDim + h*headDim + d
					kIdx := j*numKVHeads*headDim + kvh*headDim + d
					dot += qData[qIdx] * kData[kIdx]
				}
				scores[j] = scale * dot
			}
			// Softmax
			maxS := scores[0]
			for j := 1; j <= i; j++ {
				if scores[j] > maxS {
					maxS = scores[j]
				}
			}
			var sumExp float64
			for j := 0; j <= i; j++ {
				scores[j] = float32(math.Exp(float64(scores[j] - maxS)))
				sumExp += float64(scores[j])
			}
			for j := 0; j <= i; j++ {
				scores[j] /= float32(sumExp)
			}
			for j := i + 1; j < seqLen; j++ {
				scores[j] = 0
			}
			// Store
			awOff := h*seqLen*seqLen + i*seqLen
			copy(cpuAttnWeights[awOff:awOff+seqLen], scores)
		}
	}

	// CPU SDPA backward
	cpuDQ := make([]float32, seqLen*numHeads*headDim)
	cpuDK := make([]float32, seqLen*numKVHeads*headDim)
	cpuDV := make([]float32, seqLen*numKVHeads*headDim)

	for h := 0; h < numHeads; h++ {
		kvh := h / headsPerGroup
		for i := 0; i < seqLen; i++ {
			awOff := h*seqLen*seqLen + i*seqLen
			qOff := i*numHeads*headDim + h*headDim

			// dV[kvh][j][d] += sum over i: aw[h][i][j] * dOut[h][i][d]
			for j := 0; j <= i; j++ {
				aw := cpuAttnWeights[awOff+j]
				vOff := j*numKVHeads*headDim + kvh*headDim
				for d := 0; d < headDim; d++ {
					cpuDV[vOff+d] += aw * dOutData[qOff+d]
				}
			}

			// D[h][i] = sum_j(aw * daw) where daw = sum_d(dOut * V)
			var D float64
			for j := 0; j <= i; j++ {
				aw := cpuAttnWeights[awOff+j]
				vOff := j*numKVHeads*headDim + kvh*headDim
				var daw float64
				for d := 0; d < headDim; d++ {
					daw += float64(dOutData[qOff+d]) * float64(vData[vOff+d])
				}
				D += float64(aw) * daw
			}

			// dQ and dK
			for j := 0; j <= i; j++ {
				aw := cpuAttnWeights[awOff+j]
				kvOff := j*numKVHeads*headDim + kvh*headDim

				var daw float64
				for d := 0; d < headDim; d++ {
					daw += float64(dOutData[qOff+d]) * float64(vData[kvOff+d])
				}
				dScore := float32(float64(aw) * (daw - D))

				for d := 0; d < headDim; d++ {
					cpuDQ[qOff+d] += scale * dScore * kData[kvOff+d]
					cpuDK[kvOff+d] += scale * dScore * qData[qOff+d]
				}
			}
		}
	}

	// ---- GPU kernel ----
	Q := gpuBackend.AllocPermanent(len(qData) * 4)
	K := gpuBackend.AllocPermanent(len(kData) * 4)
	V := gpuBackend.AllocPermanent(len(vData) * 4)
	dOut := gpuBackend.AllocPermanent(len(dOutData) * 4)
	attnW := gpuBackend.AllocPermanent(len(cpuAttnWeights) * 4)
	dQ := gpuBackend.AllocPermanent(seqLen * numHeads * headDim * 4)
	dK := gpuBackend.AllocPermanent(seqLen * numKVHeads * headDim * 4)
	dV := gpuBackend.AllocPermanent(seqLen * numKVHeads * headDim * 4)

	gpuBackend.ToDevice(Q, float32SliceToBytes(qData))
	gpuBackend.ToDevice(K, float32SliceToBytes(kData))
	gpuBackend.ToDevice(V, float32SliceToBytes(vData))
	gpuBackend.ToDevice(dOut, float32SliceToBytes(dOutData))
	gpuBackend.ToDevice(attnW, float32SliceToBytes(cpuAttnWeights))

	training.SDPABackward(dOut, Q, K, V, attnW, dQ, dK, dV, seqLen, headDim, numHeads, numKVHeads)
	gpuBackend.Sync()

	gpuDQ := downloadF32(gpuBackend, dQ, seqLen*numHeads*headDim)
	gpuDK := downloadF32(gpuBackend, dK, seqLen*numKVHeads*headDim)
	gpuDV := downloadF32(gpuBackend, dV, seqLen*numKVHeads*headDim)

	// Compare
	compareTensors := func(name string, cpu, gpu []float32) {
		var maxErr, cpuNorm, gpuNorm float64
		for i := range cpu {
			diff := math.Abs(float64(cpu[i] - gpu[i]))
			if diff > maxErr {
				maxErr = diff
			}
			cpuNorm += float64(cpu[i]) * float64(cpu[i])
			gpuNorm += float64(gpu[i]) * float64(gpu[i])
		}
		cpuNorm = math.Sqrt(cpuNorm)
		gpuNorm = math.Sqrt(gpuNorm)
		relErr := math.Abs(cpuNorm-gpuNorm) / (cpuNorm + 1e-8)
		t.Logf("%s: cpuNorm=%.6f gpuNorm=%.6f relNormErr=%.6f maxElementErr=%.8f",
			name, cpuNorm, gpuNorm, relErr, maxErr)
		if relErr > 0.05 {
			// Print first few mismatched values
			for i := 0; i < min(10, len(cpu)); i++ {
				if math.Abs(float64(cpu[i]-gpu[i])) > 1e-6 {
					t.Logf("  [%d] cpu=%.8f gpu=%.8f diff=%.8f", i, cpu[i], gpu[i], cpu[i]-gpu[i])
				}
			}
			t.Errorf("%s: norm relative error %.4f > 0.05", name, relErr)
		}
	}

	compareTensors("dQ", cpuDQ, gpuDQ)
	compareTensors("dK", cpuDK, gpuDK)
	compareTensors("dV", cpuDV, gpuDV)
}
