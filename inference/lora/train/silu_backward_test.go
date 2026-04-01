//go:build metal && darwin && cgo

package train

import (
	"math"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
)

func TestSiLUMulBackward(t *testing.T) {
	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()
	training := backend.TrainingOps(gpuBackend)

	n := 16
	gateData := make([]float32, n)
	upData := make([]float32, n)
	dOutData := make([]float32, n)
	for i := range gateData {
		gateData[i] = 0.5 * float32(i%7-3)
	}
	for i := range upData {
		upData[i] = 0.3 * float32(i%5-2)
	}
	for i := range dOutData {
		dOutData[i] = 0.1 * float32(i%9-4)
	}

	// CPU reference
	silu := func(x float64) float64 { return x / (1 + math.Exp(-x)) }
	cpuDGate := make([]float32, n)
	cpuDUp := make([]float32, n)
	for i := 0; i < n; i++ {
		g := float64(gateData[i])
		u := float64(upData[i])
		dy := float64(dOutData[i])
		sig := 1.0 / (1.0 + math.Exp(-g))
		cpuDGate[i] = float32(dy * u * sig * (1.0 + g*(1.0-sig)))
		cpuDUp[i] = float32(dy * silu(g))
	}

	gate := gpuBackend.AllocPermanent(n * 4)
	up := gpuBackend.AllocPermanent(n * 4)
	dOut := gpuBackend.AllocPermanent(n * 4)
	dGate := gpuBackend.AllocPermanent(n * 4)
	dUp := gpuBackend.AllocPermanent(n * 4)

	gpuBackend.ToDevice(gate, float32SliceToBytes(gateData))
	gpuBackend.ToDevice(up, float32SliceToBytes(upData))
	gpuBackend.ToDevice(dOut, float32SliceToBytes(dOutData))

	training.SiLUMulBackward(dOut, gate, up, dGate, dUp, n)
	gpuBackend.Sync()

	gpuDGate := downloadF32(gpuBackend, dGate, n)
	gpuDUp := downloadF32(gpuBackend, dUp, n)

	var maxErrG, maxErrU float64
	for i := range cpuDGate {
		dg := math.Abs(float64(cpuDGate[i] - gpuDGate[i]))
		du := math.Abs(float64(cpuDUp[i] - gpuDUp[i]))
		if dg > maxErrG { maxErrG = dg }
		if du > maxErrU { maxErrU = du }
	}
	t.Logf("SiLUMulBackward: dGate maxErr=%.8f dUp maxErr=%.8f", maxErrG, maxErrU)
	if maxErrG > 1e-5 || maxErrU > 1e-5 {
		t.Errorf("SiLUMulBackward error too large")
	}
}
