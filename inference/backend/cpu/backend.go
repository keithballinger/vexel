package cpu

import (
	"math"
	"vexel/inference/tensor"
)

// cpuBackend implements the Backend interface for CPU execution.
type cpuBackend struct{}

// NewBackend creates a new CPU backend.
func NewBackend() Backend {
	return &cpuBackend{}
}

// CreateStream creates a new execution stream (dummy for CPU).
func (b *cpuBackend) CreateStream() (interface{}, error) {
	return "CPUStream", nil
}

// Device returns the CPU device description.
func (b *cpuBackend) Device() tensor.Device {
	return tensor.NewDevice(tensor.CPU, 0)
}

// Matmul performs matrix multiplication: C = A * B
func (b *cpuBackend) Matmul(a, bData, out []float32, m, n, k int) {
	for i := range out {
		out[i] = 0
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * bData[p*n+j]
			}
			out[i*n+j] = sum
		}
	}
}

// RMSNorm performs Root Mean Square Normalization.
// out = x * weight / sqrt(mean(x^2) + eps)
func (b *cpuBackend) RMSNorm(x, weight, out []float32, rows, cols int, eps float32) {
	for i := 0; i < rows; i++ {
		// 1. Calculate sum of squares
		var sumSquares float32
		offset := i * cols
		for j := 0; j < cols; j++ {
			val := x[offset+j]
			sumSquares += val * val
		}

		// 2. Calculate RMS
		mean := sumSquares / float32(cols)
		rms := float32(math.Sqrt(float64(mean + eps)))
		
		// 3. Normalize and scale
		for j := 0; j < cols; j++ {
			out[offset+j] = (x[offset+j] / rms) * weight[j]
		}
	}
}
