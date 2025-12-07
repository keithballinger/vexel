package cpu

import (
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
// A: [m, k], B: [k, n], C: [m, n]
// B is assumed to be in column-major or we handle indices carefully.
// Standard naive implementation:
// C[i, j] = sum(A[i, p] * B[p, j]) for p in 0..k
func (b *cpuBackend) Matmul(a, bData, out []float32, m, n, k int) {
	// A is Row-Major: A[i, p] -> a[i*k + p]
	// B is Row-Major: B[p, j] -> b[p*n + j]
	// C is Row-Major: C[i, j] -> out[i*n + j]

	// Zero out output first (safe practice)
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