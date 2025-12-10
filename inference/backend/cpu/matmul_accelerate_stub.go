//go:build !darwin || !cgo

package cpu

import "vexel/inference/tensor"

// MatMulAccelerate is a stub for non-Darwin platforms.
func (b *CPUBackend) MatMulAccelerate(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.MatMul(a, bMat, out, m, n, k)
}

// MatMulTransposedAccelerate is a stub for non-Darwin platforms.
func (b *CPUBackend) MatMulTransposedAccelerate(a, bMat, out tensor.DevicePtr, m, n, k int) {
	b.matMulTransposedNaive(a, bMat, out, m, n, k)
}

// useAccelerate indicates that this build does NOT have Accelerate support
const useAccelerate = false
