//go:build !darwin || !metal

package metal

import (
	"fmt"
	"unsafe"

	"vexel/inference/tensor"
)

// Backend is a stub for non-darwin platforms.
type Backend struct{}

// NewBackend returns an error on non-darwin platforms.
func NewBackend(deviceID int) (*Backend, error) {
	return nil, fmt.Errorf("Metal is only available on macOS/iOS")
}

// Close is a no-op on stubs.
func (b *Backend) Close() {}

// DeviceName returns empty string on stubs.
func (b *Backend) DeviceName() string { return "" }

// Device returns an empty device on stubs.
func (b *Backend) Device() tensor.Device { return tensor.Device{} }

// CreateStream returns an error on stubs.
func (b *Backend) CreateStream() (interface{}, error) {
	return nil, fmt.Errorf("Metal not available")
}

// AllocBuffer returns nil on stubs.
func (b *Backend) AllocBuffer(size int) unsafe.Pointer { return nil }

// FreeBuffer is a no-op on stubs.
func (b *Backend) FreeBuffer(buf unsafe.Pointer) {}

// CopyToDevice is a no-op on stubs.
func (b *Backend) CopyToDevice(dst unsafe.Pointer, src []float32) {}

// CopyFromDevice is a no-op on stubs.
func (b *Backend) CopyFromDevice(dst []float32, src unsafe.Pointer) {}

// Sync is a no-op on stubs.
func (b *Backend) Sync() {}

// MatMul is a no-op on stubs.
func (b *Backend) MatMul(a, bMat, c unsafe.Pointer, M, N, K int) {}

// RMSNorm is a no-op on stubs.
func (b *Backend) RMSNorm(x, weight, out unsafe.Pointer, batchSize, dim int, eps float32) {}

// RoPE is a no-op on stubs.
func (b *Backend) RoPE(q, k unsafe.Pointer, batchSize, seqLen, numHeads, headDim, startPos int, theta float32) {
}

// Softmax is a no-op on stubs.
func (b *Backend) Softmax(x, out unsafe.Pointer, batchSize, dim int) {}

// SiLU is a no-op on stubs.
func (b *Backend) SiLU(x, out unsafe.Pointer, n int) {}

// Add is a no-op on stubs.
func (b *Backend) Add(a, bIn, out unsafe.Pointer, n int) {}

// Mul is a no-op on stubs.
func (b *Backend) Mul(a, bIn, out unsafe.Pointer, n int) {}

// Embedding is a no-op on stubs.
func (b *Backend) Embedding(tokens []int, table, out unsafe.Pointer, vocabSize, dim int) {}

// ScaledDotProductAttention is a no-op on stubs.
func (b *Backend) ScaledDotProductAttention(q, k, v, out unsafe.Pointer,
	batchSize, numHeads, seqLen, headDim int, scale float32, causal bool) {
}
