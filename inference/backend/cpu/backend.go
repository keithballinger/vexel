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
func (b *cpuBackend) RMSNorm(x, weight, out []float32, rows, cols int, eps float32) {
	for i := 0; i < rows; i++ {
		var sumSquares float32
		offset := i * cols
		for j := 0; j < cols; j++ {
			val := x[offset+j]
			sumSquares += val * val
		}

		mean := sumSquares / float32(cols)
		rms := float32(math.Sqrt(float64(mean + eps)))
		
		for j := 0; j < cols; j++ {
			out[offset+j] = (x[offset+j] / rms) * weight[j]
		}
	}
}

// RoPE applies Rotary Positional Embeddings in-place.
func (b *cpuBackend) RoPE(q, k []float32, headDim, seqLen, startPos int, theta float32) {
	totalVectors := len(q) / headDim
	
	for i := 0; i < totalVectors; i++ {
		pos := startPos + i
		offset := i * headDim
		halfDim := headDim / 2
		
		for j := 0; j < halfDim; j++ {
			exp := float64(2*j) / float64(headDim)
			freq := float32(1.0 / math.Pow(float64(theta), exp))
			angle := float32(pos) * freq
			cos := float32(math.Cos(float64(angle)))
			sin := float32(math.Sin(float64(angle)))
			
			val1 := q[offset+j]
			val2 := q[offset+j+halfDim]
			q[offset+j] = val1*cos - val2*sin
			q[offset+j+halfDim] = val1*sin + val2*cos
		}
	}
	
	if k != nil {
		totalVectorsK := len(k) / headDim
		for i := 0; i < totalVectorsK; i++ {
			pos := startPos + i
			offset := i * headDim
			halfDim := headDim / 2
			for j := 0; j < halfDim; j++ {
				exp := float64(2*j) / float64(headDim)
				freq := float32(1.0 / math.Pow(float64(theta), exp))
				angle := float32(pos) * freq
				cos := float32(math.Cos(float64(angle)))
				sin := float32(math.Sin(float64(angle)))
				val1 := k[offset+j]
				val2 := k[offset+j+halfDim]
				k[offset+j] = val1*cos - val2*sin
				k[offset+j+halfDim] = val1*sin + val2*cos
			}
		}
	}
}

// SiLU applies the Sigmoid Linear Unit activation function element-wise.
// out = x * sigmoid(x) = x / (1 + exp(-x))
func (b *cpuBackend) SiLU(x, out []float32, n int) {
	for i := 0; i < n; i++ {
		val := x[i]
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-val))))
		out[i] = val * sigmoid
	}
}
