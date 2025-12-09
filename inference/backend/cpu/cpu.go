// Package cpu provides a CPU-based backend implementation.
// This is primarily for debugging and testing when GPU is unavailable.
package cpu

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// Ensure Backend implements the interface
var _ backend.Backend = (*CPUBackend)(nil)

// CPUBackend implements the Backend interface using CPU execution.
type CPUBackend struct{}

// NewBackend creates a new CPU backend.
func NewCPUBackend() *CPUBackend {
	return &CPUBackend{}
}

// Device returns the CPU device description.
func (b *CPUBackend) Device() tensor.Device {
	return tensor.NewDevice(tensor.CPU, 0)
}

// =============================================================================
// Memory Management
// =============================================================================

// Alloc allocates memory on CPU and returns a DevicePtr.
func (b *CPUBackend) Alloc(bytes int) tensor.DevicePtr {
	if bytes <= 0 {
		return tensor.DevicePtr{}
	}
	// Allocate aligned memory for better performance
	data := make([]byte, bytes)
	return tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&data[0])))
}

// Free is a no-op on CPU (GC handles it).
func (b *CPUBackend) Free(ptr tensor.DevicePtr) {
	// No-op: Go GC handles CPU memory
}

// ToDevice is a no-op on CPU (memory is already accessible).
func (b *CPUBackend) ToDevice(dst tensor.DevicePtr, src []byte) {
	if dst.IsNil() || len(src) == 0 {
		return
	}
	dstSlice := unsafe.Slice((*byte)(unsafe.Pointer(dst.Addr())), len(src))
	copy(dstSlice, src)
}

// ToHost copies from CPU memory to host slice.
func (b *CPUBackend) ToHost(dst []byte, src tensor.DevicePtr) {
	if src.IsNil() || len(dst) == 0 {
		return
	}
	srcSlice := unsafe.Slice((*byte)(unsafe.Pointer(src.Addr())), len(dst))
	copy(dst, srcSlice)
}

// Sync is a no-op on CPU.
func (b *CPUBackend) Sync() {
	// No-op: CPU operations are synchronous
}

// =============================================================================
// Helper Functions
// =============================================================================

// ptrToFloat32Slice converts a DevicePtr to a []float32 slice.
func ptrToFloat32Slice(ptr tensor.DevicePtr, n int) []float32 {
	if ptr.IsNil() {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr.Addr())), n)
}

// =============================================================================
// Compute Kernels
// =============================================================================

// MatMul performs C = A @ B where A is [M,K], B is [K,N], C is [M,N].
func (b *CPUBackend) MatMul(a, bMat, out tensor.DevicePtr, m, n, k int) {
	aData := ptrToFloat32Slice(a, m*k)
	bData := ptrToFloat32Slice(bMat, k*n)
	outData := ptrToFloat32Slice(out, m*n)

	for i := range outData {
		outData[i] = 0
	}

	numWorkers := runtime.NumCPU()
	if m < numWorkers {
		numWorkers = m
	}

	var wg sync.WaitGroup
	chunkSize := (m + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * chunkSize
		endRow := startRow + chunkSize
		if endRow > m {
			endRow = m
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < n; j++ {
					var sum float32
					for p := 0; p < k; p++ {
						sum += aData[i*k+p] * bData[p*n+j]
					}
					outData[i*n+j] = sum
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
}

// MatMulTransposed performs C = A @ B^T where A is [M,K], B is [N,K], C is [M,N].
func (b *CPUBackend) MatMulTransposed(a, bMat, out tensor.DevicePtr, m, n, k int) {
	aData := ptrToFloat32Slice(a, m*k)
	bData := ptrToFloat32Slice(bMat, n*k)
	outData := ptrToFloat32Slice(out, m*n)

	for i := range outData {
		outData[i] = 0
	}

	numWorkers := runtime.NumCPU()
	if m < numWorkers {
		numWorkers = m
	}

	var wg sync.WaitGroup
	chunkSize := (m + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * chunkSize
		endRow := startRow + chunkSize
		if endRow > m {
			endRow = m
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < n; j++ {
					var sum float32
					for p := 0; p < k; p++ {
						sum += aData[i*k+p] * bData[j*k+p]
					}
					outData[i*n+j] = sum
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
}

// RMSNorm performs RMS normalization.
func (b *CPUBackend) RMSNorm(x, weight, out tensor.DevicePtr, rows, cols int, eps float32) {
	xData := ptrToFloat32Slice(x, rows*cols)
	wData := ptrToFloat32Slice(weight, cols)
	outData := ptrToFloat32Slice(out, rows*cols)

	for i := 0; i < rows; i++ {
		var sumSquares float32
		offset := i * cols
		for j := 0; j < cols; j++ {
			val := xData[offset+j]
			sumSquares += val * val
		}

		mean := sumSquares / float32(cols)
		rms := float32(math.Sqrt(float64(mean + eps)))

		for j := 0; j < cols; j++ {
			outData[offset+j] = (xData[offset+j] / rms) * wData[j]
		}
	}
}

// RoPE applies Rotary Positional Embeddings in-place.
func (b *CPUBackend) RoPE(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos int, theta float32) {
	qData := ptrToFloat32Slice(q, seqLen*numHeads*headDim)
	totalVectors := len(qData) / headDim

	for i := 0; i < totalVectors; i++ {
		seqPos := i / numHeads
		pos := startPos + seqPos
		offset := i * headDim
		halfDim := headDim / 2

		for j := 0; j < halfDim; j++ {
			exp := float64(2*j) / float64(headDim)
			freq := float32(1.0 / math.Pow(float64(theta), exp))
			angle := float32(pos) * freq
			cos := float32(math.Cos(float64(angle)))
			sin := float32(math.Sin(float64(angle)))

			val1 := qData[offset+j]
			val2 := qData[offset+j+halfDim]
			qData[offset+j] = val1*cos - val2*sin
			qData[offset+j+halfDim] = val1*sin + val2*cos
		}
	}

	if !k.IsNil() {
		kData := ptrToFloat32Slice(k, seqLen*numKVHeads*headDim)
		totalVectorsK := len(kData) / headDim
		for i := 0; i < totalVectorsK; i++ {
			seqPos := i / numKVHeads
			pos := startPos + seqPos
			offset := i * headDim
			halfDim := headDim / 2
			for j := 0; j < halfDim; j++ {
				exp := float64(2*j) / float64(headDim)
				freq := float32(1.0 / math.Pow(float64(theta), exp))
				angle := float32(pos) * freq
				cos := float32(math.Cos(float64(angle)))
				sin := float32(math.Sin(float64(angle)))
				val1 := kData[offset+j]
				val2 := kData[offset+j+halfDim]
				kData[offset+j] = val1*cos - val2*sin
				kData[offset+j+halfDim] = val1*sin + val2*cos
			}
		}
	}
}

// SiLU applies the Sigmoid Linear Unit activation function.
func (b *CPUBackend) SiLU(x, out tensor.DevicePtr, n int) {
	xData := ptrToFloat32Slice(x, n)
	outData := ptrToFloat32Slice(out, n)

	for i := 0; i < n; i++ {
		val := xData[i]
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-val))))
		outData[i] = val * sigmoid
	}
}

// Softmax applies the softmax function row-wise.
func (b *CPUBackend) Softmax(x, out tensor.DevicePtr, rows, cols int) {
	xData := ptrToFloat32Slice(x, rows*cols)
	outData := ptrToFloat32Slice(out, rows*cols)

	for i := 0; i < rows; i++ {
		offset := i * cols
		row := xData[offset : offset+cols]
		outRow := outData[offset : offset+cols]

		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}

		var sum float32
		for j, v := range row {
			exp := float32(math.Exp(float64(v - maxVal)))
			outRow[j] = exp
			sum += exp
		}

		invSum := 1.0 / sum
		for j := range outRow {
			outRow[j] *= invSum
		}
	}
}

// Add performs element-wise addition.
func (b *CPUBackend) Add(a, bIn, out tensor.DevicePtr, n int) {
	aData := ptrToFloat32Slice(a, n)
	bData := ptrToFloat32Slice(bIn, n)
	outData := ptrToFloat32Slice(out, n)

	for i := 0; i < n; i++ {
		outData[i] = aData[i] + bData[i]
	}
}

// Mul performs element-wise multiplication.
func (b *CPUBackend) Mul(a, bIn, out tensor.DevicePtr, n int) {
	aData := ptrToFloat32Slice(a, n)
	bData := ptrToFloat32Slice(bIn, n)
	outData := ptrToFloat32Slice(out, n)

	for i := 0; i < n; i++ {
		outData[i] = aData[i] * bData[i]
	}
}

// Embedding performs embedding lookup.
func (b *CPUBackend) Embedding(ids tensor.DevicePtr, numTokens int, table, out tensor.DevicePtr, vocabSize, dim int) {
	idsData := unsafe.Slice((*int32)(unsafe.Pointer(ids.Addr())), numTokens)
	tableData := ptrToFloat32Slice(table, vocabSize*dim)
	outData := ptrToFloat32Slice(out, numTokens*dim)

	for i := 0; i < numTokens; i++ {
		id := int(idsData[i])
		if id < 0 || id >= vocabSize {
			continue
		}
		copy(outData[i*dim:(i+1)*dim], tableData[id*dim:(id+1)*dim])
	}
}

// SDPA performs Scaled Dot-Product Attention for decode.
func (b *CPUBackend) SDPA(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32) {
	qData := ptrToFloat32Slice(q, numQHeads*headDim)
	kData := ptrToFloat32Slice(k, kvLen*numKVHeads*headDim)
	vData := ptrToFloat32Slice(v, kvLen*numKVHeads*headDim)
	outData := ptrToFloat32Slice(out, numQHeads*headDim)

	headsPerKV := numQHeads / numKVHeads
	scores := make([]float32, kvLen)

	for h := 0; h < numQHeads; h++ {
		kvHead := h / headsPerKV

		qOffset := h * headDim
		qHead := qData[qOffset : qOffset+headDim]
		outOffset := h * headDim
		outHead := outData[outOffset : outOffset+headDim]

		for pos := 0; pos < kvLen; pos++ {
			kOffset := pos*numKVHeads*headDim + kvHead*headDim
			kVec := kData[kOffset : kOffset+headDim]

			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * kVec[d]
			}
			scores[pos] = dot * scale
		}

		maxScore := scores[0]
		for _, s := range scores {
			if s > maxScore {
				maxScore = s
			}
		}

		var sumExp float32
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
			sumExp += scores[i]
		}

		invSum := 1.0 / sumExp
		for i := range scores {
			scores[i] *= invSum
		}

		for d := 0; d < headDim; d++ {
			outHead[d] = 0
		}

		for pos := 0; pos < kvLen; pos++ {
			vOffset := pos*numKVHeads*headDim + kvHead*headDim
			vVec := vData[vOffset : vOffset+headDim]

			weight := scores[pos]
			for d := 0; d < headDim; d++ {
				outHead[d] += weight * vVec[d]
			}
		}
	}
}

// SDPAPrefill performs SDPA for prefill with causal masking.
func (b *CPUBackend) SDPAPrefill(q, k, v, out tensor.DevicePtr, seqLen, numQHeads, numKVHeads, headDim int, scale float32) {
	qData := ptrToFloat32Slice(q, seqLen*numQHeads*headDim)
	kData := ptrToFloat32Slice(k, seqLen*numKVHeads*headDim)
	vData := ptrToFloat32Slice(v, seqLen*numKVHeads*headDim)
	outData := ptrToFloat32Slice(out, seqLen*numQHeads*headDim)

	headsPerKV := numQHeads / numKVHeads

	for qPos := 0; qPos < seqLen; qPos++ {
		for h := 0; h < numQHeads; h++ {
			kvHead := h / headsPerKV

			qOffset := qPos*numQHeads*headDim + h*headDim
			qHead := qData[qOffset : qOffset+headDim]

			outOffset := qPos*numQHeads*headDim + h*headDim
			outHead := outData[outOffset : outOffset+headDim]

			// Causal: only attend to positions <= qPos
			maxKLen := qPos + 1
			scores := make([]float32, maxKLen)

			for kPos := 0; kPos < maxKLen; kPos++ {
				kOffset := kPos*numKVHeads*headDim + kvHead*headDim
				kVec := kData[kOffset : kOffset+headDim]

				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qHead[d] * kVec[d]
				}
				scores[kPos] = dot * scale
			}

			maxScore := scores[0]
			for _, s := range scores {
				if s > maxScore {
					maxScore = s
				}
			}

			var sumExp float32
			for i := range scores {
				scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
				sumExp += scores[i]
			}

			invSum := 1.0 / sumExp
			for i := range scores {
				scores[i] *= invSum
			}

			for d := 0; d < headDim; d++ {
				outHead[d] = 0
			}

			for kPos := 0; kPos < maxKLen; kPos++ {
				vOffset := kPos*numKVHeads*headDim + kvHead*headDim
				vVec := vData[vOffset : vOffset+headDim]

				weight := scores[kPos]
				for d := 0; d < headDim; d++ {
					outHead[d] += weight * vVec[d]
				}
			}
		}
	}
}
