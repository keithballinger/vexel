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
var _ backend.QuantizedMatMul = (*CPUBackend)(nil)
var _ backend.LayerNormOps = (*CPUBackend)(nil)
var _ backend.GELUOps = (*CPUBackend)(nil)
var _ backend.BiasOps = (*CPUBackend)(nil)

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
// Uses Accelerate framework on macOS for optimal performance.
func (b *CPUBackend) MatMul(a, bMat, out tensor.DevicePtr, m, n, k int) {
	if useAccelerate {
		b.MatMulAccelerate(a, bMat, out, m, n, k)
		return
	}
	b.matMulNaive(a, bMat, out, m, n, k)
}

// matMulNaive is the fallback implementation without BLAS.
func (b *CPUBackend) matMulNaive(a, bMat, out tensor.DevicePtr, m, n, k int) {
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
// Uses Accelerate framework on macOS for optimal performance.
func (b *CPUBackend) MatMulTransposed(a, bMat, out tensor.DevicePtr, m, n, k int) {
	if useAccelerate {
		b.MatMulTransposedAccelerate(a, bMat, out, m, n, k)
		return
	}
	b.matMulTransposedNaive(a, bMat, out, m, n, k)
}

// matMulTransposedNaive is the fallback implementation without BLAS.
func (b *CPUBackend) matMulTransposedNaive(a, bMat, out tensor.DevicePtr, m, n, k int) {
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
// ropeDim specifies how many dimensions to rotate (0 = full headDim for LLaMA-style,
// otherwise partial rotation for Phi-2 where only first ropeDim are rotated).
// ropeNeox: true = NEOX-style (split pairs: i, i+dim/2), false = LLaMA-style (interleaved: 2i, 2i+1)
func (b *CPUBackend) RoPE(q, k tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float32, ropeNeox bool) {
	// For partial RoPE (like Phi-2), only rotate first ropeDim dimensions
	effectiveRopeDim := ropeDim
	if effectiveRopeDim == 0 {
		effectiveRopeDim = headDim
	}
	halfRopeDim := effectiveRopeDim / 2

	qData := ptrToFloat32Slice(q, seqLen*numHeads*headDim)
	totalVectors := len(qData) / headDim

	for i := 0; i < totalVectors; i++ {
		seqPos := i / numHeads
		pos := startPos + seqPos
		offset := i * headDim

		// Only rotate first effectiveRopeDim dimensions
		for j := 0; j < halfRopeDim; j++ {
			// Compute indices based on RoPE style
			var idx0, idx1 int
			if ropeNeox {
				// NEOX-style: pairs are (j, j + halfRopeDim)
				idx0 = j
				idx1 = j + halfRopeDim
			} else {
				// LLaMA-style (interleaved): pairs are (2j, 2j+1)
				idx0 = j * 2
				idx1 = j*2 + 1
			}

			exp := float64(2*j) / float64(effectiveRopeDim)
			freq := float32(1.0 / math.Pow(float64(theta), exp))
			angle := float32(pos) * freq
			cos := float32(math.Cos(float64(angle)))
			sin := float32(math.Sin(float64(angle)))

			val1 := qData[offset+idx0]
			val2 := qData[offset+idx1]
			qData[offset+idx0] = val1*cos - val2*sin
			qData[offset+idx1] = val1*sin + val2*cos
		}
	}

	if !k.IsNil() {
		kData := ptrToFloat32Slice(k, seqLen*numKVHeads*headDim)
		totalVectorsK := len(kData) / headDim
		for i := 0; i < totalVectorsK; i++ {
			seqPos := i / numKVHeads
			pos := startPos + seqPos
			offset := i * headDim

			for j := 0; j < halfRopeDim; j++ {
				// Compute indices based on RoPE style
				var idx0, idx1 int
				if ropeNeox {
					// NEOX-style: pairs are (j, j + halfRopeDim)
					idx0 = j
					idx1 = j + halfRopeDim
				} else {
					// LLaMA-style (interleaved): pairs are (2j, 2j+1)
					idx0 = j * 2
					idx1 = j*2 + 1
				}

				exp := float64(2*j) / float64(effectiveRopeDim)
				freq := float32(1.0 / math.Pow(float64(theta), exp))
				angle := float32(pos) * freq
				cos := float32(math.Cos(float64(angle)))
				sin := float32(math.Sin(float64(angle)))

				val1 := kData[offset+idx0]
				val2 := kData[offset+idx1]
				kData[offset+idx0] = val1*cos - val2*sin
				kData[offset+idx1] = val1*sin + val2*cos
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

// SiLUMul performs fused silu(gate) * up operation.
func (b *CPUBackend) SiLUMul(gate, up, out tensor.DevicePtr, n int) {
	gateData := ptrToFloat32Slice(gate, n)
	upData := ptrToFloat32Slice(up, n)
	outData := ptrToFloat32Slice(out, n)

	for i := 0; i < n; i++ {
		g := gateData[i]
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-g))))
		outData[i] = (g * sigmoid) * upData[i]
	}
}

// LayerNorm performs Layer Normalization with mean subtraction.
// Formula: out = (x - mean) / sqrt(var + eps) * weight + bias
func (b *CPUBackend) LayerNorm(x, weight, bias, out tensor.DevicePtr, rows, cols int, eps float32) {
	xData := ptrToFloat32Slice(x, rows*cols)
	wData := ptrToFloat32Slice(weight, cols)
	bData := ptrToFloat32Slice(bias, cols)
	outData := ptrToFloat32Slice(out, rows*cols)

	for i := 0; i < rows; i++ {
		offset := i * cols
		row := xData[offset : offset+cols]

		// Compute mean
		var sum float32
		for j := 0; j < cols; j++ {
			sum += row[j]
		}
		mean := sum / float32(cols)

		// Compute variance
		var varSum float32
		for j := 0; j < cols; j++ {
			diff := row[j] - mean
			varSum += diff * diff
		}
		variance := varSum / float32(cols)

		// Normalize and apply weight/bias
		invStd := 1.0 / float32(math.Sqrt(float64(variance+eps)))
		for j := 0; j < cols; j++ {
			normalized := (row[j] - mean) * invStd
			outData[offset+j] = normalized*wData[j] + bData[j]
		}
	}
}

// GELU applies the Gaussian Error Linear Unit activation function.
// Uses fast approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
func (b *CPUBackend) GELU(x, out tensor.DevicePtr, n int) {
	xData := ptrToFloat32Slice(x, n)
	outData := ptrToFloat32Slice(out, n)

	const sqrtTwoPi = 0.7978845608 // sqrt(2/π)
	const coeff = 0.044715

	for i := 0; i < n; i++ {
		val := xData[i]
		x3 := val * val * val
		tanhArg := sqrtTwoPi * (val + coeff*x3)
		tanhVal := float32(math.Tanh(float64(tanhArg)))
		outData[i] = 0.5 * val * (1.0 + tanhVal)
	}
}

// AddBias performs row-wise bias addition: out[i] = x[i] + bias[i % cols]
func (b *CPUBackend) AddBias(x, bias, out tensor.DevicePtr, rows, cols int) {
	xData := ptrToFloat32Slice(x, rows*cols)
	biasData := ptrToFloat32Slice(bias, cols)
	outData := ptrToFloat32Slice(out, rows*cols)

	for i := 0; i < rows; i++ {
		offset := i * cols
		for j := 0; j < cols; j++ {
			outData[offset+j] = xData[offset+j] + biasData[j]
		}
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
// KV cache is in head-major layout: [numKVHeads, maxSeqLen, headDim]
// kvHeadStride = maxSeqLen * headDim
func (b *CPUBackend) SDPA(q, k, v, out tensor.DevicePtr, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int) {
	qData := ptrToFloat32Slice(q, numQHeads*headDim)
	kData := ptrToFloat32Slice(k, numKVHeads*kvHeadStride)
	vData := ptrToFloat32Slice(v, numKVHeads*kvHeadStride)
	outData := ptrToFloat32Slice(out, numQHeads*headDim)

	headsPerKV := numQHeads / numKVHeads
	scores := make([]float32, kvLen)

	for h := 0; h < numQHeads; h++ {
		kvHead := h / headsPerKV

		qOffset := h * headDim
		qHead := qData[qOffset : qOffset+headDim]
		outOffset := h * headDim
		outHead := outData[outOffset : outOffset+headDim]

		// KV cache layout: [numKVHeads, maxSeqLen, headDim]
		// For head h, position p: offset = h * kvHeadStride + p * headDim
		kvHeadBase := kvHead * kvHeadStride

		for pos := 0; pos < kvLen; pos++ {
			kOffset := kvHeadBase + pos*headDim
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
			vOffset := kvHeadBase + pos*headDim
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

// MatMulQ4_0 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_0 format.
// Q4_0 format: 32 elements per block, 18 bytes per block (2 byte f16 scale + 16 bytes nibbles).
// Nibble ordering (matching llama.cpp):
// - Low nibbles (bits 0-3) go to positions 0..15
// - High nibbles (bits 4-7) go to positions 16..31
func (b *CPUBackend) MatMulQ4_0(a, bQ4, out tensor.DevicePtr, m, n, k int) {
	aData := ptrToFloat32Slice(a, m*k)
	outData := ptrToFloat32Slice(out, m*n)
	bRaw := ptrToByteSlice(bQ4, n*k/32*18) // 18 bytes per 32 elements

	// Zero output
	for i := range outData {
		outData[i] = 0
	}

	const blockSize = 32
	const blockBytes = 18 // 2 bytes f16 scale + 16 bytes nibbles
	blocksPerRow := k / blockSize

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
				aRow := aData[i*k : (i+1)*k]

				for j := 0; j < n; j++ {
					var sum float32
					bRowOffset := j * blocksPerRow * blockBytes

					for blk := 0; blk < blocksPerRow; blk++ {
						blockOffset := bRowOffset + blk*blockBytes

						// Read scale (float16 -> float32)
						scaleF16 := uint16(bRaw[blockOffset]) | uint16(bRaw[blockOffset+1])<<8
						scale := float16ToFloat32(scaleF16)

						// Dequantize following llama.cpp nibble ordering:
						// Low nibbles -> positions 0..15, High nibbles -> positions 16..31
						baseIdx := blk * 32
						for byteI := 0; byteI < 16; byteI++ {
							byteVal := bRaw[blockOffset+2+byteI]

							// Low nibble -> position byteI
							lowNibble := int(byteVal & 0x0F)
							dequantLow := float32(lowNibble-8) * scale
							sum += aRow[baseIdx+byteI] * dequantLow

							// High nibble -> position byteI + 16
							highNibble := int((byteVal >> 4) & 0x0F)
							dequantHigh := float32(highNibble-8) * scale
							sum += aRow[baseIdx+byteI+16] * dequantHigh
						}
					}
					outData[i*n+j] = sum
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
}

// MatMulQ6_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q6_K format.
// This is a CPU fallback implementation - used if GPU Q6_K kernel not available.
// For now, we panic since Q6_K should always be handled by GPU or dequantized to F32.
func (b *CPUBackend) MatMulQ6_K(a, bQ6K, out tensor.DevicePtr, m, n, k int) {
	// Q6_K is typically only used for lm_head which should use GPU kernel.
	// If we reach here, fall back to panic or implement CPU dequant.
	panic("MatMulQ6_K not implemented for CPU backend - use GPU backend or dequantize to F32")
}

// MatMulQ4_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q4_K format.
// This is a CPU fallback implementation - panics since Q4_K should be handled by GPU.
func (b *CPUBackend) MatMulQ4_K(a, bQ4K, out tensor.DevicePtr, m, n, k int) {
	// Q4_K should be handled by GPU kernel.
	// If we reach here, fall back to panic or implement CPU dequant.
	panic("MatMulQ4_K not implemented for CPU backend - use GPU backend or dequantize to F32")
}

// MatMulQ5_K performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q5_K format.
func (b *CPUBackend) MatMulQ5_K(a, bQ5K, out tensor.DevicePtr, m, n, k int) {
	// Q5_K should be handled by GPU kernel.
	panic("MatMulQ5_K not implemented for CPU backend - use GPU backend or dequantize to F32")
}

// MatMulQ8_0 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in Q8_0 format.
func (b *CPUBackend) MatMulQ8_0(a, bQ8, out tensor.DevicePtr, m, n, k int) {
	// Q8_0 should be handled by GPU kernel.
	panic("MatMulQ8_0 not implemented for CPU backend - use GPU backend or dequantize to F32")
}

// MatMulBF16 performs C = A @ B^T where A is [M,K] in F32, B is [N,K] in BF16 format.
func (b *CPUBackend) MatMulBF16(a, bBF16, out tensor.DevicePtr, m, n, k int) {
	// BF16 should be handled by GPU kernel.
	panic("MatMulBF16 not implemented for CPU backend - use GPU backend or dequantize to F32")
}

// ptrToByteSlice converts a DevicePtr to a byte slice.
func ptrToByteSlice(ptr tensor.DevicePtr, n int) []byte {
	if ptr.IsNil() {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr.Addr())), n)
}

// float16ToFloat32 converts a float16 value (stored as uint16) to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1F)
	frac := uint32(h & 0x3FF)

	if exp == 0 {
		if frac == 0 {
			// Zero
			return math.Float32frombits(sign << 31)
		}
		// Denormalized number
		exp = 1
		for (frac & 0x400) == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
	} else if exp == 31 {
		if frac == 0 {
			// Infinity
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		// NaN
		return math.Float32frombits((sign << 31) | 0x7FC00000 | (frac << 13))
	}

	exp = exp + (127 - 15)
	frac = frac << 13

	return math.Float32frombits((sign << 31) | (exp << 23) | frac)
}
