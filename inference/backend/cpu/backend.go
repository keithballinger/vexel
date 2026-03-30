package cpu

import (
	"math"
	"runtime"
	"sync"

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
	// ... (implementation exists)
	for i := range out {
		out[i] = 0
	}

	// Parallelize over M (rows of A)
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
						sum += a[i*k+p] * bData[p*n+j]
					}
					out[i*n+j] = sum
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
}

// MatmulTransposeB performs C = A * B^T
func (b *cpuBackend) MatmulTransposeB(a, bData, out []float32, m, n, k int) {
	for i := range out {
		out[i] = 0
	}

	// Parallelize
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
					// Inner loop over K
					// Access B at [j, p] instead of [p, j]
					// B is [N, K]
					// B index = j*k + p
					for p := 0; p < k; p++ {
						sum += a[i*k+p] * bData[j*k+p]
					}
					out[i*n+j] = sum
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
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
// Data layout: [seqLen, numHeads, headDim]
// All heads at the same sequence position use the same RoPE position.
// Uses interleaved (NEOX-style) layout where pairs are (0,1), (2,3), (4,5), ...
// This matches llama.cpp's implementation for Llama-family models.
func (b *cpuBackend) RoPE(q, k []float32, headDim, numHeads, seqLen, startPos int, theta float32) {
	totalVectors := len(q) / headDim

	for i := 0; i < totalVectors; i++ {
		// Compute sequence position: all heads at same seq pos use same RoPE pos
		seqPos := i / numHeads
		pos := startPos + seqPos
		offset := i * headDim

		// Interleaved layout: pairs are (0,1), (2,3), (4,5), ...
		for j := 0; j < headDim/2; j++ {
			idx := j * 2
			exp := float64(2*j) / float64(headDim)
			freq := float32(1.0 / math.Pow(float64(theta), exp))
			angle := float32(pos) * freq
			cos := float32(math.Cos(float64(angle)))
			sin := float32(math.Sin(float64(angle)))

			val1 := q[offset+idx]
			val2 := q[offset+idx+1]
			q[offset+idx] = val1*cos - val2*sin
			q[offset+idx+1] = val1*sin + val2*cos
		}
	}

	if k != nil {
		// K may have different numHeads (GQA), recalculate
		numKVHeads := len(k) / (seqLen * headDim)
		if numKVHeads == 0 {
			numKVHeads = 1
		}
		totalVectorsK := len(k) / headDim
		for i := 0; i < totalVectorsK; i++ {
			seqPos := i / numKVHeads
			pos := startPos + seqPos
			offset := i * headDim

			for j := 0; j < headDim/2; j++ {
				idx := j * 2
				exp := float64(2*j) / float64(headDim)
				freq := float32(1.0 / math.Pow(float64(theta), exp))
				angle := float32(pos) * freq
				cos := float32(math.Cos(float64(angle)))
				sin := float32(math.Sin(float64(angle)))

				val1 := k[offset+idx]
				val2 := k[offset+idx+1]
				k[offset+idx] = val1*cos - val2*sin
				k[offset+idx+1] = val1*sin + val2*cos
			}
		}
	}
}

// SiLU applies the Sigmoid Linear Unit activation function element-wise.
func (b *cpuBackend) SiLU(x, out []float32, n int) {
	for i := 0; i < n; i++ {
		val := x[i]
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-val))))
		out[i] = val * sigmoid
	}
}

// Embedding performs lookup: out[i] = table[ids[i]]
func (b *cpuBackend) Embedding(ids []int, table []float32, out []float32, dim int) {
	for i, id := range ids {
		start := id * dim
		end := start + dim
		if start < 0 || end > len(table) {
			continue
		}
		src := table[start:end]
		dst := out[i*dim : (i+1)*dim]
		copy(dst, src)
	}
}

// Softmax applies the softmax function row-wise.
func (b *cpuBackend) Softmax(x, out []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		offset := i * cols
		row := x[offset : offset+cols]
		outRow := out[offset : offset+cols]

		// 1. Find max for numerical stability
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}

		// 2. Compute exponentials and sum
		var sum float32
		for j, v := range row {
			exp := float32(math.Exp(float64(v - maxVal)))
			outRow[j] = exp
			sum += exp
		}

		// 3. Normalize
		invSum := 1.0 / sum
		for j := range outRow {
			outRow[j] *= invSum
		}
	}
}

// RoPEShift applies a uniform RoPE position shift to K vectors in place.
// This enables fragment caching by transforming RoPE(p) -> RoPE(p+shift).
// Uses interleaved (NEOX-style) layout where pairs are (0,1), (2,3), (4,5), ...
func (b *cpuBackend) RoPEShift(k []float32, headDim, numKVHeads, numTokens, shift int, theta float32) {
	if shift == 0 {
		return // No shift needed
	}

	// Apply RoPE(shift) to each K vector
	// K layout: [numTokens, numKVHeads, headDim]
	for t := 0; t < numTokens; t++ {
		for h := 0; h < numKVHeads; h++ {
			offset := t*numKVHeads*headDim + h*headDim

			// Interleaved layout: pairs are (0,1), (2,3), (4,5), ...
			for j := 0; j < headDim/2; j++ {
				idx := j * 2
				// Compute rotation angle for this dimension at position `shift`
				exp := float64(2*j) / float64(headDim)
				freq := float32(1.0 / math.Pow(float64(theta), exp))
				angle := float32(shift) * freq
				cos := float32(math.Cos(float64(angle)))
				sin := float32(math.Sin(float64(angle)))

				// Apply rotation
				val1 := k[offset+idx]
				val2 := k[offset+idx+1]
				k[offset+idx] = val1*cos - val2*sin
				k[offset+idx+1] = val1*sin + val2*cos
			}
		}
	}
}

// SDPA performs Scaled Dot-Product Attention for a single query position.
// This implements causal attention: each query can only attend to positions <= its own.
// For decode (single query), this means attending to all kvLen positions.
//
// Q: [numQHeads, headDim] - query vectors
// K: [numKVHeads, maxSeqLen, headDim] - key cache (head-major layout)
// V: [numKVHeads, maxSeqLen, headDim] - value cache (head-major layout)
// out: [numQHeads, headDim] - output vectors
// kvHeadStride: stride between KV heads (maxSeqLen * headDim)
func (b *cpuBackend) SDPA(q, k, v, out []float32, kvLen, numQHeads, numKVHeads, headDim int, scale float32, kvHeadStride int) {
	// GQA: map Q heads to KV heads
	headsPerKV := numQHeads / numKVHeads

	// Temporary buffer for attention scores
	scores := make([]float32, kvLen)

	// Process each query head
	for h := 0; h < numQHeads; h++ {
		kvHead := h / headsPerKV

		// Pointer to this Q head: [headDim]
		qOffset := h * headDim
		qHead := q[qOffset : qOffset+headDim]

		// Output pointer for this head
		outOffset := h * headDim
		outHead := out[outOffset : outOffset+headDim]

		// KV cache layout: [numKVHeads, maxSeqLen, headDim]
		// For head h, position p: offset = h * kvHeadStride + p * headDim
		kvHeadBase := kvHead * kvHeadStride

		// Compute attention scores: Q dot K for each position
		for pos := 0; pos < kvLen; pos++ {
			// K at position pos for this KV head
			kOffset := kvHeadBase + pos*headDim
			kVec := k[kOffset : kOffset+headDim]

			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * kVec[d]
			}
			scores[pos] = dot * scale
		}

		// Softmax over scores
		// Find max for numerical stability
		maxScore := scores[0]
		for _, s := range scores {
			if s > maxScore {
				maxScore = s
			}
		}

		// Compute exp and sum
		var sumExp float32
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
			sumExp += scores[i]
		}

		// Normalize
		invSum := 1.0 / sumExp
		for i := range scores {
			scores[i] *= invSum
		}

		// Compute weighted sum of V
		// Zero output first
		for d := 0; d < headDim; d++ {
			outHead[d] = 0
		}

		for pos := 0; pos < kvLen; pos++ {
			// V at position pos for this KV head
			vOffset := kvHeadBase + pos*headDim
			vVec := v[vOffset : vOffset+headDim]

			weight := scores[pos]
			for d := 0; d < headDim; d++ {
				outHead[d] += weight * vVec[d]
			}
		}
	}
}
