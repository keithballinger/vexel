//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"testing"

	"vexel/inference/tensor"
)

// cpuRoPEWithFreqs is a reference implementation for RoPE with pre-computed frequencies.
// q: [seqLen, numHeads, headDim], freqs: [headDim/2]
func cpuRoPEWithFreqs(q []float32, freqs []float32, headDim, numHeads, seqLen, startPos int, ropeNeox bool) {
	halfDim := headDim / 2
	for s := 0; s < seqLen; s++ {
		pos := startPos + s
		for h := 0; h < numHeads; h++ {
			offset := (s*numHeads + h) * headDim
			for j := 0; j < halfDim; j++ {
				var idx0, idx1 int
				if ropeNeox {
					idx0 = j
					idx1 = j + halfDim
				} else {
					idx0 = j * 2
					idx1 = j*2 + 1
				}

				angle := float32(pos) * freqs[j]
				cos := float32(math.Cos(float64(angle)))
				sin := float32(math.Sin(float64(angle)))

				v0 := q[offset+idx0]
				v1 := q[offset+idx1]
				q[offset+idx0] = v0*cos - v1*sin
				q[offset+idx1] = v0*sin + v1*cos
			}
		}
	}
}

// TestScaledRoPE verifies RoPE with pre-computed inverse frequencies for Gemma 2.
//
// Track 6: Gemma Architecture, Phase 3 Task 1.
func TestScaledRoPE(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}

	t.Run("theta_equivalence", func(t *testing.T) {
		// Verify that when we pre-compute frequencies from theta, the scaled kernel
		// produces the same output as the standard RoPE kernel.
		headDim := 128
		numQHeads := 8
		numKVHeads := 4
		seqLen := 1
		startPos := 5
		theta := float32(10000.0)
		halfDim := headDim / 2

		// Pre-compute inverse frequencies from theta (same formula as standard kernel)
		freqs := make([]float32, halfDim)
		for j := 0; j < halfDim; j++ {
			freqs[j] = 1.0 / float32(math.Pow(float64(theta), float64(2*j)/float64(headDim)))
		}

		// Create identical Q and K data for both kernels
		qSize := seqLen * numQHeads * headDim
		kSize := seqLen * numKVHeads * headDim

		qStd := make([]float32, qSize)
		kStd := make([]float32, kSize)
		qScaled := make([]float32, qSize)
		kScaled := make([]float32, kSize)

		for i := range qStd {
			qStd[i] = float32(i%17) * 0.1
			qScaled[i] = qStd[i]
		}
		for i := range kStd {
			kStd[i] = float32(i%13) * 0.15
			kScaled[i] = kStd[i]
		}

		// Upload to GPU
		qStdBuf := allocAndUpload(be, qStd)
		kStdBuf := allocAndUpload(be, kStd)
		qScaledBuf := allocAndUpload(be, qScaled)
		kScaledBuf := allocAndUpload(be, kScaled)
		freqBuf := allocAndUpload(be, freqs)

		// Run standard kernel
		be.RoPE(qStdBuf, kStdBuf, headDim, numQHeads, numKVHeads, seqLen, startPos, 0, theta, false)

		// Run scaled kernel
		be.RoPEWithFreqs(qScaledBuf, kScaledBuf, freqBuf, headDim, numQHeads, numKVHeads, seqLen, startPos, false)

		be.Sync()

		// Compare outputs
		qStdOut := downloadFloat32(be, qStdBuf, qSize)
		kStdOut := downloadFloat32(be, kStdBuf, kSize)
		qScaledOut := downloadFloat32(be, qScaledBuf, qSize)
		kScaledOut := downloadFloat32(be, kScaledBuf, kSize)

		maxDiffQ := maxAbsDiff(qStdOut, qScaledOut)
		maxDiffK := maxAbsDiff(kStdOut, kScaledOut)

		t.Logf("Q maxDiff: %e, K maxDiff: %e", maxDiffQ, maxDiffK)

		if maxDiffQ > 1e-5 {
			t.Errorf("Q outputs differ: maxDiff=%e (want < 1e-5)", maxDiffQ)
		}
		if maxDiffK > 1e-5 {
			t.Errorf("K outputs differ: maxDiff=%e (want < 1e-5)", maxDiffK)
		}
	})

	t.Run("custom_freqs", func(t *testing.T) {
		// Verify that custom (non-theta-derived) frequencies produce correct results
		// by comparing GPU against CPU reference.
		headDim := 64
		numQHeads := 4
		numKVHeads := 2
		seqLen := 3
		startPos := 10

		halfDim := headDim / 2

		// Create custom learned frequencies (not derivable from any single theta)
		freqs := make([]float32, halfDim)
		for j := 0; j < halfDim; j++ {
			// Mix of different frequencies to simulate learned values
			freqs[j] = 0.01 * float32(j+1)
		}

		qSize := seqLen * numQHeads * headDim
		kSize := seqLen * numKVHeads * headDim

		qGPU := make([]float32, qSize)
		kGPU := make([]float32, kSize)
		qCPU := make([]float32, qSize)
		kCPU := make([]float32, kSize)

		for i := range qGPU {
			v := float32(i%23) * 0.05
			qGPU[i] = v
			qCPU[i] = v
		}
		for i := range kGPU {
			v := float32(i%19) * 0.07
			kGPU[i] = v
			kCPU[i] = v
		}

		// GPU path
		qBuf := allocAndUpload(be, qGPU)
		kBuf := allocAndUpload(be, kGPU)
		freqBuf := allocAndUpload(be, freqs)

		be.RoPEWithFreqs(qBuf, kBuf, freqBuf, headDim, numQHeads, numKVHeads, seqLen, startPos, false)
		be.Sync()

		qOut := downloadFloat32(be, qBuf, qSize)
		kOut := downloadFloat32(be, kBuf, kSize)

		// CPU reference
		cpuRoPEWithFreqs(qCPU, freqs, headDim, numQHeads, seqLen, startPos, false)
		cpuRoPEWithFreqs(kCPU, freqs, headDim, numKVHeads, seqLen, startPos, false)

		maxDiffQ := maxAbsDiff(qOut, qCPU)
		maxDiffK := maxAbsDiff(kOut, kCPU)

		t.Logf("Q maxDiff: %e, K maxDiff: %e", maxDiffQ, maxDiffK)

		if maxDiffQ > 1e-5 {
			t.Errorf("Q GPU vs CPU differ: maxDiff=%e (want < 1e-5)", maxDiffQ)
		}
		if maxDiffK > 1e-5 {
			t.Errorf("K GPU vs CPU differ: maxDiff=%e (want < 1e-5)", maxDiffK)
		}
	})

	t.Run("neox_style", func(t *testing.T) {
		// Verify NEOX-style (split pairs) works with custom frequencies.
		headDim := 32
		numQHeads := 2
		numKVHeads := 2
		seqLen := 1
		startPos := 3

		halfDim := headDim / 2
		freqs := make([]float32, halfDim)
		for j := 0; j < halfDim; j++ {
			freqs[j] = 0.1 * float32(j+1)
		}

		qSize := seqLen * numQHeads * headDim
		kSize := seqLen * numKVHeads * headDim

		qGPU := make([]float32, qSize)
		kGPU := make([]float32, kSize)
		qCPU := make([]float32, qSize)
		kCPU := make([]float32, kSize)
		for i := range qGPU {
			v := float32(i%11) * 0.2
			qGPU[i] = v
			qCPU[i] = v
		}
		for i := range kGPU {
			v := float32(i%7) * 0.3
			kGPU[i] = v
			kCPU[i] = v
		}

		qBuf := allocAndUpload(be, qGPU)
		kBuf := allocAndUpload(be, kGPU)
		freqBuf := allocAndUpload(be, freqs)

		be.RoPEWithFreqs(qBuf, kBuf, freqBuf, headDim, numQHeads, numKVHeads, seqLen, startPos, true)
		be.Sync()

		qOut := downloadFloat32(be, qBuf, qSize)
		kOut := downloadFloat32(be, kBuf, kSize)

		// CPU reference with NEOX style
		cpuRoPEWithFreqs(qCPU, freqs, headDim, numQHeads, seqLen, startPos, true)
		cpuRoPEWithFreqs(kCPU, freqs, headDim, numKVHeads, seqLen, startPos, true)

		maxDiffQ := maxAbsDiff(qOut, qCPU)
		maxDiffK := maxAbsDiff(kOut, kCPU)
		t.Logf("NEOX Q maxDiff: %e, K maxDiff: %e", maxDiffQ, maxDiffK)

		if maxDiffQ > 1e-5 {
			t.Errorf("NEOX Q GPU vs CPU differ: maxDiff=%e (want < 1e-5)", maxDiffQ)
		}
		if maxDiffK > 1e-5 {
			t.Errorf("NEOX K GPU vs CPU differ: maxDiff=%e (want < 1e-5)", maxDiffK)
		}
	})

	t.Run("production_size", func(t *testing.T) {
		// Gemma 2 9B typical sizes: headDim=256, 16 Q heads, 8 KV heads
		headDim := 256
		numQHeads := 16
		numKVHeads := 8
		seqLen := 1
		startPos := 127

		halfDim := headDim / 2
		freqs := make([]float32, halfDim)
		for j := 0; j < halfDim; j++ {
			freqs[j] = 1.0 / float32(math.Pow(10000.0, float64(2*j)/float64(headDim)))
		}

		qSize := seqLen * numQHeads * headDim
		kSize := seqLen * numKVHeads * headDim

		qGPU := make([]float32, qSize)
		kGPU := make([]float32, kSize)
		qCPU := make([]float32, qSize)
		kCPU := make([]float32, kSize)

		for i := range qGPU {
			v := float32(i%31) * 0.03
			qGPU[i] = v
			qCPU[i] = v
		}
		for i := range kGPU {
			v := float32(i%29) * 0.04
			kGPU[i] = v
			kCPU[i] = v
		}

		qBuf := allocAndUpload(be, qGPU)
		kBuf := allocAndUpload(be, kGPU)
		freqBuf := allocAndUpload(be, freqs)

		be.RoPEWithFreqs(qBuf, kBuf, freqBuf, headDim, numQHeads, numKVHeads, seqLen, startPos, false)
		be.Sync()

		qOut := downloadFloat32(be, qBuf, qSize)
		kOut := downloadFloat32(be, kBuf, kSize)

		cpuRoPEWithFreqs(qCPU, freqs, headDim, numQHeads, seqLen, startPos, false)
		cpuRoPEWithFreqs(kCPU, freqs, headDim, numKVHeads, seqLen, startPos, false)

		maxDiffQ := maxAbsDiff(qOut, qCPU)
		maxDiffK := maxAbsDiff(kOut, kCPU)

		t.Logf("Production Q maxDiff: %e, K maxDiff: %e", maxDiffQ, maxDiffK)

		if maxDiffQ > 1e-5 {
			t.Errorf("Production Q GPU vs CPU differ: maxDiff=%e (want < 1e-5)", maxDiffQ)
		}
		if maxDiffK > 1e-5 {
			t.Errorf("Production K GPU vs CPU differ: maxDiff=%e (want < 1e-5)", maxDiffK)
		}
	})
}

// allocAndUpload creates a GPU buffer from a float32 slice.
func allocAndUpload(be *Backend, data []float32) tensor.DevicePtr {
	sizeBytes := len(data) * 4
	buf := be.Alloc(sizeBytes)
	bytes := make([]byte, sizeBytes)
	for i, v := range data {
		binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(v))
	}
	be.ToDevice(buf, bytes)
	return buf
}

// downloadFloat32 reads float32 data from a GPU buffer.
func downloadFloat32(be *Backend, ptr tensor.DevicePtr, n int) []float32 {
	sizeBytes := n * 4
	bytes := make([]byte, sizeBytes)
	be.ToHost(bytes, ptr)
	result := make([]float32, n)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
	}
	return result
}

// maxAbsDiff returns the maximum absolute difference between two slices.
func maxAbsDiff(a, b []float32) float64 {
	maxD := 0.0
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxD {
			maxD = d
		}
	}
	return maxD
}
