//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"
)

// cpuSoftCap computes logit soft-capping: cap * tanh(score / cap)
func cpuSoftCap(score, cap float32) float32 {
	if cap <= 0 {
		return score
	}
	return cap * float32(math.Tanh(float64(score/cap)))
}

// cpuSDPASoftCap computes SDPA with logit soft-capping (reference implementation).
// Q: [numQHeads, headDim], K/V: [numKVHeads, kvLen, headDim] (head-major)
// out: [numQHeads, headDim]
func cpuSDPASoftCap(q, k, v []float32, kvLen, numQHeads, numKVHeads, headDim int, scale, softcap float32, kvHeadStride int) []float32 {
	out := make([]float32, numQHeads*headDim)

	gqaRatio := numQHeads / numKVHeads

	for h := 0; h < numQHeads; h++ {
		kvHead := h / gqaRatio

		// Compute QK^T scores
		scores := make([]float64, kvLen)
		maxScore := math.Inf(-1)

		for pos := 0; pos < kvLen; pos++ {
			dot := float32(0)
			for d := 0; d < headDim; d++ {
				dot += q[h*headDim+d] * k[kvHead*kvHeadStride+pos*headDim+d]
			}
			score := dot * scale

			// Apply soft-capping
			if softcap > 0 {
				score = cpuSoftCap(score, softcap)
			}

			scores[pos] = float64(score)
			if scores[pos] > maxScore {
				maxScore = scores[pos]
			}
		}

		// Softmax
		sumExp := 0.0
		for pos := 0; pos < kvLen; pos++ {
			scores[pos] = math.Exp(scores[pos] - maxScore)
			sumExp += scores[pos]
		}
		for pos := 0; pos < kvLen; pos++ {
			scores[pos] /= sumExp
		}

		// Weighted sum of V
		for d := 0; d < headDim; d++ {
			sum := float64(0)
			for pos := 0; pos < kvLen; pos++ {
				sum += scores[pos] * float64(v[kvHead*kvHeadStride+pos*headDim+d])
			}
			out[h*headDim+d] = float32(sum)
		}
	}

	return out
}

// TestSDPASoftCap verifies SDPA with logit soft-capping for Gemma 2.
//
// Track 6: Gemma Architecture, Phase 2 Task 1.
func TestSDPASoftCap(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	t.Run("softcap_zero_matches_regular", func(t *testing.T) {
		// With softcap=0, output should match regular SDPA
		numQHeads := 2
		numKVHeads := 1
		headDim := 64
		kvLen := 8
		kvHeadStride := kvLen * headDim

		q := make([]float32, numQHeads*headDim)
		k := make([]float32, numKVHeads*kvHeadStride)
		v := make([]float32, numKVHeads*kvHeadStride)

		// Fill with deterministic values
		for i := range q {
			q[i] = float32(i%7-3) * 0.1
		}
		for i := range k {
			k[i] = float32(i%11-5) * 0.1
		}
		for i := range v {
			v[i] = float32(i%13-6) * 0.05
		}

		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		// CPU reference (no softcap)
		expected := cpuSDPASoftCap(q, k, v, kvLen, numQHeads, numKVHeads, headDim, scale, 0, kvHeadStride)

		// GPU with softcap=0
		qBuf := be.Alloc(len(q) * 4)
		kBuf := be.Alloc(len(k) * 4)
		vBuf := be.Alloc(len(v) * 4)
		outBuf := be.Alloc(numQHeads * headDim * 4)

		be.ToDevice(qBuf, float32ToBytes(q))
		be.ToDevice(kBuf, float32ToBytes(k))
		be.ToDevice(vBuf, float32ToBytes(v))
		be.Sync()

		be.SDPASoftCap(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, 0, kvHeadStride)
		be.Sync()

		outBytes := make([]byte, numQHeads*headDim*4)
		be.ToHost(outBytes, outBuf)
		result := bytesToFloat32(outBytes)

		maxDiff := float32(0)
		for i := range result {
			diff := float32(math.Abs(float64(result[i] - expected[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff > 1e-3 {
			t.Errorf("softcap=0 maxDiff=%e (should match regular SDPA)", maxDiff)
		}
		t.Logf("softcap=0 maxDiff=%e", maxDiff)

		be.Free(qBuf)
		be.Free(kBuf)
		be.Free(vBuf)
		be.Free(outBuf)
	})

	t.Run("softcap_bounds_scores", func(t *testing.T) {
		// With extreme scores, softcap should limit attention to be more uniform
		numQHeads := 1
		numKVHeads := 1
		headDim := 4
		kvLen := 4
		kvHeadStride := kvLen * headDim

		q := []float32{1.0, 0.0, 0.0, 0.0}
		// K[0] has very high score, others zero
		k := []float32{
			10.0, 0.0, 0.0, 0.0, // pos 0: dot=10
			0.0, 0.0, 0.0, 0.0, // pos 1: dot=0
			0.0, 0.0, 0.0, 0.0, // pos 2: dot=0
			0.0, 0.0, 0.0, 0.0, // pos 3: dot=0
		}
		v := []float32{
			1.0, 0.0, 0.0, 0.0, // pos 0
			0.0, 1.0, 0.0, 0.0, // pos 1
			0.0, 0.0, 1.0, 0.0, // pos 2
			0.0, 0.0, 0.0, 1.0, // pos 3
		}

		scale := float32(1.0) // No scaling, raw dot products
		softcap := float32(5.0)

		// CPU reference
		expected := cpuSDPASoftCap(q, k, v, kvLen, numQHeads, numKVHeads, headDim, scale, softcap, kvHeadStride)

		// GPU
		qBuf := be.Alloc(len(q) * 4)
		kBuf := be.Alloc(len(k) * 4)
		vBuf := be.Alloc(len(v) * 4)
		outBuf := be.Alloc(numQHeads * headDim * 4)

		be.ToDevice(qBuf, float32ToBytes(q))
		be.ToDevice(kBuf, float32ToBytes(k))
		be.ToDevice(vBuf, float32ToBytes(v))
		be.Sync()

		be.SDPASoftCap(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, softcap, kvHeadStride)
		be.Sync()

		outBytes := make([]byte, numQHeads*headDim*4)
		be.ToHost(outBytes, outBuf)
		result := bytesToFloat32(outBytes)

		maxDiff := float32(0)
		for i := range result {
			diff := float32(math.Abs(float64(result[i] - expected[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff > 1e-3 {
			t.Errorf("softcap=%f maxDiff=%e (exceeds tolerance)", softcap, maxDiff)
		}

		// Verify that softcap reduced score dominance compared to uncapped.
		// Without softcap: score[0]=10, softmax ≈ 0.9999 (nearly all weight on pos 0).
		// With softcap=5: score[0]=5*tanh(10/5)≈4.82, softmax ≈ 0.976 (still dominant but less).
		// The key metric: softcap reduced the attention weight from ~1.0 to ~0.976.
		if result[0] > 0.999 {
			t.Errorf("softcap didn't reduce dominance: out[0]=%f (expected < 0.999)", result[0])
		}

		t.Logf("softcap=%f: out=%v, maxDiff=%e", softcap, result, maxDiff)
	})

	t.Run("softcap_gemma2_typical", func(t *testing.T) {
		// Test with Gemma 2 typical values: softcap=30.0, headDim=256, 8 heads
		numQHeads := 8
		numKVHeads := 4
		headDim := 128
		kvLen := 32
		kvHeadStride := kvLen * headDim
		softcap := float32(30.0)

		q := make([]float32, numQHeads*headDim)
		k := make([]float32, numKVHeads*kvHeadStride)
		v := make([]float32, numKVHeads*kvHeadStride)

		for i := range q {
			q[i] = float32(i%17-8) * 0.1
		}
		for i := range k {
			k[i] = float32(i%23-11) * 0.1
		}
		for i := range v {
			v[i] = float32(i%19-9) * 0.05
		}

		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		// CPU reference
		expected := cpuSDPASoftCap(q, k, v, kvLen, numQHeads, numKVHeads, headDim, scale, softcap, kvHeadStride)

		// GPU
		qBuf := be.Alloc(len(q) * 4)
		kBuf := be.Alloc(len(k) * 4)
		vBuf := be.Alloc(len(v) * 4)
		outBuf := be.Alloc(numQHeads * headDim * 4)

		be.ToDevice(qBuf, float32ToBytes(q))
		be.ToDevice(kBuf, float32ToBytes(k))
		be.ToDevice(vBuf, float32ToBytes(v))
		be.Sync()

		be.SDPASoftCap(qBuf, kBuf, vBuf, outBuf, kvLen, numQHeads, numKVHeads, headDim, scale, softcap, kvHeadStride)
		be.Sync()

		outBytes := make([]byte, numQHeads*headDim*4)
		be.ToHost(outBytes, outBuf)
		result := bytesToFloat32(outBytes)

		maxDiff := float32(0)
		for i := range result {
			diff := float32(math.Abs(float64(result[i] - expected[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
			if math.IsNaN(float64(result[i])) {
				t.Fatalf("NaN at index %d", i)
			}
		}

		if maxDiff > 1e-3 {
			t.Errorf("Gemma2 typical softcap=%f: maxDiff=%e", softcap, maxDiff)
		}

		t.Logf("Gemma2 typical (numQHeads=%d, numKVHeads=%d, headDim=%d, kvLen=%d, softcap=%f): maxDiff=%e",
			numQHeads, numKVHeads, headDim, kvLen, softcap, maxDiff)
	})
}

// TestSDPAPrefillSoftCap verifies prefill SDPA with logit soft-capping.
func TestSDPAPrefillSoftCap(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	t.Run("prefill_softcap_basic", func(t *testing.T) {
		numQHeads := 4
		numKVHeads := 2
		headDim := 64
		seqLen := 16
		softcap := float32(30.0)

		q := make([]float32, seqLen*numQHeads*headDim)
		k := make([]float32, seqLen*numKVHeads*headDim)
		v := make([]float32, seqLen*numKVHeads*headDim)

		for i := range q {
			q[i] = float32(i%17-8) * 0.1
		}
		for i := range k {
			k[i] = float32(i%23-11) * 0.1
		}
		for i := range v {
			v[i] = float32(i%19-9) * 0.05
		}

		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		// CPU reference for prefill with causal masking + soft-capping
		expected := cpuSDPAPrefillSoftCap(q, k, v, seqLen, numQHeads, numKVHeads, headDim, scale, softcap)

		// GPU
		qBuf := be.Alloc(len(q) * 4)
		kBuf := be.Alloc(len(k) * 4)
		vBuf := be.Alloc(len(v) * 4)
		outBuf := be.Alloc(seqLen * numQHeads * headDim * 4)

		be.ToDevice(qBuf, float32ToBytes(q))
		be.ToDevice(kBuf, float32ToBytes(k))
		be.ToDevice(vBuf, float32ToBytes(v))
		be.Sync()

		be.SDPAPrefillSoftCap(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale, softcap)
		be.Sync()

		outBytes := make([]byte, seqLen*numQHeads*headDim*4)
		be.ToHost(outBytes, outBuf)
		result := bytesToFloat32(outBytes)

		maxDiff := float32(0)
		for i := range result {
			diff := float32(math.Abs(float64(result[i] - expected[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
			if math.IsNaN(float64(result[i])) {
				t.Fatalf("NaN at index %d", i)
			}
		}

		if maxDiff > 1e-2 {
			t.Errorf("Prefill softcap maxDiff=%e (exceeds tolerance)", maxDiff)
		}

		t.Logf("Prefill softcap=%f: seqLen=%d, maxDiff=%e", softcap, seqLen, maxDiff)
	})

	// Test with Gemma 2 dimensions (headDim=256, 8 Q heads, 4 KV heads)
	t.Run("prefill_softcap_gemma2_dims", func(t *testing.T) {
		numQHeads := 8
		numKVHeads := 4
		headDim := 256
		seqLen := 32
		softcap := float32(50.0) // Gemma 2 uses softcap=50

		q := make([]float32, seqLen*numQHeads*headDim)
		k := make([]float32, seqLen*numKVHeads*headDim)
		v := make([]float32, seqLen*numKVHeads*headDim)

		for i := range q {
			q[i] = float32(i%17-8) * 0.01
		}
		for i := range k {
			k[i] = float32(i%23-11) * 0.01
		}
		for i := range v {
			v[i] = float32(i%19-9) * 0.005
		}

		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		expected := cpuSDPAPrefillSoftCap(q, k, v, seqLen, numQHeads, numKVHeads, headDim, scale, softcap)

		qBuf := be.Alloc(len(q) * 4)
		kBuf := be.Alloc(len(k) * 4)
		vBuf := be.Alloc(len(v) * 4)
		outBuf := be.Alloc(seqLen * numQHeads * headDim * 4)

		be.ToDevice(qBuf, float32ToBytes(q))
		be.ToDevice(kBuf, float32ToBytes(k))
		be.ToDevice(vBuf, float32ToBytes(v))
		be.Sync()

		be.SDPAPrefillSoftCap(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale, softcap)
		be.Sync()

		outBytes := make([]byte, seqLen*numQHeads*headDim*4)
		be.ToHost(outBytes, outBuf)
		result := bytesToFloat32(outBytes)

		maxDiff := float32(0)
		for i := range result {
			diff := float32(math.Abs(float64(result[i] - expected[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
			if math.IsNaN(float64(result[i])) {
				t.Fatalf("NaN at index %d", i)
			}
		}

		if maxDiff > 1e-2 {
			t.Errorf("Prefill softcap Gemma2 dims maxDiff=%e (exceeds tolerance)", maxDiff)
		}

		t.Logf("Prefill softcap=%f: seqLen=%d, headDim=%d, numQHeads=%d, numKVHeads=%d, maxDiff=%e",
			softcap, seqLen, headDim, numQHeads, numKVHeads, maxDiff)
	})

	// Test with longer sequence to verify causal masking correctness
	t.Run("prefill_softcap_long_seq", func(t *testing.T) {
		numQHeads := 8
		numKVHeads := 4
		headDim := 256
		seqLen := 64
		softcap := float32(50.0)

		q := make([]float32, seqLen*numQHeads*headDim)
		k := make([]float32, seqLen*numKVHeads*headDim)
		v := make([]float32, seqLen*numKVHeads*headDim)

		for i := range q {
			q[i] = float32(i%17-8) * 0.01
		}
		for i := range k {
			k[i] = float32(i%23-11) * 0.01
		}
		for i := range v {
			v[i] = float32(i%19-9) * 0.005
		}

		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		expected := cpuSDPAPrefillSoftCap(q, k, v, seqLen, numQHeads, numKVHeads, headDim, scale, softcap)

		qBuf := be.Alloc(len(q) * 4)
		kBuf := be.Alloc(len(k) * 4)
		vBuf := be.Alloc(len(v) * 4)
		outBuf := be.Alloc(seqLen * numQHeads * headDim * 4)

		be.ToDevice(qBuf, float32ToBytes(q))
		be.ToDevice(kBuf, float32ToBytes(k))
		be.ToDevice(vBuf, float32ToBytes(v))
		be.Sync()

		be.SDPAPrefillSoftCap(qBuf, kBuf, vBuf, outBuf, seqLen, numQHeads, numKVHeads, headDim, scale, softcap)
		be.Sync()

		outBytes := make([]byte, seqLen*numQHeads*headDim*4)
		be.ToHost(outBytes, outBuf)
		result := bytesToFloat32(outBytes)

		maxDiff := float32(0)
		for i := range result {
			diff := float32(math.Abs(float64(result[i] - expected[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
			if math.IsNaN(float64(result[i])) {
				t.Fatalf("NaN at index %d", i)
			}
		}

		if maxDiff > 1e-2 {
			t.Errorf("Prefill softcap long seq maxDiff=%e (exceeds tolerance)", maxDiff)
		}

		t.Logf("Prefill softcap=%f: seqLen=%d, headDim=%d, maxDiff=%e", softcap, seqLen, headDim, maxDiff)
	})
}

// cpuSDPAPrefillSoftCap computes prefill SDPA with causal masking and soft-capping.
// Q/K/V: [seqLen, numHeads, headDim], out: [seqLen, numQHeads, headDim]
func cpuSDPAPrefillSoftCap(q, k, v []float32, seqLen, numQHeads, numKVHeads, headDim int, scale, softcap float32) []float32 {
	out := make([]float32, seqLen*numQHeads*headDim)
	gqaRatio := numQHeads / numKVHeads

	for pos := 0; pos < seqLen; pos++ {
		for h := 0; h < numQHeads; h++ {
			kvHead := h / gqaRatio

			// Compute scores against all causal positions (0..pos)
			scores := make([]float64, pos+1)
			maxScore := math.Inf(-1)

			for kPos := 0; kPos <= pos; kPos++ {
				dot := float32(0)
				for d := 0; d < headDim; d++ {
					qi := q[pos*numQHeads*headDim+h*headDim+d]
					ki := k[kPos*numKVHeads*headDim+kvHead*headDim+d]
					dot += qi * ki
				}
				score := dot * scale

				if softcap > 0 {
					score = cpuSoftCap(score, softcap)
				}

				scores[kPos] = float64(score)
				if scores[kPos] > maxScore {
					maxScore = scores[kPos]
				}
			}

			// Softmax
			sumExp := 0.0
			for kPos := 0; kPos <= pos; kPos++ {
				scores[kPos] = math.Exp(scores[kPos] - maxScore)
				sumExp += scores[kPos]
			}
			for kPos := 0; kPos <= pos; kPos++ {
				scores[kPos] /= sumExp
			}

			// Weighted sum
			for d := 0; d < headDim; d++ {
				sum := float64(0)
				for kPos := 0; kPos <= pos; kPos++ {
					vi := v[kPos*numKVHeads*headDim+kvHead*headDim+d]
					sum += scores[kPos] * float64(vi)
				}
				out[pos*numQHeads*headDim+h*headDim+d] = float32(sum)
			}
		}
	}

	return out
}
