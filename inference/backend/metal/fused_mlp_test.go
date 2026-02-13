//go:build metal && darwin && cgo

package metal

import (
	"math"
	"math/rand"
	"testing"
)

func TestFusedMLP(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Dimensions
	N := 32  // Intermediate size (must be multiple of 32 for kernel)
	K := 128 // Hidden size (must be multiple of 32 for Q4_0)
	
	// Create weights (Q4_0)
	// We'll create random float weights, then quantize them
	// Actually, easier to mock quantization: 
	// Create Q4_0 buffer directly with known values.
	// But Q4_0 format is complex.
	// Let's use Backend.QuantizeF32ToQ4_0? We don't have that exposed easily for test.
	// We can manually construct Q4_0 blocks.
	
	// Q4_0 block: 18 bytes. 2 bytes scale (f16), 16 bytes data (32 nibbles).
	// Let's set scale=1.0 and data=0 (all zeros) for simplicity first?
	// No, we want to verify computation.
	// Let's rely on random values and fuzzy comparison.
	
	// Allocate buffers
	xBuf := b.Alloc(K * 4) // FP32 input
	
	// W1, W3: [N, K] Q4_0
	// Blocks per row = K / 32
	// Bytes per row = (K/32) * 18
	rowBytes := (K / 32) * 18
	wBytes := N * rowBytes
	
	w1Buf := b.Alloc(wBytes)
	w3Buf := b.Alloc(wBytes)
	outBuf := b.Alloc(N * 4) // FP32 output
	
	defer b.Free(xBuf)
	defer b.Free(w1Buf)
	defer b.Free(w3Buf)
	defer b.Free(outBuf)
	
	// Initialize input x
	xFloats := make([]float32, K)
	for i := range xFloats {
		xFloats[i] = rand.Float32() - 0.5
	}
	b.ToDevice(xBuf, float32ToBytes(xFloats))
	
	// Initialize weights (fill with random bytes)
	// This will result in valid but random Q4_0 weights
	w1Bytes := make([]byte, wBytes)
	w3Bytes := make([]byte, wBytes)
	rand.Read(w1Bytes)
	rand.Read(w3Bytes)
	
	// Ensure scales are positive and not Inf/NaN to avoid issues
	// Scale is first 2 bytes of each 18-byte block (FP16)
	// We can set them to 1.0 (0x3C00)
	for i := 0; i < len(w1Bytes); i += 18 {
		w1Bytes[i] = 0x00
		w1Bytes[i+1] = 0x3C // 1.0 in FP16
		w3Bytes[i] = 0x00
		w3Bytes[i+1] = 0x3C // 1.0 in FP16
	}
	
	b.ToDevice(w1Buf, w1Bytes)
	b.ToDevice(w3Buf, w3Bytes)
	
	// Run fused kernel
	b.MatMulQ4_0_FusedMLP(xBuf, w1Buf, w3Buf, outBuf, 1, N, K)
	b.Sync()
	
	// Read result
	outBytes := make([]byte, N*4)
	b.ToHost(outBytes, outBuf)
	result := bytesToFloat32(outBytes)
	
	// Calculate expected result on CPU
	// This requires dequantizing Q4_0 which is tedious in Go test.
	// Alternative: Verify properties or use a reference run (non-fused).
	
	// Let's run the non-fused path on GPU to compare!
	// 1. MatVecQ4_0 W1 -> gate
	// 2. MatVecQ4_0 W3 -> up
	// 3. SiLU(gate) * up -> out
	
	gateBuf := b.Alloc(N * 4)
	upBuf := b.Alloc(N * 4)
	refOutBuf := b.Alloc(N * 4)
	defer b.Free(gateBuf)
	defer b.Free(upBuf)
	defer b.Free(refOutBuf)
	
	// W1 -> gate
	b.MatVecQ4_0MultiOutput(xBuf, w1Buf, gateBuf, N, K)
	
	// W3 -> up
	b.MatVecQ4_0MultiOutput(xBuf, w3Buf, upBuf, N, K)
	
	// SiLU(gate) * up
	b.SiLUMul(gateBuf, upBuf, refOutBuf, N)
	b.Sync()
	
	// Read reference
	refBytes := make([]byte, N*4)
	b.ToHost(refBytes, refOutBuf)
	refResult := bytesToFloat32(refBytes)
	
	// Compare
	for i := 0; i < N; i++ {
		diff := math.Abs(float64(result[i] - refResult[i]))
		if diff > 1e-3 {
			t.Errorf("Mismatch at index %d: fused=%f, ref=%f, diff=%f", i, result[i], refResult[i], diff)
		}
	}
}
