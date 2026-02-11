package metal

import (
	"testing"
)

func TestSDPAF16_HeadDim80(t *testing.T) {
	b, err := NewBackend(0) // Device 0
	if err != nil {
		t.Skipf("Skipping test, backend creation failed: %v", err)
	}

	// Phi-2 parameters
	kvLen := 10
	numQHeads := 32
	numKVHeads := 32
	headDim := 80
	scale := float32(0.1)
	kvHeadStride := kvLen * headDim

	// Allocate FP16 tensors (2 bytes per element)
	qSize := numQHeads * headDim * 2
	kvSize := numKVHeads * kvLen * headDim * 2
	outSize := numQHeads * headDim * 2

	q := b.Alloc(qSize)
	k := b.Alloc(kvSize)
	v := b.Alloc(kvSize)
	out := b.Alloc(outSize)

	// We can't easily validate numerical correctness without a reference implementation here,
	// but we can ensure it runs without crashing and check for basic sanity if needed.
	// For now, this verifies the kernel launch and shared memory sizing for headDim=80.

	b.SDPAF16(q, k, v, out, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
	b.Sync()

	t.Log("SDPAF16 with HeadDim=80 completed successfully")
}
