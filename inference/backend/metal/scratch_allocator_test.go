//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"testing"

	"vexel/inference/tensor"
)

func TestScratchAllocatorCreation(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	// Create a 1MB scratch allocator
	sa, err := NewScratchAllocator(b, 1024*1024)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	if sa.Capacity() != 1024*1024 {
		t.Errorf("capacity: got %d, want %d", sa.Capacity(), 1024*1024)
	}
	if sa.Used() != 0 {
		t.Errorf("used: got %d, want 0", sa.Used())
	}
}

func TestScratchAllocatorBumpAlloc(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	sa, err := NewScratchAllocator(b, 64*1024)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	// Allocate 4KB
	ptr1 := sa.Alloc(4096)
	if ptr1.IsNil() {
		t.Fatal("first allocation returned nil")
	}
	if ptr1.Location() != tensor.Metal {
		t.Errorf("location: got %v, want Metal", ptr1.Location())
	}
	if ptr1.Offset() != 0 {
		t.Errorf("first offset: got %d, want 0", ptr1.Offset())
	}

	// Allocate another 4KB — should be at offset 4096 (aligned)
	ptr2 := sa.Alloc(4096)
	if ptr2.IsNil() {
		t.Fatal("second allocation returned nil")
	}
	if ptr2.Offset() < 4096 {
		t.Errorf("second offset: got %d, want >= 4096", ptr2.Offset())
	}

	// Both should share the same base buffer
	if ptr1.Addr() != ptr2.Addr() {
		t.Error("allocations should share the same base MTLBuffer")
	}

	// Used should reflect allocations
	if sa.Used() < 8192 {
		t.Errorf("used: got %d, want >= 8192", sa.Used())
	}
}

func TestScratchAllocatorAlignment(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	sa, err := NewScratchAllocator(b, 64*1024)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	// Allocate odd size (not aligned)
	_ = sa.Alloc(100) // 100 bytes
	ptr2 := sa.Alloc(200)

	// Second allocation should be aligned to 256 bytes (Metal optimal)
	if ptr2.Offset()%256 != 0 {
		t.Errorf("offset %d not aligned to 256 bytes", ptr2.Offset())
	}
}

func TestScratchAllocatorReset(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	sa, err := NewScratchAllocator(b, 64*1024)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	// Allocate some space
	sa.Alloc(4096)
	sa.Alloc(8192)
	usedBefore := sa.Used()
	if usedBefore == 0 {
		t.Fatal("expected non-zero used after allocations")
	}

	// Reset should bring offset back to 0
	sa.Reset()

	if sa.Used() != 0 {
		t.Errorf("used after reset: got %d, want 0", sa.Used())
	}

	// Should be able to allocate again from the beginning
	ptr := sa.Alloc(4096)
	if ptr.IsNil() {
		t.Fatal("allocation after reset returned nil")
	}
	if ptr.Offset() != 0 {
		t.Errorf("offset after reset: got %d, want 0", ptr.Offset())
	}
}

func TestScratchAllocatorOOM(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	// Tiny scratch space
	sa, err := NewScratchAllocator(b, 4096)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	// First alloc fits
	ptr := sa.Alloc(2048)
	if ptr.IsNil() {
		t.Fatal("first allocation should succeed")
	}

	// Second alloc exceeds capacity
	ptr2 := sa.Alloc(4096)
	if !ptr2.IsNil() {
		t.Error("over-capacity allocation should return nil DevicePtr")
	}
}

func TestScratchAllocatorDataIsolation(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	sa, err := NewScratchAllocator(b, 64*1024)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	// Allocate two sub-regions from scratch space
	ptr1 := sa.Alloc(16) // 4 floats
	ptr2 := sa.Alloc(16) // 4 floats

	if ptr1.IsNil() || ptr2.IsNil() {
		t.Fatal("allocations returned nil")
	}

	// Regions should not overlap: ptr2.Offset() >= ptr1.Offset() + 16
	if ptr2.Offset() < ptr1.Offset()+16 {
		t.Errorf("regions overlap: ptr1.Offset=%d, ptr2.Offset=%d",
			ptr1.Offset(), ptr2.Offset())
	}

	// Write to both using ScratchToDevice (shared memory, can use memcpy via offset)
	data1 := make([]byte, 16)
	data2 := make([]byte, 16)
	for i := range data1 {
		data1[i] = byte(i + 1)
		data2[i] = byte(i + 17)
	}

	sa.WriteAt(ptr1, data1)
	sa.WriteAt(ptr2, data2)

	// Read back
	readBuf1 := sa.ReadAt(ptr1, 16)
	readBuf2 := sa.ReadAt(ptr2, 16)

	for i := range data1 {
		if readBuf1[i] != data1[i] {
			t.Errorf("region1 byte %d: got %d, want %d", i, readBuf1[i], data1[i])
		}
	}
	for i := range data2 {
		if readBuf2[i] != data2[i] {
			t.Errorf("region2 byte %d: got %d, want %d", i, readBuf2[i], data2[i])
		}
	}
}

func TestScratchAllocatorManyAllocs(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	// 8MB scratch space — simulate forward pass allocations
	sa, err := NewScratchAllocator(b, 8*1024*1024)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	// Simulate 32 layers × 9 intermediate buffers per layer
	sizes := []int{
		4096 * 4,  // hidden state
		4096 * 4,  // Q
		4096 * 4,  // K
		4096 * 4,  // V
		4096 * 4,  // attn output
		11008 * 4, // gate proj
		11008 * 4, // up proj
		11008 * 4, // SiLU×Mul
		4096 * 4,  // down proj
	}

	ptrs := make([]tensor.DevicePtr, 0)
	for layer := 0; layer < 32; layer++ {
		for _, sz := range sizes {
			ptr := sa.Alloc(sz)
			if ptr.IsNil() {
				t.Fatalf("layer %d: allocation of %d bytes failed (used: %d / %d)",
					layer, sz, sa.Used(), sa.Capacity())
			}
			ptrs = append(ptrs, ptr)
		}
	}

	// All should share same base buffer
	baseAddr := ptrs[0].Addr()
	for i, ptr := range ptrs {
		if ptr.Addr() != baseAddr {
			t.Errorf("ptr[%d] base addr differs: got %x, want %x", i, ptr.Addr(), baseAddr)
			break
		}
	}

	// Each should have unique offset
	offsets := make(map[int]bool)
	for _, ptr := range ptrs {
		if offsets[ptr.Offset()] {
			t.Errorf("duplicate offset: %d", ptr.Offset())
		}
		offsets[ptr.Offset()] = true
	}

	t.Logf("Allocated %d buffers from %d bytes scratch (%.1f%% utilized)",
		len(ptrs), sa.Capacity(), float64(sa.Used())/float64(sa.Capacity())*100)
}

func TestScratchAllocatorVsPoolBenchmark(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	p := b.DispatchProfiler()
	p.Enable()

	sizes := []int{4096 * 4, 4096 * 4, 11008 * 4, 11008 * 4, 4096 * 4}
	numLayers := 32
	numAllocs := numLayers * len(sizes)

	// Benchmark pool-based allocation (second pass = all pool hits)
	p.BeginPass()
	for layer := 0; layer < numLayers; layer++ {
		for _, sz := range sizes {
			_ = b.Alloc(sz)
		}
	}
	_ = p.EndPass()
	b.ResetPool()

	p.BeginPass()
	for layer := 0; layer < numLayers; layer++ {
		for _, sz := range sizes {
			_ = b.Alloc(sz)
		}
	}
	poolProfile := p.EndPass()
	b.ResetPool()

	// Benchmark scratch-based allocation
	sa, err := NewScratchAllocator(b, 8*1024*1024)
	if err != nil {
		t.Fatalf("NewScratchAllocator: %v", err)
	}

	p.BeginPass()
	for layer := 0; layer < numLayers; layer++ {
		for _, sz := range sizes {
			_ = sa.Alloc(sz)
		}
	}
	_ = p.EndPass() // scratch allocs don't go through profiler

	t.Logf("Pool-based: %d allocs, %v overhead (%.2f µs/alloc), %d pool hits",
		poolProfile.AllocCount,
		poolProfile.AllocTime,
		float64(poolProfile.AllocTime.Microseconds())/float64(max(poolProfile.AllocCount, 1)),
		poolProfile.AllocPoolHits)
	t.Logf("Scratch-based: %d allocs from single buffer (no profiler overhead)",
		numAllocs)
	t.Logf("Scratch utilization: %d / %d bytes (%.1f%%)",
		sa.Used(), sa.Capacity(), float64(sa.Used())/float64(sa.Capacity())*100)
}

func TestScratchAllocatorAddKernel(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	// Initialize scratch
	if err := b.InitScratch(64 * 1024); err != nil {
		t.Fatalf("InitScratch: %v", err)
	}

	n := 8 // 8 floats
	sz := n * 4

	// Allocate from scratch (all share same base MTLBuffer)
	aPtr := b.ScratchAlloc(sz)
	bPtr := b.ScratchAlloc(sz)
	outPtr := b.ScratchAlloc(sz)

	if aPtr.IsNil() || bPtr.IsNil() || outPtr.IsNil() {
		t.Fatal("scratch allocation returned nil")
	}
	if aPtr.Addr() != bPtr.Addr() || bPtr.Addr() != outPtr.Addr() {
		t.Fatal("scratch allocations should share the same base buffer")
	}

	// Write test data: a = [1,2,3,...,8], b = [10,20,30,...,80]
	sa := b.ScratchAllocator()
	aData := make([]byte, sz)
	bData := make([]byte, sz)
	for i := 0; i < n; i++ {
		aVal := float32(i + 1)
		bVal := float32((i + 1) * 10)
		putFloat32LE(aData[i*4:], aVal)
		putFloat32LE(bData[i*4:], bVal)
	}
	sa.WriteAt(aPtr, aData)
	sa.WriteAt(bPtr, bData)

	// Run offset-aware Add kernel: out = a + b
	b.AddOffset(aPtr, bPtr, outPtr, n)
	b.Sync()

	// Read back result
	outData := sa.ReadAt(outPtr, sz)
	for i := 0; i < n; i++ {
		got := getFloat32LE(outData[i*4:])
		want := float32(i+1) + float32((i+1)*10)
		if got != want {
			t.Errorf("out[%d]: got %f, want %f", i, got, want)
		}
	}
}

func TestScratchAllocatorRMSNormKernel(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	if err := b.InitScratch(64 * 1024); err != nil {
		t.Fatalf("InitScratch: %v", err)
	}

	dim := 4
	eps := float32(1e-5)

	// Allocate from scratch
	xPtr := b.ScratchAlloc(dim * 4)
	outPtr := b.ScratchAlloc(dim * 4)
	// Weight uses regular alloc (simulating permanent weight buffer)
	wPtr := b.Alloc(dim * 4)

	sa := b.ScratchAllocator()

	// x = [1, 2, 3, 4]
	xData := make([]byte, dim*4)
	for i := 0; i < dim; i++ {
		putFloat32LE(xData[i*4:], float32(i+1))
	}
	sa.WriteAt(xPtr, xData)

	// weight = [1, 1, 1, 1]
	wData := make([]byte, dim*4)
	for i := 0; i < dim; i++ {
		putFloat32LE(wData[i*4:], 1.0)
	}
	b.ToDevice(wPtr, wData)

	// Run offset-aware RMSNorm
	b.RMSNormOffset(xPtr, wPtr, outPtr, 1, dim, eps)
	b.Sync()

	// Read result
	outData := sa.ReadAt(outPtr, dim*4)

	// RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
	// out[i] = x[i] / rms * weight[i]
	rms := float32(2.7386127)
	for i := 0; i < dim; i++ {
		got := getFloat32LE(outData[i*4:])
		want := float32(i+1) / rms
		diff := got - want
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.01 {
			t.Errorf("out[%d]: got %f, want ~%f (diff=%f)", i, got, want, diff)
		}
	}
}

func TestBackendInitScratch(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	if b.ScratchAllocator() != nil {
		t.Error("scratch should be nil before InitScratch")
	}

	if err := b.InitScratch(1024 * 1024); err != nil {
		t.Fatalf("InitScratch: %v", err)
	}

	sa := b.ScratchAllocator()
	if sa == nil {
		t.Fatal("scratch should be non-nil after InitScratch")
	}
	if sa.Capacity() != 1024*1024 {
		t.Errorf("capacity: got %d, want %d", sa.Capacity(), 1024*1024)
	}
}

func TestScratchAllocFallback(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	// ScratchAlloc without InitScratch should fall back to pool
	ptr := b.ScratchAlloc(4096)
	if ptr.IsNil() {
		t.Fatal("ScratchAlloc fallback returned nil")
	}
	// Pool-allocated buffers have offset 0
	if ptr.Offset() != 0 {
		t.Errorf("pool-allocated buffer should have offset 0, got %d", ptr.Offset())
	}
}

// TestScratchAllocatorMatMulQ4_0 tests that MatMulQ4_0 works correctly when
// the input (A) and output (C) are scratch-sub-allocated DevicePtrs with
// non-zero offsets. Weights (B) are pool-allocated (offset=0) since they
// are not per-layer temporaries.
//
// This is the key test for P1 (GPU scratch sub-allocation):
// If this passes, the offset-aware MatMulQ4_0 dispatch is working.
func TestScratchAllocatorMatMulQ4_0(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	// Initialize scratch with enough capacity
	if err := b.InitScratch(256 * 1024); err != nil {
		t.Fatalf("InitScratch: %v", err)
	}

	// Small matmul: A=[1, K=64], B=[N=32, K=64] Q4_0, C=[1, N=32]
	m, n, k := 1, 32, 64

	// Create Q4_0 weight data (B): N rows of K elements each
	numBlocksPerRow := k / Q4BlockSize // 64/32 = 2 blocks per row
	bytesPerRow := numBlocksPerRow * Q4BytesPerBlock
	bData := make([]byte, n*bytesPerRow)
	for row := 0; row < n; row++ {
		for blk := 0; blk < numBlocksPerRow; blk++ {
			// Use scale = 0.1 and values centered at 8 (neutral for Q4)
			values := make([]int, 32)
			for i := range values {
				values[i] = 8 + (row%3 - 1) // values in {7, 8, 9}
			}
			block := createQ4_0Block(0.1, values)
			copy(bData[(row*numBlocksPerRow+blk)*Q4BytesPerBlock:], block)
		}
	}

	// Allocate B (weights) from pool (not scratch) — weights are persistent
	bBufSize := len(bData)
	bPtr := b.Alloc(bBufSize)
	if bPtr.IsNil() {
		t.Fatal("failed to allocate B buffer")
	}
	b.ToDevice(bPtr, bData)

	// Create A data: [1, K=64] float32 = all 1.0
	aDataF32 := make([]float32, k)
	for i := range aDataF32 {
		aDataF32[i] = 1.0
	}
	aBytes := make([]byte, k*4)
	for i, v := range aDataF32 {
		putFloat32LE(aBytes[i*4:], v)
	}

	// --- Test 1: Pool-allocated A and C (baseline, offset=0) ---
	aPoolPtr := b.Alloc(k * 4)
	cPoolPtr := b.Alloc(n * 4)
	b.ToDevice(aPoolPtr, aBytes)

	b.MatMulQ4_0(aPoolPtr, bPtr, cPoolPtr, m, n, k)
	b.Sync()

	cPoolData := make([]byte, n*4)
	b.ToHost(cPoolData, cPoolPtr)
	poolResults := make([]float32, n)
	for i := range poolResults {
		poolResults[i] = getFloat32LE(cPoolData[i*4:])
	}

	// --- Test 2: Scratch-allocated A and C (non-zero offsets) ---
	sa := b.ScratchAllocator()
	sa.Reset()

	// Add a dummy allocation first to ensure non-zero offsets
	_ = b.ScratchAlloc(1024) // Dummy: pushes offset to 1024

	aScratchPtr := b.ScratchAlloc(k * 4)
	cScratchPtr := b.ScratchAlloc(n * 4)
	if aScratchPtr.IsNil() || cScratchPtr.IsNil() {
		t.Fatal("scratch allocation returned nil")
	}

	// Verify non-zero offsets (the whole point of this test)
	if aScratchPtr.Offset() == 0 {
		t.Error("expected non-zero offset for A, got 0")
	}
	if cScratchPtr.Offset() == 0 {
		t.Error("expected non-zero offset for C, got 0")
	}
	t.Logf("A scratch offset: %d, C scratch offset: %d", aScratchPtr.Offset(), cScratchPtr.Offset())

	// Write A data to scratch
	sa.WriteAt(aScratchPtr, aBytes)

	// Run offset-aware MatMulQ4_0 — this is what we're testing
	b.MatMulQ4_0Offset(aScratchPtr, bPtr, cScratchPtr, m, n, k)
	b.Sync()

	// Read back results
	cScratchData := sa.ReadAt(cScratchPtr, n*4)
	scratchResults := make([]float32, n)
	for i := range scratchResults {
		scratchResults[i] = getFloat32LE(cScratchData[i*4:])
	}

	// --- Compare ---
	t.Logf("Pool result[0..3]: %v", poolResults[:4])
	t.Logf("Scratch result[0..3]: %v", scratchResults[:4])

	for i := 0; i < n; i++ {
		diff := math.Abs(float64(poolResults[i] - scratchResults[i]))
		if diff > 0.001 {
			t.Errorf("output[%d]: pool=%f, scratch=%f, diff=%f", i, poolResults[i], scratchResults[i], diff)
		}
	}
}

// TestScratchAllocatorRoPE tests that RoPE works correctly when Q and K
// are scratch-sub-allocated with non-zero offsets.
func TestScratchAllocatorRoPE(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	if err := b.InitScratch(256 * 1024); err != nil {
		t.Fatalf("InitScratch: %v", err)
	}

	// Q=[1, numHeads=4, headDim=8], K=[1, numKVHeads=4, headDim=8]
	seqLen := 1
	numHeads := 4
	numKVHeads := 4
	headDim := 8
	startPos := 5
	theta := float32(10000.0)

	qSize := seqLen * numHeads * headDim
	kSize := seqLen * numKVHeads * headDim

	// Create test data
	qData := make([]byte, qSize*4)
	kData := make([]byte, kSize*4)
	for i := 0; i < qSize; i++ {
		putFloat32LE(qData[i*4:], float32(i)*0.01)
	}
	for i := 0; i < kSize; i++ {
		putFloat32LE(kData[i*4:], float32(i)*0.01)
	}

	// --- Pool-allocated (baseline) ---
	qPoolPtr := b.Alloc(qSize * 4)
	kPoolPtr := b.Alloc(kSize * 4)
	b.ToDevice(qPoolPtr, qData)
	b.ToDevice(kPoolPtr, kData)

	b.RoPE(qPoolPtr, kPoolPtr, headDim, numHeads, numKVHeads, seqLen, startPos, 0, theta, false)
	b.Sync()

	qPoolOut := make([]byte, qSize*4)
	kPoolOut := make([]byte, kSize*4)
	b.ToHost(qPoolOut, qPoolPtr)
	b.ToHost(kPoolOut, kPoolPtr)

	// --- Scratch-allocated ---
	sa := b.ScratchAllocator()
	sa.Reset()
	_ = b.ScratchAlloc(512) // Dummy for non-zero offset

	qScratchPtr := b.ScratchAlloc(qSize * 4)
	kScratchPtr := b.ScratchAlloc(kSize * 4)
	if qScratchPtr.Offset() == 0 || kScratchPtr.Offset() == 0 {
		t.Error("expected non-zero offsets for scratch Q/K")
	}

	sa.WriteAt(qScratchPtr, qData)
	sa.WriteAt(kScratchPtr, kData)

	b.RoPEOffset(qScratchPtr, kScratchPtr, headDim, numHeads, numKVHeads, seqLen, startPos, 0, theta, false)
	b.Sync()

	qScratchOut := sa.ReadAt(qScratchPtr, qSize*4)
	kScratchOut := sa.ReadAt(kScratchPtr, kSize*4)

	// Compare
	for i := 0; i < qSize; i++ {
		pool := getFloat32LE(qPoolOut[i*4:])
		scratch := getFloat32LE(qScratchOut[i*4:])
		diff := math.Abs(float64(pool - scratch))
		if diff > 0.001 {
			t.Errorf("Q[%d]: pool=%f, scratch=%f", i, pool, scratch)
		}
	}
	for i := 0; i < kSize; i++ {
		pool := getFloat32LE(kPoolOut[i*4:])
		scratch := getFloat32LE(kScratchOut[i*4:])
		diff := math.Abs(float64(pool - scratch))
		if diff > 0.001 {
			t.Errorf("K[%d]: pool=%f, scratch=%f", i, pool, scratch)
		}
	}
}

// TestScratchAllocatorSDPA tests SDPA decode with scratch-allocated Q and output.
func TestScratchAllocatorSDPA(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	if err := b.InitScratch(256 * 1024); err != nil {
		t.Fatalf("InitScratch: %v", err)
	}

	numQHeads := 4
	numKVHeads := 4
	headDim := 8
	kvLen := 3
	maxSeqLen := 16
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvHeadStride := maxSeqLen * headDim

	qSize := numQHeads * headDim
	kvTotalSize := numKVHeads * maxSeqLen * headDim

	// Create Q data
	qData := make([]byte, qSize*4)
	for i := 0; i < qSize; i++ {
		putFloat32LE(qData[i*4:], float32(i)*0.01+0.1)
	}

	// Create KV cache data (head-major layout)
	kData := make([]byte, kvTotalSize*4)
	vData := make([]byte, kvTotalSize*4)
	for h := 0; h < numKVHeads; h++ {
		for p := 0; p < kvLen; p++ {
			for d := 0; d < headDim; d++ {
				idx := h*kvHeadStride + p*headDim + d
				putFloat32LE(kData[idx*4:], float32(h*100+p*10+d)*0.01)
				putFloat32LE(vData[idx*4:], float32(h*100+p*10+d)*0.02)
			}
		}
	}

	// Pool-allocated KV cache (always pool-allocated, not scratch)
	kPtr := b.Alloc(kvTotalSize * 4)
	vPtr := b.Alloc(kvTotalSize * 4)
	b.ToDevice(kPtr, kData)
	b.ToDevice(vPtr, vData)

	// --- Pool-allocated Q and output (baseline) ---
	qPoolPtr := b.Alloc(qSize * 4)
	outPoolPtr := b.Alloc(qSize * 4)
	b.ToDevice(qPoolPtr, qData)

	b.SDPA(qPoolPtr, kPtr, vPtr, outPoolPtr, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
	b.Sync()

	outPoolData := make([]byte, qSize*4)
	b.ToHost(outPoolData, outPoolPtr)

	// --- Scratch-allocated Q and output ---
	sa := b.ScratchAllocator()
	sa.Reset()
	_ = b.ScratchAlloc(512) // Dummy for non-zero offset

	qScratchPtr := b.ScratchAlloc(qSize * 4)
	outScratchPtr := b.ScratchAlloc(qSize * 4)

	sa.WriteAt(qScratchPtr, qData)

	b.SDPAOffset(qScratchPtr, kPtr, vPtr, outScratchPtr, kvLen, numQHeads, numKVHeads, headDim, scale, kvHeadStride)
	b.Sync()

	outScratchData := sa.ReadAt(outScratchPtr, qSize*4)

	// Compare
	for i := 0; i < qSize; i++ {
		pool := getFloat32LE(outPoolData[i*4:])
		scratch := getFloat32LE(outScratchData[i*4:])
		diff := math.Abs(float64(pool - scratch))
		if diff > 0.01 {
			t.Errorf("SDPA out[%d]: pool=%f, scratch=%f, diff=%f", i, pool, scratch, diff)
		}
	}
}

// putFloat32LE writes a float32 as little-endian bytes.
func putFloat32LE(buf []byte, v float32) {
	binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
}

// getFloat32LE reads a float32 from little-endian bytes.
func getFloat32LE(buf []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(buf))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
