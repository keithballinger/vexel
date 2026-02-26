//go:build metal && darwin && cgo

package metal

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestDispatchProfilerDisabledByDefault(t *testing.T) {
	p := NewDispatchProfiler()
	if p.IsEnabled() {
		t.Error("profiler should be disabled by default")
	}

	// Operations should be no-ops when disabled
	p.RecordDispatch("MatMul")
	p.RecordAlloc(1024, false, time.Microsecond)
	profile := p.EndPass()
	if profile.TotalDispatches != 0 {
		t.Errorf("got %d dispatches, want 0 (profiler disabled)", profile.TotalDispatches)
	}
}

func TestDispatchProfilerCounting(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()

	p.BeginPass()
	p.RecordDispatch("MatMulQ4_0")
	p.RecordDispatch("MatMulQ4_0")
	p.RecordDispatch("MatMulQ4_0")
	p.RecordDispatch("RMSNorm")
	p.RecordDispatch("RoPE")
	p.RecordDispatch("SDPA")

	profile := p.EndPass()

	if profile.TotalDispatches != 6 {
		t.Errorf("total dispatches: got %d, want 6", profile.TotalDispatches)
	}
	if profile.OpCounts["MatMulQ4_0"] != 3 {
		t.Errorf("MatMulQ4_0 count: got %d, want 3", profile.OpCounts["MatMulQ4_0"])
	}
	if profile.OpCounts["RMSNorm"] != 1 {
		t.Errorf("RMSNorm count: got %d, want 1", profile.OpCounts["RMSNorm"])
	}
	if profile.OpCounts["SDPA"] != 1 {
		t.Errorf("SDPA count: got %d, want 1", profile.OpCounts["SDPA"])
	}
}

func TestDispatchProfilerSequence(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()

	p.BeginPass()
	ops := []string{"RMSNorm", "MatMulQ4_0", "MatMulQ4_0", "MatMulQ4_0", "RoPE", "ScatterKV", "SDPA"}
	for _, op := range ops {
		p.RecordDispatch(op)
	}
	profile := p.EndPass()

	if len(profile.OpSequence) != len(ops) {
		t.Fatalf("sequence length: got %d, want %d", len(profile.OpSequence), len(ops))
	}
	for i, op := range ops {
		if profile.OpSequence[i] != op {
			t.Errorf("sequence[%d]: got %q, want %q", i, profile.OpSequence[i], op)
		}
	}
}

func TestDispatchProfilerAllocTracking(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()

	p.BeginPass()
	p.RecordAlloc(4096, false, 10*time.Microsecond)   // new allocation
	p.RecordAlloc(4096, true, time.Microsecond)         // pool hit
	p.RecordAlloc(8192, true, time.Microsecond)         // pool hit
	p.RecordAlloc(16384, false, 15*time.Microsecond)   // new allocation

	profile := p.EndPass()

	if profile.AllocCount != 4 {
		t.Errorf("alloc count: got %d, want 4", profile.AllocCount)
	}
	if profile.AllocBytes != 4096+4096+8192+16384 {
		t.Errorf("alloc bytes: got %d, want %d", profile.AllocBytes, 4096+4096+8192+16384)
	}
	if profile.AllocPoolHits != 2 {
		t.Errorf("pool hits: got %d, want 2", profile.AllocPoolHits)
	}
	if profile.AllocTime != 27*time.Microsecond {
		t.Errorf("alloc time: got %v, want %v", profile.AllocTime, 27*time.Microsecond)
	}
}

func TestDispatchProfilerReset(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()

	p.BeginPass()
	p.RecordDispatch("MatMul")
	p.RecordDispatch("MatMul")
	p.RecordAlloc(1024, false, time.Microsecond)

	// Reset should clear everything
	p.Reset()

	profile := p.EndPass()
	if profile.TotalDispatches != 0 {
		t.Errorf("dispatches after reset: got %d, want 0", profile.TotalDispatches)
	}
	if profile.AllocCount != 0 {
		t.Errorf("allocs after reset: got %d, want 0", profile.AllocCount)
	}
}

func TestDispatchProfilerBeginPassResets(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()

	// First pass
	p.BeginPass()
	p.RecordDispatch("MatMul")
	p.RecordDispatch("MatMul")
	_ = p.EndPass()

	// Second pass should start fresh
	p.BeginPass()
	p.RecordDispatch("RMSNorm")
	profile := p.EndPass()

	if profile.TotalDispatches != 1 {
		t.Errorf("second pass dispatches: got %d, want 1", profile.TotalDispatches)
	}
	if profile.OpCounts["MatMul"] != 0 {
		t.Errorf("MatMul count in second pass: got %d, want 0", profile.OpCounts["MatMul"])
	}
	if profile.OpCounts["RMSNorm"] != 1 {
		t.Errorf("RMSNorm count in second pass: got %d, want 1", profile.OpCounts["RMSNorm"])
	}
}

func TestDispatchProfilerPassDuration(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()

	p.BeginPass()
	time.Sleep(5 * time.Millisecond) // Small sleep to ensure measurable duration
	p.RecordDispatch("MatMul")
	profile := p.EndPass()

	if profile.PassDuration < 5*time.Millisecond {
		t.Errorf("pass duration too short: got %v, want >= 5ms", profile.PassDuration)
	}
}

func TestForwardPassProfilePrintReport(t *testing.T) {
	fp := ForwardPassProfile{
		TotalDispatches: 42,
		OpCounts: map[string]int{
			"MatMulQ4_0": 28,
			"RMSNorm":    4,
			"RoPE":       2,
			"SDPA":       2,
			"SiLUMul":    2,
			"Add":        4,
		},
		OpSequence:    []string{"RMSNorm", "MatMulQ4_0"},
		AllocCount:    10,
		AllocBytes:    1048576,
		AllocPoolHits: 7,
		AllocTime:     100 * time.Microsecond,
		PassDuration:  5 * time.Millisecond,
	}

	var buf bytes.Buffer
	fp.PrintReport(&buf)
	output := buf.String()

	// Verify key sections are present
	expectations := []string{
		"DISPATCH PROFILE",
		"Total kernel dispatches: 42",
		"MatMulQ4_0",
		"RMSNorm",
		"Allocation Statistics",
		"Total allocations:",
		"Pool hit rate:",
	}
	for _, exp := range expectations {
		if !strings.Contains(output, exp) {
			t.Errorf("report missing %q\nGot:\n%s", exp, output)
		}
	}
}

func TestBackendDispatchProfiler(t *testing.T) {
	// Verify the Backend has a dispatch profiler accessible
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	p := b.DispatchProfiler()
	if p == nil {
		t.Fatal("Backend.DispatchProfiler() returned nil")
	}
}

func TestBackendAllocWithProfiler(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	p := b.DispatchProfiler()
	p.Enable()
	p.BeginPass()

	// Allocate some buffers
	ptr1 := b.Alloc(1024)
	ptr2 := b.Alloc(2048)
	ptr3 := b.Alloc(4096)

	profile := p.EndPass()

	if profile.AllocCount != 3 {
		t.Errorf("alloc count: got %d, want 3", profile.AllocCount)
	}
	expectedBytes := int64(1024 + 2048 + 4096)
	if profile.AllocBytes != expectedBytes {
		t.Errorf("alloc bytes: got %d, want %d", profile.AllocBytes, expectedBytes)
	}

	// Clean up
	_ = ptr1
	_ = ptr2
	_ = ptr3
}

func TestBackendPoolReuseTracking(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	p := b.DispatchProfiler()
	p.Enable()

	// First pass: allocate fresh buffers (all misses)
	p.BeginPass()
	_ = b.Alloc(1024)
	_ = b.Alloc(2048)
	_ = b.Alloc(1024) // same size, but first pass so still a new alloc

	pass1 := p.EndPass()
	if pass1.AllocPoolHits != 0 {
		t.Errorf("pass 1 pool hits: got %d, want 0 (all fresh)", pass1.AllocPoolHits)
	}
	if pass1.AllocCount != 3 {
		t.Errorf("pass 1 alloc count: got %d, want 3", pass1.AllocCount)
	}

	// Reset pool — returns all in-use buffers to available pool
	b.ResetPool()

	// Second pass: same sizes should get pool hits
	p.BeginPass()
	_ = b.Alloc(1024) // pool hit (exact size match)
	_ = b.Alloc(2048) // pool hit
	_ = b.Alloc(1024) // pool hit (we had two 1024-byte buffers)
	_ = b.Alloc(4096) // pool miss (new size)

	pass2 := p.EndPass()
	if pass2.AllocPoolHits != 3 {
		t.Errorf("pass 2 pool hits: got %d, want 3", pass2.AllocPoolHits)
	}
	if pass2.AllocCount != 4 {
		t.Errorf("pass 2 alloc count: got %d, want 4", pass2.AllocCount)
	}
}

func TestBackendAllocOverheadMeasurement(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	p := b.DispatchProfiler()
	p.Enable()

	// Simulate a realistic forward pass allocation pattern:
	// 32 layers × ~5 intermediate buffers per layer + embedding/logits
	sizes := []int{
		4096 * 4,   // hidden state (4096 float32)
		4096 * 4,   // Q projection output
		4096 * 4,   // K projection output
		4096 * 4,   // V projection output
		4096 * 4,   // attention output
		11008 * 4,  // gate projection (MLP intermediate)
		11008 * 4,  // up projection
		11008 * 4,  // SiLU×Mul output
		4096 * 4,   // down projection output
	}

	// First pass: all fresh allocations
	p.BeginPass()
	for layer := 0; layer < 32; layer++ {
		for _, sz := range sizes {
			_ = b.Alloc(sz)
		}
	}
	fresh := p.EndPass()

	// Reset pool
	b.ResetPool()

	// Second pass: should get pool hits for all same-size allocations
	p.BeginPass()
	for layer := 0; layer < 32; layer++ {
		for _, sz := range sizes {
			_ = b.Alloc(sz)
		}
	}
	reused := p.EndPass()

	totalAllocs := 32 * len(sizes)
	if fresh.AllocCount != totalAllocs {
		t.Errorf("fresh alloc count: got %d, want %d", fresh.AllocCount, totalAllocs)
	}
	if reused.AllocCount != totalAllocs {
		t.Errorf("reused alloc count: got %d, want %d", reused.AllocCount, totalAllocs)
	}

	// All allocations in second pass should be pool hits
	if reused.AllocPoolHits != totalAllocs {
		t.Errorf("reused pool hits: got %d, want %d", reused.AllocPoolHits, totalAllocs)
	}

	// Pool reuse should be faster than fresh allocation
	if fresh.AllocTime > 0 && reused.AllocTime > 0 {
		t.Logf("Fresh allocation overhead:  %v (%d allocs, %.2f µs/alloc)",
			fresh.AllocTime, fresh.AllocCount,
			float64(fresh.AllocTime.Microseconds())/float64(fresh.AllocCount))
		t.Logf("Pooled allocation overhead: %v (%d allocs, %.2f µs/alloc)",
			reused.AllocTime, reused.AllocCount,
			float64(reused.AllocTime.Microseconds())/float64(reused.AllocCount))
		t.Logf("Pool hit rate: %.1f%%",
			float64(reused.AllocPoolHits)/float64(reused.AllocCount)*100)
		t.Logf("Total bytes per pass: %.2f MB", float64(reused.AllocBytes)/1024/1024)
	}
}

func TestBackendKernelDispatchCounting(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Close()

	p := b.DispatchProfiler()
	p.Enable()
	p.BeginPass()

	// Allocate test buffers
	n := 256
	aPtr := b.Alloc(n * 4) // [1, 256] F32
	bPtr := b.Alloc(n * 4) // [256, 1] F32
	cPtr := b.Alloc(1 * 4) // [1, 1] F32

	// Dispatch a matmul
	b.MatMul(aPtr, bPtr, cPtr, 1, 1, n)
	b.Sync()

	profile := p.EndPass()

	if profile.TotalDispatches < 1 {
		t.Errorf("expected at least 1 dispatch from MatMul, got %d", profile.TotalDispatches)
	}
	if profile.OpCounts["MatMul"] != 1 {
		t.Errorf("MatMul dispatch count: got %d, want 1", profile.OpCounts["MatMul"])
	}
}
