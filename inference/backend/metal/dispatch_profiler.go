//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// DispatchProfiler tracks Metal kernel dispatch counts and allocation overhead
// per forward pass. Enable with VEXEL_DISPATCH_PROFILE=1 or by calling Enable().
type DispatchProfiler struct {
	mu      sync.Mutex
	enabled atomic.Bool

	// Per-operation dispatch counts (e.g., "MatMulQ4_0" -> 32)
	opCounts map[string]int

	// Ordered operation log for sequencing analysis
	opLog []dispatchEntry

	// Allocation tracking
	allocCount    int
	allocBytes    int64
	allocPoolHits int
	allocTime     time.Duration

	// Forward pass timing
	passStart time.Time
}

// dispatchEntry records a single kernel dispatch for sequencing.
type dispatchEntry struct {
	Op   string
	Time time.Time
}

// ForwardPassProfile is the result of profiling a single forward pass.
type ForwardPassProfile struct {
	// Kernel dispatch statistics
	TotalDispatches int
	OpCounts        map[string]int
	OpSequence      []string // Ordered list of dispatched operations

	// Allocation statistics
	AllocCount    int
	AllocBytes    int64
	AllocPoolHits int
	AllocTime     time.Duration

	// Overall timing
	PassDuration time.Duration
}

// NewDispatchProfiler creates a new profiler (disabled by default).
func NewDispatchProfiler() *DispatchProfiler {
	return &DispatchProfiler{
		opCounts: make(map[string]int),
	}
}

// Enable turns on dispatch profiling.
func (p *DispatchProfiler) Enable() {
	p.enabled.Store(true)
	p.Reset()
}

// Disable turns off dispatch profiling.
func (p *DispatchProfiler) Disable() {
	p.enabled.Store(false)
}

// IsEnabled returns whether profiling is active.
func (p *DispatchProfiler) IsEnabled() bool {
	return p.enabled.Load()
}

// Reset clears all accumulated profiling data.
func (p *DispatchProfiler) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.opCounts = make(map[string]int)
	p.opLog = p.opLog[:0]
	p.allocCount = 0
	p.allocBytes = 0
	p.allocPoolHits = 0
	p.allocTime = 0
	p.passStart = time.Time{}
}

// BeginPass marks the start of a forward pass.
func (p *DispatchProfiler) BeginPass() {
	if !p.enabled.Load() {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.opCounts = make(map[string]int)
	p.opLog = p.opLog[:0]
	p.allocCount = 0
	p.allocBytes = 0
	p.allocPoolHits = 0
	p.allocTime = 0
	p.passStart = time.Now()
}

// RecordDispatch records a kernel dispatch for the named operation.
func (p *DispatchProfiler) RecordDispatch(op string) {
	if !p.enabled.Load() {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.opCounts[op]++
	p.opLog = append(p.opLog, dispatchEntry{Op: op, Time: time.Now()})
}

// RecordAlloc records a buffer allocation.
func (p *DispatchProfiler) RecordAlloc(bytes int, poolHit bool, duration time.Duration) {
	if !p.enabled.Load() {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.allocCount++
	p.allocBytes += int64(bytes)
	if poolHit {
		p.allocPoolHits++
	}
	p.allocTime += duration
}

// EndPass finalizes the forward pass and returns the profile.
func (p *DispatchProfiler) EndPass() ForwardPassProfile {
	if !p.enabled.Load() {
		return ForwardPassProfile{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	total := 0
	opCounts := make(map[string]int, len(p.opCounts))
	for k, v := range p.opCounts {
		opCounts[k] = v
		total += v
	}

	seq := make([]string, len(p.opLog))
	for i, e := range p.opLog {
		seq[i] = e.Op
	}

	var dur time.Duration
	if !p.passStart.IsZero() {
		dur = time.Since(p.passStart)
	}

	return ForwardPassProfile{
		TotalDispatches: total,
		OpCounts:        opCounts,
		OpSequence:      seq,
		AllocCount:      p.allocCount,
		AllocBytes:      p.allocBytes,
		AllocPoolHits:   p.allocPoolHits,
		AllocTime:       p.allocTime,
		PassDuration:    dur,
	}
}

// PrintReport writes a formatted dispatch profile report to w.
func (fp ForwardPassProfile) PrintReport(w io.Writer) {
	fmt.Fprintf(w, "\n[DISPATCH PROFILE] Forward Pass Summary\n")
	fmt.Fprintf(w, "========================================\n")
	fmt.Fprintf(w, "  Total kernel dispatches: %d\n", fp.TotalDispatches)
	fmt.Fprintf(w, "  Pass duration:           %v\n", fp.PassDuration)
	if fp.TotalDispatches > 0 {
		avgUs := float64(fp.PassDuration.Microseconds()) / float64(fp.TotalDispatches)
		fmt.Fprintf(w, "  Avg dispatch overhead:   %.1f µs/dispatch\n", avgUs)
	}

	// Sort operations by count (descending)
	type entry struct {
		name  string
		count int
	}
	var entries []entry
	for name, count := range fp.OpCounts {
		entries = append(entries, entry{name, count})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].count > entries[j].count
	})

	fmt.Fprintf(w, "\n  Per-Operation Dispatch Counts:\n")
	fmt.Fprintf(w, "  %-30s %6s %6s\n", "Operation", "Count", "%%")
	fmt.Fprintf(w, "  %-30s %6s %6s\n", "─────────", "─────", "──")
	for _, e := range entries {
		pct := float64(e.count) / float64(fp.TotalDispatches) * 100
		fmt.Fprintf(w, "  %-30s %6d %5.1f%%\n", e.name, e.count, pct)
	}

	fmt.Fprintf(w, "\n  Allocation Statistics:\n")
	fmt.Fprintf(w, "  %-30s %d\n", "Total allocations:", fp.AllocCount)
	fmt.Fprintf(w, "  %-30s %.2f MB\n", "Total bytes allocated:", float64(fp.AllocBytes)/1024/1024)
	fmt.Fprintf(w, "  %-30s %d\n", "Pool hits:", fp.AllocPoolHits)
	if fp.AllocCount > 0 {
		hitRate := float64(fp.AllocPoolHits) / float64(fp.AllocCount) * 100
		fmt.Fprintf(w, "  %-30s %.1f%%\n", "Pool hit rate:", hitRate)
	}
	fmt.Fprintf(w, "  %-30s %v\n", "Allocation overhead:", fp.AllocTime)
	fmt.Fprintf(w, "========================================\n")
}
