package runtime

import (
	"fmt"
	"os"
	"sort"
	"sync"
	"time"
)

// ProfileStats tracks cumulative timing for different operation categories.
type ProfileStats struct {
	mu       sync.Mutex
	counts   map[string]int
	times    map[string]time.Duration
	enabled  bool
	layerIdx int // Track which layer we're in for per-layer breakdown
}

// Global profiler instance (controlled by DEBUG_PROFILE env var)
var profiler = &ProfileStats{
	counts:  make(map[string]int),
	times:   make(map[string]time.Duration),
	enabled: os.Getenv("DEBUG_PROFILE") == "1",
}

func init() {
	if profiler.enabled {
		fmt.Println("[PROFILE] Profiling enabled (DEBUG_PROFILE=1)")
	}
}

// EnableProfiling turns on operation timing.
func EnableProfiling() {
	profiler.mu.Lock()
	defer profiler.mu.Unlock()
	profiler.enabled = true
	profiler.counts = make(map[string]int)
	profiler.times = make(map[string]time.Duration)
}

// DisableProfiling turns off operation timing.
func DisableProfiling() {
	profiler.mu.Lock()
	defer profiler.mu.Unlock()
	profiler.enabled = false
}

// ResetProfile clears all accumulated stats.
func ResetProfile() {
	profiler.mu.Lock()
	defer profiler.mu.Unlock()
	profiler.counts = make(map[string]int)
	profiler.times = make(map[string]time.Duration)
}

// RecordOp records timing for a named operation.
func RecordOp(name string, d time.Duration) {
	if !profiler.enabled {
		return
	}
	profiler.mu.Lock()
	defer profiler.mu.Unlock()
	profiler.counts[name]++
	profiler.times[name] += d
}

// TimeOp is a helper that returns a function to call when the operation is done.
// Usage: defer TimeOp("matmul")()
func TimeOp(name string) func() {
	if !profiler.enabled {
		return func() {}
	}
	start := time.Now()
	return func() {
		RecordOp(name, time.Since(start))
	}
}

// ProfileEntry represents timing data for a single operation.
type ProfileEntry struct {
	Name  string
	Total time.Duration
	Count int
	AvgUs float64
}

// GetProfileData returns a copy of the current profiling data.
func GetProfileData() []ProfileEntry {
	profiler.mu.Lock()
	defer profiler.mu.Unlock()
	var entries []ProfileEntry
	for name, t := range profiler.times {
		entries = append(entries, ProfileEntry{
			Name:  name,
			Total: t,
			Count: profiler.counts[name],
			AvgUs: float64(t.Microseconds()) / float64(profiler.counts[name]),
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Total > entries[j].Total
	})
	return entries
}

// PrintProfile prints a summary of operation timings.
func PrintProfile() {
	profiler.mu.Lock()
	defer profiler.mu.Unlock()

	if len(profiler.times) == 0 {
		fmt.Println("[PROFILE] No data collected")
		return
	}

	// Sort by total time (descending)
	type entry struct {
		name  string
		time  time.Duration
		count int
	}
	var entries []entry
	var total time.Duration
	for name, t := range profiler.times {
		entries = append(entries, entry{name, t, profiler.counts[name]})
		total += t
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].time > entries[j].time
	})

	fmt.Println("\n[PROFILE] Operation Timing Summary:")
	fmt.Println("=====================================")
	for _, e := range entries {
		pct := float64(e.time) / float64(total) * 100
		avgUs := float64(e.time.Microseconds()) / float64(e.count)
		fmt.Printf("  %-20s %8.1f ms (%5.1f%%) | %6d calls | %.1f us/call\n",
			e.name, float64(e.time.Milliseconds()), pct, e.count, avgUs)
	}
	fmt.Printf("=====================================\n")
	fmt.Printf("  %-20s %8.1f ms\n", "TOTAL", float64(total.Milliseconds()))
}
