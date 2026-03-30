// Package debug provides a configurable debug harness for tracing inference.
package debug

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
)

// Config specifies what to debug during inference.
type Config struct {
	// Targeting - what to capture
	Layers    []int    // Empty = all layers, or specific layer indices
	Positions []int    // Empty = all positions, or specific positions
	Ops       []string // Empty = all ops, or specific: "norm", "qkv", "rope", "sdpa", "wo", "mlp", "add"

	// Output control
	OutputPath string // Path to write JSON output (empty = stderr)
	Verbose    bool   // Print human-readable summary in addition to JSON
	MaxValues  int    // Max values to include in tensor dumps (default 16)

	// Thresholds for automatic flagging
	NaNThreshold float32 // Flag if NaN count exceeds this
	InfThreshold float32 // Flag if Inf count exceeds this
	MaxThreshold float32 // Flag if max value exceeds this (0 = disabled)
	MinThreshold float32 // Flag if min value below this (0 = disabled)
}

// TensorSnapshot captures tensor state at a point in inference.
type TensorSnapshot struct {
	Layer    int         `json:"layer"`
	Position int         `json:"position"`
	Op       string      `json:"op"`
	Name     string      `json:"name"`
	Size     int         `json:"size"`
	Stats    TensorStats `json:"stats"`
	Values   []float32   `json:"values,omitempty"` // First N values
	Flags    []string    `json:"flags,omitempty"`  // Any warnings
}

// TensorStats holds summary statistics for a tensor.
type TensorStats struct {
	Min  float32 `json:"min"`
	Max  float32 `json:"max"`
	Mean float32 `json:"mean"`
	NaN  int     `json:"nan"`
	Inf  int     `json:"inf"`
	Zero int     `json:"zero"`
}

// Trace holds all snapshots for a single inference run.
type Trace struct {
	Model     string           `json:"model"`
	Prompt    string           `json:"prompt"`
	Snapshots []TensorSnapshot `json:"snapshots"`
}

// Harness is the main debug controller.
type Harness struct {
	config  Config
	trace   Trace
	mu      sync.Mutex
	enabled bool
	outFile *os.File
}

var (
	globalHarness *Harness
	globalMu      sync.Mutex
)

// Init initializes the global debug harness with the given config.
func Init(cfg Config) error {
	globalMu.Lock()
	defer globalMu.Unlock()

	h := &Harness{
		config:  cfg,
		enabled: true,
	}

	if cfg.MaxValues == 0 {
		h.config.MaxValues = 16
	}

	if cfg.OutputPath != "" && cfg.OutputPath != "-" {
		f, err := os.Create(cfg.OutputPath)
		if err != nil {
			return fmt.Errorf("failed to create output file: %w", err)
		}
		h.outFile = f
	}

	globalHarness = h
	return nil
}

// Close flushes and closes the debug harness.
func Close() error {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalHarness == nil {
		return nil
	}

	h := globalHarness

	// Write final trace
	if h.outFile != nil {
		enc := json.NewEncoder(h.outFile)
		enc.SetIndent("", "  ")
		enc.Encode(h.trace)
		h.outFile.Close()
	}

	globalHarness = nil
	return nil
}

// Enabled returns true if debug harness is active.
func Enabled() bool {
	globalMu.Lock()
	defer globalMu.Unlock()
	return globalHarness != nil && globalHarness.enabled
}

// SetModel sets the model name for the trace.
func SetModel(name string) {
	globalMu.Lock()
	defer globalMu.Unlock()
	if globalHarness != nil {
		globalHarness.trace.Model = name
	}
}

// SetPrompt sets the prompt for the trace.
func SetPrompt(prompt string) {
	globalMu.Lock()
	defer globalMu.Unlock()
	if globalHarness != nil {
		globalHarness.trace.Prompt = prompt
	}
}

// ShouldCapture returns true if we should capture this layer/position/op.
func ShouldCapture(layer, position int, op string) bool {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalHarness == nil || !globalHarness.enabled {
		return false
	}

	cfg := globalHarness.config

	// Check layer filter
	if len(cfg.Layers) > 0 {
		found := false
		for _, l := range cfg.Layers {
			if l == layer {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check position filter
	if len(cfg.Positions) > 0 {
		found := false
		for _, p := range cfg.Positions {
			if p == position {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check op filter
	if len(cfg.Ops) > 0 {
		found := false
		for _, o := range cfg.Ops {
			if o == op {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// Capture records a tensor snapshot.
func Capture(layer, position int, op, name string, data []float32) {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalHarness == nil || !globalHarness.enabled {
		return
	}

	cfg := globalHarness.config

	// Compute stats
	stats := computeStats(data)

	// Get first N values
	var values []float32
	if len(data) > 0 {
		n := cfg.MaxValues
		if n > len(data) {
			n = len(data)
		}
		values = make([]float32, n)
		copy(values, data[:n])
	}

	// Check for flags
	var flags []string
	if stats.NaN > 0 {
		flags = append(flags, fmt.Sprintf("NaN=%d", stats.NaN))
	}
	if stats.Inf > 0 {
		flags = append(flags, fmt.Sprintf("Inf=%d", stats.Inf))
	}
	if cfg.MaxThreshold > 0 && stats.Max > cfg.MaxThreshold {
		flags = append(flags, fmt.Sprintf("max_exceeded=%.2f", stats.Max))
	}
	if cfg.MinThreshold != 0 && stats.Min < cfg.MinThreshold {
		flags = append(flags, fmt.Sprintf("min_exceeded=%.2f", stats.Min))
	}

	snap := TensorSnapshot{
		Layer:    layer,
		Position: position,
		Op:       op,
		Name:     name,
		Size:     len(data),
		Stats:    stats,
		Values:   values,
		Flags:    flags,
	}

	globalHarness.trace.Snapshots = append(globalHarness.trace.Snapshots, snap)

	// Print verbose output
	if cfg.Verbose {
		flagStr := ""
		if len(flags) > 0 {
			flagStr = fmt.Sprintf(" [%v]", flags)
		}
		fmt.Fprintf(os.Stderr, "[DEBUG] L%d pos=%d %s %s: min=%.4f max=%.4f mean=%.4f nan=%d%s\n",
			layer, position, op, name, stats.Min, stats.Max, stats.Mean, stats.NaN, flagStr)
	}
}

// CaptureWithFlag is like Capture but also returns true if any flags were triggered.
func CaptureWithFlag(layer, position int, op, name string, data []float32) bool {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalHarness == nil || !globalHarness.enabled {
		return false
	}

	cfg := globalHarness.config
	stats := computeStats(data)

	var values []float32
	if len(data) > 0 {
		n := cfg.MaxValues
		if n > len(data) {
			n = len(data)
		}
		values = make([]float32, n)
		copy(values, data[:n])
	}

	var flags []string
	if stats.NaN > 0 {
		flags = append(flags, fmt.Sprintf("NaN=%d", stats.NaN))
	}
	if stats.Inf > 0 {
		flags = append(flags, fmt.Sprintf("Inf=%d", stats.Inf))
	}
	if cfg.MaxThreshold > 0 && stats.Max > cfg.MaxThreshold {
		flags = append(flags, fmt.Sprintf("max_exceeded=%.2f", stats.Max))
	}
	if cfg.MinThreshold != 0 && stats.Min < cfg.MinThreshold {
		flags = append(flags, fmt.Sprintf("min_exceeded=%.2f", stats.Min))
	}

	snap := TensorSnapshot{
		Layer:    layer,
		Position: position,
		Op:       op,
		Name:     name,
		Size:     len(data),
		Stats:    stats,
		Values:   values,
		Flags:    flags,
	}

	globalHarness.trace.Snapshots = append(globalHarness.trace.Snapshots, snap)

	if cfg.Verbose {
		flagStr := ""
		if len(flags) > 0 {
			flagStr = fmt.Sprintf(" [%v]", flags)
		}
		fmt.Fprintf(os.Stderr, "[DEBUG] L%d pos=%d %s %s: min=%.4f max=%.4f mean=%.4f nan=%d%s\n",
			layer, position, op, name, stats.Min, stats.Max, stats.Mean, stats.NaN, flagStr)
	}

	return len(flags) > 0
}

func computeStats(data []float32) TensorStats {
	if len(data) == 0 {
		return TensorStats{}
	}

	var min, max, sum float32
	var nan, inf, zero int
	min = float32(math.MaxFloat32)
	max = float32(-math.MaxFloat32)

	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nan++
			continue
		}
		if math.IsInf(float64(v), 0) {
			inf++
			continue
		}
		if v == 0 {
			zero++
		}
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}

	validCount := len(data) - nan - inf
	var mean float32
	if validCount > 0 {
		mean = sum / float32(validCount)
	}

	// Handle case where all values are NaN/Inf
	if validCount == 0 {
		min = float32(math.NaN())
		max = float32(math.NaN())
	}

	return TensorStats{
		Min:  min,
		Max:  max,
		Mean: mean,
		NaN:  nan,
		Inf:  inf,
		Zero: zero,
	}
}
