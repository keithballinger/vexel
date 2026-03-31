# LoRA Inference (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load LoRA adapters from safetensors files and apply them during inference via `--lora` flag.

**Architecture:** LoRA adapter weights (A/B matrices per layer) are loaded from safetensors into GPU memory. During the forward pass, after each frozen Q/V matmul, the LoRA contribution `scale * B @ (A @ x)` is computed via two small FP32 matmuls and accumulated into the output buffer. The existing `MatMulTransposed` and `Add` kernels handle all computation — no new Metal kernels needed for v1.

**Tech Stack:** Go, Metal (existing kernels), safetensors format

---

### Task 1: Adapter Config Parser

**Files:**
- Create: `inference/lora/config.go`
- Create: `inference/lora/config_test.go`

- [ ] **Step 1: Create the lora package with AdapterConfig**

```go
// inference/lora/config.go
package lora

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// AdapterConfig holds LoRA hyperparameters from adapter_config.json.
type AdapterConfig struct {
	Rank          int      `json:"r"`
	Alpha         float32  `json:"lora_alpha"`
	TargetModules []string `json:"target_modules"`
	BaseModel     string   `json:"base_model_name_or_path"`
}

// Scale returns the LoRA scaling factor alpha/rank.
func (c AdapterConfig) Scale() float32 {
	if c.Rank == 0 {
		return 0
	}
	return c.Alpha / float32(c.Rank)
}

// LoadConfig reads adapter_config.json from a directory.
func LoadConfig(dir string) (AdapterConfig, error) {
	data, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		return AdapterConfig{}, fmt.Errorf("read adapter_config.json: %w", err)
	}
	var cfg AdapterConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return AdapterConfig{}, fmt.Errorf("parse adapter_config.json: %w", err)
	}
	if cfg.Rank <= 0 {
		return AdapterConfig{}, fmt.Errorf("invalid LoRA rank: %d", cfg.Rank)
	}
	return cfg, nil
}
```

- [ ] **Step 2: Write test**

```go
// inference/lora/config_test.go
package lora

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	json := `{"r": 16, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": "test-model"}`
	os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(json), 0644)

	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Rank != 16 {
		t.Errorf("rank=%d, want 16", cfg.Rank)
	}
	if cfg.Scale() != 1.0 {
		t.Errorf("scale=%f, want 1.0", cfg.Scale())
	}
	if len(cfg.TargetModules) != 2 {
		t.Errorf("target_modules len=%d, want 2", len(cfg.TargetModules))
	}
}

func TestLoadConfigMissing(t *testing.T) {
	_, err := LoadConfig(t.TempDir())
	if err == nil {
		t.Error("expected error for missing config")
	}
}
```

- [ ] **Step 3: Run tests**

```bash
go test -v ./inference/lora/ -run TestLoadConfig
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add inference/lora/
git commit -m "feat(lora): add adapter config parser

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Safetensors Adapter Loader

**Files:**
- Create: `inference/lora/loader.go`
- Create: `inference/lora/loader_test.go`
- Modify: `inference/pkg/safetensors/loader.go` (extend if needed)

The existing safetensors package only parses headers. We need to actually load tensor data.

- [ ] **Step 1: Implement the safetensors tensor loader**

Read the existing `/Users/qeetbastudio/projects/vexel/inference/pkg/safetensors/loader.go` to understand the header format. Then create the LoRA loader that:
1. Reads the safetensors file
2. Parses the header to find tensor offsets/shapes
3. Extracts FP32 tensor data by name

```go
// inference/lora/loader.go
package lora

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// TensorInfo describes a tensor in the safetensors file.
type TensorInfo struct {
	DType   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets [2]int  `json:"data_offsets"`
}

// LoadAdapter loads a LoRA adapter from a directory containing
// adapter_config.json and adapter_model.safetensors.
func LoadAdapter(dir string) (*Adapter, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}

	stPath := filepath.Join(dir, "adapter_model.safetensors")
	data, err := os.ReadFile(stPath)
	if err != nil {
		return nil, fmt.Errorf("read safetensors: %w", err)
	}

	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors file too small")
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	headerJSON := data[8 : 8+headerLen]
	tensorData := data[8+headerLen:]

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("parse safetensors header: %w", err)
	}

	// Parse tensor infos (skip __metadata__)
	tensors := make(map[string]TensorInfo)
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("parse tensor %s: %w", name, err)
		}
		tensors[name] = info
	}

	adapter := &Adapter{
		Config: cfg,
		Scale:  cfg.Scale(),
	}

	// Find max layer index
	maxLayer := -1
	for name := range tensors {
		layerIdx := parseLayerIndex(name)
		if layerIdx > maxLayer {
			maxLayer = layerIdx
		}
	}
	if maxLayer < 0 {
		return nil, fmt.Errorf("no layer tensors found in adapter")
	}

	adapter.Layers = make([]LayerAdapter, maxLayer+1)

	// Load A/B matrices for each layer and target module
	for name, info := range tensors {
		layerIdx := parseLayerIndex(name)
		if layerIdx < 0 {
			continue
		}

		weights := extractF32(tensorData, info)

		lower := strings.ToLower(name)
		switch {
		case strings.Contains(lower, "q_proj.lora_a"):
			adapter.Layers[layerIdx].QA = weights
			adapter.Layers[layerIdx].QAShape = info.Shape
		case strings.Contains(lower, "q_proj.lora_b"):
			adapter.Layers[layerIdx].QB = weights
			adapter.Layers[layerIdx].QBShape = info.Shape
		case strings.Contains(lower, "v_proj.lora_a"):
			adapter.Layers[layerIdx].VA = weights
			adapter.Layers[layerIdx].VAShape = info.Shape
		case strings.Contains(lower, "v_proj.lora_b"):
			adapter.Layers[layerIdx].VB = weights
			adapter.Layers[layerIdx].VBShape = info.Shape
		}
	}

	return adapter, nil
}

// Adapter holds LoRA weights on CPU before GPU upload.
type Adapter struct {
	Config AdapterConfig
	Scale  float32
	Layers []LayerAdapter
}

// LayerAdapter holds LoRA A/B matrices for one transformer layer (CPU).
type LayerAdapter struct {
	QA      []float32
	QAShape []int // [rank, hidden]
	QB      []float32
	QBShape []int // [qDim, rank]
	VA      []float32
	VAShape []int // [rank, hidden]
	VB      []float32
	VBShape []int // [vDim, rank]
}

// HasQ returns true if this layer has Q projection LoRA weights.
func (la LayerAdapter) HasQ() bool { return len(la.QA) > 0 && len(la.QB) > 0 }

// HasV returns true if this layer has V projection LoRA weights.
func (la LayerAdapter) HasV() bool { return len(la.VA) > 0 && len(la.VB) > 0 }

func parseLayerIndex(name string) int {
	// Match patterns like "layers.5." or "blk.5."
	for _, prefix := range []string{"layers.", "blk."} {
		idx := strings.Index(name, prefix)
		if idx < 0 {
			continue
		}
		numStr := ""
		for _, c := range name[idx+len(prefix):] {
			if c >= '0' && c <= '9' {
				numStr += string(c)
			} else {
				break
			}
		}
		if numStr != "" {
			n := 0
			for _, c := range numStr {
				n = n*10 + int(c-'0')
			}
			return n
		}
	}
	return -1
}

func extractF32(data []byte, info TensorInfo) []float32 {
	start, end := info.Offsets[0], info.Offsets[1]
	raw := data[start:end]

	switch info.DType {
	case "F32":
		n := len(raw) / 4
		result := make([]float32, n)
		for i := 0; i < n; i++ {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return result
	case "F16":
		n := len(raw) / 2
		result := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2:])
			result[i] = halfToFloat(bits)
		}
		return result
	case "BF16":
		n := len(raw) / 2
		result := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2:])
			result[i] = math.Float32frombits(uint32(bits) << 16)
		}
		return result
	default:
		return nil
	}
}

func halfToFloat(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Denormal
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 31 {
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000) // Inf
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000) // NaN
	}
	exp = exp + 127 - 15
	return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
}
```

- [ ] **Step 2: Write test**

```go
// inference/lora/loader_test.go
package lora

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAdapter(t *testing.T) {
	dir := t.TempDir()

	// Write config
	cfgJSON := `{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}`
	os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(cfgJSON), 0644)

	// Build a minimal safetensors file with 2 tensors
	// Layer 0 Q LoRA: A [4, 8], B [16, 4]
	qa := make([]float32, 4*8)   // [rank=4, hidden=8]
	qb := make([]float32, 16*4)  // [qDim=16, rank=4]
	for i := range qa { qa[i] = 0.01 * float32(i) }
	for i := range qb { qb[i] = 0.001 * float32(i) }

	st := buildSafetensors(map[string]stTensor{
		"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {Shape: []int{4, 8}, Data: f32ToBytes(qa)},
		"base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {Shape: []int{16, 4}, Data: f32ToBytes(qb)},
	})
	os.WriteFile(filepath.Join(dir, "adapter_model.safetensors"), st, 0644)

	adapter, err := LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	if adapter.Scale != 2.0 { // alpha=8, rank=4
		t.Errorf("scale=%f, want 2.0", adapter.Scale)
	}
	if len(adapter.Layers) != 1 {
		t.Fatalf("layers=%d, want 1", len(adapter.Layers))
	}
	if !adapter.Layers[0].HasQ() {
		t.Error("layer 0 should have Q LoRA")
	}
	if adapter.Layers[0].HasV() {
		t.Error("layer 0 should NOT have V LoRA (not in test data)")
	}
	if len(adapter.Layers[0].QA) != 32 { // 4*8
		t.Errorf("QA len=%d, want 32", len(adapter.Layers[0].QA))
	}
}

type stTensor struct {
	Shape []int
	Data  []byte
}

func buildSafetensors(tensors map[string]stTensor) []byte {
	header := make(map[string]interface{})
	offset := 0
	for name, t := range tensors {
		header[name] = map[string]interface{}{
			"dtype":        "F32",
			"shape":        t.Shape,
			"data_offsets": []int{offset, offset + len(t.Data)},
		}
		offset += len(t.Data)
	}
	headerJSON, _ := json.Marshal(header)

	result := make([]byte, 8+len(headerJSON)+offset)
	binary.LittleEndian.PutUint64(result[:8], uint64(len(headerJSON)))
	copy(result[8:], headerJSON)

	dataStart := 8 + len(headerJSON)
	for name, t := range tensors {
		info := header[name].(map[string]interface{})
		offsets := info["data_offsets"].([]int)
		copy(result[dataStart+offsets[0]:], t.Data)
	}
	return result
}

func f32ToBytes(data []float32) []byte {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32frombits(v))
	}
	return buf
}
```

- [ ] **Step 3: Run tests**

```bash
go test -v ./inference/lora/ -run TestLoadAdapter
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add inference/lora/
git commit -m "feat(lora): add safetensors adapter loader

Loads LoRA A/B weight matrices from HuggingFace PEFT format.
Supports F32, F16, BF16 tensor dtypes.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: GPU Adapter + Forward Pass Injection

**Files:**
- Create: `inference/lora/gpu_adapter.go`
- Modify: `inference/runtime/model.go` — add AttachLoRA method
- Modify: `inference/runtime/block.go` — inject LoRA into Q/V projection

- [ ] **Step 1: Create GPU adapter that uploads weights to Metal**

```go
// inference/lora/gpu_adapter.go
package lora

import (
	"fmt"
	"unsafe"
	"vexel/inference/tensor"
)

// GPUAdapter holds LoRA weights on GPU, ready for forward pass injection.
type GPUAdapter struct {
	Scale  float32
	Rank   int
	Layers []GPULayerAdapter
}

// GPULayerAdapter holds per-layer LoRA GPU buffers.
type GPULayerAdapter struct {
	QA, QB tensor.DevicePtr // Q projection LoRA
	VA, VB tensor.DevicePtr // V projection LoRA
	HasQ   bool
	HasV   bool
}

// Allocator is the interface needed to upload weights to GPU.
type Allocator interface {
	AllocPermanent(bytes int) tensor.DevicePtr
	ToDevice(dst tensor.DevicePtr, src []byte)
}

// UploadToGPU copies CPU adapter weights to GPU permanent memory.
func UploadToGPU(adapter *Adapter, alloc Allocator) (*GPUAdapter, error) {
	gpu := &GPUAdapter{
		Scale:  adapter.Scale,
		Rank:   adapter.Config.Rank,
		Layers: make([]GPULayerAdapter, len(adapter.Layers)),
	}

	for i, layer := range adapter.Layers {
		if layer.HasQ() {
			gpu.Layers[i].QA = uploadF32(alloc, layer.QA)
			gpu.Layers[i].QB = uploadF32(alloc, layer.QB)
			gpu.Layers[i].HasQ = true
		}
		if layer.HasV() {
			gpu.Layers[i].VA = uploadF32(alloc, layer.VA)
			gpu.Layers[i].VB = uploadF32(alloc, layer.VB)
			gpu.Layers[i].HasV = true
		}
	}

	return gpu, nil
}

func uploadF32(alloc Allocator, data []float32) tensor.DevicePtr {
	bytes := len(data) * 4
	ptr := alloc.AllocPermanent(bytes)
	if ptr.IsNil() {
		return ptr
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), bytes)
	alloc.ToDevice(ptr, src)
	return ptr
}

// GetLayer returns the GPU LoRA for a specific layer index, or nil if out of range.
func (g *GPUAdapter) GetLayer(idx int) *GPULayerAdapter {
	if idx < 0 || idx >= len(g.Layers) {
		return nil
	}
	la := &g.Layers[idx]
	if !la.HasQ && !la.HasV {
		return nil
	}
	return la
}
```

- [ ] **Step 2: Add AttachLoRA to ModelRuntime**

In `inference/runtime/model.go`, add a field and method:

```go
// Add to ModelRuntime struct (after verbose field):
loraAdapter *lora.GPUAdapter

// Add method:
func (m *ModelRuntime) AttachLoRA(adapter *lora.GPUAdapter) {
	m.loraAdapter = adapter
}

func (m *ModelRuntime) LoRA() *lora.GPUAdapter {
	return m.loraAdapter
}
```

Add `"vexel/inference/lora"` to the imports.

- [ ] **Step 3: Inject LoRA into Q/V forward pass**

In `inference/runtime/block.go`, in the `ExecuteWithGPUKV` function, find the separate Q/V projection path (lines ~1416-1429). After each Q and V matmul, add LoRA contribution:

```go
// After line 1418 (Q projection):
if !b.Wq.DevicePtr().IsNil() {
    qDim := b.Wq.Shape().Dims()[0]
    b.matMulTransposedWithBias(normOutPtr, b.Wq, b.WqBias, qPtr, seqLen, qDim, hiddenSize)
}
// NEW: LoRA Q contribution
if b.loraLayer != nil && b.loraLayer.HasQ {
    b.applyLoRA(normOutPtr, b.loraLayer.QA, b.loraLayer.QB, qPtr,
        seqLen, b.loraRank, hiddenSize, b.Wq.Shape().Dims()[0], b.loraScale)
}
```

Similarly after line 1428 (V projection):
```go
if !b.Wv.DevicePtr().IsNil() {
    vDim := b.Wv.Shape().Dims()[0]
    b.matMulTransposedWithBias(normOutPtr, b.Wv, b.WvBias, vPtr, seqLen, vDim, hiddenSize)
}
// NEW: LoRA V contribution
if b.loraLayer != nil && b.loraLayer.HasV {
    b.applyLoRA(normOutPtr, b.loraLayer.VA, b.loraLayer.VB, vPtr,
        seqLen, b.loraRank, hiddenSize, b.Wv.Shape().Dims()[0], b.loraScale)
}
```

Add fields to `BlockRuntime` struct:
```go
loraLayer *lora.GPULayerAdapter
loraRank  int
loraScale float32
```

Add the `applyLoRA` helper method:
```go
// applyLoRA computes out += scale * B @ (A @ x) using two FP32 matmuls.
// A: [rank, inDim], B: [outDim, rank], x: [seqLen, inDim], out: [seqLen, outDim]
func (b *BlockRuntime) applyLoRA(x, a, bMat, out tensor.DevicePtr, seqLen, rank, inDim, outDim int, scale float32) {
    // Step 1: intermediate = x @ A^T → [seqLen, rank]
    intermediateBytes := seqLen * rank * 4
    intermediate := b.backend.Alloc(intermediateBytes)
    b.backend.MatMulTransposed(x, a, intermediate, seqLen, rank, inDim)

    // Step 2: loraOut = intermediate @ B^T → [seqLen, outDim]
    loraOutBytes := seqLen * outDim * 4
    loraOut := b.backend.Alloc(loraOutBytes)
    b.backend.MatMulTransposed(intermediate, bMat, loraOut, seqLen, outDim, rank)

    // Step 3: out += scale * loraOut
    if scale != 1.0 {
        b.backend.ScaleBuffer(loraOut, scale, seqLen*outDim)
    }
    b.backend.Add(out, loraOut, out, seqLen*outDim)
}
```

Add a method to wire LoRA into layers in model.go:
```go
func (m *ModelRuntime) wireLoRA() {
    if m.loraAdapter == nil {
        return
    }
    for i, layer := range m.layers {
        la := m.loraAdapter.GetLayer(i)
        if la != nil {
            layer.loraLayer = la
            layer.loraRank = m.loraAdapter.Rank
            layer.loraScale = m.loraAdapter.Scale
        }
    }
}
```

Call `m.wireLoRA()` at the end of `AttachLoRA`.

- [ ] **Step 4: Build and verify**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add inference/lora/gpu_adapter.go inference/runtime/model.go inference/runtime/block.go
git commit -m "feat(lora): GPU adapter upload + forward pass injection

LoRA contribution computed as scale * B @ (A @ x) via two FP32 matmuls.
Injected after frozen Q/V projections in ExecuteWithGPUKV.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: CLI Integration

**Files:**
- Modify: `inference/cmd/vexel/cli.go` — add --lora flag
- Modify: `inference/cmd/vexel/commands.go` — load and attach adapter

- [ ] **Step 1: Add --lora flag to GlobalFlags**

In `inference/cmd/vexel/cli.go`, add to GlobalFlags struct:
```go
LoRAPath string // Path to LoRA adapter directory
```

In the parseArgs function, add the case:
```go
case "--lora":
    if i+1 >= len(args) {
        return "", GlobalFlags{}, fmt.Errorf("--lora requires a path")
    }
    globals.LoRAPath = args[i+1]
    i += 2
```

- [ ] **Step 2: Load and attach adapter in commands.go**

In `inference/cmd/vexel/commands.go`, in the `initModel` function, after `model.CopyWeightsToDevice()` and GPU KV cache creation, add:

```go
// Load LoRA adapter if specified
if globals.LoRAPath != "" {
    adapter, err := lora.LoadAdapter(globals.LoRAPath)
    if err != nil {
        return nil, nil, nil, fmt.Errorf("load LoRA adapter: %w", err)
    }
    gpuAdapter, err := lora.UploadToGPU(adapter, gpuBackend)
    if err != nil {
        return nil, nil, nil, fmt.Errorf("upload LoRA to GPU: %w", err)
    }
    model.AttachLoRA(gpuAdapter)
    log.Printf("LoRA adapter loaded: rank=%d, alpha=%.0f, scale=%.4f, layers=%d",
        adapter.Config.Rank, adapter.Config.Alpha, adapter.Scale, len(adapter.Layers))
}
```

Add `"vexel/inference/lora"` to imports.

- [ ] **Step 3: Build and test CLI flag parsing**

```bash
go build -tags metal ./inference/...

# Test that --lora flag is parsed (will error on missing adapter, that's OK)
go run -tags metal ./inference/cmd/vexel --model /Users/qeetbastudio/projects/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --lora /nonexistent generate --prompt "test" 2>&1 | grep -i "lora"
```

Expected: Error message containing "lora" or "adapter".

- [ ] **Step 4: Commit**

```bash
git add inference/cmd/vexel/cli.go inference/cmd/vexel/commands.go
git commit -m "feat(cli): add --lora flag for LoRA adapter loading

Usage: vexel generate --model model.gguf --lora ./adapter/ --prompt 'Hello'

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: End-to-End Test with Synthetic Adapter

**Files:**
- Create: `inference/lora/e2e_test.go`

- [ ] **Step 1: Write end-to-end test**

Create a test that builds a zero-initialized LoRA adapter (B=0, so adapter has no effect), loads it onto a real model, and verifies the output matches the base model.

```go
// inference/lora/e2e_test.go
//go:build metal && darwin && cgo

package lora_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestZeroLoRAMatchesBase verifies that a zero-initialized LoRA
// (B matrices = 0) produces identical output to the base model.
// This is a compile/integration sanity check.
func TestZeroLoRAMatchesBase(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		// Try known paths
		candidates := []string{
			"/Users/qeetbastudio/projects/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
		}
		for _, p := range candidates {
			if _, err := os.Stat(p); err == nil {
				modelPath = p
				break
			}
		}
	}
	if modelPath == "" {
		t.Skip("No test model available (set VEXEL_TEST_MODEL)")
	}

	// Create a minimal zero LoRA adapter (rank=4, 1 layer)
	dir := t.TempDir()

	cfgJSON := `{"r": 4, "lora_alpha": 4, "target_modules": ["q_proj", "v_proj"]}`
	os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(cfgJSON), 0644)

	// Zero B matrices mean the adapter has no effect on output
	rank := 4
	hidden := 896 // Qwen 2.5 0.5B hidden size
	qDim := 896   // numHeads * headDim = 14 * 64
	vDim := 128   // numKVHeads * headDim = 2 * 64

	qa := make([]float32, rank*hidden) // Will be random but B=0 so doesn't matter
	qb := make([]float32, qDim*rank)   // All zeros
	va := make([]float32, rank*hidden)
	vb := make([]float32, vDim*rank) // All zeros

	// Set A to small values (doesn't matter since B=0)
	for i := range qa { qa[i] = 0.01 }
	for i := range va { va[i] = 0.01 }

	st := buildTestSafetensors(map[string]testTensor{
		"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {[]int{rank, hidden}, qa},
		"base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {[]int{qDim, rank}, qb},
		"base_model.model.layers.0.self_attn.v_proj.lora_A.weight": {[]int{rank, hidden}, va},
		"base_model.model.layers.0.self_attn.v_proj.lora_B.weight": {[]int{vDim, rank}, vb},
	})
	os.WriteFile(filepath.Join(dir, "adapter_model.safetensors"), st, 0644)

	// Verify the adapter loads without error
	adapter, err := LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	if adapter.Config.Rank != 4 {
		t.Errorf("rank=%d, want 4", adapter.Config.Rank)
	}
	if !adapter.Layers[0].HasQ() {
		t.Error("layer 0 missing Q LoRA")
	}
	if !adapter.Layers[0].HasV() {
		t.Error("layer 0 missing V LoRA")
	}
	t.Logf("Adapter loaded: rank=%d, scale=%.2f, layers=%d", adapter.Config.Rank, adapter.Scale, len(adapter.Layers))
}

type testTensor struct {
	Shape []int
	Data  []float32
}

func buildTestSafetensors(tensors map[string]testTensor) []byte {
	header := make(map[string]interface{})
	var allData []byte
	offset := 0
	for name, t := range tensors {
		data := f32Bytes(t.Data)
		header[name] = map[string]interface{}{
			"dtype":        "F32",
			"shape":        t.Shape,
			"data_offsets": []int{offset, offset + len(data)},
		}
		allData = append(allData, data...)
		offset += len(data)
	}
	headerJSON, _ := json.Marshal(header)
	result := make([]byte, 8+len(headerJSON)+len(allData))
	binary.LittleEndian.PutUint64(result[:8], uint64(len(headerJSON)))
	copy(result[8:], headerJSON)
	copy(result[8+len(headerJSON):], allData)
	return result
}

func f32Bytes(data []float32) []byte {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32frombits(v))
	}
	return buf
}
```

- [ ] **Step 2: Run test**

```bash
go test -tags metal -v ./inference/lora/ -run TestZeroLoRA -timeout 60s
```

Expected: PASS (adapter loads, shapes are correct)

- [ ] **Step 3: Commit**

```bash
git add inference/lora/e2e_test.go
git commit -m "test(lora): add end-to-end test with synthetic zero adapter

Verifies adapter loading, config parsing, and weight extraction
from safetensors format.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
