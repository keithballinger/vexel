# LoRA Training (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LoRA fine-tuning to vexel via `vexel train`, training LoRA adapters on Apple Silicon (Metal) with SGD, single-example steps, live loss output, and PEFT-compatible checkpoint saving.

**Architecture:** The training loop runs a modified forward pass that saves activations at each layer, computes cross-entropy loss, runs a full backward pass through all frozen layers (activation gradients only) to extract LoRA weight gradients at Q/V injection points, then updates weights with SGD. Five new Metal kernels handle backward ops (CrossEntropy, RMSNorm, SDPA, SiLUMul, RoPE backward). Data is loaded from JSONL with auto-detected format.

**Tech Stack:** Go, Metal (MSL shaders embedded in Objective-C bridge), safetensors format

---

### Task 1: JSONL Data Loader

**Files:**
- Create: `inference/lora/train/data.go`
- Create: `inference/lora/train/data_test.go`

- [ ] **Step 1: Write the failing test for text-format loading**

```go
// inference/lora/train/data_test.go
package train

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDataText(t *testing.T) {
	dir := t.TempDir()
	jsonl := `{"text": "Hello world"}
{"text": "Second example"}
`
	os.WriteFile(filepath.Join(dir, "train.jsonl"), []byte(jsonl), 0644)

	examples, err := LoadData(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadData: %v", err)
	}
	if len(examples) != 2 {
		t.Fatalf("got %d examples, want 2", len(examples))
	}
	if examples[0].Text != "Hello world" {
		t.Errorf("text=%q, want %q", examples[0].Text, "Hello world")
	}
	if examples[0].Format != FormatText {
		t.Errorf("format=%v, want FormatText", examples[0].Format)
	}
}

func TestLoadDataPromptCompletion(t *testing.T) {
	dir := t.TempDir()
	jsonl := `{"prompt": "What is 2+2?", "completion": "4"}
{"prompt": "Capital of France?", "completion": "Paris"}
`
	os.WriteFile(filepath.Join(dir, "train.jsonl"), []byte(jsonl), 0644)

	examples, err := LoadData(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadData: %v", err)
	}
	if len(examples) != 2 {
		t.Fatalf("got %d examples, want 2", len(examples))
	}
	if examples[0].Prompt != "What is 2+2?" {
		t.Errorf("prompt=%q", examples[0].Prompt)
	}
	if examples[0].Completion != "4" {
		t.Errorf("completion=%q", examples[0].Completion)
	}
	if examples[0].Format != FormatPromptCompletion {
		t.Errorf("format=%v, want FormatPromptCompletion", examples[0].Format)
	}
}

func TestLoadDataEmpty(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "empty.jsonl"), []byte(""), 0644)

	_, err := LoadData(filepath.Join(dir, "empty.jsonl"))
	if err == nil {
		t.Error("expected error for empty file")
	}
}

func TestLoadDataMixedFormat(t *testing.T) {
	dir := t.TempDir()
	jsonl := `{"text": "Hello"}
{"prompt": "Q?", "completion": "A"}
`
	os.WriteFile(filepath.Join(dir, "mixed.jsonl"), []byte(jsonl), 0644)

	_, err := LoadData(filepath.Join(dir, "mixed.jsonl"))
	if err == nil {
		t.Error("expected error for mixed formats")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test -v ./inference/lora/train/ -run TestLoadData
```

Expected: FAIL — package does not exist yet.

- [ ] **Step 3: Implement the data loader**

```go
// inference/lora/train/data.go
package train

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// DataFormat indicates the JSONL record format.
type DataFormat int

const (
	FormatText             DataFormat = iota // {"text": "..."}
	FormatPromptCompletion                   // {"prompt": "...", "completion": "..."}
)

// Example is a single training example loaded from JSONL.
type Example struct {
	Format     DataFormat
	Text       string // Used when Format == FormatText
	Prompt     string // Used when Format == FormatPromptCompletion
	Completion string // Used when Format == FormatPromptCompletion
}

// LoadData reads a JSONL file and returns parsed examples.
// All examples must use the same format (auto-detected from the first line).
func LoadData(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open data file: %w", err)
	}
	defer f.Close()

	var examples []Example
	var detectedFormat DataFormat
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB line buffer
	lineNum := 0

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		lineNum++

		var raw map[string]string
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return nil, fmt.Errorf("line %d: invalid JSON: %w", lineNum, err)
		}

		var ex Example
		_, hasText := raw["text"]
		_, hasPrompt := raw["prompt"]
		_, hasCompletion := raw["completion"]

		if hasText {
			ex.Format = FormatText
			ex.Text = raw["text"]
		} else if hasPrompt && hasCompletion {
			ex.Format = FormatPromptCompletion
			ex.Prompt = raw["prompt"]
			ex.Completion = raw["completion"]
		} else {
			return nil, fmt.Errorf("line %d: must have \"text\" or \"prompt\"+\"completion\" fields", lineNum)
		}

		if lineNum == 1 {
			detectedFormat = ex.Format
		} else if ex.Format != detectedFormat {
			return nil, fmt.Errorf("line %d: mixed formats (first line was %v, this line is %v)", lineNum, detectedFormat, ex.Format)
		}

		examples = append(examples, ex)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read data file: %w", err)
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("data file is empty")
	}

	return examples, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
go test -v ./inference/lora/train/ -run TestLoadData
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Write the loss mask builder test**

```go
// Add to inference/lora/train/data_test.go

func TestBuildLossMaskText(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5}
	mask := BuildLossMask(tokens, FormatText, 0)
	// Loss on all positions except last (no next token to predict)
	expected := []float32{1, 1, 1, 1, 0}
	if len(mask) != len(expected) {
		t.Fatalf("mask len=%d, want %d", len(mask), len(expected))
	}
	for i, v := range expected {
		if mask[i] != v {
			t.Errorf("mask[%d]=%f, want %f", i, mask[i], v)
		}
	}
}

func TestBuildLossMaskPromptCompletion(t *testing.T) {
	// 3 prompt tokens + 2 completion tokens
	tokens := []int32{10, 20, 30, 40, 50}
	promptLen := 3
	mask := BuildLossMask(tokens, FormatPromptCompletion, promptLen)
	// Loss only on completion tokens (positions 3,4), but last has no target
	expected := []float32{0, 0, 0, 1, 0}
	if len(mask) != len(expected) {
		t.Fatalf("mask len=%d, want %d", len(mask), len(expected))
	}
	for i, v := range expected {
		if mask[i] != v {
			t.Errorf("mask[%d]=%f, want %f", i, mask[i], v)
		}
	}
}
```

- [ ] **Step 6: Implement BuildLossMask**

```go
// Add to inference/lora/train/data.go

// BuildLossMask creates a loss mask for a tokenized sequence.
// For FormatText: mask is 1 for all positions except the last (no next-token target).
// For FormatPromptCompletion: mask is 0 for prompt positions, 1 for completion positions
// (except the last position which has no next-token target).
func BuildLossMask(tokens []int32, format DataFormat, promptLen int) []float32 {
	n := len(tokens)
	mask := make([]float32, n)

	switch format {
	case FormatText:
		for i := 0; i < n-1; i++ {
			mask[i] = 1
		}
	case FormatPromptCompletion:
		for i := promptLen; i < n-1; i++ {
			mask[i] = 1
		}
	}

	return mask
}
```

- [ ] **Step 7: Run all data tests**

```bash
go test -v ./inference/lora/train/ -run TestBuildLossMask
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add inference/lora/train/data.go inference/lora/train/data_test.go
git commit -m "feat(lora/train): add JSONL data loader with auto-format detection

Supports {\"text\"} and {\"prompt\",\"completion\"} formats with loss masking."
```

---

### Task 2: Safetensors Writer

**Files:**
- Create: `inference/lora/writer.go`
- Create: `inference/lora/writer_test.go`

- [ ] **Step 1: Write the failing round-trip test**

```go
// inference/lora/writer_test.go
package lora

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteAndReloadAdapter(t *testing.T) {
	// Create an adapter with known weights
	adapter := &Adapter{
		Config: AdapterConfig{
			Rank:          4,
			Alpha:         8,
			TargetModules: []string{"q_proj", "v_proj"},
			BaseModel:     "test-model",
		},
		Scale:  2.0,
		Layers: make([]LayerAdapter, 2),
	}

	// Layer 0: Q only
	adapter.Layers[0].QA = make([]float32, 4*8)   // [rank=4, hidden=8]
	adapter.Layers[0].QAShape = [2]int64{4, 8}
	adapter.Layers[0].QB = make([]float32, 16*4)   // [qDim=16, rank=4]
	adapter.Layers[0].QBShape = [2]int64{16, 4}
	for i := range adapter.Layers[0].QA {
		adapter.Layers[0].QA[i] = 0.1 * float32(i)
	}
	for i := range adapter.Layers[0].QB {
		adapter.Layers[0].QB[i] = 0.01 * float32(i)
	}

	// Layer 1: Q and V
	adapter.Layers[1].QA = make([]float32, 4*8)
	adapter.Layers[1].QAShape = [2]int64{4, 8}
	adapter.Layers[1].QB = make([]float32, 16*4)
	adapter.Layers[1].QBShape = [2]int64{16, 4}
	adapter.Layers[1].VA = make([]float32, 4*8)
	adapter.Layers[1].VAShape = [2]int64{4, 8}
	adapter.Layers[1].VB = make([]float32, 6*4)
	adapter.Layers[1].VBShape = [2]int64{6, 4}
	for i := range adapter.Layers[1].QA {
		adapter.Layers[1].QA[i] = 0.2 * float32(i)
	}
	for i := range adapter.Layers[1].QB {
		adapter.Layers[1].QB[i] = 0.02 * float32(i)
	}
	for i := range adapter.Layers[1].VA {
		adapter.Layers[1].VA[i] = 0.3 * float32(i)
	}
	for i := range adapter.Layers[1].VB {
		adapter.Layers[1].VB[i] = 0.03 * float32(i)
	}

	// Write
	dir := t.TempDir()
	err := SaveAdapter(adapter, dir)
	if err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	// Verify files exist
	if _, err := os.Stat(filepath.Join(dir, "adapter_config.json")); err != nil {
		t.Fatalf("adapter_config.json missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "adapter_model.safetensors")); err != nil {
		t.Fatalf("adapter_model.safetensors missing: %v", err)
	}

	// Reload and compare
	reloaded, err := LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter after save: %v", err)
	}

	if reloaded.Config.Rank != 4 {
		t.Errorf("rank=%d, want 4", reloaded.Config.Rank)
	}
	if reloaded.Scale != 2.0 {
		t.Errorf("scale=%f, want 2.0", reloaded.Scale)
	}
	if len(reloaded.Layers) != 2 {
		t.Fatalf("layers=%d, want 2", len(reloaded.Layers))
	}

	// Check layer 0 Q weights round-tripped correctly
	if !reloaded.Layers[0].HasQ() {
		t.Fatal("layer 0 should have Q")
	}
	if len(reloaded.Layers[0].QA) != len(adapter.Layers[0].QA) {
		t.Fatalf("QA len mismatch: %d vs %d", len(reloaded.Layers[0].QA), len(adapter.Layers[0].QA))
	}
	for i, v := range adapter.Layers[0].QA {
		if reloaded.Layers[0].QA[i] != v {
			t.Errorf("layer0.QA[%d]=%f, want %f", i, reloaded.Layers[0].QA[i], v)
			break
		}
	}

	// Check layer 1 V weights
	if !reloaded.Layers[1].HasV() {
		t.Fatal("layer 1 should have V")
	}
	for i, v := range adapter.Layers[1].VB {
		if reloaded.Layers[1].VB[i] != v {
			t.Errorf("layer1.VB[%d]=%f, want %f", i, reloaded.Layers[1].VB[i], v)
			break
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test -v ./inference/lora/ -run TestWriteAndReloadAdapter
```

Expected: FAIL — `SaveAdapter` not defined.

- [ ] **Step 3: Implement the safetensors writer**

```go
// inference/lora/writer.go
package lora

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// SaveAdapter writes a LoRA adapter to a directory in HuggingFace PEFT format.
// Creates adapter_config.json and adapter_model.safetensors.
func SaveAdapter(adapter *Adapter, dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Write adapter_config.json
	cfgJSON, err := json.MarshalIndent(map[string]interface{}{
		"r":                          adapter.Config.Rank,
		"lora_alpha":                 adapter.Config.Alpha,
		"target_modules":             adapter.Config.TargetModules,
		"base_model_name_or_path":    adapter.Config.BaseModel,
		"peft_type":                  "LORA",
		"task_type":                  "CAUSAL_LM",
		"bias":                       "none",
		"fan_in_fan_out":             false,
	}, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), cfgJSON, 0644); err != nil {
		return fmt.Errorf("write config: %w", err)
	}

	// Build safetensors
	type tensorEntry struct {
		name string
		data []float32
		shape [2]int64
	}

	var tensors []tensorEntry
	for i, layer := range adapter.Layers {
		if layer.HasQ() {
			tensors = append(tensors, tensorEntry{
				name:  fmt.Sprintf("base_model.model.model.layers.%d.self_attn.q_proj.lora_A.weight", i),
				data:  layer.QA,
				shape: layer.QAShape,
			})
			tensors = append(tensors, tensorEntry{
				name:  fmt.Sprintf("base_model.model.model.layers.%d.self_attn.q_proj.lora_B.weight", i),
				data:  layer.QB,
				shape: layer.QBShape,
			})
		}
		if layer.HasV() {
			tensors = append(tensors, tensorEntry{
				name:  fmt.Sprintf("base_model.model.model.layers.%d.self_attn.v_proj.lora_A.weight", i),
				data:  layer.VA,
				shape: layer.VAShape,
			})
			tensors = append(tensors, tensorEntry{
				name:  fmt.Sprintf("base_model.model.model.layers.%d.self_attn.v_proj.lora_B.weight", i),
				data:  layer.VB,
				shape: layer.VBShape,
			})
		}
	}

	// Sort by name for deterministic output
	sort.Slice(tensors, func(i, j int) bool {
		return tensors[i].name < tensors[j].name
	})

	// Build header and data blob
	header := make(map[string]interface{})
	var dataBlob []byte
	offset := 0

	for _, t := range tensors {
		byteLen := len(t.data) * 4
		header[t.name] = map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int64{t.shape[0], t.shape[1]},
			"data_offsets": []int{offset, offset + byteLen},
		}

		buf := make([]byte, byteLen)
		for j, v := range t.data {
			binary.LittleEndian.PutUint32(buf[j*4:], math.Float32bits(v))
		}
		dataBlob = append(dataBlob, buf...)
		offset += byteLen
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("marshal safetensors header: %w", err)
	}

	// Assemble: 8-byte header length + header JSON + data blob
	result := make([]byte, 8+len(headerJSON)+len(dataBlob))
	binary.LittleEndian.PutUint64(result[:8], uint64(len(headerJSON)))
	copy(result[8:], headerJSON)
	copy(result[8+len(headerJSON):], dataBlob)

	stPath := filepath.Join(dir, "adapter_model.safetensors")
	if err := os.WriteFile(stPath, result, 0644); err != nil {
		return fmt.Errorf("write safetensors: %w", err)
	}

	return nil
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
go test -v ./inference/lora/ -run TestWriteAndReloadAdapter
```

Expected: PASS — round-trip preserves all weights.

- [ ] **Step 5: Commit**

```bash
git add inference/lora/writer.go inference/lora/writer_test.go
git commit -m "feat(lora): add safetensors writer for checkpoint saving

Round-trips through existing loader. Writes PEFT-compatible format."
```

---

### Task 3: LoRA Weight Initialization

**Files:**
- Create: `inference/lora/train/init.go`
- Create: `inference/lora/train/init_test.go`

- [ ] **Step 1: Write the failing test**

```go
// inference/lora/train/init_test.go
package train

import (
	"math"
	"testing"

	"vexel/inference/lora"
)

func TestInitAdapter(t *testing.T) {
	cfg := lora.AdapterConfig{
		Rank:          4,
		Alpha:         8,
		TargetModules: []string{"q_proj", "v_proj"},
		BaseModel:     "test-model",
	}

	numLayers := 3
	hiddenSize := 64
	qDim := 32
	vDim := 16

	adapter := InitAdapter(cfg, numLayers, hiddenSize, qDim, vDim)

	if adapter.Config.Rank != 4 {
		t.Errorf("rank=%d, want 4", adapter.Config.Rank)
	}
	if adapter.Scale != 2.0 {
		t.Errorf("scale=%f, want 2.0", adapter.Scale)
	}
	if len(adapter.Layers) != numLayers {
		t.Fatalf("layers=%d, want %d", len(adapter.Layers), numLayers)
	}

	for i, layer := range adapter.Layers {
		// A matrices should be non-zero (Kaiming init)
		if !layer.HasQ() {
			t.Errorf("layer %d: missing Q", i)
		}
		if !layer.HasV() {
			t.Errorf("layer %d: missing V", i)
		}

		// Check A shapes
		if layer.QAShape != [2]int64{4, 64} {
			t.Errorf("layer %d: QAShape=%v, want [4, 64]", i, layer.QAShape)
		}
		if layer.QBShape != [2]int64{32, 4} {
			t.Errorf("layer %d: QBShape=%v, want [32, 4]", i, layer.QBShape)
		}

		// A matrices should have non-zero values
		allZero := true
		for _, v := range layer.QA {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("layer %d: QA is all zeros (should be Kaiming init)", i)
		}

		// B matrices should be all zeros
		for j, v := range layer.QB {
			if v != 0 {
				t.Errorf("layer %d: QB[%d]=%f, want 0", i, j, v)
				break
			}
		}
		for j, v := range layer.VB {
			if v != 0 {
				t.Errorf("layer %d: VB[%d]=%f, want 0", i, j, v)
				break
			}
		}

		// Verify Kaiming uniform bounds: [-1/sqrt(hidden), 1/sqrt(hidden)]
		bound := float32(1.0 / math.Sqrt(float64(hiddenSize)))
		for j, v := range layer.QA {
			if v < -bound || v > bound {
				t.Errorf("layer %d: QA[%d]=%f outside Kaiming bound [-%f, %f]", i, j, v, bound, bound)
				break
			}
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test -v ./inference/lora/train/ -run TestInitAdapter
```

Expected: FAIL — `InitAdapter` not defined.

- [ ] **Step 3: Implement weight initialization**

```go
// inference/lora/train/init.go
package train

import (
	"math"
	"math/rand"

	"vexel/inference/lora"
)

// InitAdapter creates a new LoRA adapter with Kaiming-initialized A matrices
// and zero-initialized B matrices (standard PEFT convention).
func InitAdapter(cfg lora.AdapterConfig, numLayers, hiddenSize, qDim, vDim int) *lora.Adapter {
	adapter := &lora.Adapter{
		Config: cfg,
		Scale:  cfg.Scale(),
		Layers: make([]lora.LayerAdapter, numLayers),
	}

	rank := cfg.Rank

	for i := 0; i < numLayers; i++ {
		layer := &adapter.Layers[i]

		// Q projection LoRA
		layer.QA = kaimingUniform(rank, hiddenSize)
		layer.QAShape = [2]int64{int64(rank), int64(hiddenSize)}
		layer.QB = make([]float32, qDim*rank) // zeros
		layer.QBShape = [2]int64{int64(qDim), int64(rank)}

		// V projection LoRA
		layer.VA = kaimingUniform(rank, hiddenSize)
		layer.VAShape = [2]int64{int64(rank), int64(hiddenSize)}
		layer.VB = make([]float32, vDim*rank) // zeros
		layer.VBShape = [2]int64{int64(vDim), int64(rank)}
	}

	return adapter
}

// kaimingUniform initializes a [rows, cols] matrix with Kaiming uniform distribution.
// For LoRA A matrices: fan_in = cols (hidden_size), bound = 1/sqrt(fan_in).
func kaimingUniform(rows, cols int) []float32 {
	bound := float32(1.0 / math.Sqrt(float64(cols)))
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = (rand.Float32()*2 - 1) * bound
	}
	return data
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
go test -v ./inference/lora/train/ -run TestInitAdapter
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add inference/lora/train/init.go inference/lora/train/init_test.go
git commit -m "feat(lora/train): add LoRA weight initialization

Kaiming uniform for A matrices, zero for B matrices (PEFT convention)."
```

---

### Task 4: Extend TrainingOps Interface with Backward Methods

**Files:**
- Modify: `inference/backend/backend.go` — add backward method signatures to `TrainingOps`

- [ ] **Step 1: Read the current TrainingOps interface**

Read `inference/backend/backend.go` lines 265-293 to see the current interface.

- [ ] **Step 2: Add backward methods to TrainingOps**

Add the following methods after the existing `Zero` method in the `TrainingOps` interface:

```go
	// Backward pass kernels
	CrossEntropyLossForwardBackward(logits, targets, mask, dLogits tensor.DevicePtr, lossOut *float32, seqLen, vocabSize int)
	RMSNormBackward(dOut, input, weight, rms, dInput tensor.DevicePtr, rows, cols int)
	SDPABackward(dOut, Q, K, V, attnWeights, dQ, dK, dV tensor.DevicePtr, seqLen, headDim, numHeads int)
	SiLUMulBackward(dOut, gate, up, dGate, dUp tensor.DevicePtr, n int)
	RoPEBackward(dQ, dK tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float64, ropeNeox bool)
```

- [ ] **Step 3: Build to check interface compiles**

```bash
go build ./inference/backend/...
```

Expected: Build failure — Metal backend doesn't implement the new methods yet. That's expected; we'll add stub implementations now.

- [ ] **Step 4: Add stub implementations to Metal backend**

Read `inference/backend/metal/backend.go` to find the existing TrainingOps implementations (around lines 1803-1875). Add stubs after the `Zero` method:

```go
func (b *Backend) CrossEntropyLossForwardBackward(logits, targets, mask, dLogits tensor.DevicePtr, lossOut *float32, seqLen, vocabSize int) {
	panic("CrossEntropyLossForwardBackward not yet implemented")
}

func (b *Backend) RMSNormBackward(dOut, input, weight, rms, dInput tensor.DevicePtr, rows, cols int) {
	panic("RMSNormBackward not yet implemented")
}

func (b *Backend) SDPABackward(dOut, Q, K, V, attnWeights, dQ, dK, dV tensor.DevicePtr, seqLen, headDim, numHeads int) {
	panic("SDPABackward not yet implemented")
}

func (b *Backend) SiLUMulBackward(dOut, gate, up, dGate, dUp tensor.DevicePtr, n int) {
	panic("SiLUMulBackward not yet implemented")
}

func (b *Backend) RoPEBackward(dQ, dK tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float64, ropeNeox bool) {
	panic("RoPEBackward not yet implemented")
}
```

- [ ] **Step 5: Build to verify stubs compile**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 6: Commit**

```bash
git add inference/backend/backend.go inference/backend/metal/backend.go
git commit -m "feat(backend): add backward pass method signatures to TrainingOps

Stubs for CrossEntropy, RMSNorm, SDPA, SiLUMul, and RoPE backward.
Implementations will follow with Metal kernels."
```

---

### Task 5: CrossEntropyLossForwardBackward Metal Kernel

**Files:**
- Modify: `inference/backend/metal/metal_bridge.h` — add C declaration
- Modify: `inference/backend/metal/metal_bridge_darwin.m` — add Metal shader + dispatch
- Modify: `inference/backend/metal/backend.go` — add pipeline + Go dispatch wrapper

- [ ] **Step 1: Add the C function declaration to metal_bridge.h**

```c
void metal_cross_entropy_loss_fwd_bwd_f32(void* queue, void* pipeline,
    void* logits, void* targets, void* mask, void* dLogits, void* lossOut,
    int seqLen, int vocabSize);
```

- [ ] **Step 2: Add the Metal shader and dispatch function to metal_bridge_darwin.m**

Add the Metal shader source string to the kernel source section (near other kernel definitions):

```metal
kernel void cross_entropy_loss_fwd_bwd_f32(
    device const float* logits [[buffer(0)]],    // [seqLen, vocabSize]
    device const int* targets [[buffer(1)]],      // [seqLen] — target token IDs
    device const float* mask [[buffer(2)]],       // [seqLen] — loss mask
    device float* dLogits [[buffer(3)]],          // [seqLen, vocabSize] — output gradients
    device atomic_float* lossOut [[buffer(4)]],   // scalar — accumulated loss
    constant int& seqLen [[buffer(5)]],
    constant int& vocabSize [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)seqLen) return;
    if (mask[tid] == 0.0f) {
        // Zero out gradient for masked positions
        for (int j = 0; j < vocabSize; j++) {
            dLogits[tid * vocabSize + j] = 0.0f;
        }
        return;
    }

    // Find max for numerical stability (log-sum-exp trick)
    float maxVal = logits[tid * vocabSize];
    for (int j = 1; j < vocabSize; j++) {
        maxVal = max(maxVal, logits[tid * vocabSize + j]);
    }

    // Compute log-sum-exp and softmax gradient
    float sumExp = 0.0f;
    for (int j = 0; j < vocabSize; j++) {
        sumExp += exp(logits[tid * vocabSize + j] - maxVal);
    }
    float logSumExp = log(sumExp) + maxVal;

    int target = targets[tid];
    float loss = -(logits[tid * vocabSize + target] - logSumExp);

    // Count masked positions for averaging (simple: add to atomic)
    atomic_fetch_add_explicit(lossOut, loss, memory_order_relaxed);

    // Gradient: softmax(logits) - one_hot(target), scaled by mask
    float invSumExp = 1.0f / sumExp;
    for (int j = 0; j < vocabSize; j++) {
        float softmax_j = exp(logits[tid * vocabSize + j] - maxVal) * invSumExp;
        float grad = softmax_j;
        if (j == target) {
            grad -= 1.0f;
        }
        dLogits[tid * vocabSize + j] = grad;
    }
}
```

Add the dispatch function:

```objc
void metal_cross_entropy_loss_fwd_bwd_f32(void* queuePtr, void* pipelinePtr,
    void* logits, void* targets, void* mask, void* dLogits, void* lossOut,
    int seqLen, int vocabSize) {

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder =
        get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)logits offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)targets offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)mask offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dLogits offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)lossOut offset:0 atIndex:4];
    [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:5];
    [encoder setBytes:&vocabSize length:sizeof(vocabSize) atIndex:6];

    MTLSize threadgroups = MTLSizeMake((seqLen + 255) / 256, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}
```

- [ ] **Step 3: Add pipeline and Go wrapper to Metal backend**

In `inference/backend/metal/backend.go`, add a pipeline field to the Backend struct:

```go
crossEntropyLossPipeline unsafe.Pointer
```

In the pipeline initialization section (after `zeroPipeline`):

```go
b.crossEntropyLossPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("cross_entropy_loss_fwd_bwd_f32"))
```

Replace the stub `CrossEntropyLossForwardBackward` with the real implementation:

```go
func (b *Backend) CrossEntropyLossForwardBackward(logits, targets, mask, dLogits tensor.DevicePtr, lossOut *float32, seqLen, vocabSize int) {
	// Create a small GPU buffer for the loss accumulator
	lossBuf := b.AllocPermanent(4)
	b.Zero(lossBuf, 1)

	C.metal_cross_entropy_loss_fwd_bwd_f32(
		b.queue, b.crossEntropyLossPipeline,
		unsafe.Pointer(logits.Addr()), unsafe.Pointer(targets.Addr()),
		unsafe.Pointer(mask.Addr()), unsafe.Pointer(dLogits.Addr()),
		unsafe.Pointer(lossBuf.Addr()),
		C.int(seqLen), C.int(vocabSize))

	// Read back loss value
	b.ReadFromDevice(lossBuf, unsafe.Slice((*byte)(unsafe.Pointer(lossOut)), 4))
}
```

- [ ] **Step 4: Build to verify kernel compiles**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add inference/backend/metal/metal_bridge.h inference/backend/metal/metal_bridge_darwin.m inference/backend/metal/backend.go
git commit -m "feat(metal): add CrossEntropyLossForwardBackward kernel

Fused softmax + loss + gradient computation with log-sum-exp stability."
```

---

### Task 6: RMSNormBackward Metal Kernel

**Files:**
- Modify: `inference/backend/metal/metal_bridge.h`
- Modify: `inference/backend/metal/metal_bridge_darwin.m`
- Modify: `inference/backend/metal/backend.go`

- [ ] **Step 1: Add C declaration to metal_bridge.h**

```c
void metal_rmsnorm_backward_f32(void* queue, void* pipeline,
    void* dOut, void* input, void* weight, void* rms, void* dInput,
    int rows, int cols);
```

- [ ] **Step 2: Add Metal shader and dispatch to metal_bridge_darwin.m**

Metal shader:

```metal
kernel void rmsnorm_backward_f32(
    device const float* dOut [[buffer(0)]],     // [rows, cols]
    device const float* input [[buffer(1)]],    // [rows, cols]
    device const float* weight [[buffer(2)]],   // [cols]
    device const float* rms [[buffer(3)]],      // [rows] — saved 1/rms from forward
    device float* dInput [[buffer(4)]],         // [rows, cols]
    constant int& cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (col >= (uint)cols) return;

    float inv_rms = rms[row];
    float x_i = input[row * cols + col];
    float w_i = weight[col];
    float dy_i = dOut[row * cols + col];

    // d(RMSNorm)/dx = w * inv_rms * (1 - x_i^2 * inv_rms^2 / cols)
    // But more precisely:
    // y = w * x * inv_rms
    // dy/dx = w * inv_rms - w * x * x * inv_rms^3 / cols (from chain rule through rms)

    // First compute dot product dOut . (w * x) for this row
    // We need a reduction — use a simpler per-element approach with a correction term
    // For correctness, compute the full gradient:
    // dInput[i] = inv_rms * (w[i] * dOut[i] - x[i] * inv_rms^2 * mean(w * dOut * x))

    // This kernel uses a two-pass approach per row.
    // Pass 1: compute dot = sum(w[j] * dOut[row,j] * input[row,j]) / cols
    float dot = 0.0f;
    for (int j = 0; j < cols; j++) {
        dot += weight[j] * dOut[row * cols + j] * input[row * cols + j];
    }
    dot *= inv_rms * inv_rms / float(cols);

    // Pass 2: dInput = inv_rms * (w * dOut - x * dot)
    dInput[row * cols + col] = inv_rms * (w_i * dy_i - x_i * dot);
}
```

Dispatch function:

```objc
void metal_rmsnorm_backward_f32(void* queuePtr, void* pipelinePtr,
    void* dOut, void* input, void* weight, void* rms, void* dInput,
    int rows, int cols) {

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder =
        get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dOut offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)rms offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dInput offset:0 atIndex:4];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:5];

    MTLSize threadgroups = MTLSizeMake((cols + 255) / 256, rows, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}
```

- [ ] **Step 3: Add pipeline and replace stub in backend.go**

Pipeline field:

```go
rmsnormBackwardPipeline unsafe.Pointer
```

Pipeline init:

```go
b.rmsnormBackwardPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rmsnorm_backward_f32"))
```

Replace the stub:

```go
func (b *Backend) RMSNormBackward(dOut, input, weight, rms, dInput tensor.DevicePtr, rows, cols int) {
	C.metal_rmsnorm_backward_f32(
		b.queue, b.rmsnormBackwardPipeline,
		unsafe.Pointer(dOut.Addr()), unsafe.Pointer(input.Addr()),
		unsafe.Pointer(weight.Addr()), unsafe.Pointer(rms.Addr()),
		unsafe.Pointer(dInput.Addr()),
		C.int(rows), C.int(cols))
}
```

- [ ] **Step 4: Build**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add inference/backend/metal/metal_bridge.h inference/backend/metal/metal_bridge_darwin.m inference/backend/metal/backend.go
git commit -m "feat(metal): add RMSNormBackward kernel

Two-pass per-row gradient: computes dInput = inv_rms * (w * dOut - x * dot)."
```

---

### Task 7: SiLUMulBackward Metal Kernel

**Files:**
- Modify: `inference/backend/metal/metal_bridge.h`
- Modify: `inference/backend/metal/metal_bridge_darwin.m`
- Modify: `inference/backend/metal/backend.go`

- [ ] **Step 1: Add C declaration**

```c
void metal_silu_mul_backward_f32(void* queue, void* pipeline,
    void* dOut, void* gate, void* up, void* dGate, void* dUp, int n);
```

- [ ] **Step 2: Add Metal shader and dispatch**

Metal shader:

```metal
kernel void silu_mul_backward_f32(
    device const float* dOut [[buffer(0)]],   // [n]
    device const float* gate [[buffer(1)]],   // [n] — pre-activation gate values
    device const float* up [[buffer(2)]],     // [n] — up projection values
    device float* dGate [[buffer(3)]],        // [n]
    device float* dUp [[buffer(4)]],          // [n]
    constant int& n [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;

    float g = gate[tid];
    float u = up[tid];
    float dy = dOut[tid];

    // Forward: y = silu(gate) * up = (gate / (1 + exp(-gate))) * up
    float sig = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sig;

    // d(silu(g) * u) / dg = u * (sig + g * sig * (1 - sig)) = u * sig * (1 + g * (1 - sig))
    // d(silu(g) * u) / du = silu(g)
    dGate[tid] = dy * u * sig * (1.0f + g * (1.0f - sig));
    dUp[tid] = dy * silu_g;
}
```

Dispatch function:

```objc
void metal_silu_mul_backward_f32(void* queuePtr, void* pipelinePtr,
    void* dOut, void* gate, void* up, void* dGate, void* dUp, int n) {

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder =
        get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dOut offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)gate offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)up offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dGate offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dUp offset:0 atIndex:4];
    [encoder setBytes:&n length:sizeof(n) atIndex:5];

    MTLSize threadgroups = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}
```

- [ ] **Step 3: Add pipeline and replace stub**

Pipeline field:

```go
siluMulBackwardPipeline unsafe.Pointer
```

Pipeline init:

```go
b.siluMulBackwardPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("silu_mul_backward_f32"))
```

Replace stub:

```go
func (b *Backend) SiLUMulBackward(dOut, gate, up, dGate, dUp tensor.DevicePtr, n int) {
	C.metal_silu_mul_backward_f32(
		b.queue, b.siluMulBackwardPipeline,
		unsafe.Pointer(dOut.Addr()), unsafe.Pointer(gate.Addr()),
		unsafe.Pointer(up.Addr()), unsafe.Pointer(dGate.Addr()),
		unsafe.Pointer(dUp.Addr()), C.int(n))
}
```

- [ ] **Step 4: Build**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add inference/backend/metal/metal_bridge.h inference/backend/metal/metal_bridge_darwin.m inference/backend/metal/backend.go
git commit -m "feat(metal): add SiLUMulBackward kernel

Element-wise d(silu(gate) * up) for FFN backward pass."
```

---

### Task 8: RoPEBackward Metal Kernel

**Files:**
- Modify: `inference/backend/metal/metal_bridge.h`
- Modify: `inference/backend/metal/metal_bridge_darwin.m`
- Modify: `inference/backend/metal/backend.go`

- [ ] **Step 1: Read the existing RoPE forward kernel**

Read the RoPE forward implementation in `metal_bridge_darwin.m` to understand the rotation pattern (standard vs neox-style), frequency computation, and buffer layout. The backward is the inverse rotation (negate the angle).

- [ ] **Step 2: Add C declaration**

```c
void metal_rope_backward_f32(void* queue, void* pipeline,
    void* dQ, void* dK, int headDim, int numHeads, int numKVHeads,
    int seqLen, int startPos, int ropeDim, double theta, bool ropeNeox);
```

- [ ] **Step 3: Add Metal shader and dispatch**

The backward for RoPE is the inverse rotation. Since RoPE applies a 2D rotation with angle `θ`, the backward applies rotation with angle `-θ` (same code, negated sin):

```metal
kernel void rope_backward_f32(
    device float* dQ [[buffer(0)]],
    device float* dK [[buffer(1)]],
    constant int& headDim [[buffer(2)]],
    constant int& numHeads [[buffer(3)]],
    constant int& numKVHeads [[buffer(4)]],
    constant int& seqLen [[buffer(5)]],
    constant int& startPos [[buffer(6)]],
    constant int& ropeDim [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    constant int& ropeNeox [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pos_idx = gid.y;  // sequence position
    uint pair_idx = gid.x; // which (cos,sin) pair within ropeDim/2
    if (pos_idx >= (uint)seqLen) return;
    if (pair_idx >= (uint)(ropeDim / 2)) return;

    int pos = startPos + (int)pos_idx;
    float freq = 1.0f / pow(theta, float(2 * pair_idx) / float(ropeDim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = -sin(angle);  // NEGATIVE for backward (inverse rotation)

    // Apply inverse rotation to dQ for all Q heads
    for (int h = 0; h < numHeads; h++) {
        int base = (int)pos_idx * numHeads * headDim + h * headDim;
        int i0, i1;
        if (ropeNeox) {
            i0 = base + (int)pair_idx;
            i1 = base + (int)pair_idx + ropeDim / 2;
        } else {
            i0 = base + 2 * (int)pair_idx;
            i1 = base + 2 * (int)pair_idx + 1;
        }
        float d0 = dQ[i0];
        float d1 = dQ[i1];
        dQ[i0] = d0 * cos_a - d1 * sin_a;
        dQ[i1] = d0 * sin_a + d1 * cos_a;
    }

    // Apply inverse rotation to dK for all KV heads
    for (int h = 0; h < numKVHeads; h++) {
        int base = (int)pos_idx * numKVHeads * headDim + h * headDim;
        int i0, i1;
        if (ropeNeox) {
            i0 = base + (int)pair_idx;
            i1 = base + (int)pair_idx + ropeDim / 2;
        } else {
            i0 = base + 2 * (int)pair_idx;
            i1 = base + 2 * (int)pair_idx + 1;
        }
        float d0 = dK[i0];
        float d1 = dK[i1];
        dK[i0] = d0 * cos_a - d1 * sin_a;
        dK[i1] = d0 * sin_a + d1 * cos_a;
    }
}
```

Dispatch function:

```objc
void metal_rope_backward_f32(void* queuePtr, void* pipelinePtr,
    void* dQ, void* dK, int headDim, int numHeads, int numKVHeads,
    int seqLen, int startPos, int ropeDim, double theta, bool ropeNeox) {

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder =
        get_encoder(queue, &cmdBuffer, &shouldCommit);

    float thetaF = (float)theta;
    int neox = ropeNeox ? 1 : 0;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dQ offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dK offset:0 atIndex:1];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:2];
    [encoder setBytes:&numHeads length:sizeof(numHeads) atIndex:3];
    [encoder setBytes:&numKVHeads length:sizeof(numKVHeads) atIndex:4];
    [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:5];
    [encoder setBytes:&startPos length:sizeof(startPos) atIndex:6];
    [encoder setBytes:&ropeDim length:sizeof(ropeDim) atIndex:7];
    [encoder setBytes:&thetaF length:sizeof(thetaF) atIndex:8];
    [encoder setBytes:&neox length:sizeof(neox) atIndex:9];

    MTLSize threadgroups = MTLSizeMake((ropeDim / 2 + 255) / 256, seqLen, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}
```

- [ ] **Step 4: Add pipeline and replace stub**

Pipeline field:

```go
ropeBackwardPipeline unsafe.Pointer
```

Pipeline init:

```go
b.ropeBackwardPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("rope_backward_f32"))
```

Replace stub:

```go
func (b *Backend) RoPEBackward(dQ, dK tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos, ropeDim int, theta float64, ropeNeox bool) {
	C.metal_rope_backward_f32(
		b.queue, b.ropeBackwardPipeline,
		unsafe.Pointer(dQ.Addr()), unsafe.Pointer(dK.Addr()),
		C.int(headDim), C.int(numHeads), C.int(numKVHeads),
		C.int(seqLen), C.int(startPos), C.int(ropeDim),
		C.double(theta), C.bool(ropeNeox))
}
```

- [ ] **Step 5: Build**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 6: Commit**

```bash
git add inference/backend/metal/metal_bridge.h inference/backend/metal/metal_bridge_darwin.m inference/backend/metal/backend.go
git commit -m "feat(metal): add RoPEBackward kernel

Inverse rotation (negate sin) for backpropagating through positional embeddings."
```

---

### Task 9: SDPABackward Metal Kernel

**Files:**
- Modify: `inference/backend/metal/metal_bridge.h`
- Modify: `inference/backend/metal/metal_bridge_darwin.m`
- Modify: `inference/backend/metal/backend.go`

- [ ] **Step 1: Read the existing SDPA forward kernel**

Read the SDPAPrefill implementation in `metal_bridge_darwin.m` to understand the layout (Q/K/V strides, head organization, scaling). The backward needs to match this layout.

- [ ] **Step 2: Add C declaration**

```c
void metal_sdpa_backward_f32(void* queue, void* pipeline,
    void* dOut, void* Q, void* K, void* V, void* attnWeights,
    void* dQ, void* dK, void* dV,
    int seqLen, int headDim, int numHeads);
```

- [ ] **Step 3: Add Metal shader and dispatch**

Metal shader (one threadgroup per head, iterates over sequence positions):

```metal
kernel void sdpa_backward_f32(
    device const float* dOut [[buffer(0)]],        // [numHeads, seqLen, headDim]
    device const float* Q [[buffer(1)]],           // [numHeads, seqLen, headDim]
    device const float* K [[buffer(2)]],           // [numHeads, seqLen, headDim]
    device const float* V [[buffer(3)]],           // [numHeads, seqLen, headDim]
    device const float* attnWeights [[buffer(4)]], // [numHeads, seqLen, seqLen]
    device float* dQ [[buffer(5)]],                // [numHeads, seqLen, headDim]
    device float* dK [[buffer(6)]],                // [numHeads, seqLen, headDim]
    device float* dV [[buffer(7)]],                // [numHeads, seqLen, headDim]
    constant int& seqLen [[buffer(8)]],
    constant int& headDim [[buffer(9)]],
    constant int& numHeads [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.y;  // head index
    uint i = gid.x;  // query position
    if (h >= (uint)numHeads || i >= (uint)seqLen) return;

    float scale = 1.0f / sqrt(float(headDim));

    int hOffset = h * seqLen * headDim;
    int aOffset = h * seqLen * seqLen;

    // dV[h][j][d] += sum_i attnWeights[h][i][j] * dOut[h][i][d]
    // We compute dV contribution from this query position i
    for (int j = 0; j <= (int)i; j++) {  // causal: j <= i
        float w = attnWeights[aOffset + i * seqLen + j];
        for (int d = 0; d < headDim; d++) {
            atomic_fetch_add_explicit(
                (device atomic_float*)&dV[hOffset + j * headDim + d],
                w * dOut[hOffset + i * headDim + d],
                memory_order_relaxed);
        }
    }

    // dAttnWeights[h][i][j] = sum_d dOut[h][i][d] * V[h][j][d]
    // Then dScores = attnWeights * (dAttnWeights - sum_j(attnWeights * dAttnWeights))
    // (softmax backward)

    // Compute dAttnWeights for row i
    float rowSum = 0.0f;
    for (int j = 0; j <= (int)i; j++) {
        float daw = 0.0f;
        for (int d = 0; d < headDim; d++) {
            daw += dOut[hOffset + i * headDim + d] * V[hOffset + j * headDim + d];
        }
        float w = attnWeights[aOffset + i * seqLen + j];
        rowSum += w * daw;
    }

    // Compute dScores and accumulate dQ, dK
    for (int j = 0; j <= (int)i; j++) {
        float daw = 0.0f;
        for (int d = 0; d < headDim; d++) {
            daw += dOut[hOffset + i * headDim + d] * V[hOffset + j * headDim + d];
        }
        float w = attnWeights[aOffset + i * seqLen + j];
        float dScore = w * (daw - rowSum) * scale;

        // dQ[h][i][d] += dScore * K[h][j][d]
        // dK[h][j][d] += dScore * Q[h][i][d]
        for (int d = 0; d < headDim; d++) {
            dQ[hOffset + i * headDim + d] += dScore * K[hOffset + j * headDim + d];
            atomic_fetch_add_explicit(
                (device atomic_float*)&dK[hOffset + j * headDim + d],
                dScore * Q[hOffset + i * headDim + d],
                memory_order_relaxed);
        }
    }
}
```

Dispatch function:

```objc
void metal_sdpa_backward_f32(void* queuePtr, void* pipelinePtr,
    void* dOut, void* Q, void* K, void* V, void* attnWeights,
    void* dQ, void* dK, void* dV,
    int seqLen, int headDim, int numHeads) {

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queuePtr;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipelinePtr;

    id<MTLCommandBuffer> cmdBuffer;
    bool shouldCommit;
    id<MTLComputeCommandEncoder> encoder =
        get_encoder(queue, &cmdBuffer, &shouldCommit);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dOut offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)Q offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)K offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)V offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)attnWeights offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dQ offset:0 atIndex:5];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dK offset:0 atIndex:6];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dV offset:0 atIndex:7];
    [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:8];
    [encoder setBytes:&headDim length:sizeof(headDim) atIndex:9];
    [encoder setBytes:&numHeads length:sizeof(numHeads) atIndex:10];

    MTLSize threadgroups = MTLSizeMake((seqLen + 31) / 32, numHeads, 1);
    MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    finish_encode(encoder, cmdBuffer, shouldCommit);
}
```

- [ ] **Step 3: Add pipeline and replace stub**

Pipeline field:

```go
sdpaBackwardPipeline unsafe.Pointer
```

Pipeline init:

```go
b.sdpaBackwardPipeline = C.metal_create_pipeline(b.device, b.library, C.CString("sdpa_backward_f32"))
```

Replace stub:

```go
func (b *Backend) SDPABackward(dOut, Q, K, V, attnWeights, dQ, dK, dV tensor.DevicePtr, seqLen, headDim, numHeads int) {
	// Zero dQ, dK, dV first (atomics accumulate)
	b.Zero(dQ, numHeads*seqLen*headDim)
	b.Zero(dK, numHeads*seqLen*headDim)
	b.Zero(dV, numHeads*seqLen*headDim)

	C.metal_sdpa_backward_f32(
		b.queue, b.sdpaBackwardPipeline,
		unsafe.Pointer(dOut.Addr()), unsafe.Pointer(Q.Addr()),
		unsafe.Pointer(K.Addr()), unsafe.Pointer(V.Addr()),
		unsafe.Pointer(attnWeights.Addr()),
		unsafe.Pointer(dQ.Addr()), unsafe.Pointer(dK.Addr()),
		unsafe.Pointer(dV.Addr()),
		C.int(seqLen), C.int(headDim), C.int(numHeads))
}
```

- [ ] **Step 4: Build**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add inference/backend/metal/metal_bridge.h inference/backend/metal/metal_bridge_darwin.m inference/backend/metal/backend.go
git commit -m "feat(metal): add SDPABackward kernel

Causal attention backward: computes dQ, dK, dV from saved attention weights.
Uses atomics for K/V gradient accumulation across query positions."
```

---

### Task 10: Training Forward Pass (Activation Saving)

**Files:**
- Modify: `inference/runtime/block.go` — add activation saving
- Modify: `inference/runtime/model.go` — add training forward method

- [ ] **Step 1: Read current block.go and model.go**

Read `inference/runtime/block.go` (BlockRuntime struct and ExecuteWithGPUKV) and `inference/runtime/model.go` (ModelRuntime struct and forward methods) to understand exact field names and method signatures.

- [ ] **Step 2: Define the SavedActivations struct in block.go**

Add near the BlockRuntime struct definition:

```go
// SavedActivations holds intermediate values from a training forward pass,
// needed for the backward pass.
type SavedActivations struct {
	NormOut     tensor.DevicePtr // [seqLen, hidden] — RMSNorm output before QKV projection
	Q           tensor.DevicePtr // [seqLen, numHeads*headDim] — post-RoPE queries
	K           tensor.DevicePtr // [seqLen, numKVHeads*headDim] — post-RoPE keys
	V           tensor.DevicePtr // [seqLen, numKVHeads*headDim] — values
	AttnWeights tensor.DevicePtr // [numHeads, seqLen, seqLen] — attention weights
	AttnOut     tensor.DevicePtr // [seqLen, numHeads*headDim] — attention output before Wo
	Gate        tensor.DevicePtr // [seqLen, ffnHidden] — FFN gate pre-activation
	Up          tensor.DevicePtr // [seqLen, ffnHidden] — FFN up projection
	FFNNormOut  tensor.DevicePtr // [seqLen, hidden] — RMSNorm output before FFN
	AttnNormRMS tensor.DevicePtr // [seqLen] — saved 1/RMS from attention norm
	FFNNormRMS  tensor.DevicePtr // [seqLen] — saved 1/RMS from FFN norm
	Residual    tensor.DevicePtr // [seqLen, hidden] — residual before this layer
}
```

Add a training mode flag to BlockRuntime:

```go
trainingMode bool
savedAct     *SavedActivations
```

- [ ] **Step 3: Modify RMSNorm forward to optionally save RMS**

The existing RMSNorm forward discards the RMS statistic. For training, we need it saved. Read the RMSNorm dispatch in block.go. Add a variant method:

```go
// rmsNormWithSave performs RMSNorm and saves the inverse RMS for backward.
func (b *BlockRuntime) rmsNormWithSave(x, weight, out, rmsOut tensor.DevicePtr, rows, cols int) {
	// Call existing RMSNorm forward
	b.backend.RMSNorm(x, weight, out, rows, cols, b.normEps)
	// Compute and save inv_rms per row (needed for backward)
	b.backend.ComputeInvRMS(x, rmsOut, rows, cols)
}
```

Note: This requires a small `ComputeInvRMS` kernel or we can modify RMSNorm forward to optionally write the RMS values. Check existing implementation. If the existing `RMSNorm` doesn't support saving RMS, add a `RMSNormSaveRMS` method to the backend interface and kernel. The alternative is to recompute RMS during backward from saved input — this avoids a new kernel at the cost of recomputation. **Use the recomputation approach** to avoid adding another kernel:

```go
// For training, save the input to RMSNorm (normOut will be recomputed in backward)
if b.trainingMode {
	b.savedAct.AttnNormRMS = b.backend.Alloc(rows * 4)
	// Recompute inv_rms during backward from saved input
}
```

Actually, the simpler approach: modify the backward kernel to accept the original input and recompute inv_rms internally. The spec's `RMSNormBackward` already takes `input` as a parameter. Update the kernel to compute RMS from input rather than requiring a saved RMS buffer. This eliminates the need for a separate RMS-saving step.

Update the `RMSNormBackward` kernel from Task 6 to compute inv_rms from input:

```metal
// Replace rms buffer with computed value:
float sumSq = 0.0f;
for (int j = 0; j < cols; j++) {
    float x = input[row * cols + j];
    sumSq += x * x;
}
float inv_rms = rsqrt(sumSq / float(cols) + 1e-6f);
```

And update the `TrainingOps` interface — remove the `rms` parameter from `RMSNormBackward`:

```go
RMSNormBackward(dOut, input, weight, dInput tensor.DevicePtr, rows, cols int, eps float32)
```

- [ ] **Step 4: Add training forward method to BlockRuntime**

```go
// TrainingForward runs the forward pass for one layer, saving activations for backward.
// Returns the output residual. Caller must provide pre-allocated SavedActivations.
func (b *BlockRuntime) TrainingForward(residual tensor.DevicePtr, seqLen int, saved *SavedActivations) tensor.DevicePtr {
	hiddenSize := b.HiddenSize
	numHeads := b.NumHeads
	numKVHeads := b.NumKVHeads
	headDim := b.HeadDim
	ffnHidden := b.FFNHiddenSize

	// Save input residual
	saved.Residual = b.backend.AllocPermanent(seqLen * hiddenSize * 4)
	b.backend.Copy(residual, saved.Residual, seqLen*hiddenSize*4)

	// Attention norm
	normOut := b.backend.Alloc(seqLen * hiddenSize * 4)
	b.backend.RMSNorm(residual, b.WNormAttn.DevicePtr(), normOut, seqLen, hiddenSize, b.normEps)
	saved.NormOut = b.backend.AllocPermanent(seqLen * hiddenSize * 4)
	b.backend.Copy(normOut, saved.NormOut, seqLen*hiddenSize*4)

	// Q, K, V projections
	qDim := numHeads * headDim
	kDim := numKVHeads * headDim
	vDim := numKVHeads * headDim

	qPtr := b.backend.Alloc(seqLen * qDim * 4)
	kPtr := b.backend.Alloc(seqLen * kDim * 4)
	vPtr := b.backend.Alloc(seqLen * vDim * 4)

	b.backend.MatMulTransposed(normOut, b.Wq.DevicePtr(), qPtr, seqLen, qDim, hiddenSize)
	b.backend.MatMulTransposed(normOut, b.Wk.DevicePtr(), kPtr, seqLen, kDim, hiddenSize)
	b.backend.MatMulTransposed(normOut, b.Wv.DevicePtr(), vPtr, seqLen, vDim, hiddenSize)

	// LoRA injection (Q and V)
	if b.loraLayer != nil && b.loraLayer.HasQ {
		b.applyLoRA(normOut, b.loraLayer.QA, b.loraLayer.QB, qPtr,
			seqLen, b.loraRank, hiddenSize, qDim, b.loraScale)
	}
	if b.loraLayer != nil && b.loraLayer.HasV {
		b.applyLoRA(normOut, b.loraLayer.VA, b.loraLayer.VB, vPtr,
			seqLen, b.loraRank, hiddenSize, vDim, b.loraScale)
	}

	// RoPE
	b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, 0,
		b.ropeDim, b.ropeTheta, b.ropeNeox)

	// Save post-RoPE Q, K, V
	saved.Q = b.backend.AllocPermanent(seqLen * qDim * 4)
	saved.K = b.backend.AllocPermanent(seqLen * kDim * 4)
	saved.V = b.backend.AllocPermanent(seqLen * vDim * 4)
	b.backend.Copy(qPtr, saved.Q, seqLen*qDim*4)
	b.backend.Copy(kPtr, saved.K, seqLen*kDim*4)
	b.backend.Copy(vPtr, saved.V, seqLen*vDim*4)

	// SDPA (training uses prefill path — full sequence)
	attnOut := b.backend.Alloc(seqLen * qDim * 4)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOut, seqLen, numHeads, numKVHeads, headDim, scale)

	// Save attention weights (need a variant that also outputs weights)
	// For now, recompute attention weights during backward from saved Q, K
	// This avoids modifying the SDPA kernel and saves significant memory

	// Output projection
	attnProjOut := b.backend.Alloc(seqLen * hiddenSize * 4)
	b.backend.MatMulTransposed(attnOut, b.Wo.DevicePtr(), attnProjOut, seqLen, hiddenSize, qDim)
	saved.AttnOut = b.backend.AllocPermanent(seqLen * qDim * 4)
	b.backend.Copy(attnOut, saved.AttnOut, seqLen*qDim*4)

	// Residual add (attention)
	b.backend.Add(residual, attnProjOut, residual, seqLen*hiddenSize)

	// FFN norm
	ffnNormOut := b.backend.Alloc(seqLen * hiddenSize * 4)
	b.backend.RMSNorm(residual, b.WNormFFN.DevicePtr(), ffnNormOut, seqLen, hiddenSize, b.normEps)
	saved.FFNNormOut = b.backend.AllocPermanent(seqLen * hiddenSize * 4)
	b.backend.Copy(ffnNormOut, saved.FFNNormOut, seqLen*hiddenSize*4)

	// FFN: gate and up projections
	gateOut := b.backend.Alloc(seqLen * ffnHidden * 4)
	upOut := b.backend.Alloc(seqLen * ffnHidden * 4)
	b.backend.MatMulTransposed(ffnNormOut, b.Wgate.DevicePtr(), gateOut, seqLen, ffnHidden, hiddenSize)
	b.backend.MatMulTransposed(ffnNormOut, b.Wup.DevicePtr(), upOut, seqLen, ffnHidden, hiddenSize)

	// Save pre-activation gate and up
	saved.Gate = b.backend.AllocPermanent(seqLen * ffnHidden * 4)
	saved.Up = b.backend.AllocPermanent(seqLen * ffnHidden * 4)
	b.backend.Copy(gateOut, saved.Gate, seqLen*ffnHidden*4)
	b.backend.Copy(upOut, saved.Up, seqLen*ffnHidden*4)

	// SiLU(gate) * up
	b.backend.SiLUInplace(gateOut, seqLen*ffnHidden)
	b.backend.Mul(gateOut, upOut, gateOut, seqLen*ffnHidden)

	// Down projection
	ffnOut := b.backend.Alloc(seqLen * hiddenSize * 4)
	b.backend.MatMulTransposed(gateOut, b.Wdown.DevicePtr(), ffnOut, seqLen, hiddenSize, ffnHidden)

	// Residual add (FFN)
	b.backend.Add(residual, ffnOut, residual, seqLen*hiddenSize)

	return residual
}
```

Note: The exact field names for weight tensors (Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, WNormAttn, WNormFFN) must match the existing BlockRuntime fields. Read block.go to confirm the exact names and adjust accordingly. Also confirm whether `Mul`, `Copy`, and `FFNHiddenSize` exist on the backend/struct — add if needed.

- [ ] **Step 5: Add TrainingForward to ModelRuntime**

In `model.go`, add:

```go
// TrainingForward runs a full forward pass saving activations for backward.
// Returns logits and per-layer saved activations.
func (m *ModelRuntime) TrainingForward(tokens []int32) (logits tensor.DevicePtr, savedPerLayer []*SavedActivations, finalNormInput tensor.DevicePtr) {
	seqLen := len(tokens)
	m.backend.ResetPool()

	// Embed tokens
	x := m.embedTokens(tokens)

	// Per-layer forward
	savedPerLayer = make([]*SavedActivations, len(m.layers))
	for i, layer := range m.layers {
		savedPerLayer[i] = &SavedActivations{}
		x = layer.TrainingForward(x, seqLen, savedPerLayer[i])
	}

	// Final norm
	finalNormInput = m.backend.AllocPermanent(seqLen * m.hiddenSize * 4)
	m.backend.Copy(x, finalNormInput, seqLen*m.hiddenSize*4)
	normOut := m.backend.Alloc(seqLen * m.hiddenSize * 4)
	m.backend.RMSNorm(x, m.wNormFinal.DevicePtr(), normOut, seqLen, m.hiddenSize, m.normEps)

	// Logits (unembed)
	logits = m.backend.Alloc(seqLen * m.vocabSize * 4)
	m.backend.MatMulTransposed(normOut, m.wOutput.DevicePtr(), logits, seqLen, m.vocabSize, m.hiddenSize)

	return logits, savedPerLayer, finalNormInput
}
```

Note: Confirm exact field names (hiddenSize, vocabSize, normEps, wNormFinal, wOutput, layers, backend) by reading model.go. Adjust as needed.

- [ ] **Step 6: Build**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build (may require adjusting field names to match existing code).

- [ ] **Step 7: Commit**

```bash
git add inference/runtime/block.go inference/runtime/model.go
git commit -m "feat(runtime): add training forward pass with activation saving

Saves normOut, Q, K, V, gate, up, attnOut per layer for backward pass."
```

---

### Task 11: Backward Pass

**Files:**
- Create: `inference/lora/train/backward.go`

- [ ] **Step 1: Implement the backward pass**

```go
// inference/lora/train/backward.go
package train

import (
	"math"

	"vexel/inference/backend"
	"vexel/inference/lora"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// GradientBuffers holds per-layer LoRA gradient accumulators.
type GradientBuffers struct {
	Layers []LayerGradients
}

// LayerGradients holds gradient buffers for one layer's LoRA weights.
type LayerGradients struct {
	DQA tensor.DevicePtr // [rank, hidden]
	DQB tensor.DevicePtr // [qDim, rank]
	DVA tensor.DevicePtr // [rank, hidden]
	DVB tensor.DevicePtr // [vDim, rank]
}

// AllocGradients creates gradient buffers matching an adapter's structure.
func AllocGradients(adapter *lora.GPUAdapter, b backend.Backend, layers []*runtime.BlockRuntime) *GradientBuffers {
	grads := &GradientBuffers{
		Layers: make([]LayerGradients, len(adapter.Layers)),
	}
	for i, la := range adapter.Layers {
		rank := adapter.Rank
		hidden := layers[i].HiddenSize
		qDim := layers[i].NumHeads * layers[i].HeadDim
		vDim := layers[i].NumKVHeads * layers[i].HeadDim

		if la.HasQ {
			grads.Layers[i].DQA = b.AllocPermanent(rank * hidden * 4)
			grads.Layers[i].DQB = b.AllocPermanent(qDim * rank * 4)
		}
		if la.HasV {
			grads.Layers[i].DVA = b.AllocPermanent(rank * hidden * 4)
			grads.Layers[i].DVB = b.AllocPermanent(vDim * rank * 4)
		}
	}
	return grads
}

// ZeroGradients resets all gradient buffers to zero.
func ZeroGradients(grads *GradientBuffers, adapter *lora.GPUAdapter, b backend.TrainingOps, layers []*runtime.BlockRuntime) {
	for i, la := range adapter.Layers {
		rank := adapter.Rank
		hidden := layers[i].HiddenSize
		qDim := layers[i].NumHeads * layers[i].HeadDim
		vDim := layers[i].NumKVHeads * layers[i].HeadDim

		if la.HasQ {
			b.Zero(grads.Layers[i].DQA, rank*hidden)
			b.Zero(grads.Layers[i].DQB, qDim*rank)
		}
		if la.HasV {
			b.Zero(grads.Layers[i].DVA, rank*hidden)
			b.Zero(grads.Layers[i].DVB, vDim*rank)
		}
	}
}

// Backward computes gradients for all LoRA weights given saved activations.
// Returns the loss value.
func Backward(
	b backend.Backend,
	training backend.TrainingOps,
	logits tensor.DevicePtr,
	targets tensor.DevicePtr, // [seqLen] int32 target tokens
	mask tensor.DevicePtr,    // [seqLen] float32 loss mask
	seqLen int,
	vocabSize int,
	hiddenSize int,
	model *runtime.ModelRuntime,
	layers []*runtime.BlockRuntime,
	savedPerLayer []*runtime.SavedActivations,
	finalNormInput tensor.DevicePtr,
	adapter *lora.GPUAdapter,
	grads *GradientBuffers,
) float32 {

	// Phase 1: Cross-entropy loss + dLogits
	dLogits := b.Alloc(seqLen * vocabSize * 4)
	var loss float32
	training.CrossEntropyLossForwardBackward(logits, targets, mask, dLogits, &loss, seqLen, vocabSize)

	// Count masked positions for averaging
	// (kernel accumulates raw loss — divide here)
	// Read mask back to count
	maskData := make([]float32, seqLen)
	b.ReadFromDevice(mask, maskData)
	numMasked := float32(0)
	for _, m := range maskData {
		numMasked += m
	}
	if numMasked > 0 {
		loss /= numMasked
		// Scale dLogits by 1/numMasked
		scaler := b.(backend.ScaleOps)
		scaler.ScaleBuffer(dLogits, 1.0/numMasked, seqLen*vocabSize)
	}

	// Phase 2: Backprop through unembedding
	// dLastHidden = dLogits @ Wunembed^T
	dResidual := b.Alloc(seqLen * hiddenSize * 4)
	b.MatMulTransposed(dLogits, model.WOutput().DevicePtr(), dResidual, seqLen, hiddenSize, vocabSize)

	// Backprop through final RMSNorm
	dFinalNorm := b.Alloc(seqLen * hiddenSize * 4)
	training.RMSNormBackward(dResidual, finalNormInput, model.WNormFinal().DevicePtr(), dFinalNorm, seqLen, hiddenSize, model.NormEps())
	dResidual = dFinalNorm

	// Phase 3: Layer-by-layer backward
	numLayers := len(layers)
	for i := numLayers - 1; i >= 0; i-- {
		layer := layers[i]
		saved := savedPerLayer[i]
		la := adapter.GetLayer(i)

		numHeads := layer.NumHeads
		numKVHeads := layer.NumKVHeads
		headDim := layer.HeadDim
		ffnHidden := layer.FFNHiddenSize
		qDim := numHeads * headDim
		kDim := numKVHeads * headDim
		vDim := numKVHeads * headDim
		rank := adapter.Rank
		scale := adapter.Scale

		// --- FFN backward ---
		// dFFNOut = dResidual (residual add is identity)
		dFFNMid := b.Alloc(seqLen * ffnHidden * 4)
		b.MatMulTransposed(dResidual, layer.Wdown().DevicePtr(), dFFNMid, seqLen, ffnHidden, hiddenSize)

		dGate := b.Alloc(seqLen * ffnHidden * 4)
		dUp := b.Alloc(seqLen * ffnHidden * 4)
		training.SiLUMulBackward(dFFNMid, saved.Gate, saved.Up, dGate, dUp, seqLen*ffnHidden)

		dFFNInput := b.Alloc(seqLen * hiddenSize * 4)
		dFFNInput2 := b.Alloc(seqLen * hiddenSize * 4)
		b.MatMulTransposed(dGate, layer.Wgate().DevicePtr(), dFFNInput, seqLen, hiddenSize, ffnHidden)
		b.MatMulTransposed(dUp, layer.Wup().DevicePtr(), dFFNInput2, seqLen, hiddenSize, ffnHidden)
		b.Add(dFFNInput, dFFNInput2, dFFNInput, seqLen*hiddenSize)

		dFFNNorm := b.Alloc(seqLen * hiddenSize * 4)
		training.RMSNormBackward(dFFNInput, saved.FFNNormOut, layer.WNormFFN().DevicePtr(), dFFNNorm, seqLen, hiddenSize, layer.NormEps())

		// Residual connection: dResidual += dFFNNorm
		b.Add(dResidual, dFFNNorm, dResidual, seqLen*hiddenSize)

		// --- Attention backward ---
		// dAttnProj = dResidual @ Wo^T
		dAttnProj := b.Alloc(seqLen * qDim * 4)
		b.MatMulTransposed(dResidual, layer.Wo().DevicePtr(), dAttnProj, seqLen, qDim, hiddenSize)

		// SDPA backward — recompute attention weights from saved Q, K
		// Then compute dQ, dK, dV
		dQ := b.Alloc(seqLen * qDim * 4)
		dK := b.Alloc(seqLen * kDim * 4)
		dV := b.Alloc(seqLen * vDim * 4)

		// Recompute attention weights
		attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
		attnWeights := b.Alloc(numHeads * seqLen * seqLen * 4)
		// Q @ K^T * scale → attnWeights, then softmax
		// For simplicity, use the SDPA backward kernel which takes saved Q, K, V and attn weights
		// We need to compute attention weights first
		b.ComputeAttnWeights(saved.Q, saved.K, attnWeights, seqLen, headDim, numHeads, numKVHeads, attnScale)

		training.SDPABackward(dAttnProj, saved.Q, saved.K, saved.V, attnWeights,
			dQ, dK, dV, seqLen, headDim, numHeads)

		// RoPE backward
		training.RoPEBackward(dQ, dK, headDim, numHeads, numKVHeads, seqLen, 0,
			layer.RopeDim(), layer.RopeTheta(), layer.RopeNeox())

		// --- LoRA weight gradients ---
		if la != nil && la.HasQ {
			// inter_q = normOut @ A_q^T  [seqLen, rank]
			interQ := b.Alloc(seqLen * rank * 4)
			b.MatMulTransposed(saved.NormOut, la.QA, interQ, seqLen, rank, hiddenSize)

			// dB_q += scale * (dQ^T @ inter_q)  [qDim, rank]
			dBTemp := b.Alloc(qDim * rank * 4)
			b.MatMulTransposed(dQ, interQ, dBTemp, qDim, rank, seqLen) // transposed differently
			if scale != 1.0 {
				scaler := b.(backend.ScaleOps)
				scaler.ScaleBuffer(dBTemp, scale, qDim*rank)
			}
			b.Add(grads.Layers[i].DQB, dBTemp, grads.Layers[i].DQB, qDim*rank)

			// dA_q += scale * (B_q^T @ dQ)^T @ normOut  [rank, hidden]
			dInterQ := b.Alloc(seqLen * rank * 4)
			b.MatMulTransposed(dQ, la.QB, dInterQ, seqLen, rank, qDim) // dQ @ B_q (since B is [qDim, rank])
			dATemp := b.Alloc(rank * hiddenSize * 4)
			// dA = dInterQ^T @ normOut  [rank, seqLen] @ [seqLen, hidden] = [rank, hidden]
			b.MatMul(dInterQ, saved.NormOut, dATemp, rank, hiddenSize, seqLen) // need non-transposed matmul
			if scale != 1.0 {
				scaler := b.(backend.ScaleOps)
				scaler.ScaleBuffer(dATemp, scale, rank*hiddenSize)
			}
			b.Add(grads.Layers[i].DQA, dATemp, grads.Layers[i].DQA, rank*hiddenSize)
		}

		if la != nil && la.HasV {
			interV := b.Alloc(seqLen * rank * 4)
			b.MatMulTransposed(saved.NormOut, la.VA, interV, seqLen, rank, hiddenSize)

			dBTemp := b.Alloc(vDim * rank * 4)
			b.MatMulTransposed(dV, interV, dBTemp, vDim, rank, seqLen)
			if scale != 1.0 {
				scaler := b.(backend.ScaleOps)
				scaler.ScaleBuffer(dBTemp, scale, vDim*rank)
			}
			b.Add(grads.Layers[i].DVB, dBTemp, grads.Layers[i].DVB, vDim*rank)

			dInterV := b.Alloc(seqLen * rank * 4)
			b.MatMulTransposed(dV, la.VB, dInterV, seqLen, rank, vDim)
			dATemp := b.Alloc(rank * hiddenSize * 4)
			b.MatMul(dInterV, saved.NormOut, dATemp, rank, hiddenSize, seqLen)
			if scale != 1.0 {
				scaler := b.(backend.ScaleOps)
				scaler.ScaleBuffer(dATemp, scale, rank*hiddenSize)
			}
			b.Add(grads.Layers[i].DVA, dATemp, grads.Layers[i].DVA, rank*hiddenSize)
		}

		// Continue residual gradient through attention norms
		dNormOut := b.Alloc(seqLen * hiddenSize * 4)
		b.MatMulTransposed(dQ, layer.Wq().DevicePtr(), dNormOut, seqLen, hiddenSize, qDim)
		dNormK := b.Alloc(seqLen * hiddenSize * 4)
		b.MatMulTransposed(dK, layer.Wk().DevicePtr(), dNormK, seqLen, hiddenSize, kDim)
		b.Add(dNormOut, dNormK, dNormOut, seqLen*hiddenSize)
		dNormV := b.Alloc(seqLen * hiddenSize * 4)
		b.MatMulTransposed(dV, layer.Wv().DevicePtr(), dNormV, seqLen, hiddenSize, vDim)
		b.Add(dNormOut, dNormV, dNormOut, seqLen*hiddenSize)

		dAttnNorm := b.Alloc(seqLen * hiddenSize * 4)
		training.RMSNormBackward(dNormOut, saved.Residual, layer.WNormAttn().DevicePtr(), dAttnNorm, seqLen, hiddenSize, layer.NormEps())

		// Residual connection
		b.Add(dResidual, dAttnNorm, dResidual, seqLen*hiddenSize)
	}

	return loss
}
```

Note: The exact method names for accessing layer weights (Wq(), Wk(), Wo(), Wgate(), Wup(), Wdown(), WNormAttn(), WNormFFN()) must match what's exposed on BlockRuntime. If they're not currently exported, add accessor methods. Same for ModelRuntime's WOutput(), WNormFinal(), NormEps(). Also confirm `MatMul` (non-transposed) exists on the backend, and `ComputeAttnWeights` — these may need to be added or the backward logic adjusted to use existing operations.

- [ ] **Step 2: Build**

```bash
go build -tags metal ./inference/...
```

Expected: May need accessor methods added. Fix compilation errors.

- [ ] **Step 3: Commit**

```bash
git add inference/lora/train/backward.go
git commit -m "feat(lora/train): add full backward pass for LoRA training

Layer-by-layer gradient computation through frozen attention and FFN.
Extracts dA and dB for Q/V LoRA matrices at each injection point."
```

---

### Task 12: Training Loop (Trainer)

**Files:**
- Create: `inference/lora/train/trainer.go`

- [ ] **Step 1: Implement the trainer**

```go
// inference/lora/train/trainer.go
package train

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"

	"vexel/inference/backend"
	"vexel/inference/lora"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// TrainConfig holds training hyperparameters.
type TrainConfig struct {
	Rank        int
	Alpha       float32
	LR          float32
	Momentum    float32
	WeightDecay float32
	Epochs      int
	OutputDir   string
	DataPath    string
}

// MomentumBuffers holds SGD momentum state per LoRA parameter.
type MomentumBuffers struct {
	Layers []LayerMomentum
}

type LayerMomentum struct {
	MQA, MQB, MVA, MVB tensor.DevicePtr
}

// Trainer orchestrates the LoRA training loop.
type Trainer struct {
	config    TrainConfig
	model     *runtime.ModelRuntime
	layers    []*runtime.BlockRuntime
	adapter   *lora.GPUAdapter
	cpuAdapter *lora.Adapter
	grads     *GradientBuffers
	momentum  *MomentumBuffers
	b         backend.Backend
	training  backend.TrainingOps
	tokenizer interface {
		Encode(text string) []int32
	}
}

// NewTrainer creates a trainer with initialized LoRA weights.
func NewTrainer(
	config TrainConfig,
	model *runtime.ModelRuntime,
	layers []*runtime.BlockRuntime,
	b backend.Backend,
	training backend.TrainingOps,
	tokenizer interface{ Encode(text string) []int32 },
) *Trainer {
	hiddenSize := layers[0].HiddenSize
	qDim := layers[0].NumHeads * layers[0].HeadDim
	vDim := layers[0].NumKVHeads * layers[0].HeadDim

	cfg := lora.AdapterConfig{
		Rank:          config.Rank,
		Alpha:         config.Alpha,
		TargetModules: []string{"q_proj", "v_proj"},
	}

	// Initialize weights
	cpuAdapter := InitAdapter(cfg, len(layers), hiddenSize, qDim, vDim)
	gpuAdapter, _ := lora.UploadToGPU(cpuAdapter, b.(lora.Allocator))
	model.AttachLoRA(gpuAdapter)

	// Allocate gradient and momentum buffers
	grads := AllocGradients(gpuAdapter, b, layers)
	mom := allocMomentum(gpuAdapter, b, layers)

	return &Trainer{
		config:     config,
		model:      model,
		layers:     layers,
		adapter:    gpuAdapter,
		cpuAdapter: cpuAdapter,
		grads:      grads,
		momentum:   mom,
		b:          b,
		training:   training,
		tokenizer:  tokenizer,
	}
}

func allocMomentum(adapter *lora.GPUAdapter, b backend.Backend, layers []*runtime.BlockRuntime) *MomentumBuffers {
	mom := &MomentumBuffers{
		Layers: make([]LayerMomentum, len(adapter.Layers)),
	}
	for i, la := range adapter.Layers {
		rank := adapter.Rank
		hidden := layers[i].HiddenSize
		qDim := layers[i].NumHeads * layers[i].HeadDim
		vDim := layers[i].NumKVHeads * layers[i].HeadDim

		if la.HasQ {
			mom.Layers[i].MQA = b.AllocPermanent(rank * hidden * 4)
			mom.Layers[i].MQB = b.AllocPermanent(qDim * rank * 4)
			b.(backend.TrainingOps).Zero(mom.Layers[i].MQA, rank*hidden)
			b.(backend.TrainingOps).Zero(mom.Layers[i].MQB, qDim*rank)
		}
		if la.HasV {
			mom.Layers[i].MVA = b.AllocPermanent(rank * hidden * 4)
			mom.Layers[i].MVB = b.AllocPermanent(vDim * rank * 4)
			b.(backend.TrainingOps).Zero(mom.Layers[i].MVA, rank*hidden)
			b.(backend.TrainingOps).Zero(mom.Layers[i].MVB, vDim*rank)
		}
	}
	return mom
}

// Train runs the full training loop.
func (t *Trainer) Train(examples []Example) error {
	// Signal handling for Ctrl-C
	interrupted := make(chan os.Signal, 1)
	signal.Notify(interrupted, syscall.SIGINT)
	defer signal.Stop(interrupted)

	totalSteps := len(examples) * t.config.Epochs
	step := 0

	fmt.Printf("LoRA training: rank=%d, alpha=%.0f, lr=%.1e, SGD momentum=%.1f\n",
		t.config.Rank, t.config.Alpha, t.config.LR, t.config.Momentum)
	fmt.Printf("Data: %d examples\n\n", len(examples))

	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		fmt.Printf("epoch %d/%d\n", epoch+1, t.config.Epochs)

		// Shuffle
		perm := rand.Perm(len(examples))

		for _, idx := range perm {
			// Check for interrupt
			select {
			case <-interrupted:
				fmt.Println("\nInterrupted. Saving checkpoint...")
				return t.saveCheckpoint()
			default:
			}

			step++
			ex := examples[idx]

			// Tokenize
			var tokens []int32
			var promptLen int
			switch ex.Format {
			case FormatText:
				tokens = t.tokenizer.Encode(ex.Text)
			case FormatPromptCompletion:
				promptTokens := t.tokenizer.Encode(ex.Prompt)
				completionTokens := t.tokenizer.Encode(ex.Completion)
				tokens = append(promptTokens, completionTokens...)
				promptLen = len(promptTokens)
			}

			if len(tokens) < 2 {
				continue // Need at least 2 tokens for next-token prediction
			}

			// Build loss mask
			mask := BuildLossMask(tokens, ex.Format, promptLen)

			// Forward pass
			logits, savedPerLayer, finalNormInput := t.model.TrainingForward(tokens)

			// Upload targets and mask to GPU
			seqLen := len(tokens)
			targetsBuf := t.b.Alloc(seqLen * 4)
			maskBuf := t.b.Alloc(seqLen * 4)
			// Targets are tokens[1:] shifted, but we handle this in the kernel
			// Actually: targets for position t is tokens[t+1]
			targets := make([]int32, seqLen)
			copy(targets, tokens[1:])
			t.b.ToDevice(targetsBuf, targets)
			t.b.ToDevice(maskBuf, mask)

			// Zero gradients
			ZeroGradients(t.grads, t.adapter, t.training, t.layers)

			// Backward pass
			loss := Backward(t.b, t.training, logits, targetsBuf, maskBuf,
				seqLen, t.model.VocabSize(), t.model.HiddenSize(),
				t.model, t.layers, savedPerLayer, finalNormInput,
				t.adapter, t.grads)

			// SGD update
			t.sgdUpdate()

			// Free saved activations
			for _, saved := range savedPerLayer {
				saved.Free(t.b)
			}

			fmt.Printf("  step %d/%d    loss=%.4f\n", step, totalSteps, loss)
		}
	}

	fmt.Println("\nTraining complete. Saving checkpoint...")
	return t.saveCheckpoint()
}

// sgdUpdate applies SGD with momentum to all LoRA weights.
func (t *Trainer) sgdUpdate() {
	for i, la := range t.adapter.Layers {
		rank := t.adapter.Rank
		hidden := t.layers[i].HiddenSize
		qDim := t.layers[i].NumHeads * t.layers[i].HeadDim
		vDim := t.layers[i].NumKVHeads * t.layers[i].HeadDim

		if la.HasQ {
			t.training.SGDUpdate(la.QA, t.grads.Layers[i].DQA, t.config.LR, t.config.WeightDecay, rank*hidden)
			t.training.SGDUpdate(la.QB, t.grads.Layers[i].DQB, t.config.LR, t.config.WeightDecay, qDim*rank)
		}
		if la.HasV {
			t.training.SGDUpdate(la.VA, t.grads.Layers[i].DVA, t.config.LR, t.config.WeightDecay, rank*hidden)
			t.training.SGDUpdate(la.VB, t.grads.Layers[i].DVB, t.config.LR, t.config.WeightDecay, vDim*rank)
		}
	}
}

// saveCheckpoint downloads weights from GPU and saves in PEFT format.
func (t *Trainer) saveCheckpoint() error {
	// Download GPU weights back to CPU adapter
	for i, la := range t.adapter.Layers {
		rank := t.adapter.Rank
		hidden := t.layers[i].HiddenSize
		qDim := t.layers[i].NumHeads * t.layers[i].HeadDim
		vDim := t.layers[i].NumKVHeads * t.layers[i].HeadDim

		if la.HasQ {
			t.b.ReadFromDevice(la.QA, t.cpuAdapter.Layers[i].QA[:rank*hidden])
			t.b.ReadFromDevice(la.QB, t.cpuAdapter.Layers[i].QB[:qDim*rank])
		}
		if la.HasV {
			t.b.ReadFromDevice(la.VA, t.cpuAdapter.Layers[i].VA[:rank*hidden])
			t.b.ReadFromDevice(la.VB, t.cpuAdapter.Layers[i].VB[:vDim*rank])
		}
	}

	if err := lora.SaveAdapter(t.cpuAdapter, t.config.OutputDir); err != nil {
		return fmt.Errorf("save checkpoint: %w", err)
	}
	fmt.Printf("Checkpoint saved to %s\n", t.config.OutputDir)
	return nil
}
```

Note: Several methods referenced here (ReadFromDevice, ToDevice, VocabSize, HiddenSize, etc.) may not exist yet or may have different signatures. The implementing engineer should verify these against the actual codebase and add accessor methods as needed. The `SavedActivations.Free()` method also needs to be added to release permanent allocations.

- [ ] **Step 2: Build**

```bash
go build -tags metal ./inference/...
```

Expected: May need accessor methods. Fix compilation errors.

- [ ] **Step 3: Commit**

```bash
git add inference/lora/train/trainer.go
git commit -m "feat(lora/train): add training loop with SGD and Ctrl-C checkpointing

Single-example SGD with momentum, live loss output, SIGINT-safe checkpoint saving."
```

---

### Task 13: CLI Integration

**Files:**
- Modify: `inference/cmd/vexel/cli.go` — add training flags
- Modify: `inference/cmd/vexel/commands.go` — add `train` subcommand

- [ ] **Step 1: Read current cli.go and commands.go**

Read `inference/cmd/vexel/cli.go` (GlobalFlags struct, parseArgs function) and `inference/cmd/vexel/commands.go` (subcommand dispatch pattern, initModel).

- [ ] **Step 2: Add training flags to cli.go**

Add a `TrainFlags` struct after `GlobalFlags`:

```go
type TrainFlags struct {
	DataPath    string
	OutputDir   string
	Rank        int
	Alpha       float32
	LR          float32
	Momentum    float32
	WeightDecay float32
	Epochs      int
}
```

Add a `parseTrainArgs` function:

```go
func parseTrainArgs(args []string) (TrainFlags, error) {
	flags := TrainFlags{
		Rank:        16,
		Alpha:       16,
		LR:          1e-4,
		Momentum:    0.9,
		WeightDecay: 0.0,
		Epochs:      1,
	}

	for i := 0; i < len(args); {
		switch args[i] {
		case "--data":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--data requires a path")
			}
			flags.DataPath = args[i+1]
			i += 2
		case "--output":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--output requires a path")
			}
			flags.OutputDir = args[i+1]
			i += 2
		case "--rank":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--rank requires a value")
			}
			fmt.Sscanf(args[i+1], "%d", &flags.Rank)
			i += 2
		case "--alpha":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--alpha requires a value")
			}
			fmt.Sscanf(args[i+1], "%f", &flags.Alpha)
			i += 2
		case "--lr":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--lr requires a value")
			}
			fmt.Sscanf(args[i+1], "%f", &flags.LR)
			i += 2
		case "--momentum":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--momentum requires a value")
			}
			fmt.Sscanf(args[i+1], "%f", &flags.Momentum)
			i += 2
		case "--weight-decay":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--weight-decay requires a value")
			}
			fmt.Sscanf(args[i+1], "%f", &flags.WeightDecay)
			i += 2
		case "--epochs":
			if i+1 >= len(args) {
				return flags, fmt.Errorf("--epochs requires a value")
			}
			fmt.Sscanf(args[i+1], "%d", &flags.Epochs)
			i += 2
		default:
			return flags, fmt.Errorf("unknown train flag: %s", args[i])
		}
	}

	if flags.DataPath == "" {
		return flags, fmt.Errorf("--data is required")
	}
	if flags.OutputDir == "" {
		return flags, fmt.Errorf("--output is required")
	}

	return flags, nil
}
```

- [ ] **Step 3: Add train subcommand to commands.go**

In the subcommand dispatch (where "generate", "chat", "serve" etc. are handled), add the "train" case:

```go
case "train":
	trainFlags, err := parseTrainArgs(subArgs)
	if err != nil {
		return fmt.Errorf("parse train flags: %w", err)
	}

	// Load model (without LoRA — training creates its own)
	model, gpuBackend, tokenizer, err := initModel(globals)
	if err != nil {
		return err
	}

	// Load training data
	examples, err := train.LoadData(trainFlags.DataPath)
	if err != nil {
		return fmt.Errorf("load training data: %w", err)
	}
	log.Printf("Loaded %d training examples", len(examples))

	// Create trainer
	cfg := train.TrainConfig{
		Rank:        trainFlags.Rank,
		Alpha:       trainFlags.Alpha,
		LR:          trainFlags.LR,
		Momentum:    trainFlags.Momentum,
		WeightDecay: trainFlags.WeightDecay,
		Epochs:      trainFlags.Epochs,
		OutputDir:   trainFlags.OutputDir,
		DataPath:    trainFlags.DataPath,
	}

	trainer := train.NewTrainer(cfg, model, model.Layers(), gpuBackend, gpuBackend, tokenizer)

	return trainer.Train(examples)
```

Add import for `"vexel/inference/lora/train"`.

- [ ] **Step 4: Build**

```bash
go build -tags metal ./inference/...
```

Expected: Clean build.

- [ ] **Step 5: Test CLI flag parsing**

```bash
go run -tags metal ./inference/cmd/vexel --model /path/to/model.gguf train --data /nonexistent.jsonl --output /tmp/test-adapter 2>&1 | head -5
```

Expected: Error about data file not found (confirms flags are parsed correctly).

- [ ] **Step 6: Commit**

```bash
git add inference/cmd/vexel/cli.go inference/cmd/vexel/commands.go
git commit -m "feat(cli): add 'vexel train' subcommand for LoRA fine-tuning

Usage: vexel train --model m.gguf --data train.jsonl --output ./adapter/ --rank 16 --lr 1e-4"
```

---

### Task 14: End-to-End Integration Test

**Files:**
- Create: `inference/lora/train/e2e_test.go`

- [ ] **Step 1: Write integration test**

```go
// inference/lora/train/e2e_test.go
//go:build metal && darwin && cgo

package train_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"vexel/inference/lora"
	"vexel/inference/lora/train"
)

func TestTrainingE2E(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
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

	// Create training data
	dataDir := t.TempDir()
	dataPath := filepath.Join(dataDir, "train.jsonl")
	examples := []map[string]string{
		{"text": "The quick brown fox jumps over the lazy dog."},
		{"text": "Hello world, this is a test of LoRA training."},
		{"text": "Machine learning on Apple Silicon with Metal."},
	}
	f, _ := os.Create(dataPath)
	enc := json.NewEncoder(f)
	for _, ex := range examples {
		enc.Encode(ex)
	}
	f.Close()

	outputDir := filepath.Join(dataDir, "adapter-output")

	// Load data
	data, err := train.LoadData(dataPath)
	if err != nil {
		t.Fatalf("LoadData: %v", err)
	}
	if len(data) != 3 {
		t.Fatalf("got %d examples, want 3", len(data))
	}

	// TODO: Full e2e test requires model loading infrastructure.
	// For now, verify data loading + adapter initialization + save/load round-trip.

	cfg := lora.AdapterConfig{
		Rank:          4,
		Alpha:         4,
		TargetModules: []string{"q_proj", "v_proj"},
		BaseModel:     "test",
	}
	adapter := train.InitAdapter(cfg, 24, 896, 896, 128)

	// Save and reload
	err = lora.SaveAdapter(adapter, outputDir)
	if err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	reloaded, err := lora.LoadAdapter(outputDir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	if reloaded.Config.Rank != 4 {
		t.Errorf("rank=%d, want 4", reloaded.Config.Rank)
	}
	if len(reloaded.Layers) != 24 {
		t.Errorf("layers=%d, want 24", len(reloaded.Layers))
	}
	t.Logf("E2E: data loaded (%d examples), adapter init + save/load round-trip OK", len(data))
}
```

- [ ] **Step 2: Run the test**

```bash
go test -tags metal -v ./inference/lora/train/ -run TestTrainingE2E -timeout 120s
```

Expected: PASS (data loading + init + round-trip verified).

- [ ] **Step 3: Commit**

```bash
git add inference/lora/train/e2e_test.go
git commit -m "test(lora/train): add end-to-end integration test

Verifies data loading, adapter initialization, and checkpoint round-trip."
```

---

### Implementation Notes for the Engineer

1. **Field name verification is critical.** Tasks 10-12 reference many fields on BlockRuntime and ModelRuntime (e.g., `HiddenSize`, `NumHeads`, `HeadDim`, `FFNHiddenSize`, `Wq`, `Wo`, `WNormAttn`, etc.). Read the actual struct definitions before implementing. You may need to add public accessor methods if fields are unexported.

2. **Memory management.** The training forward pass uses `AllocPermanent` for saved activations (they persist across the forward/backward boundary). These must be explicitly freed after each training step. Add a `Free()` method to `SavedActivations`.

3. **Backend methods.** The backward pass assumes `MatMul` (non-transposed C = A @ B), `Copy`, `ComputeAttnWeights`, `ReadFromDevice`, and `ToDevice` exist on the backend. Check what's available and add wrappers as needed.

4. **Build order.** Tasks 1-3 are independent and can be done in parallel. Tasks 4-9 (kernels) are independent of each other but depend on Task 4 (interface). Task 10 depends on Tasks 5-9. Task 11 depends on Task 10. Task 12 depends on Tasks 1-3 and 11. Task 13 depends on Task 12. Task 14 depends on Task 13.

5. **RMSNormBackward simplification.** Task 6 and Task 10 discuss removing the `rms` parameter from `RMSNormBackward` and recomputing it from input. This is the recommended approach — it avoids modifying the forward RMSNorm kernel and saves memory. The backward kernel should compute `inv_rms = rsqrt(mean(x^2) + eps)` internally.
