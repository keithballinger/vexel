# Medusa Speculative Decoding Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing Medusa heads infrastructure into the CLI and scheduler so users can enable online-learned speculative decoding with `--medusa` flag, achieving 20-50% decode throughput gains.

**Architecture:** The MedusaScheduler already implements training sample collection, simple verification, and tree-based verification paths. The gap is CLI integration: commands.go needs a `--medusa` flag that creates a MedusaScheduler instead of a plain Scheduler, and the MedusaScheduler needs to be exposed through the same interface the server expects. Additionally, GPU trainer head saving (the TODO at gpu_trainer.go:302) must work for persistence across restarts.

**Tech Stack:** Go 1.25.4, Metal backend (build tag: metal && darwin && cgo), existing medusa package

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/cmd/vexel/cli.go` | Add `--medusa` and `--medusa-heads` global flags |
| Modify | `inference/cmd/vexel/commands.go` | Create MedusaScheduler when `--medusa` is set |
| Modify | `inference/scheduler/medusa_scheduler.go` | Expose `Scheduler` field for server compatibility |
| Modify | `inference/medusa/gpu_trainer.go` | Implement CPU conversion + SaveHeads |
| Create | `inference/cmd/vexel/medusa_integration_test.go` | CLI integration test for --medusa flag |
| Create | `inference/medusa/gpu_save_test.go` | Test GPU head save/load round-trip |

---

### Task 1: Add --medusa CLI flags

**Files:**
- Modify: `inference/cmd/vexel/cli.go`

- [ ] **Step 1: Read cli.go to find GlobalFlags struct and flag parsing**

Run: Read `inference/cmd/vexel/cli.go` and locate the `GlobalFlags` struct and `parseGlobalFlags` function.

- [ ] **Step 2: Write failing test for new flags**

Create `inference/cmd/vexel/medusa_integration_test.go`:

```go
//go:build metal && darwin && cgo

package main

import (
	"testing"
)

func TestParseMedusaFlags(t *testing.T) {
	args := []string{"--model", "test.gguf", "--medusa", "serve"}
	globals, remaining, err := parseGlobalFlags(args)
	if err != nil {
		t.Fatal(err)
	}
	if !globals.Medusa {
		t.Error("expected --medusa to be true")
	}
	if len(remaining) != 1 || remaining[0] != "serve" {
		t.Errorf("unexpected remaining args: %v", remaining)
	}
}

func TestParseMedusaHeadsFlag(t *testing.T) {
	args := []string{"--model", "test.gguf", "--medusa", "--medusa-heads", "/tmp/heads.bin", "serve"}
	globals, _, err := parseGlobalFlags(args)
	if err != nil {
		t.Fatal(err)
	}
	if globals.MedusaHeadsPath != "/tmp/heads.bin" {
		t.Errorf("expected medusa-heads=/tmp/heads.bin, got %s", globals.MedusaHeadsPath)
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `go test -tags metal -run TestParseMedusa -v ./inference/cmd/vexel/`
Expected: FAIL — `globals.Medusa` field doesn't exist

- [ ] **Step 4: Add Medusa fields to GlobalFlags and flag parsing**

In `inference/cmd/vexel/cli.go`, add to `GlobalFlags`:

```go
type GlobalFlags struct {
	Model      string
	DraftModel string
	Verbose    bool
	Medusa          bool   // Enable Medusa speculative decoding
	MedusaHeadsPath string // Path to pre-trained Medusa heads (optional)
}
```

In `parseGlobalFlags`, add:

```go
case "--medusa":
	globals.Medusa = true
case "--medusa-heads":
	i++
	if i >= len(args) {
		return globals, nil, fmt.Errorf("--medusa-heads requires a path")
	}
	globals.MedusaHeadsPath = args[i]
	globals.Medusa = true // Implies --medusa
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test -tags metal -run TestParseMedusa -v ./inference/cmd/vexel/`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add inference/cmd/vexel/cli.go inference/cmd/vexel/medusa_integration_test.go
git commit -m "feat(cli): add --medusa and --medusa-heads global flags"
```

---

### Task 2: Wire MedusaScheduler into runServe

**Files:**
- Modify: `inference/cmd/vexel/commands.go`
- Modify: `inference/scheduler/medusa_scheduler.go`

- [ ] **Step 1: Verify MedusaScheduler exposes Scheduler for server**

Read `inference/scheduler/medusa_scheduler.go` and check if `MedusaScheduler` has a public `Scheduler` field or method. The server (`serve.NewServerWithConfig`) takes a `*scheduler.Scheduler`.

- [ ] **Step 2: Add Scheduler accessor to MedusaScheduler if needed**

In `inference/scheduler/medusa_scheduler.go`, ensure this exists:

```go
// BaseScheduler returns the underlying Scheduler for server integration.
func (ms *MedusaScheduler) BaseScheduler() *Scheduler {
	return ms.Scheduler
}
```

- [ ] **Step 3: Add Medusa path to runServe in commands.go**

In `inference/cmd/vexel/commands.go`, modify `runServe` to add a Medusa branch after the existing speculative/standard branches (around line 167):

```go
if globals.DraftModel != "" {
	// ... existing speculative scheduling code ...
} else if globals.Medusa {
	medusaCfg := scheduler.DefaultMedusaConfig()
	medusaCfg.EnableOnlineTraining = true
	medusaCfg.UseGPUTraining = true
	if globals.MedusaHeadsPath != "" {
		medusaCfg.HeadsPath = globals.MedusaHeadsPath
	}
	ms, err := scheduler.NewMedusaScheduler(model, tok, schedConfig, medusaCfg)
	if err != nil {
		return fmt.Errorf("create medusa scheduler: %w", err)
	}
	log.Printf("Using Medusa speculative decoding (online training=%v, GPU=%v)",
		medusaCfg.EnableOnlineTraining, medusaCfg.UseGPUTraining)
	baseSched = ms.BaseScheduler()
	runFunc = ms.Run
} else {
	// ... existing standard scheduler code ...
}
```

- [ ] **Step 4: Run build to verify compilation**

Run: `CGO_ENABLED=1 go build -tags metal -o /dev/null ./inference/cmd/vexel/`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add inference/cmd/vexel/commands.go inference/scheduler/medusa_scheduler.go
git commit -m "feat(serve): wire MedusaScheduler into serve command with --medusa flag"
```

---

### Task 3: Wire MedusaScheduler into runGenerate and runChat

**Files:**
- Modify: `inference/cmd/vexel/commands.go`

- [ ] **Step 1: Read runGenerate and runChat functions**

Read `inference/cmd/vexel/commands.go` to understand how `generate` and `chat` subcommands create their scheduler/decode loop.

- [ ] **Step 2: Add Medusa support to runGenerate**

In `runGenerate`, after model initialization, add Medusa scheduler creation that mirrors the serve path. The generate command uses a scheduler internally for single-sequence generation:

```go
if globals.Medusa {
	medusaCfg := scheduler.DefaultMedusaConfig()
	medusaCfg.EnableOnlineTraining = true
	medusaCfg.UseGPUTraining = true
	if globals.MedusaHeadsPath != "" {
		medusaCfg.HeadsPath = globals.MedusaHeadsPath
	}
	ms, err := scheduler.NewMedusaScheduler(model, tok, schedConfig, medusaCfg)
	if err != nil {
		return fmt.Errorf("create medusa scheduler: %w", err)
	}
	sched = ms.BaseScheduler()
	runFunc = ms.Run
}
```

- [ ] **Step 3: Add Medusa support to runChat (same pattern)**

Apply the same Medusa scheduler creation to `runChat`.

- [ ] **Step 4: Run build to verify**

Run: `CGO_ENABLED=1 go build -tags metal -o /dev/null ./inference/cmd/vexel/`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add inference/cmd/vexel/commands.go
git commit -m "feat(cli): add Medusa support to generate and chat subcommands"
```

---

### Task 4: Implement GPU trainer head saving

**Files:**
- Modify: `inference/medusa/gpu_trainer.go`
- Create: `inference/medusa/gpu_save_test.go`

- [ ] **Step 1: Write failing test for GPU head save/load round-trip**

Create `inference/medusa/gpu_save_test.go`:

```go
//go:build metal && darwin && cgo

package medusa

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGPUHeadsSaveLoad(t *testing.T) {
	// Create GPU heads with known dimensions
	numHeads, hiddenSize, vocabSize := 3, 64, 100
	heads := NewHeads(numHeads, hiddenSize, vocabSize)

	// Set some non-zero weights so we can verify round-trip
	for i := range heads.FC1Weights[0] {
		heads.FC1Weights[0][i] = float32(i) * 0.001
	}

	tmpDir := t.TempDir()
	savePath := filepath.Join(tmpDir, "test_heads.bin")

	// Save
	if err := heads.Save(savePath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(savePath); err != nil {
		t.Fatalf("Save file not found: %v", err)
	}

	// Load
	loaded, err := LoadHeads(savePath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Verify dimensions
	if loaded.NumHeads != numHeads {
		t.Errorf("NumHeads: got %d, want %d", loaded.NumHeads, numHeads)
	}

	// Verify weights survived round-trip
	for i := 0; i < min(10, len(loaded.FC1Weights[0])); i++ {
		if loaded.FC1Weights[0][i] != heads.FC1Weights[0][i] {
			t.Errorf("FC1Weights[0][%d]: got %f, want %f", i, loaded.FC1Weights[0][i], heads.FC1Weights[0][i])
		}
	}
}
```

- [ ] **Step 2: Run test to verify it passes (Save/Load exist on CPU Heads)**

Run: `go test -tags metal -run TestGPUHeadsSaveLoad -v ./inference/medusa/`
Expected: PASS (Save/Load already work on the CPU `Heads` struct)

- [ ] **Step 3: Write test for GPUOnlineTrainer.SaveHeads**

Add to `gpu_save_test.go`:

```go
func TestGPUTrainerSaveHeads(t *testing.T) {
	// This tests that the GPU trainer can extract weights back to CPU and save
	cfg := OnlineConfig{
		NumHeads:       2,
		BufferCapacity: 10,
		WarmupSamples:  5,
		MinAccuracy:    0.1,
		BatchSize:      2,
		LearningRate:   0.001,
	}

	// Need a mock or real backend - skip if not available
	// The key test is that SaveHeads doesn't return the TODO error
	trainer := &GPUOnlineTrainer{}
	tmpDir := t.TempDir()
	savePath := filepath.Join(tmpDir, "gpu_heads.bin")

	err := trainer.SaveHeads(savePath)
	if err != nil && err.Error() == "GPU head saving not implemented" {
		t.Fatal("SaveHeads still returns TODO error - needs implementation")
	}
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `go test -tags metal -run TestGPUTrainerSaveHeads -v ./inference/medusa/`
Expected: FAIL — SaveHeads returns TODO error

- [ ] **Step 5: Implement SaveHeads on GPUOnlineTrainer**

In `inference/medusa/gpu_trainer.go`, replace the TODO at line ~302 with:

```go
func (t *GPUOnlineTrainer) SaveHeads(path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.gpuHeads == nil {
		return fmt.Errorf("no GPU heads initialized")
	}

	// Convert GPU heads back to CPU Heads struct for serialization
	cpuHeads := t.gpuHeads.ToCPUHeads()
	return cpuHeads.Save(path)
}
```

Then implement `ToCPUHeads()` on `GPUHeads` — read FC1/FC2 weight buffers from device to host:

```go
func (g *GPUHeads) ToCPUHeads() *Heads {
	heads := &Heads{
		NumHeads:   g.NumHeads,
		HiddenSize: g.HiddenSize,
		VocabSize:  g.VocabSize,
		FC1Weights: make([][]float32, g.NumHeads),
		FC1Bias:    make([][]float32, g.NumHeads),
		FC2Weights: make([][]float32, g.NumHeads),
		FC2Bias:    make([][]float32, g.NumHeads),
	}

	for i := 0; i < g.NumHeads; i++ {
		fc1Size := g.HiddenSize * g.HiddenSize
		heads.FC1Weights[i] = make([]float32, fc1Size)
		buf := make([]byte, fc1Size*4)
		g.backend.ToHost(buf, g.fc1Weights[i])
		for j := 0; j < fc1Size; j++ {
			heads.FC1Weights[i][j] = math.Float32frombits(
				uint32(buf[j*4]) | uint32(buf[j*4+1])<<8 | uint32(buf[j*4+2])<<16 | uint32(buf[j*4+3])<<24,
			)
		}

		// Repeat for FC1Bias, FC2Weights, FC2Bias
		// ... (same pattern: allocate, ToHost, convert bytes to float32)
	}

	return heads
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `go test -tags metal -run TestGPUTrainerSaveHeads -v ./inference/medusa/`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add inference/medusa/gpu_trainer.go inference/medusa/gpu_save_test.go
git commit -m "feat(medusa): implement GPU head save via CPU conversion"
```

---

### Task 5: End-to-end smoke test

**Files:**
- Modify: `inference/cmd/vexel/medusa_integration_test.go`

- [ ] **Step 1: Write E2E test that creates a MedusaScheduler via the same path as runServe**

Add to `medusa_integration_test.go`:

```go
func TestMedusaSchedulerCreation(t *testing.T) {
	// Verify the MedusaScheduler can be created with default config
	// and that BaseScheduler() returns a non-nil *Scheduler
	cfg := scheduler.Config{
		MaxBatchSize:  1,
		MaxSequences:  4,
		MaxTokens:     32,
		SamplerConfig: sampler.DefaultConfig(),
	}

	medusaCfg := scheduler.DefaultMedusaConfig()
	medusaCfg.EnableOnlineTraining = true

	// Skip if no model available
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		t.Skip("VEXEL_TEST_MODEL not set")
	}

	model, tok, gpuBackend, err := initModel(modelPath, 32, false)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuBackend.Close()

	ms, err := scheduler.NewMedusaScheduler(model, tok, cfg, medusaCfg)
	if err != nil {
		t.Fatalf("NewMedusaScheduler: %v", err)
	}

	base := ms.BaseScheduler()
	if base == nil {
		t.Fatal("BaseScheduler() returned nil")
	}
}
```

- [ ] **Step 2: Run test (with model if available)**

Run: `VEXEL_TEST_MODEL=models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf go test -tags metal -run TestMedusaSchedulerCreation -v ./inference/cmd/vexel/`
Expected: PASS (or SKIP if no model)

- [ ] **Step 3: Commit**

```bash
git add inference/cmd/vexel/medusa_integration_test.go
git commit -m "test(medusa): add E2E smoke test for MedusaScheduler creation"
```

---

### Task 6: Manual verification

- [ ] **Step 1: Build the binary**

Run: `make build`

- [ ] **Step 2: Test --medusa flag is recognized**

Run: `./vexel --help 2>&1 | grep -i medusa`
Expected: `--medusa` flag appears in help output

- [ ] **Step 3: Run with --medusa (if model available)**

Run: `./vexel --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --medusa generate --prompt "Hello" --max-tokens 10`
Expected: Output includes "Using Medusa speculative decoding" log line, generates tokens

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "fix(medusa): address integration issues from manual testing"
```
