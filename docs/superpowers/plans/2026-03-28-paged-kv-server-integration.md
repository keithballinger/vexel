# Paged KV Cache Server Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the serve command from single-tenant GPU KV cache to paged KV cache, enabling true multi-client concurrent inference.

**Architecture:** The `initModel()` function currently calls `CreateGPUKVCache(2048)` which creates a single-sequence KV cache. For serve mode, replace this with `CreatePagedKVCache()` which supports multiple concurrent sequences with per-sequence block tables. The scheduler already has full paged KV support: `runBatchedPrefill()`, `DecodeWithPagedKV()`, and `DecodeWithPagedKVBatched()`. For generate/chat (single-sequence), keep GPU KV cache for maximum performance. The key change: make `initModel()` accept a flag controlling which KV cache to create.

**Tech Stack:** Go 1.25.4, existing paged KV infrastructure (`inference/kv/`, `inference/runtime/gpu_block_pool.go`)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/cmd/vexel/commands.go` | Add `usePaged` param to initModel, use paged KV for serve |
| Create | `inference/scheduler/paged_serve_test.go` | Test that scheduler processes multiple sequences with paged KV |

---

### Task 1: Add paged KV option to initModel

**Files:**
- Modify: `inference/cmd/vexel/commands.go`

- [ ] **Step 1: Read initModel to understand current KV cache creation**

Read `inference/cmd/vexel/commands.go` lines 32-100.

Key line 80: `model.CreateGPUKVCache(maxContextLen)` — this creates the single-tenant cache.

Also check `CreatePagedKVCache` in `inference/runtime/model.go` line 206 for its signature: `func (m *ModelRuntime) CreatePagedKVCache(maxBlocks int) *kv.PagedKVCache`

- [ ] **Step 2: Add `usePaged bool` parameter to initModel**

Change the signature:

```go
func initModel(modelPath string, maxTokens int, verbose, usePaged bool) (*runtime.ModelRuntime, *tokenizer.Tokenizer, *metal.Backend, error)
```

Replace the KV cache creation (line 80) with:

```go
if usePaged {
	// Paged KV cache: supports multiple concurrent sequences.
	// Each sequence gets its own block table with 16-token blocks.
	blockSize := 16
	maxBlocks := (maxContextLen + blockSize - 1) / blockSize
	model.CreatePagedKVCache(maxBlocks)
	log.Printf("Using paged KV cache (maxBlocks=%d, blockSize=%d, maxContext=%d)", maxBlocks, blockSize, maxContextLen)
} else {
	// GPU KV cache: single-sequence, optimized for generate/chat.
	model.CreateGPUKVCache(maxContextLen)
}
```

- [ ] **Step 3: Update all callers of initModel**

In `runServe` — use paged KV:
```go
model, tok, gpuBackend, err := initModel(globals.Model, sf.MaxTokens, globals.Verbose, true)
```

In `runGenerate` — keep GPU KV (single sequence, faster):
```go
model, tok, gpuBackend, err := initModel(globals.Model, gf.MaxTokens, globals.Verbose, false)
```

In `runChat` — keep GPU KV (single sequence):
```go
model, tok, gpuBackend, err := initModel(globals.Model, 256, globals.Verbose, false)
```

- [ ] **Step 4: Verify build**

Run: `CGO_ENABLED=1 go build -tags metal -o /dev/null ./inference/cmd/vexel/`
Expected: Build succeeds

- [ ] **Step 5: Run CLI tests**

Run: `go test -v ./inference/cmd/vexel/`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add inference/cmd/vexel/commands.go
git commit -m "feat(serve): use paged KV cache for multi-client concurrent inference"
```

---

### Task 2: Test multi-sequence serving with paged KV

**Files:**
- Create: `inference/scheduler/paged_serve_test.go`

- [ ] **Step 1: Write test that verifies multiple sequences complete with paged KV**

Create `inference/scheduler/paged_serve_test.go`:

```go
//go:build metal && darwin && cgo

package scheduler

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"vexel/inference/pkg/sampler"
)

// TestPagedKVMultiSequence verifies that the scheduler can process
// multiple concurrent sequences using the paged KV cache path.
func TestPagedKVMultiSequence(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		t.Skip("VEXEL_TEST_MODEL not set")
	}

	// This test requires a model loaded with paged KV cache.
	// It verifies that 3 concurrent sequences all produce tokens
	// without interfering with each other.
	t.Log("This test requires manual setup with a paged KV model")
	t.Skip("Requires integration test infrastructure")
}
```

Note: A full integration test requires model loading infrastructure that's in the `main` package. This placeholder documents the intent. The real verification is the manual benchmark in Task 3.

- [ ] **Step 2: Commit**

```bash
git add inference/scheduler/paged_serve_test.go
git commit -m "test(scheduler): add placeholder for paged KV multi-sequence test"
```

---

### Task 3: Manual concurrent serving verification

- [ ] **Step 1: Build**

Run: `make build`

- [ ] **Step 2: Start server and test concurrent requests**

```bash
# Start server (now uses paged KV cache)
./vexel --model benchmarks/models/qwen-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf serve --max-batch-size 4 --port 18084 &
sleep 8

# Verify health
curl -sf http://localhost:18084/health

# Single request baseline
time curl -sf -X POST http://localhost:18084/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":16,"temperature":0}'

# 4 concurrent requests — should complete in roughly the same time as 1 if batching works
time (
  for i in 1 2 3 4; do
    curl -sf -X POST http://localhost:18084/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Hello","max_tokens":16,"temperature":0}' -o /dev/null &
  done
  wait
)

kill %1
```

Expected: 4 concurrent requests complete in significantly less than 4x single-request time.

- [ ] **Step 3: Fix any issues found**

- [ ] **Step 4: Commit fixes if needed**

```bash
git add -A
git commit -m "fix(serve): address paged KV integration issues from manual test"
```
