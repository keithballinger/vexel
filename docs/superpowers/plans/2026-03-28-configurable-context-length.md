# Configurable Context Length Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded 2048 max context length with a configurable `--context-len` CLI flag, allowing Vexel to handle longer prompts.

**Architecture:** Add `ContextLen int` to `GlobalFlags`, pass it through `initModel()` and `loadDraftModel()` to `CreateGPUKVCache()`. The scratch arena sizing already scales correctly with `maxBatchSize` which equals `maxContextLen`. Default remains 2048 for backward compatibility. Update examples and docs.

**Tech Stack:** Go 1.25.4, CLI flag parsing, existing KV cache infrastructure

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/cmd/vexel/cli.go` | Add `--context-len` flag to GlobalFlags |
| Modify | `inference/cmd/vexel/commands.go` | Pass ContextLen to initModel/loadDraftModel |
| Modify | `inference/cmd/vexel/medusa_integration_test.go` | Add test for new flag |
| Modify | `examples/server/main.go` | Use flag instead of hardcoded 2048 |
| Modify | `examples/generate/main.go` | Use flag instead of hardcoded 2048 |

---

### Task 1: Add --context-len CLI flag

**Files:**
- Modify: `inference/cmd/vexel/cli.go`
- Modify: `inference/cmd/vexel/medusa_integration_test.go`

- [ ] **Step 1: Write test for new flag**

Add to `inference/cmd/vexel/medusa_integration_test.go`:

```go
func TestParseContextLenFlag(t *testing.T) {
	args := []string{"vexel", "--model", "test.gguf", "--context-len", "4096", "serve"}
	cmd, globals, err := parseArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if globals.ContextLen != 4096 {
		t.Errorf("expected ContextLen=4096, got %d", globals.ContextLen)
	}
	if cmd != "serve" {
		t.Errorf("expected cmd=serve, got %s", cmd)
	}
}

func TestParseContextLenDefault(t *testing.T) {
	args := []string{"vexel", "--model", "test.gguf", "serve"}
	_, globals, err := parseArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if globals.ContextLen != 0 {
		t.Errorf("expected default ContextLen=0, got %d", globals.ContextLen)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test -run TestParseContextLen -v ./inference/cmd/vexel/`
Expected: FAIL — `globals.ContextLen` field doesn't exist

- [ ] **Step 3: Add ContextLen to GlobalFlags and parseArgs**

In `inference/cmd/vexel/cli.go`, add to `GlobalFlags`:

```go
ContextLen      int    // Max context length (default: 2048)
```

In `parseArgs`, add case:

```go
case "--context-len":
	if i+1 >= len(args) {
		return "", GlobalFlags{}, fmt.Errorf("--context-len requires a value")
	}
	n, err := strconv.Atoi(args[i+1])
	if err != nil {
		return "", GlobalFlags{}, fmt.Errorf("--context-len must be an integer: %v", err)
	}
	globals.ContextLen = n
	i += 2
```

Add `"strconv"` to imports if not already present.

In `printUsage`, add:

```
  --context-len  Max context length for KV cache (default: 2048)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test -run TestParseContextLen -v ./inference/cmd/vexel/`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference/cmd/vexel/cli.go inference/cmd/vexel/medusa_integration_test.go
git commit -m "feat(cli): add --context-len flag for configurable KV cache size"
```

---

### Task 2: Wire context length through initModel and loadDraftModel

**Files:**
- Modify: `inference/cmd/vexel/commands.go`

- [ ] **Step 1: Read initModel to find all hardcoded 2048 references**

Read `inference/cmd/vexel/commands.go` lines 32-100. Find:
- `maxContextLen := 2048` (line 53)
- `model.CreateGPUKVCache(maxContextLen)` (line 80)
- Same pattern in `loadDraftModel` (lines 104-139)

- [ ] **Step 2: Update initModel to accept contextLen parameter**

Change `initModel` signature from:

```go
func initModel(modelPath string, maxTokens int, verbose bool) (*runtime.ModelRuntime, *tokenizer.Tokenizer, *metal.Backend, error)
```

to:

```go
func initModel(modelPath string, maxTokens, contextLen int, verbose bool) (*runtime.ModelRuntime, *tokenizer.Tokenizer, *metal.Backend, error)
```

Replace `maxContextLen := 2048` with:

```go
maxContextLen := contextLen
if maxContextLen <= 0 {
	maxContextLen = 2048
}
```

- [ ] **Step 3: Update loadDraftModel similarly**

Change signature to accept `contextLen int`, replace hardcoded 2048:

```go
func loadDraftModel(draftPath string, gpuBackend *metal.Backend, maxTokens, contextLen int, verbose bool) (*runtime.ModelRuntime, error)
```

Replace `maxContextLen := 2048` and `draft.CreateGPUKVCache(2048)` with the parameter.

- [ ] **Step 4: Update all callers of initModel and loadDraftModel**

In `runServe`:
```go
model, tok, gpuBackend, err := initModel(globals.Model, sf.MaxTokens, globals.ContextLen, globals.Verbose)
```
and:
```go
draft, err := loadDraftModel(globals.DraftModel, gpuBackend, sf.MaxTokens, globals.ContextLen, globals.Verbose)
```

In `runGenerate`:
```go
model, tok, gpuBackend, err := initModel(globals.Model, gf.MaxTokens, globals.ContextLen, globals.Verbose)
```
and:
```go
draft, err := loadDraftModel(globals.DraftModel, gpuBackend, gf.MaxTokens, globals.ContextLen, globals.Verbose)
```

In `runChat`:
```go
model, tok, gpuBackend, err := initModel(globals.Model, 256, globals.ContextLen, globals.Verbose)
```

- [ ] **Step 5: Verify build**

Run: `CGO_ENABLED=1 go build -tags metal -o /dev/null ./inference/cmd/vexel/`
Expected: Build succeeds

- [ ] **Step 6: Run all CLI tests**

Run: `go test -v ./inference/cmd/vexel/`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add inference/cmd/vexel/commands.go
git commit -m "feat(cli): wire --context-len through initModel and loadDraftModel"
```

---

### Task 3: Update examples

**Files:**
- Modify: `examples/server/main.go`
- Modify: `examples/generate/main.go`

- [ ] **Step 1: Update examples/server/main.go**

Add a `contextLen` flag and use it:

```go
contextLen := flag.Int("context-len", 2048, "Max context length for KV cache")
```

Replace `model.CreateGPUKVCache(2048)` with `model.CreateGPUKVCache(*contextLen)`.

- [ ] **Step 2: Update examples/generate/main.go**

Add a `contextLen` flag:

```go
contextLen := flag.Int("context-len", 2048, "Max context length for KV cache")
```

Replace `cache := model.CreateGPUKVCache(2048)` with `cache := model.CreateGPUKVCache(*contextLen)`.

- [ ] **Step 3: Verify examples compile**

Run: `CGO_ENABLED=1 go build -tags metal ./examples/server/ && CGO_ENABLED=1 go build -tags metal ./examples/generate/`
Expected: Both compile

- [ ] **Step 4: Commit**

```bash
git add examples/server/main.go examples/generate/main.go
git commit -m "feat(examples): add --context-len flag, replace hardcoded 2048"
```

---

### Task 4: Update docs

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update README CLI flags section**

Add `--context-len` to the global flags listing and add an example:

```
  --context-len  Max context length for KV cache (default: 2048)
```

- [ ] **Step 2: Update CLAUDE.md environment variables**

Add note that default context length is 2048, configurable via `--context-len`.

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document --context-len flag"
```
