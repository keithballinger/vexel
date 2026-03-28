# Medusa + Paged KV Compatibility Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Medusa speculative decoding work in serve mode by using GPU KV cache (not paged) when --medusa is enabled, with a log warning about single-sequence limitation.

**Architecture:** Medusa's decode paths (`runDecodeStepWithTraining`, `runMedusaDecodeStep`, `VerifySpeculativeWithHidden`) require GPU KV cache for hidden state capture. Paged KV doesn't support hidden state extraction. The pragmatic fix: when `--medusa` is set in serve mode, use GPU KV cache (`usePaged: false`). This means Medusa serve handles one sequence at a time but gets speculation benefits. Non-Medusa serve continues using paged KV for multi-client batching. A full Medusa + paged KV integration (creating `DecodeWithPagedKVAndHidden`) is a future project.

**Tech Stack:** Go 1.25.4

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/cmd/vexel/commands.go` | Use GPU KV when --medusa in serve mode |

---

### Task 1: Use GPU KV cache when --medusa is set in serve mode

**Files:**
- Modify: `inference/cmd/vexel/commands.go`

- [ ] **Step 1: Change initModel call in runServe to use GPU KV when Medusa is enabled**

In `runServe` (line 182), change:

```go
model, tok, gpuBackend, err := initModel(globals.Model, sf.MaxTokens, globals.ContextLen, globals.Verbose, true)
```

to:

```go
// Medusa requires GPU KV cache (hidden state capture not supported in paged KV).
// Use paged KV only when Medusa is not enabled.
usePaged := !globals.Medusa && globals.DraftModel == ""
model, tok, gpuBackend, err := initModel(globals.Model, sf.MaxTokens, globals.ContextLen, globals.Verbose, usePaged)
```

Also add a log warning when Medusa disables paged KV:

```go
if globals.Medusa {
	log.Printf("Note: Medusa mode uses GPU KV cache (single-sequence). Multi-client batching requires non-Medusa mode.")
}
```

- [ ] **Step 2: Also handle draft-model (speculative also needs GPU KV)**

The `--draft-model` speculative path also uses `VerifySpeculativeWithHidden` which requires GPU KV cache. The `usePaged` logic already handles this: `!globals.Medusa && globals.DraftModel == ""`.

- [ ] **Step 3: Verify build**

Run: `CGO_ENABLED=1 go build -tags metal -o /dev/null ./inference/cmd/vexel/`

- [ ] **Step 4: Run tests**

Run: `go test -v ./inference/cmd/vexel/`

- [ ] **Step 5: Commit**

```bash
git add inference/cmd/vexel/commands.go
git commit -m "fix(serve): use GPU KV cache when --medusa is enabled (paged KV incompatible)"
```

---

### Task 2: Verify Medusa serve works

- [ ] **Step 1: Build**

Run: `make build`

- [ ] **Step 2: Test Medusa server with 0.5B model**

```bash
./vexel --model benchmarks/models/qwen-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf --medusa serve --port 18090 &
sleep 5
curl -X POST http://localhost:18090/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Hello world","max_tokens":16,"temperature":0}'
kill %1
```

Expected: Server starts, generates tokens, no crash.

- [ ] **Step 3: Run speculative benchmark**

```bash
WARMUP=1 RUNS=2 bash benchmarks/full_comparison.sh speculative
```

- [ ] **Step 4: Commit any fixes**
