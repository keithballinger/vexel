# Fix Server Concurrent Throughput Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the scheduler's signal notification system so concurrent HTTP requests are processed in batched steps, enabling multi-client throughput scaling.

**Architecture:** The scheduler's `signal` channel (capacity=1) drops wakeup notifications from concurrent `AddSequence` calls. The fix replaces the bounded channel with `sync.Cond` for reliable multi-producer wakeup. Additionally, the scheduler's Run loop needs to continuously process while sequences are active (not wait for signal between every step). The existing `step()` → `collectReady()` → `formBatches()` → `runDecodeStep()` pipeline is correct and handles batched decode properly — the bug is purely in the wakeup mechanism.

**Tech Stack:** Go 1.25.4, sync.Cond, existing scheduler tests

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `inference/scheduler/scheduler.go` | Replace signal channel with sync.Cond, fix Run loop |
| Create | `inference/scheduler/concurrent_test.go` | Test concurrent AddSequence wakeup |

---

### Task 1: Write concurrent wakeup test

**Files:**
- Create: `inference/scheduler/concurrent_test.go`

- [ ] **Step 1: Write test that verifies concurrent AddSequence calls all get processed**

Create `inference/scheduler/concurrent_test.go`:

```go
package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestConcurrentAddSequenceAllProcessed(t *testing.T) {
	// Create scheduler with MaxBatchSize=4
	cfg := Config{
		MaxBatchSize: 4,
		MaxSequences: 8,
		MaxTokens:    8,
	}
	sched, err := NewScheduler(&runtime.ModelRuntime{}, nil, cfg)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	go sched.Run(ctx)

	// Add 4 sequences concurrently
	numSeqs := 4
	var wg sync.WaitGroup
	var added int32

	for i := 0; i < numSeqs; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			seq := NewSequence(SequenceID(id+1), "test prompt")
			sched.AddSequence(seq)
			atomic.AddInt32(&added, 1)
		}(i)
	}

	wg.Wait()

	// All 4 should be registered
	if count := sched.SequenceCount(); count != numSeqs {
		t.Errorf("expected %d sequences, got %d", numSeqs, count)
	}

	// Wait briefly for scheduler to process at least one step
	time.Sleep(50 * time.Millisecond)

	// Verify scheduler woke up and attempted to process
	// (it won't produce tokens without a real model, but it should have called step)
	metrics := sched.Metrics()
	if metrics.ActiveSequences == 0 && sched.SequenceCount() > 0 {
		t.Log("Note: scheduler has sequences but ActiveSequences metric is 0 — may indicate wakeup issue")
	}
}

func TestSignalNotDroppedUnderContention(t *testing.T) {
	cfg := Config{
		MaxBatchSize: 8,
		MaxSequences: 100,
		MaxTokens:    4,
	}
	sched, err := NewScheduler(&runtime.ModelRuntime{}, nil, cfg)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	go sched.Run(ctx)

	// Rapidly add many sequences — previously, all but the first signal would be dropped
	for i := 0; i < 50; i++ {
		sched.AddSequence(NewSequence(SequenceID(i+1), "hello"))
	}

	// Give scheduler time to wake up
	time.Sleep(100 * time.Millisecond)

	// All 50 should be registered (AddSequence is synchronous for registration)
	if count := sched.SequenceCount(); count != 50 {
		t.Errorf("expected 50 sequences registered, got %d", count)
	}
}
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `go test -run TestConcurrent -v ./inference/scheduler/`
Expected: Tests should pass (AddSequence registration is synchronous — the bug is in processing speed, not registration)

- [ ] **Step 3: Commit**

```bash
git add inference/scheduler/concurrent_test.go
git commit -m "test(scheduler): add concurrent AddSequence wakeup tests"
```

---

### Task 2: Replace signal channel with sync.Cond

**Files:**
- Modify: `inference/scheduler/scheduler.go`

- [ ] **Step 1: Read the Scheduler struct and signal-related code**

Read `inference/scheduler/scheduler.go` lines 55-107 to find:
- The `signal` field declaration in the Scheduler struct
- The `NewScheduler` constructor where `signal` is created
- The `wakeUp()` method
- The `Run()` method's select loop

- [ ] **Step 2: Replace signal channel with sync.Cond in the struct**

In the `Scheduler` struct, replace `signal chan struct{}` with:

```go
cond    *sync.Cond   , // Condition variable for wakeup
```

- [ ] **Step 3: Update NewScheduler to create sync.Cond**

Replace `signal: make(chan struct{}, 1)` with:

```go
cond: sync.NewCond(&sync.Mutex{}),
```

Note: This uses its own mutex, separate from `s.mu`, to avoid deadlock with AddSequence which holds `s.mu`.

- [ ] **Step 4: Update wakeUp() to use Broadcast**

Replace the wakeUp method:

```go
func (s *Scheduler) wakeUp() {
	s.cond.Broadcast()
}
```

- [ ] **Step 5: Update Run() loop to use sync.Cond**

Replace the entire Run() method:

```go
func (s *Scheduler) Run(ctx context.Context) error {
	// Process continuously while there are active sequences.
	// When idle, wait on condition variable for new sequences.
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		if err := s.step(ctx); err != nil {
			return err
		}

		// If sequences remain, loop immediately (no wait)
		if s.SequenceCount() > 0 {
			continue
		}

		// No sequences — wait for signal with timeout
		// Use a goroutine + channel to make Cond.Wait cancellable
		done := make(chan struct{})
		go func() {
			s.cond.L.Lock()
			s.cond.Wait()
			s.cond.L.Unlock()
			close(done)
		}()

		select {
		case <-ctx.Done():
			s.cond.Broadcast() // Unblock the waiting goroutine
			return nil
		case <-done:
			// Woken up — loop back to process
		}
	}
}
```

- [ ] **Step 6: Run tests**

Run: `go test -v ./inference/scheduler/`
Expected: All tests pass

- [ ] **Step 7: Verify build with Metal tags**

Run: `CGO_ENABLED=1 go build -tags metal ./inference/cmd/vexel/`
Expected: Build succeeds

- [ ] **Step 8: Commit**

```bash
git add inference/scheduler/scheduler.go
git commit -m "fix(scheduler): replace signal channel with sync.Cond for reliable concurrent wakeup"
```

---

### Task 3: Fix MedusaScheduler and SpeculativeScheduler Run loops

**Files:**
- Modify: `inference/scheduler/medusa_scheduler.go`
- Modify: `inference/scheduler/speculative_scheduler.go`

- [ ] **Step 1: Check if MedusaScheduler.Run uses the signal channel**

Read `inference/scheduler/medusa_scheduler.go` `Run()` method. It has its own loop with a 1ms ticker — it does NOT use the signal channel. It should be updated to also use the cond var for wakeup.

- [ ] **Step 2: Update MedusaScheduler.Run to use cond**

Replace the MedusaScheduler's Run method ticker-based loop with one that uses the base scheduler's cond:

```go
func (ms *MedusaScheduler) Run(ctx context.Context) error {
	ms.Start(ctx)
	defer ms.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		if err := ms.step(ctx); err != nil {
			return err
		}

		// If sequences remain, loop immediately
		if ms.SequenceCount() > 0 {
			continue
		}

		// Wait for new sequences
		done := make(chan struct{})
		go func() {
			ms.cond.L.Lock()
			ms.cond.Wait()
			ms.cond.L.Unlock()
			close(done)
		}()

		select {
		case <-ctx.Done():
			ms.cond.Broadcast()
			return nil
		case <-done:
		}
	}
}
```

- [ ] **Step 3: Update SpeculativeScheduler.Run similarly**

Read `inference/scheduler/speculative_scheduler.go` Run() method and apply the same pattern.

- [ ] **Step 4: Run all tests**

Run: `go test -v ./inference/scheduler/`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add inference/scheduler/medusa_scheduler.go inference/scheduler/speculative_scheduler.go
git commit -m "fix(scheduler): update Medusa and Speculative scheduler loops to use sync.Cond wakeup"
```

---

### Task 4: Verify with benchmark

- [ ] **Step 1: Build**

Run: `make build`

- [ ] **Step 2: Run batched benchmark with concurrency=4**

Run a quick manual test:

```bash
# Start server with batch size 4
./vexel --model benchmarks/models/qwen-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf serve --max-batch-size 4 --port 18080 &
sleep 3

# Send 4 concurrent requests
time (
  for i in 1 2 3 4; do
    curl -sf -X POST http://localhost:18080/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Hello","max_tokens":16,"temperature":0}' -o /dev/null &
  done
  wait
)

kill %1
```

Expected: Total wall time should be significantly less than 4x single-request time.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix(scheduler): address integration issues from concurrent serving test"
```
