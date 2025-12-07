package scheduler_test

import (
	"context"
	"testing"
	"time"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
)

func TestSchedulerRunLoop(t *testing.T) {
	// Setup
	cfg := scheduler.Config{
		MaxBatchSize: 8,
		MaxSequences: 16,
	}
	// Note: In a real test, we would mock the runtime's behavior (e.g., verifying DecodeStep is called).
	// For this loop test, we just want to ensure it starts and stops cleanly via context.
	rt := &runtime.ModelRuntime{} 
	sched, _ := scheduler.NewScheduler(rt, cfg)

	// Create context with timeout to stop the loop
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	// Run the scheduler loop
	errChan := make(chan error)
	go func() {
		errChan <- sched.Run(ctx)
	}()

	// Wait for completion
	select {
	case err := <-errChan:
		if err != nil {
			t.Errorf("Scheduler.Run returned error: %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Scheduler.Run did not exit after context cancellation")
	}
}
