package serve_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
)

func TestAdmissionControl(t *testing.T) {
	// Setup real scheduler with mock runtime
	cfg := scheduler.Config{MaxBatchSize: 1, MaxSequences: 1} // Limit to 1 sequence for easy full test
	rt := &runtime.ModelRuntime{}
	sched, _ := scheduler.NewScheduler(rt, nil, cfg)

	server := serve.NewServer(sched)

	// Helper to send request
	sendRequest := func(prompt string) int {
		payload := map[string]string{"prompt": prompt}
		body, _ := json.Marshal(payload)
		req := httptest.NewRequest("POST", "/generate", bytes.NewBuffer(body))
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		return w.Code
	}

	// 1. Send first request.
	// Since handleGenerate blocks until close, we run it in a goroutine
	// so we can check the scheduler state while it's active.
	respCodeChan := make(chan int, 1)
	go func() {
		respCodeChan <- sendRequest("Request 1")
	}()

	// Wait for sequence to appear in scheduler
	for i := 0; i < 100; i++ {
		if sched.SequenceCount() == 1 {
			break
		}
		time.Sleep(time.Millisecond)
	}

	// Verify scheduler has 1 sequence
	if sched.SequenceCount() != 1 {
		t.Errorf("Expected scheduler to have 1 sequence, got %d", sched.SequenceCount())
	}

	// Now close the sequence so handleGenerate can finish
	seqs := sched.GetSequences()
	if len(seqs) > 0 {
		seqs[0].PushToken("OK")
		seqs[0].Close()
	}

	code1 := <-respCodeChan
	if code1 != http.StatusOK {
		t.Errorf("Expected first request to succeed, got %d", code1)
	}
}
