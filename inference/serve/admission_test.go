package serve_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
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

	// 1. Send first request. It should be accepted (200 OK)
	// Note: Currently handler is mocked to return 200 immediately.
	// We want to verify that it *registers* with the scheduler.
	// Since we can't inspect scheduler state easily from outside without helper,
	// we will rely on integration behavior or add a helper to Scheduler to count sequences.
	
	// However, if we implement admission control, maybe we return 503 if full?
	// But our scheduler is queue-based. It should accept until MaxSequences is hit?
	
	// Let's test that the handler actually calls AddSequence.
	// We can check this by verifying the Scheduler has the sequence after the call.
	
	code1 := sendRequest("Request 1")
	if code1 != http.StatusOK {
		t.Errorf("Expected first request to succeed, got %d", code1)
	}

	// Verify scheduler has 1 sequence
	if sched.SequenceCount() != 1 {
		t.Errorf("Expected scheduler to have 1 sequence, got %d", sched.SequenceCount())
	}
}
