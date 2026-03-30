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

func TestGenerateEndpoint(t *testing.T) {
	// Setup server
	rt := &runtime.ModelRuntime{}
	cfg := scheduler.Config{MaxBatchSize: 1, MaxSequences: 1}
	sched, _ := scheduler.NewScheduler(rt, nil, cfg)

	server := serve.NewServer(sched)

	// Create request payload
	payload := map[string]string{
		"prompt": "Hello world",
	}
	body, _ := json.Marshal(payload)

	req := httptest.NewRequest("POST", "/generate", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	// Push token in background
	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("Response")
				seqs[0].Close()
				break
			}
			time.Sleep(time.Millisecond)
		}
	}()

	// Handler
	server.ServeHTTP(w, req)

	// Assertions
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var resp map[string]string
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if resp["text"] == "" {
		t.Error("Expected response text, got empty")
	}
}
