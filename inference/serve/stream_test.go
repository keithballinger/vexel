package serve_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
)

func TestStreamEndpoint(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	server := serve.NewServer(sched)

	payload := map[string]string{
		"prompt": "Hello world",
	}
	body, _ := json.Marshal(payload)

	req := httptest.NewRequest("POST", "/stream", bytes.NewBuffer(body))
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

	server.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	// Verify SSE headers
	if w.Header().Get("Content-Type") != "text/event-stream" {
		t.Errorf("Expected Content-Type text/event-stream, got %s", w.Header().Get("Content-Type"))
	}

	// Verify stream content
	// We expect multiple chunks starting with "data: "
	scanner := bufio.NewScanner(w.Body)
	foundData := false
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			foundData = true
			break
		}
	}

	if !foundData {
		t.Error("Expected SSE data stream, got none")
	}
}
