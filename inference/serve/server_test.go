package serve

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
)

func TestServer_handleGenerate(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	server := NewServer(sched)

	go func() {
		// Wait for sequence to appear
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("Hi")
				seqs[0].Close()
				break
			}
			time.Sleep(1 * time.Millisecond)
		}
	}()

	reqBody, _ := json.Marshal(map[string]string{
		"prompt": "Hello",
	})
	req, _ := http.NewRequest("POST", "/generate", bytes.NewBuffer(reqBody))
	rr := httptest.NewRecorder()

	server.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var resp map[string]string
	json.NewDecoder(rr.Body).Decode(&resp)
	if resp["text"] != "Hi" {
		t.Errorf("expected Hi, got %s", resp["text"])
	}
}

func TestServer_handleStream(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	server := NewServer(sched)

	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("Token1")
				seqs[0].PushToken("Token2")
				seqs[0].Close()
				break
			}
			time.Sleep(1 * time.Millisecond)
		}
	}()

	reqBody, _ := json.Marshal(map[string]string{
		"prompt": "Stream this",
	})
	req, _ := http.NewRequest("POST", "/stream", bytes.NewBuffer(reqBody))
	rr := httptest.NewRecorder()

	server.ServeHTTP(rr, req)

	if contentType := rr.Header().Get("Content-Type"); contentType != "text/event-stream" {
		t.Errorf("expected Content-Type text/event-stream, got %s", contentType)
	}

	body := rr.Body.String()
	if !contains(body, "Token1") || !contains(body, "Token2") {
		t.Errorf("body missing tokens: %s", body)
	}
}

func contains(s, substr string) bool {
	return bytes.Contains([]byte(s), []byte(substr))
}
