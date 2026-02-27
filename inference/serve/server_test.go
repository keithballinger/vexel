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

// --- Phase 1: Configurable Timeout & Health Endpoint Tests ---

func TestServerConfig_DefaultTimeout(t *testing.T) {
	// NewServer with no config should use a default timeout of 120s.
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	srv := NewServer(sched)
	if srv.config.RequestTimeout != 120*time.Second {
		t.Errorf("default timeout: got %v, want 120s", srv.config.RequestTimeout)
	}
}

func TestServerConfig_CustomTimeout(t *testing.T) {
	// NewServerWithConfig should accept a custom timeout.
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	cfg := Config{RequestTimeout: 60 * time.Second}
	srv := NewServerWithConfig(sched, cfg)
	if srv.config.RequestTimeout != 60*time.Second {
		t.Errorf("custom timeout: got %v, want 60s", srv.config.RequestTimeout)
	}
}

func TestServerConfig_ZeroTimeoutMeansNoTimeout(t *testing.T) {
	// A zero timeout should mean no request timeout (unlimited).
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	cfg := Config{RequestTimeout: 0}
	srv := NewServerWithConfig(sched, cfg)

	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				// Respond quickly — the point is that no timeout fires
				seqs[0].PushToken("ok")
				seqs[0].Close()
				break
			}
			time.Sleep(1 * time.Millisecond)
		}
	}()

	reqBody, _ := json.Marshal(map[string]string{"prompt": "test"})
	req, _ := http.NewRequest("POST", "/generate", bytes.NewBuffer(reqBody))
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("zero timeout: got status %d, want 200", rr.Code)
	}
}

func TestServer_handleGenerateTimeout(t *testing.T) {
	// With a very short timeout and a slow scheduler, the request should time out.
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	cfg := Config{RequestTimeout: 10 * time.Millisecond}
	srv := NewServerWithConfig(sched, cfg)

	// Don't push any tokens — let the request time out.
	reqBody, _ := json.Marshal(map[string]string{"prompt": "test"})
	req, _ := http.NewRequest("POST", "/generate", bytes.NewBuffer(reqBody))
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestTimeout {
		t.Errorf("short timeout: got status %d, want %d", rr.Code, http.StatusRequestTimeout)
	}
}

func TestServer_handleHealth(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	srv := NewServer(sched)

	req, _ := http.NewRequest("GET", "/health", nil)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("health: got status %d, want 200", rr.Code)
	}

	var resp map[string]string
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("health: failed to decode response: %v", err)
	}
	if resp["status"] != "ok" {
		t.Errorf("health: got status=%q, want ok", resp["status"])
	}
}

func TestServer_handleHealthContentType(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	srv := NewServer(sched)

	req, _ := http.NewRequest("GET", "/health", nil)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("health content-type: got %q, want application/json", ct)
	}
}
