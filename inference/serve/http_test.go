package serve_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"vexel/inference/serve"
)

func TestGenerateEndpoint(t *testing.T) {
	// Setup server
	// We need a mock scheduler or a way to inject it.
	// For now, assuming NewServer takes a Scheduler (or interface).
	
	server := serve.NewServer(nil) // Passing nil scheduler for now, expecting structural setup

	// Create request payload
	payload := map[string]string{
		"prompt": "Hello world",
	}
	body, _ := json.Marshal(payload)

	req := httptest.NewRequest("POST", "/generate", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

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
