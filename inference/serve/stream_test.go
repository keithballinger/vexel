package serve_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"vexel/inference/serve"
)

func TestStreamEndpoint(t *testing.T) {
	server := serve.NewServer(nil)

	payload := map[string]string{
		"prompt": "Hello world",
	}
	body, _ := json.Marshal(payload)

	req := httptest.NewRequest("POST", "/stream", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

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
