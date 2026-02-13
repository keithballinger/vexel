package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNew(t *testing.T) {
	baseURL := "http://localhost:8080"
	timeout := 10 * time.Second

	c := New(Config{
		BaseURL: baseURL,
		Timeout: timeout,
	})

	if c.BaseURL() != baseURL {
		t.Errorf("expected BaseURL %s, got %s", baseURL, c.BaseURL())
	}

	if c.Timeout() != timeout {
		t.Errorf("expected Timeout %v, got %v", timeout, c.Timeout())
	}
}

func TestDefaultConfig(t *testing.T) {
	c := New(Config{})

	if c.BaseURL() == "" {
		t.Error("expected default BaseURL, got empty string")
	}

	if c.Timeout() == 0 {
		t.Error("expected default Timeout, got 0")
	}
}

func TestGenerate(t *testing.T) {
	mockResponse := map[string]string{
		"text": "Hello, I am Vexel!",
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/generate" {
			t.Errorf("expected /generate, got %s", r.URL.Path)
		}

		var reqBody map[string]string
		json.NewDecoder(r.Body).Decode(&reqBody)
		if reqBody["prompt"] != "Hello" {
			t.Errorf("expected prompt Hello, got %s", reqBody["prompt"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(mockResponse)
	}))
	defer ts.Close()

	c := New(Config{BaseURL: ts.URL})
	resp, err := c.Generate(context.Background(), "Hello", nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp != mockResponse["text"] {
		t.Errorf("expected %s, got %s", mockResponse["text"], resp)
	}
}

func TestGenerate_ServerErrors(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		respBody   string
		expectedErr string
	}{
		{
			name:       "InternalServerError",
			status:     http.StatusInternalServerError,
			respBody:   "Internal error",
			expectedErr: "server returned status 500",
		},
		{
			name:       "InvalidJSON",
			status:     http.StatusOK,
			respBody:   "{invalid-json}",
			expectedErr: "failed to decode response",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				w.Write([]byte(tt.respBody))
			}))
			defer ts.Close()

			c := New(Config{BaseURL: ts.URL})
			_, err := c.Generate(context.Background(), "Hello", nil)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !contains(err.Error(), tt.expectedErr) {
				t.Errorf("expected error containing %s, got %s", tt.expectedErr, err.Error())
			}
		})
	}
}

func TestGenerate_RequestError(t *testing.T) {
	c := New(Config{BaseURL: "http://invalid-url-that-should-fail"})
	_, err := c.Generate(context.Background(), "Hello", nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestStream(t *testing.T) {
	mockTokens := []string{"Hello", " ", "world", "!"}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		for _, token := range mockTokens {
			data := map[string]string{"token": token}
			buf, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", buf)
			w.(http.Flusher).Flush()
		}
	}))
	defer ts.Close()

	c := New(Config{BaseURL: ts.URL})
	tokenChan, err := c.Stream(context.Background(), "Hello", nil)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var receivedTokens []string
	for token := range tokenChan {
		receivedTokens = append(receivedTokens, token)
	}

	if len(receivedTokens) != len(mockTokens) {
		t.Errorf("expected %d tokens, got %d", len(mockTokens), len(receivedTokens))
	}

	for i, token := range receivedTokens {
		if token != mockTokens[i] {
			t.Errorf("expected token %s, got %s", mockTokens[i], token)
		}
	}
}

func TestStream_ServerErrors(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		expectedErr string
	}{
		{
			name:       "InternalServerError",
			status:     http.StatusInternalServerError,
			expectedErr: "server returned status 500",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
			}))
			defer ts.Close()

			c := New(Config{BaseURL: ts.URL})
			_, err := c.Stream(context.Background(), "Hello", nil)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !contains(err.Error(), tt.expectedErr) {
				t.Errorf("expected error containing %s, got %s", tt.expectedErr, err.Error())
			}
		})
	}
}

func TestStream_RequestError(t *testing.T) {
	c := New(Config{BaseURL: "http://invalid-url-that-should-fail"})
	_, err := c.Stream(context.Background(), "Hello", nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}
