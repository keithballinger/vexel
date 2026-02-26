package client_test

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"vexel/inference/client"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
	"vexel/inference/serve/pb"

	"google.golang.org/grpc"
)

// startTestGRPCServer starts a gRPC server for testing and returns the address.
func startTestGRPCServer(t *testing.T, sched *scheduler.Scheduler) (string, func()) {
	t.Helper()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	s := grpc.NewServer()
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)
	go func() { _ = s.Serve(lis) }()
	cleanup := func() {
		s.Stop()
		lis.Close()
	}
	return lis.Addr().String(), cleanup
}

// simulateTokens watches for sequences and pushes tokens to the first one.
func simulateTokens(sched *scheduler.Scheduler, tokens []string) {
	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				for _, tok := range tokens {
					seqs[0].PushToken(tok)
				}
				seqs[0].Close()
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()
}

// TestGRPCClientGenerate tests the gRPC client's Generate method.
func TestGRPCClientGenerate(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"Hello", " ", "world"})

	addr, cleanup := startTestGRPCServer(t, sched)
	defer cleanup()

	c, err := client.NewGRPCClient(addr, client.WithInsecure())
	if err != nil {
		t.Fatalf("NewGRPCClient: %v", err)
	}
	defer c.Close()

	resp, err := c.Generate(context.Background(), "test prompt")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp != "Hello world" {
		t.Errorf("got %q, want %q", resp, "Hello world")
	}
}

// TestGRPCClientStream tests the gRPC client's Stream method.
func TestGRPCClientStream(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"a", "b", "c"})

	addr, cleanup := startTestGRPCServer(t, sched)
	defer cleanup()

	c, err := client.NewGRPCClient(addr, client.WithInsecure())
	if err != nil {
		t.Fatalf("NewGRPCClient: %v", err)
	}
	defer c.Close()

	tokenCh, errCh := c.Stream(context.Background(), "test")
	var tokens []string
	for tok := range tokenCh {
		tokens = append(tokens, tok)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(tokens) != 3 {
		t.Errorf("got %d tokens, want 3", len(tokens))
	}
}

// TestHTTPClientGenerate tests the HTTP client's Generate method.
func TestHTTPClientGenerate(t *testing.T) {
	// Create a simple HTTP server that returns a JSON response
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"text": "hello from http"})
	}))
	defer ts.Close()

	c := client.NewHTTPClient(ts.URL)
	defer c.Close()

	resp, err := c.Generate(context.Background(), "test prompt")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp != "hello from http" {
		t.Errorf("got %q, want %q", resp, "hello from http")
	}
}

// TestHTTPClientStream tests the HTTP client's Stream method with SSE.
func TestHTTPClientStream(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, tok := range []string{"x", "y", "z"} {
			data, _ := json.Marshal(map[string]string{"token": tok})
			w.Write([]byte("data: " + string(data) + "\n\n"))
			flusher.Flush()
		}
	}))
	defer ts.Close()

	c := client.NewHTTPClient(ts.URL)
	defer c.Close()

	tokenCh, errCh := c.Stream(context.Background(), "test")
	var tokens []string
	for tok := range tokenCh {
		tokens = append(tokens, tok)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(tokens) != 3 || tokens[0] != "x" || tokens[1] != "y" || tokens[2] != "z" {
		t.Errorf("got tokens=%v, want [x y z]", tokens)
	}
}

// TestClientInterface verifies both client types implement the Client interface.
func TestClientInterface(t *testing.T) {
	// This is a compile-time check
	var _ client.Client = (*client.GRPCClient)(nil)
	var _ client.Client = (*client.HTTPClient)(nil)
}

// TestAutoClient tests that AutoClient tries gRPC first, then falls back to HTTP.
func TestAutoClientFallbackToHTTP(t *testing.T) {
	// Set up only an HTTP server (no gRPC)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"text": "http fallback"})
	}))
	defer ts.Close()

	c, err := client.NewAutoClient(ts.URL, "localhost:0") // invalid gRPC addr
	if err != nil {
		t.Fatalf("NewAutoClient: %v", err)
	}
	defer c.Close()

	resp, err := c.Generate(context.Background(), "test")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp != "http fallback" {
		t.Errorf("got %q, want %q", resp, "http fallback")
	}
}

// TestAutoClientPrefersGRPC tests that AutoClient uses gRPC when available.
func TestAutoClientPrefersGRPC(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"grpc", " ", "preferred"})

	grpcAddr, cleanup := startTestGRPCServer(t, sched)
	defer cleanup()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"text": "http not used"})
	}))
	defer ts.Close()

	c, err := client.NewAutoClient(ts.URL, grpcAddr)
	if err != nil {
		t.Fatalf("NewAutoClient: %v", err)
	}
	defer c.Close()

	resp, err := c.Generate(context.Background(), "test")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp != "grpc preferred" {
		t.Errorf("got %q, want %q", resp, "grpc preferred")
	}
}

// TestGRPCClientClose tests that Close properly cleans up resources.
func TestGRPCClientClose(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	addr, cleanup := startTestGRPCServer(t, sched)
	defer cleanup()

	c, err := client.NewGRPCClient(addr, client.WithInsecure())
	if err != nil {
		t.Fatalf("NewGRPCClient: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

// TestNewGRPCClientInvalidAddr tests error handling for invalid addresses.
func TestNewGRPCClientInvalidAddr(t *testing.T) {
	// With insecure + non-blocking dial, connection might not fail immediately.
	// But Generate should fail.
	c, err := client.NewGRPCClient("localhost:0", client.WithInsecure())
	if err != nil {
		return // If it errors on creation, that's fine too
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err = c.Generate(ctx, "test")
	if err == nil {
		t.Error("expected error for invalid address")
	}
}
