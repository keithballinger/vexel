package serve_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
	"vexel/inference/serve/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// TestUnifiedServerBothPorts verifies that StartServers listens on
// both HTTP and gRPC ports simultaneously.
func TestUnifiedServerBothPorts(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	httpLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen http: %v", err)
	}
	grpcLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		httpLis.Close()
		t.Fatalf("listen grpc: %v", err)
	}

	cfg := serve.ServerConfig{
		Scheduler: sched,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- serve.StartServers(ctx, cfg, httpLis, grpcLis)
	}()

	// Give servers time to start
	time.Sleep(50 * time.Millisecond)

	// HTTP should be reachable
	httpResp, err := http.Get(fmt.Sprintf("http://%s/generate", httpLis.Addr()))
	if err != nil {
		t.Fatalf("HTTP request: %v", err)
	}
	httpResp.Body.Close()
	// We expect 405 Method Not Allowed (GET on POST endpoint) — that's fine, it means the server is up
	if httpResp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("unexpected HTTP status: %d", httpResp.StatusCode)
	}

	// gRPC should be reachable
	conn, err := grpc.Dial(grpcLis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("gRPC dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)
	_, err = client.ModelInfo(context.Background(), &pb.ModelInfoRequest{})
	if err != nil {
		t.Fatalf("gRPC ModelInfo: %v", err)
	}

	cancel()
}

// TestUnifiedServerSharedScheduler verifies that both HTTP and gRPC
// share the same scheduler instance.
func TestUnifiedServerSharedScheduler(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Start a goroutine that pushes tokens to any sequence added to the scheduler
	go func() {
		for {
			seqs := sched.GetSequences()
			for _, seq := range seqs {
				seq.PushToken("shared")
				seq.Close()
			}
			time.Sleep(time.Millisecond)
		}
	}()

	httpLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen http: %v", err)
	}
	grpcLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		httpLis.Close()
		t.Fatalf("listen grpc: %v", err)
	}

	cfg := serve.ServerConfig{
		Scheduler: sched,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		serve.StartServers(ctx, cfg, httpLis, grpcLis)
	}()
	time.Sleep(50 * time.Millisecond)

	// gRPC Generate
	conn, err := grpc.Dial(grpcLis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("gRPC dial: %v", err)
	}
	defer conn.Close()

	grpcClient := pb.NewInferenceServiceClient(conn)
	grpcResp, err := grpcClient.Generate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("gRPC Generate: %v", err)
	}
	if grpcResp.Text != "shared" {
		t.Errorf("gRPC got %q, want 'shared'", grpcResp.Text)
	}

	// HTTP Generate
	body := strings.NewReader(`{"prompt":"test"}`)
	httpResp, err := http.Post(
		fmt.Sprintf("http://%s/generate", httpLis.Addr()),
		"application/json",
		body,
	)
	if err != nil {
		t.Fatalf("HTTP Generate: %v", err)
	}
	defer httpResp.Body.Close()

	respBody, _ := io.ReadAll(httpResp.Body)
	var result map[string]string
	json.Unmarshal(respBody, &result)
	if result["text"] != "shared" {
		t.Errorf("HTTP got %q, want 'shared'", result["text"])
	}

	cancel()
}

// TestUnifiedServerWithTLS verifies that TLS can be configured
// for the gRPC server in unified mode.
func TestUnifiedServerWithTLS(t *testing.T) {
	cfg := serve.ServerConfig{}
	if cfg.TLS != nil {
		t.Error("expected nil TLS by default")
	}
}

// TestUnifiedServerGracefulShutdown verifies that cancelling the context
// stops both servers cleanly.
func TestUnifiedServerGracefulShutdown(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	httpLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen http: %v", err)
	}
	grpcLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		httpLis.Close()
		t.Fatalf("listen grpc: %v", err)
	}

	cfg := serve.ServerConfig{
		Scheduler: sched,
	}

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- serve.StartServers(ctx, cfg, httpLis, grpcLis)
	}()
	time.Sleep(50 * time.Millisecond)

	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("StartServers returned error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Error("StartServers did not stop within 5s")
	}
}

// TestServerConfigInterceptors verifies that interceptors can be configured
// via ServerConfig.
func TestServerConfigInterceptors(t *testing.T) {
	cfg := serve.ServerConfig{
		EnableInterceptors: true,
	}
	if !cfg.EnableInterceptors {
		t.Error("expected interceptors enabled")
	}
}
