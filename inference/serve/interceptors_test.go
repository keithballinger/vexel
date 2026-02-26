package serve_test

import (
	"context"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
	"vexel/inference/serve/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// TestUnaryInterceptorAddsRequestID verifies that the unary interceptor
// adds a request-id to the response metadata.
func TestUnaryInterceptorAddsRequestID(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"test"})

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()

	s := grpc.NewServer(grpc.UnaryInterceptor(serve.UnaryLoggingInterceptor()))
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)
	go func() { _ = s.Serve(lis) }()
	defer s.Stop()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)

	// Use a header to capture response metadata
	var header metadata.MD
	_, err = client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "test"}, grpc.Header(&header))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	reqIDs := header.Get("x-request-id")
	if len(reqIDs) == 0 {
		t.Error("expected x-request-id in response metadata")
	}
	if len(reqIDs) > 0 && reqIDs[0] == "" {
		t.Error("x-request-id should not be empty")
	}
}

// TestStreamInterceptorAddsRequestID verifies that the stream interceptor
// adds a request-id to the response metadata.
func TestStreamInterceptorAddsRequestID(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"tok"})

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()

	s := grpc.NewServer(grpc.StreamInterceptor(serve.StreamLoggingInterceptor()))
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)
	go func() { _ = s.Serve(lis) }()
	defer s.Stop()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)
	stream, err := client.StreamGenerate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("StreamGenerate: %v", err)
	}

	// Read header metadata
	header, err := stream.Header()
	if err != nil {
		t.Fatalf("get stream header: %v", err)
	}

	reqIDs := header.Get("x-request-id")
	if len(reqIDs) == 0 {
		t.Error("expected x-request-id in stream metadata")
	}

	// Drain stream
	for {
		_, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("recv: %v", err)
		}
	}
}

// TestUnaryInterceptorUsesClientRequestID verifies that the interceptor
// preserves a client-provided request ID instead of generating a new one.
func TestUnaryInterceptorUsesClientRequestID(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"ok"})

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()

	s := grpc.NewServer(grpc.UnaryInterceptor(serve.UnaryLoggingInterceptor()))
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)
	go func() { _ = s.Serve(lis) }()
	defer s.Stop()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)

	// Send request with client-provided request ID
	ctx := metadata.AppendToOutgoingContext(context.Background(), "x-request-id", "client-req-123")
	var header metadata.MD
	_, err = client.Generate(ctx, &pb.GenerateRequest{Prompt: "test"}, grpc.Header(&header))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	reqIDs := header.Get("x-request-id")
	if len(reqIDs) == 0 || reqIDs[0] != "client-req-123" {
		t.Errorf("expected preserved request ID 'client-req-123', got %v", reqIDs)
	}
}

// TestUnaryInterceptorLogsDuration verifies that the interceptor records
// request duration in the response trailer.
func TestUnaryInterceptorLogsDuration(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Simulate slightly delayed token generation
	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				time.Sleep(5 * time.Millisecond) // small delay
				seqs[0].PushToken("done")
				seqs[0].Close()
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()

	s := grpc.NewServer(grpc.UnaryInterceptor(serve.UnaryLoggingInterceptor()))
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)
	go func() { _ = s.Serve(lis) }()
	defer s.Stop()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)

	var trailer metadata.MD
	_, err = client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "test"}, grpc.Trailer(&trailer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	durations := trailer.Get("x-request-duration")
	if len(durations) == 0 {
		t.Error("expected x-request-duration in trailer")
	}
	if len(durations) > 0 && !strings.Contains(durations[0], "ms") && !strings.Contains(durations[0], "s") {
		t.Errorf("unexpected duration format: %q", durations[0])
	}
}

// TestKeepaliveConfig verifies that keepalive parameters can be configured.
func TestKeepaliveConfig(t *testing.T) {
	ka := serve.DefaultKeepalive()
	if ka.MaxConnectionIdle == 0 {
		t.Error("expected non-zero MaxConnectionIdle")
	}
	if ka.MaxConnectionAge == 0 {
		t.Error("expected non-zero MaxConnectionAge")
	}
	if ka.Time == 0 {
		t.Error("expected non-zero Time")
	}
	if ka.Timeout == 0 {
		t.Error("expected non-zero Timeout")
	}
}
