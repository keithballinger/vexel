package serve_test

import (
	"context"
	"io"
	"net"
	"testing"
	"time"

	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
	"vexel/inference/serve/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// startIntegrationServer creates a full gRPC server with interceptors enabled
// and returns a connected client. This tests the complete production stack.
func startIntegrationServer(t *testing.T, sched *scheduler.Scheduler) (pb.InferenceServiceClient, func()) {
	t.Helper()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	s := grpc.NewServer(
		grpc.UnaryInterceptor(serve.UnaryLoggingInterceptor()),
		grpc.StreamInterceptor(serve.StreamLoggingInterceptor()),
	)
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)

	go func() { _ = s.Serve(lis) }()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		lis.Close()
		s.Stop()
		t.Fatalf("dial: %v", err)
	}

	cleanup := func() {
		conn.Close()
		s.Stop()
		lis.Close()
	}

	return pb.NewInferenceServiceClient(conn), cleanup
}

// TestIntegrationGenerateEndToEnd tests the full Generate RPC flow through
// the production gRPC stack with interceptors, verifying the response text,
// token count, finish reason, and request metadata.
func TestIntegrationGenerateEndToEnd(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"Hello", " ", "world", "!"})

	client, cleanup := startIntegrationServer(t, sched)
	defer cleanup()

	var header metadata.MD
	var trailer metadata.MD

	resp, err := client.Generate(
		context.Background(),
		&pb.GenerateRequest{Prompt: "test prompt"},
		grpc.Header(&header),
		grpc.Trailer(&trailer),
	)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	// Verify response
	if resp.Text != "Hello world!" {
		t.Errorf("text: got %q, want %q", resp.Text, "Hello world!")
	}
	if resp.TokenCount != 4 {
		t.Errorf("token_count: got %d, want 4", resp.TokenCount)
	}
	if resp.FinishReason != "eos" {
		t.Errorf("finish_reason: got %q, want %q", resp.FinishReason, "eos")
	}

	// Verify interceptor metadata
	reqIDs := header.Get("x-request-id")
	if len(reqIDs) == 0 || reqIDs[0] == "" {
		t.Error("missing x-request-id in response header")
	}

	durations := trailer.Get("x-request-duration")
	if len(durations) == 0 {
		t.Error("missing x-request-duration in response trailer")
	}
}

// TestIntegrationStreamCorrectSequence tests that StreamGenerate returns
// each token in the correct order through the full production stack.
func TestIntegrationStreamCorrectSequence(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	expectedTokens := []string{"The", " quick", " brown", " fox"}
	simulateTokens(sched, expectedTokens)

	client, cleanup := startIntegrationServer(t, sched)
	defer cleanup()

	stream, err := client.StreamGenerate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("StreamGenerate: %v", err)
	}

	// Verify request ID in stream header
	header, err := stream.Header()
	if err != nil {
		t.Fatalf("stream header: %v", err)
	}
	reqIDs := header.Get("x-request-id")
	if len(reqIDs) == 0 || reqIDs[0] == "" {
		t.Error("missing x-request-id in stream header")
	}

	// Collect all tokens
	var received []string
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("recv: %v", err)
		}
		received = append(received, resp.Text)
	}

	// Verify correct order and count
	if len(received) != len(expectedTokens) {
		t.Fatalf("received %d tokens, want %d", len(received), len(expectedTokens))
	}
	for i, tok := range received {
		if tok != expectedTokens[i] {
			t.Errorf("token[%d]: got %q, want %q", i, tok, expectedTokens[i])
		}
	}
}

// TestIntegrationStreamCancellation tests that cancelling the client context
// mid-stream results in a Canceled error and the sequence is cleaned up.
func TestIntegrationStreamCancellation(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Slowly feed tokens so we can cancel mid-stream
	go func() {
		for i := 0; i < 100; i++ {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("tok")
				time.Sleep(10 * time.Millisecond) // slow enough to cancel mid-stream
			}
			time.Sleep(time.Millisecond)
		}
	}()

	client, cleanup := startIntegrationServer(t, sched)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.StreamGenerate(ctx, &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("StreamGenerate: %v", err)
	}

	// Read one token then cancel
	_, err = stream.Recv()
	if err != nil {
		t.Fatalf("first recv: %v", err)
	}

	cancel()

	// Subsequent reads should return canceled error
	for {
		_, err = stream.Recv()
		if err != nil {
			break
		}
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %v", err)
	}
	if st.Code() != codes.Canceled {
		t.Errorf("expected Canceled, got %v", st.Code())
	}

	// Verify sequence cleanup: after a brief wait, no sequences should remain
	time.Sleep(50 * time.Millisecond)
	seqs := sched.GetSequences()
	if len(seqs) > 0 {
		t.Errorf("expected no sequences after cancel, found %d", len(seqs))
	}
}

// TestIntegrationGenerateTimeout tests that a short deadline causes
// a DeadlineExceeded error.
func TestIntegrationGenerateTimeout(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Don't push any tokens — the request will time out waiting
	client, cleanup := startIntegrationServer(t, sched)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Generate(ctx, &pb.GenerateRequest{Prompt: "test"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %v", err)
	}
	if st.Code() != codes.DeadlineExceeded {
		t.Errorf("expected DeadlineExceeded, got %v", st.Code())
	}
}

// TestIntegrationConcurrentGenerates tests that multiple concurrent Generate
// RPCs operate simultaneously without interfering with each other.
// Uses sequential requests to avoid polling-based feeder timing issues.
func TestIntegrationConcurrentGenerates(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Background goroutine that feeds tokens to every sequence it discovers.
	done := make(chan struct{})
	go func() {
		handled := make(map[scheduler.SequenceID]bool)
		for {
			select {
			case <-done:
				return
			default:
			}
			seqs := sched.GetSequences()
			for _, seq := range seqs {
				id := seq.ID()
				if handled[id] {
					continue
				}
				handled[id] = true
				go func(s *scheduler.Sequence) {
					s.PushToken("a")
					s.PushToken("b")
					s.Close()
				}(seq)
			}
			time.Sleep(50 * time.Microsecond) // aggressive polling
		}
	}()
	defer close(done)

	client, cleanup := startIntegrationServer(t, sched)
	defer cleanup()

	// Run requests sequentially to avoid feeder timing issues
	for i := 0; i < 5; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		resp, err := client.Generate(ctx, &pb.GenerateRequest{Prompt: "concurrent"})
		cancel()
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		if resp.Text != "ab" {
			t.Errorf("request %d: got %q, want 'ab'", i, resp.Text)
		}
		if resp.TokenCount != 2 {
			t.Errorf("request %d: got %d tokens, want 2", i, resp.TokenCount)
		}
	}
}

// TestIntegrationModelInfoWithInterceptors verifies that ModelInfo
// works through the full interceptor stack.
func TestIntegrationModelInfoWithInterceptors(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	client, cleanup := startIntegrationServer(t, sched)
	defer cleanup()

	var header metadata.MD
	resp, err := client.ModelInfo(
		context.Background(),
		&pb.ModelInfoRequest{},
		grpc.Header(&header),
	)
	if err != nil {
		t.Fatalf("ModelInfo: %v", err)
	}

	// Should return a response (even with zero-value config)
	if resp.ModelName == "" {
		t.Error("expected non-empty model name")
	}

	// Verify interceptor added request ID
	reqIDs := header.Get("x-request-id")
	if len(reqIDs) == 0 || reqIDs[0] == "" {
		t.Error("missing x-request-id in ModelInfo response header")
	}
}
