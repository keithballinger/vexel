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
	"google.golang.org/grpc/status"
)

// startGRPCServer creates a test gRPC server and client, returning the client
// and a cleanup function. The scheduler should already have a background
// goroutine simulating token generation if needed.
func startGRPCServer(t *testing.T, sched *scheduler.Scheduler) (pb.InferenceServiceClient, func()) {
	t.Helper()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)

	go func() {
		_ = s.Serve(lis)
	}()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		s.Stop()
		lis.Close()
		t.Fatalf("failed to dial: %v", err)
	}

	client := pb.NewInferenceServiceClient(conn)
	cleanup := func() {
		conn.Close()
		s.Stop()
		lis.Close()
	}
	return client, cleanup
}

// simulateTokens runs a background goroutine that watches the scheduler
// for new sequences and pushes the given tokens to the first one found.
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

// TestGRPCGenerateCollectsTokens verifies that Generate wires to the scheduler,
// collects all tokens from the sequence, and returns the concatenated text.
func TestGRPCGenerateCollectsTokens(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"Hello", " ", "world"})

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	resp, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Text != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", resp.Text)
	}
}

// TestGRPCGenerateEmptyPrompt verifies that an empty prompt returns
// InvalidArgument status code.
func TestGRPCGenerateEmptyPrompt(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	_, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: ""})
	if err == nil {
		t.Fatal("expected error for empty prompt")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got: %v", err)
	}
	if st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", st.Code())
	}
}

// TestGRPCGenerateTimeout verifies that Generate respects context deadline
// and returns DeadlineExceeded when the scheduler takes too long.
func TestGRPCGenerateTimeout(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	// Don't simulate any tokens — sequence will never complete

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Generate(ctx, &pb.GenerateRequest{Prompt: "slow request"})
	if err == nil {
		t.Fatal("expected timeout error")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got: %v", err)
	}
	if st.Code() != codes.DeadlineExceeded {
		t.Errorf("expected DeadlineExceeded, got %v", st.Code())
	}
}

// TestGRPCGenerateNilScheduler verifies that Generate returns Internal
// error when no scheduler is configured.
func TestGRPCGenerateNilScheduler(t *testing.T) {
	client, cleanup := startGRPCServer(t, nil)
	defer cleanup()

	_, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err == nil {
		t.Fatal("expected error for nil scheduler")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got: %v", err)
	}
	if st.Code() != codes.Internal {
		t.Errorf("expected Internal, got %v", st.Code())
	}
}

// TestGRPCGenerateCleanup verifies that RemoveSequence is called after
// Generate completes, so the scheduler doesn't leak sequences.
func TestGRPCGenerateCleanup(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Simulate quick completion
	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("done")
				seqs[0].Close()
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	_, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "cleanup test"})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// Give a moment for deferred cleanup
	time.Sleep(10 * time.Millisecond)

	if count := sched.SequenceCount(); count != 0 {
		t.Errorf("expected 0 sequences after Generate, got %d", count)
	}
}

// TestGRPCStreamGenerateTokens verifies that StreamGenerate streams tokens
// individually from the scheduler sequence.
func TestGRPCStreamGenerateTokens(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"one", "two", "three"})

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	stream, err := client.StreamGenerate(context.Background(), &pb.GenerateRequest{Prompt: "stream test"})
	if err != nil {
		t.Fatalf("StreamGenerate failed: %v", err)
	}

	var tokens []string
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream recv failed: %v", err)
		}
		tokens = append(tokens, resp.Text)
	}

	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d", len(tokens))
	}
	if tokens[0] != "one" || tokens[1] != "two" || tokens[2] != "three" {
		t.Errorf("unexpected tokens: %v", tokens)
	}
}

// TestGRPCStreamGenerateEmptyPrompt verifies that StreamGenerate returns
// InvalidArgument for an empty prompt.
func TestGRPCStreamGenerateEmptyPrompt(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	stream, err := client.StreamGenerate(context.Background(), &pb.GenerateRequest{Prompt: ""})
	if err != nil {
		t.Fatalf("StreamGenerate call failed: %v", err)
	}

	// The error should come on first Recv
	_, err = stream.Recv()
	if err == nil {
		t.Fatal("expected error for empty prompt")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got: %v", err)
	}
	if st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", st.Code())
	}
}

// TestGRPCStreamGenerateCancellation verifies that cancelling the client
// context mid-stream causes the server to clean up the sequence.
func TestGRPCStreamGenerateCancellation(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Simulate slow token generation — push tokens with delays
	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("tok1")
				time.Sleep(100 * time.Millisecond)
				seqs[0].PushToken("tok2")
				time.Sleep(100 * time.Millisecond)
				seqs[0].PushToken("tok3")
				seqs[0].Close()
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.StreamGenerate(ctx, &pb.GenerateRequest{Prompt: "cancel me"})
	if err != nil {
		t.Fatalf("StreamGenerate call failed: %v", err)
	}

	// Read first token, then cancel
	resp, err := stream.Recv()
	if err != nil {
		t.Fatalf("first recv failed: %v", err)
	}
	if resp.Text != "tok1" {
		t.Errorf("expected tok1, got %q", resp.Text)
	}

	cancel()

	// Subsequent reads should fail
	_, err = stream.Recv()
	if err == nil {
		t.Error("expected error after cancellation")
	}

	// Allow cleanup time
	time.Sleep(50 * time.Millisecond)

	if count := sched.SequenceCount(); count != 0 {
		t.Errorf("expected 0 sequences after cancellation, got %d", count)
	}
}

// TestGRPCStreamGenerateNilScheduler verifies that StreamGenerate returns
// Internal error when no scheduler is configured.
func TestGRPCStreamGenerateNilScheduler(t *testing.T) {
	client, cleanup := startGRPCServer(t, nil)
	defer cleanup()

	stream, err := client.StreamGenerate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("StreamGenerate call failed: %v", err)
	}

	_, err = stream.Recv()
	if err == nil {
		t.Fatal("expected error for nil scheduler")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got: %v", err)
	}
	if st.Code() != codes.Internal {
		t.Errorf("expected Internal, got %v", st.Code())
	}
}

// --- Tests for expanded proto schema ---

// TestGRPCGenerateTokenCount verifies that Generate returns the correct
// token_count in the response.
func TestGRPCGenerateTokenCount(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"Hello", " ", "world"})

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	resp, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.TokenCount != 3 {
		t.Errorf("expected token_count=3, got %d", resp.TokenCount)
	}
}

// TestGRPCGenerateFinishReason verifies that Generate sets finish_reason
// to "eos" when generation completes normally.
func TestGRPCGenerateFinishReason(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"done"})

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	resp, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.FinishReason != "eos" {
		t.Errorf("expected finish_reason='eos', got %q", resp.FinishReason)
	}
}

// TestGRPCStreamTokenCount verifies that StreamGenerate sends incrementing
// token_count with each message.
func TestGRPCStreamTokenCount(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"a", "b", "c"})

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	stream, err := client.StreamGenerate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("StreamGenerate failed: %v", err)
	}

	var counts []int32
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("recv failed: %v", err)
		}
		counts = append(counts, resp.TokenCount)
	}

	if len(counts) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(counts))
	}
	for i, c := range counts {
		if c != int32(i+1) {
			t.Errorf("message %d: expected token_count=%d, got %d", i, i+1, c)
		}
	}
}

// TestGRPCStreamFinishReason verifies that the last streamed message
// has finish_reason="eos" and intermediate messages have it empty.
func TestGRPCStreamFinishReason(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"tok1", "tok2", "tok3"})

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	stream, err := client.StreamGenerate(context.Background(), &pb.GenerateRequest{Prompt: "test"})
	if err != nil {
		t.Fatalf("StreamGenerate failed: %v", err)
	}

	var responses []*pb.GenerateResponse
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("recv failed: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) != 3 {
		t.Fatalf("expected 3 responses, got %d", len(responses))
	}

	// Intermediate messages should have no finish_reason
	for i := 0; i < len(responses)-1; i++ {
		if responses[i].FinishReason != "" {
			t.Errorf("message %d: expected empty finish_reason, got %q", i, responses[i].FinishReason)
		}
	}

	// Last message should have finish_reason="eos"
	last := responses[len(responses)-1]
	if last.FinishReason != "eos" {
		t.Errorf("last message: expected finish_reason='eos', got %q", last.FinishReason)
	}
}

// TestGRPCModelInfo verifies that ModelInfo returns model metadata.
func TestGRPCModelInfo(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	resp, err := client.ModelInfo(context.Background(), &pb.ModelInfoRequest{})
	if err != nil {
		t.Fatalf("ModelInfo failed: %v", err)
	}

	// Zero-value runtime defaults: NormRMSNorm=0, MLPSwiGLU=0 → "llama"
	// VocabSize=0 → model_name="unknown"
	if resp.ModelName != "unknown" {
		t.Errorf("expected model_name='unknown' for zero config, got %q", resp.ModelName)
	}
	if resp.Architecture != "llama" {
		t.Errorf("expected architecture='llama' for zero config, got %q", resp.Architecture)
	}
	if resp.Quantization != "F32" {
		t.Errorf("expected quantization='F32' for zero config, got %q", resp.Quantization)
	}
}

// TestGRPCModelInfoNilScheduler verifies ModelInfo returns Internal
// error when no scheduler is configured.
func TestGRPCModelInfoNilScheduler(t *testing.T) {
	client, cleanup := startGRPCServer(t, nil)
	defer cleanup()

	_, err := client.ModelInfo(context.Background(), &pb.ModelInfoRequest{})
	if err == nil {
		t.Fatal("expected error for nil scheduler")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got: %v", err)
	}
	if st.Code() != codes.Internal {
		t.Errorf("expected Internal, got %v", st.Code())
	}
}

// TestGRPCSamplingParamsAccepted verifies that requests with sampling
// parameters are accepted without error.
func TestGRPCSamplingParamsAccepted(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	simulateTokens(sched, []string{"ok"})

	client, cleanup := startGRPCServer(t, sched)
	defer cleanup()

	resp, err := client.Generate(context.Background(), &pb.GenerateRequest{
		Prompt: "test with params",
		SamplingParams: &pb.SamplingParams{
			Temperature: 0.7,
			TopK:        50,
			TopP:        0.9,
			MaxTokens:   100,
		},
	})
	if err != nil {
		t.Fatalf("Generate with sampling params failed: %v", err)
	}
	if resp.Text != "ok" {
		t.Errorf("expected 'ok', got %q", resp.Text)
	}
}
