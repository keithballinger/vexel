package serve_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
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

// benchGRPCServer starts a gRPC server for benchmarking.
func benchGRPCServer(b *testing.B, sched *scheduler.Scheduler) (pb.InferenceServiceClient, func()) {
	b.Helper()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		b.Fatalf("listen: %v", err)
	}
	s := grpc.NewServer()
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)
	go func() { _ = s.Serve(lis) }()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		lis.Close()
		s.Stop()
		b.Fatalf("dial: %v", err)
	}
	cleanup := func() {
		conn.Close()
		s.Stop()
		lis.Close()
	}
	return pb.NewInferenceServiceClient(conn), cleanup
}

// benchHTTPServer starts an HTTP server for benchmarking.
func benchHTTPServer(b *testing.B, sched *scheduler.Scheduler) (*httptest.Server, func()) {
	b.Helper()
	srv := serve.NewServer(sched)
	ts := httptest.NewServer(srv)
	return ts, ts.Close
}

// continuousTokenFeeder runs a background goroutine that feeds single tokens
// to any sequence it discovers. Returns a done channel to stop it.
func continuousTokenFeeder(sched *scheduler.Scheduler) chan struct{} {
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
					s.PushToken("tok")
					s.Close()
				}(seq)
			}
			time.Sleep(100 * time.Microsecond)
		}
	}()
	return done
}

// BenchmarkGRPCSingleToken measures gRPC latency for single-token generation.
// This is a sequential benchmark — one request at a time.
func BenchmarkGRPCSingleToken(b *testing.B) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	done := continuousTokenFeeder(sched)
	defer close(done)

	client, cleanup := benchGRPCServer(b, sched)
	defer cleanup()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "bench"})
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
	}
}

// BenchmarkHTTPSingleToken measures HTTP latency for single-token generation.
// This is a sequential benchmark — one request at a time.
func BenchmarkHTTPSingleToken(b *testing.B) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	done := continuousTokenFeeder(sched)
	defer close(done)

	ts, cleanup := benchHTTPServer(b, sched)
	defer cleanup()

	httpClient := &http.Client{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		body := strings.NewReader(`{"prompt":"bench"}`)
		resp, err := httpClient.Post(ts.URL+"/generate", "application/json", body)
		if err != nil {
			b.Fatalf("HTTP: %v", err)
		}
		io.ReadAll(resp.Body)
		resp.Body.Close()
	}
}

// BenchmarkDirectSchedulerSingleToken measures direct scheduler access
// for single-token generation, as a baseline for overhead comparison.
func BenchmarkDirectSchedulerSingleToken(b *testing.B) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	done := continuousTokenFeeder(sched)
	defer close(done)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seqID := scheduler.SequenceID(time.Now().UnixNano() + int64(i))
		seq := scheduler.NewSequence(seqID, "bench")
		sched.AddSequence(seq)
		for range seq.TokenChan() {
		}
		sched.RemoveSequence(seqID)
	}
}

// BenchmarkGRPCWithInterceptors measures gRPC latency with interceptors enabled.
func BenchmarkGRPCWithInterceptors(b *testing.B) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	done := continuousTokenFeeder(sched)
	defer close(done)

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		b.Fatalf("listen: %v", err)
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
		b.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	defer s.Stop()
	defer lis.Close()

	client := pb.NewInferenceServiceClient(conn)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "bench"})
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
	}
}

// TestBenchmarkSmokeTest runs a quick sanity check that benchmark helpers work.
func TestBenchmarkSmokeTest(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	done := continuousTokenFeeder(sched)

	// Direct scheduler smoke test
	seqID := scheduler.SequenceID(time.Now().UnixNano())
	seq := scheduler.NewSequence(seqID, "smoke")
	sched.AddSequence(seq)
	for range seq.TokenChan() {
	}
	sched.RemoveSequence(seqID)

	close(done)
}
