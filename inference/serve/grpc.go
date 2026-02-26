package serve

import (
	"context"
	"strings"
	"time"

	"vexel/inference/scheduler"
	"vexel/inference/serve/pb"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GRPCServer implements the InferenceService gRPC interface.
type GRPCServer struct {
	pb.UnimplementedInferenceServiceServer
	scheduler *scheduler.Scheduler
}

// NewGRPCServer creates a new gRPC server wired to the given scheduler.
func NewGRPCServer(sched *scheduler.Scheduler) *GRPCServer {
	return &GRPCServer{
		scheduler: sched,
	}
}

// Generate handles non-streaming generation. It creates a sequence in the
// scheduler, collects all generated tokens, and returns the concatenated text.
// Respects context deadline and returns proper gRPC status codes.
func (s *GRPCServer) Generate(ctx context.Context, req *pb.GenerateRequest) (*pb.GenerateResponse, error) {
	if req.Prompt == "" {
		return nil, status.Errorf(codes.InvalidArgument, "prompt must not be empty")
	}
	if s.scheduler == nil {
		return nil, status.Errorf(codes.Internal, "scheduler not available")
	}

	// Create and register sequence
	seqID := scheduler.SequenceID(time.Now().UnixNano())
	seq := scheduler.NewSequence(seqID, req.Prompt)
	s.scheduler.AddSequence(seq)
	defer s.scheduler.RemoveSequence(seqID)

	// Collect tokens with context awareness
	var tokens []string
	for {
		select {
		case token, ok := <-seq.TokenChan():
			if !ok {
				// Channel closed — generation complete
				return &pb.GenerateResponse{
					Text: strings.Join(tokens, ""),
				}, nil
			}
			tokens = append(tokens, token)
		case <-ctx.Done():
			return nil, status.Errorf(codes.DeadlineExceeded, "request timed out")
		}
	}
}

// StreamGenerate handles streaming generation. Each generated token is sent
// as an individual response on the stream. Handles client cancellation and
// cleans up the scheduler sequence on exit.
func (s *GRPCServer) StreamGenerate(req *pb.GenerateRequest, stream pb.InferenceService_StreamGenerateServer) error {
	if req.Prompt == "" {
		return status.Errorf(codes.InvalidArgument, "prompt must not be empty")
	}
	if s.scheduler == nil {
		return status.Errorf(codes.Internal, "scheduler not available")
	}

	// Create and register sequence
	seqID := scheduler.SequenceID(time.Now().UnixNano())
	seq := scheduler.NewSequence(seqID, req.Prompt)
	s.scheduler.AddSequence(seq)
	defer s.scheduler.RemoveSequence(seqID)

	ctx := stream.Context()

	// Stream tokens as they arrive
	for {
		select {
		case token, ok := <-seq.TokenChan():
			if !ok {
				// Channel closed — generation complete
				return nil
			}
			if err := stream.Send(&pb.GenerateResponse{Text: token}); err != nil {
				return err
			}
		case <-ctx.Done():
			// Client cancelled or deadline exceeded
			return status.Errorf(codes.Canceled, "client cancelled")
		}
	}
}
