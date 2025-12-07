package serve

import (
	"context"
	"vexel/inference/scheduler"
	"vexel/inference/serve/pb"
)

// GRPCServer implements the InferenceService gRPC interface.
type GRPCServer struct {
	pb.UnimplementedInferenceServiceServer
	scheduler *scheduler.Scheduler
}

// NewGRPCServer creates a new gRPC server.
func NewGRPCServer(sched *scheduler.Scheduler) *GRPCServer {
	return &GRPCServer{
		scheduler: sched,
	}
}

// Generate handles non-streaming generation.
func (s *GRPCServer) Generate(ctx context.Context, req *pb.GenerateRequest) (*pb.GenerateResponse, error) {
	// TODO: Integrate with scheduler
	return &pb.GenerateResponse{
		Text: "Mock gRPC response for: " + req.Prompt,
	}, nil
}

// StreamGenerate handles streaming generation.
func (s *GRPCServer) StreamGenerate(req *pb.GenerateRequest, stream pb.InferenceService_StreamGenerateServer) error {
	// TODO: Integrate with scheduler subscription
	// Mock stream
	tokens := []string{"Mock", " ", "gRPC", " ", "stream", " ", "for: ", req.Prompt}
	for _, t := range tokens {
		if err := stream.Send(&pb.GenerateResponse{Text: t}); err != nil {
			return err
		}
	}
	return nil
}
