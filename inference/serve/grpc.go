package serve

import (
	"context"
	"fmt"
	"strings"
	"time"

	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve/pb"
	"vexel/inference/tensor"

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
// scheduler, collects all generated tokens, and returns the concatenated text
// with token count and finish reason.
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

	// Apply per-request sampling params if provided
	if params := req.GetSamplingParams(); params != nil {
		if params.MaxTokens > 0 {
			seq.SetMaxTokens(int(params.MaxTokens))
		}
		if params.Temperature > 0 {
			seq.SetSamplingParams(params.Temperature, int(params.TopK), params.TopP)
		}
	}

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
					Text:         strings.Join(tokens, ""),
					TokenCount:   int32(len(tokens)),
					FinishReason: "eos",
				}, nil
			}
			tokens = append(tokens, token)
		case <-ctx.Done():
			return nil, status.Errorf(codes.DeadlineExceeded, "request timed out")
		}
	}
}

// StreamGenerate handles streaming generation. Each generated token is sent
// as an individual response on the stream with incrementing token count.
// The final message includes finish_reason="eos".
// Handles client cancellation and cleans up the scheduler sequence on exit.
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

	// Apply per-request sampling params if provided
	if params := req.GetSamplingParams(); params != nil {
		if params.MaxTokens > 0 {
			seq.SetMaxTokens(int(params.MaxTokens))
		}
		if params.Temperature > 0 {
			seq.SetSamplingParams(params.Temperature, int(params.TopK), params.TopP)
		}
	}

	s.scheduler.AddSequence(seq)
	defer s.scheduler.RemoveSequence(seqID)

	ctx := stream.Context()

	// Stream tokens as they arrive. We buffer one token ahead to detect
	// when the channel closes so we can mark the final message with finish_reason.
	var tokenCount int32
	var pending *string // buffered token waiting to be sent

	for {
		select {
		case token, ok := <-seq.TokenChan():
			if !ok {
				// Channel closed — send last pending token with finish_reason
				if pending != nil {
					tokenCount++
					if err := stream.Send(&pb.GenerateResponse{
						Text:         *pending,
						TokenCount:   tokenCount,
						FinishReason: "eos",
					}); err != nil {
						return err
					}
				}
				return nil
			}
			// Send the previously buffered token (without finish_reason)
			if pending != nil {
				tokenCount++
				if err := stream.Send(&pb.GenerateResponse{
					Text:       *pending,
					TokenCount: tokenCount,
				}); err != nil {
					return err
				}
			}
			t := token
			pending = &t
		case <-ctx.Done():
			return status.Errorf(codes.Canceled, "client cancelled")
		}
	}
}

// ModelInfo returns metadata about the loaded model.
func (s *GRPCServer) ModelInfo(ctx context.Context, req *pb.ModelInfoRequest) (*pb.ModelInfoResponse, error) {
	if s.scheduler == nil {
		return nil, status.Errorf(codes.Internal, "scheduler not available")
	}

	cfg := s.scheduler.ModelConfig()

	return &pb.ModelInfoResponse{
		ModelName:        inferModelName(cfg),
		Architecture:     inferArchitecture(cfg),
		Quantization:     dtypeString(cfg.DType),
		MaxContextLength: int32(cfg.MaxSeqLen),
	}, nil
}

// inferArchitecture determines architecture name from model config.
func inferArchitecture(cfg runtime.ModelConfig) string {
	normType := cfg.NormType.String()
	mlpType := cfg.MLPType.String()

	switch {
	case normType == "RMSNorm" && mlpType == "SwiGLU":
		return "llama"
	case normType == "LayerNorm" && mlpType == "GELU":
		return "phi"
	case normType == "RMSNorm" && mlpType == "GELU":
		return "gemma"
	default:
		if normType != "" {
			return normType + "/" + mlpType
		}
		return "unknown"
	}
}

// inferModelName generates a descriptive model name from config parameters.
func inferModelName(cfg runtime.ModelConfig) string {
	if cfg.VocabSize == 0 {
		return "unknown"
	}
	arch := inferArchitecture(cfg)
	return fmt.Sprintf("%s-%dh-%dl", arch, cfg.HiddenSize, cfg.NumHiddenLayers)
}

// dtypeString converts a tensor DType to a human-readable quantization string.
func dtypeString(d tensor.DType) string {
	switch d {
	case tensor.Float32:
		return "F32"
	case tensor.Float16:
		return "F16"
	case tensor.BFloat16:
		return "BF16"
	case tensor.Int8:
		return "Q8_0"
	case tensor.Uint8:
		return "Q4_0"
	default:
		return "unknown"
	}
}
