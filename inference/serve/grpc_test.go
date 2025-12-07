package serve_test

import (
	"context"
	"testing"
	"vexel/inference/serve"
	"vexel/inference/serve/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"net"
)

func TestGRPCGenerate(t *testing.T) {
	// Setup listener
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	defer lis.Close()

	// Create and register server
	s := grpc.NewServer()
	srv := serve.NewGRPCServer(nil) // Mock scheduler
	pb.RegisterInferenceServiceServer(s, srv)
	
	go func() {
		if err := s.Serve(lis); err != nil {
			// t.Errorf here might be racey if test ends, but good for debug
		}
	}()
	defer s.Stop()

	// Client
	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)

	// Call Generate
	req := &pb.GenerateRequest{Prompt: "Hello"}
	resp, err := client.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.Text == "" {
		t.Error("Expected response text")
	}
}
