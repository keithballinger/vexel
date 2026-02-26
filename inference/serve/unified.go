package serve

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"

	"vexel/inference/scheduler"
	"vexel/inference/serve/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// ServerConfig holds configuration for the unified HTTP+gRPC server.
type ServerConfig struct {
	// Scheduler is the shared scheduler instance used by both servers.
	Scheduler *scheduler.Scheduler

	// TLS is an optional TLS configuration for the gRPC server.
	// When nil, gRPC serves without TLS.
	TLS *tls.Config

	// EnableInterceptors enables request tracing and logging interceptors
	// on the gRPC server.
	EnableInterceptors bool
}

// StartServers starts both an HTTP and gRPC server using the provided
// listeners and shared scheduler. It blocks until the context is cancelled,
// then performs graceful shutdown of both servers.
func StartServers(ctx context.Context, cfg ServerConfig, httpLis, grpcLis net.Listener) error {
	// Build gRPC server options
	var grpcOpts []grpc.ServerOption

	if cfg.TLS != nil {
		grpcOpts = append(grpcOpts, grpc.Creds(credentials.NewTLS(cfg.TLS)))
	}

	if cfg.EnableInterceptors {
		grpcOpts = append(grpcOpts,
			grpc.UnaryInterceptor(UnaryLoggingInterceptor()),
			grpc.StreamInterceptor(StreamLoggingInterceptor()),
		)
	}

	grpcServer := grpc.NewServer(grpcOpts...)
	grpcSvc := NewGRPCServer(cfg.Scheduler)
	pb.RegisterInferenceServiceServer(grpcServer, grpcSvc)

	// Build HTTP server
	httpSvc := NewServer(cfg.Scheduler)
	httpServer := &http.Server{Handler: httpSvc}

	// Channel to collect first error
	errCh := make(chan error, 2)

	// Start gRPC
	go func() {
		if err := grpcServer.Serve(grpcLis); err != nil {
			errCh <- err
		}
	}()

	// Start HTTP
	go func() {
		if err := httpServer.Serve(httpLis); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	// Wait for context cancellation or error
	select {
	case <-ctx.Done():
		// Graceful shutdown
		grpcServer.GracefulStop()
		httpServer.Close()
		return nil
	case err := <-errCh:
		grpcServer.GracefulStop()
		httpServer.Close()
		return err
	}
}
