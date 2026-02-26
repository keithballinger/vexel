package serve

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// UnaryLoggingInterceptor returns a gRPC unary server interceptor that:
//   - Assigns or preserves a request ID (x-request-id) in response headers
//   - Records request duration (x-request-duration) in response trailers
func UnaryLoggingInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		start := time.Now()

		// Extract or generate request ID
		requestID := extractRequestID(ctx)

		// Send request ID in response header
		header := metadata.Pairs("x-request-id", requestID)
		if err := grpc.SendHeader(ctx, header); err != nil {
			// Non-fatal: log but continue
			_ = err
		}

		// Call the actual handler
		resp, err := handler(ctx, req)

		// Record duration in trailer
		duration := time.Since(start)
		trailer := metadata.Pairs("x-request-duration", formatDuration(duration))
		if setErr := grpc.SetTrailer(ctx, trailer); setErr != nil {
			_ = setErr
		}

		return resp, err
	}
}

// StreamLoggingInterceptor returns a gRPC stream server interceptor that:
//   - Assigns or preserves a request ID (x-request-id) in stream headers
func StreamLoggingInterceptor() grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		// Extract or generate request ID
		requestID := extractRequestID(ss.Context())

		// Send request ID in stream header
		header := metadata.Pairs("x-request-id", requestID)
		if err := ss.SendHeader(header); err != nil {
			_ = err
		}

		return handler(srv, ss)
	}
}

// extractRequestID checks incoming metadata for a client-provided x-request-id.
// If none is found, it generates a new UUID.
func extractRequestID(ctx context.Context) string {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if vals := md.Get("x-request-id"); len(vals) > 0 && vals[0] != "" {
			return vals[0]
		}
	}
	return uuid.New().String()
}

// formatDuration formats a duration as a human-readable string with units.
func formatDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%dµs", d.Microseconds())
	}
	return fmt.Sprintf("%.2fms", float64(d.Microseconds())/1000.0)
}

// KeepaliveParams holds gRPC keepalive configuration.
type KeepaliveParams struct {
	MaxConnectionIdle time.Duration
	MaxConnectionAge  time.Duration
	Time              time.Duration
	Timeout           time.Duration
}

// DefaultKeepalive returns sensible default keepalive parameters for the
// inference gRPC server.
func DefaultKeepalive() KeepaliveParams {
	return KeepaliveParams{
		MaxConnectionIdle: 5 * time.Minute,
		MaxConnectionAge:  30 * time.Minute,
		Time:              1 * time.Minute,
		Timeout:           20 * time.Second,
	}
}
