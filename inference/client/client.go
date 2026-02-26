// Package client provides HTTP and gRPC clients for the Vexel inference server.
//
// The package supports three client types:
//   - GRPCClient: connects via gRPC for efficient binary streaming
//   - HTTPClient: connects via HTTP/SSE for browser-compatible streaming
//   - AutoClient: tries gRPC first, falls back to HTTP if unavailable
//
// All clients implement the Client interface with Generate() and Stream() methods.
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"vexel/inference/serve/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is the interface implemented by all Vexel inference clients.
type Client interface {
	// Generate sends a prompt and returns the complete generated text.
	Generate(ctx context.Context, prompt string) (string, error)

	// Stream sends a prompt and returns channels for streaming tokens.
	// The token channel receives individual tokens. The error channel
	// receives any error (or nil on success) after the token channel closes.
	Stream(ctx context.Context, prompt string) (<-chan string, <-chan error)

	// Close releases any resources held by the client.
	Close() error
}

// GRPCOption configures a GRPCClient.
type GRPCOption func(*grpcOptions)

type grpcOptions struct {
	insecure bool
}

// WithInsecure disables TLS for the gRPC connection.
func WithInsecure() GRPCOption {
	return func(o *grpcOptions) {
		o.insecure = true
	}
}

// GRPCClient connects to the Vexel server via gRPC.
type GRPCClient struct {
	conn   *grpc.ClientConn
	client pb.InferenceServiceClient
}

// NewGRPCClient creates a new gRPC client connected to the given address.
func NewGRPCClient(addr string, opts ...GRPCOption) (*GRPCClient, error) {
	var cfg grpcOptions
	for _, opt := range opts {
		opt(&cfg)
	}

	var dialOpts []grpc.DialOption
	if cfg.insecure {
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	conn, err := grpc.Dial(addr, dialOpts...)
	if err != nil {
		return nil, fmt.Errorf("grpc dial: %w", err)
	}

	return &GRPCClient{
		conn:   conn,
		client: pb.NewInferenceServiceClient(conn),
	}, nil
}

// Generate sends a prompt and returns the complete generated text via gRPC.
func (c *GRPCClient) Generate(ctx context.Context, prompt string) (string, error) {
	resp, err := c.client.Generate(ctx, &pb.GenerateRequest{Prompt: prompt})
	if err != nil {
		return "", fmt.Errorf("grpc generate: %w", err)
	}
	return resp.Text, nil
}

// Stream sends a prompt and streams tokens back via gRPC server streaming.
func (c *GRPCClient) Stream(ctx context.Context, prompt string) (<-chan string, <-chan error) {
	tokenCh := make(chan string, 100)
	errCh := make(chan error, 1)

	go func() {
		defer close(tokenCh)
		defer close(errCh)

		stream, err := c.client.StreamGenerate(ctx, &pb.GenerateRequest{Prompt: prompt})
		if err != nil {
			errCh <- fmt.Errorf("grpc stream: %w", err)
			return
		}

		for {
			resp, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				errCh <- fmt.Errorf("grpc recv: %w", err)
				return
			}
			tokenCh <- resp.Text
		}
	}()

	return tokenCh, errCh
}

// Close closes the gRPC connection.
func (c *GRPCClient) Close() error {
	return c.conn.Close()
}

// HTTPClient connects to the Vexel server via HTTP/SSE.
type HTTPClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewHTTPClient creates a new HTTP client targeting the given base URL.
func NewHTTPClient(baseURL string) *HTTPClient {
	return &HTTPClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

// Generate sends a prompt via HTTP POST and returns the complete generated text.
func (c *HTTPClient) Generate(ctx context.Context, prompt string) (string, error) {
	body, _ := json.Marshal(map[string]string{"prompt": prompt})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/generate", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("http request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("http generate: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("http status %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("http decode: %w", err)
	}
	return result.Text, nil
}

// Stream sends a prompt via HTTP POST and streams SSE tokens back.
func (c *HTTPClient) Stream(ctx context.Context, prompt string) (<-chan string, <-chan error) {
	tokenCh := make(chan string, 100)
	errCh := make(chan error, 1)

	go func() {
		defer close(tokenCh)
		defer close(errCh)

		body, _ := json.Marshal(map[string]string{"prompt": prompt})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/stream", bytes.NewReader(body))
		if err != nil {
			errCh <- fmt.Errorf("http request: %w", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")

		// Use a client without timeout for streaming
		streamClient := &http.Client{}
		resp, err := streamClient.Do(req)
		if err != nil {
			errCh <- fmt.Errorf("http stream: %w", err)
			return
		}
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			var event struct {
				Token string `json:"token"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue // skip malformed events
			}
			if event.Token != "" {
				tokenCh <- event.Token
			}
		}
	}()

	return tokenCh, errCh
}

// Close is a no-op for HTTP clients.
func (c *HTTPClient) Close() error {
	return nil
}

// AutoClient tries gRPC first, falling back to HTTP if gRPC is unavailable.
type AutoClient struct {
	active Client
	grpc   *GRPCClient
	http   *HTTPClient
}

// NewAutoClient creates a client that prefers gRPC but falls back to HTTP.
// httpURL is the HTTP server address and grpcAddr is the gRPC server address.
func NewAutoClient(httpURL, grpcAddr string) (*AutoClient, error) {
	ac := &AutoClient{
		http: NewHTTPClient(httpURL),
	}

	// Try to connect via gRPC
	grpcClient, err := NewGRPCClient(grpcAddr, WithInsecure())
	if err == nil {
		// Test the connection with a quick ModelInfo call
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		_, err = grpcClient.client.ModelInfo(ctx, &pb.ModelInfoRequest{})
		if err == nil {
			ac.grpc = grpcClient
			ac.active = grpcClient
			return ac, nil
		}
		grpcClient.Close()
	}

	// Fall back to HTTP
	ac.active = ac.http
	return ac, nil
}

// Generate delegates to the active client (gRPC or HTTP).
func (c *AutoClient) Generate(ctx context.Context, prompt string) (string, error) {
	return c.active.Generate(ctx, prompt)
}

// Stream delegates to the active client (gRPC or HTTP).
func (c *AutoClient) Stream(ctx context.Context, prompt string) (<-chan string, <-chan error) {
	return c.active.Stream(ctx, prompt)
}

// Close closes the active client and any connections.
func (c *AutoClient) Close() error {
	if c.grpc != nil {
		return c.grpc.Close()
	}
	return c.http.Close()
}
