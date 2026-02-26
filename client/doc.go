// Package client provides a high-level Go client for the Vexel inference server.
//
// The client communicates with a running Vexel server over HTTP, supporting
// both blocking generation and real-time SSE token streaming.
//
// # Quick Start
//
// Create a client and generate text:
//
//	c := client.New(client.Config{
//	    BaseURL: "http://localhost:8080",
//	})
//
//	// Blocking: wait for full response
//	text, err := c.Generate(ctx, "What is Go?", nil)
//
//	// Streaming: receive tokens as they are generated
//	tokens, err := c.Stream(ctx, "Hello!", nil)
//	for tok := range tokens {
//	    fmt.Print(tok)
//	}
//
// # Generation Options
//
// Both Generate and Stream accept optional [GenerateOptions] to control
// sampling parameters:
//
//	opts := &client.GenerateOptions{
//	    Temperature: 0.7,
//	    MaxTokens:   100,
//	}
//	text, err := c.Generate(ctx, prompt, opts)
//
// # Thread Safety
//
// A single [Client] instance can be used concurrently from multiple goroutines.
// Each request creates its own HTTP connection.
package client
