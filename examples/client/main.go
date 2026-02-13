package main

import (
	"context"
	"fmt"
	"log"
	"vexel/client"
)

func main() {
	// Create a new client
	c := client.New(client.Config{
		BaseURL: "http://localhost:8080",
	})

	ctx := context.Background()

	// 1. Simple generation
	fmt.Println("--- Simple Generation ---")
	resp, err := c.Generate(ctx, "What is the capital of France?", nil)
	if err != nil {
		log.Printf("Generate failed (is the server running?): %v", err)
	} else {
		fmt.Printf("Response: %s
", resp)
	}

	fmt.Println()

	// 2. Streaming generation
	fmt.Println("--- Streaming Generation ---")
	tokenChan, err := c.Stream(ctx, "Tell me a short story.", nil)
	if err != nil {
		log.Fatalf("Stream failed: %v", err)
	}

	fmt.Print("Response: ")
	for token := range tokenChan {
		fmt.Print(token)
	}
	fmt.Println("
--- Done ---")
}
