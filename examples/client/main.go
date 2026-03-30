package main

import (
	"context"
	"fmt"
	"log"
	"time"
	"vexel/client"
)

func main() {
	// Create a new client pointing to the local Vexel server
	c := client.New(client.Config{
		BaseURL: "http://localhost:8080",
		Timeout: 60 * time.Second,
	})

	ctx := context.Background()

	// 1. Simple generation
	fmt.Println("--- Simple Generation ---")
	prompt := "What is the capital of France?"
	fmt.Printf("Prompt: %s\n", prompt)

	resp, err := c.Generate(ctx, prompt, &client.GenerateOptions{
		Temperature: 0.7,
		MaxTokens:   50,
	})
	if err != nil {
		log.Printf("Generate failed (ensure server is running at localhost:8080): %v", err)
	} else {
		fmt.Printf("Response: %s\n", resp)
	}

	fmt.Println()

	// 2. Streaming generation
	fmt.Println("--- Streaming Generation ---")
	streamPrompt := "Write a haiku about coding."
	fmt.Printf("Prompt: %s\n", streamPrompt)

	tokenChan, err := c.Stream(ctx, streamPrompt, &client.GenerateOptions{
		Temperature: 0.9,
		MaxTokens:   100,
	})
	if err != nil {
		log.Fatalf("Stream failed: %v", err)
	}

	fmt.Print("Response: ")
	for token := range tokenChan {
		fmt.Print(token)
	}
	fmt.Println("\n--- Done ---")
}
