//go:build metal && darwin && cgo

package main

import (
	"fmt"
	"vexel/inference/backend"
	"vexel/inference/backend/metal"
)

// createMetalBackend creates a Metal GPU backend for inference.
func createMetalBackend() (backend.Backend, error) {
	b, err := metal.NewBackend(0) // Use first GPU device
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal backend: %w", err)
	}
	return b, nil
}
