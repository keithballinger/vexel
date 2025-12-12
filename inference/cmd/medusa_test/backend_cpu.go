//go:build !metal || !darwin || !cgo

package main

import (
	"errors"
	"vexel/inference/backend"
)

// createMetalBackend returns an error when Metal is not available.
func createMetalBackend() (backend.Backend, error) {
	return nil, errors.New("Metal backend not available - build with: go build -tags metal")
}
