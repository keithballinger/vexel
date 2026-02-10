//go:build !metal || !darwin || !cgo

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("Medusa test requires Metal backend. Build with: go run -tags metal ./inference/cmd/medusa_test")
	os.Exit(1)
}
