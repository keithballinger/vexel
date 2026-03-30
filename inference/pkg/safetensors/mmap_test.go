package safetensors_test

import (
	"os"
	"testing"
	"vexel/inference/pkg/safetensors"
)

func TestMmap(t *testing.T) {
	// Create a temp file
	f, err := os.CreateTemp("", "mmap_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	// Write some data
	data := []byte("Hello Mmap World")
	f.Write(data)
	f.Close()

	// Mmap it
	mapped, err := safetensors.Mmap(f.Name())
	if err != nil {
		t.Fatalf("Mmap failed: %v", err)
	}
	defer mapped.Close()

	// Verify content
	if string(mapped.Bytes()) != string(data) {
		t.Errorf("Expected %s, got %s", data, mapped.Bytes())
	}
}
