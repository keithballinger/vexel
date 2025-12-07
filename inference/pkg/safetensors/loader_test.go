package safetensors_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"testing"
	"vexel/inference/pkg/safetensors"
)

func TestLoad(t *testing.T) {
	// Create a mock safetensors file
	// Format: <uint64 header_len><json header><data>
	
	header := map[string]interface{}{
		"weight": map[string]interface{}{
			"dtype": "F16",
			"shape": []int{10, 10},
			"data_offsets": []int{0, 200},
		},
	}
	headerBytes, _ := json.Marshal(header)
	
	var buf bytes.Buffer
	var headerLen uint64 = uint64(len(headerBytes))
	binary.Write(&buf, binary.LittleEndian, headerLen)
	buf.Write(headerBytes)
	
	// Data (dummy)
	buf.Write(make([]byte, 200))

	// Test loader
	meta, err := safetensors.Load(&buf)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	
	if meta == nil {
		t.Fatal("Expected metadata, got nil")
	}
}
