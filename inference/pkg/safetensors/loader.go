package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

// Load parses a safetensors file from the given reader.
func Load(r io.Reader) (map[string]interface{}, error) {
	// 1. Read JSON header size (uint64)
	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	// 2. Read JSON header
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	// 3. Parse JSON
	var header map[string]interface{}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	return header, nil
}