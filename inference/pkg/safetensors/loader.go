package safetensors

import (
	"bytes"
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

// ParseHeader parses the header from a byte slice and returns the header and data start offset.
func ParseHeader(data []byte) (map[string]interface{}, int, error) {
	if len(data) < 8 {
		return nil, 0, fmt.Errorf("data too short")
	}

	r := bytes.NewReader(data)
	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return nil, 0, err
	}

	if uint64(len(data)) < 8+headerLen {
		return nil, 0, fmt.Errorf("header length exceeds data size")
	}

	headerBytes := data[8 : 8+headerLen]
	var header map[string]interface{}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, 0, err
	}

	return header, int(8 + headerLen), nil
}
