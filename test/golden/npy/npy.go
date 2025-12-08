// Package npy provides utilities for loading NumPy .npy files.
package npy

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// LoadFloat32 loads a .npy file containing float32 data.
// Returns the data as a flat []float32 slice and the shape.
func LoadFloat32(path string) ([]float32, []int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	// Read magic number
	magic := make([]byte, 6)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if string(magic) != "\x93NUMPY" {
		return nil, nil, fmt.Errorf("invalid numpy magic number")
	}

	// Read version
	version := make([]byte, 2)
	if _, err := io.ReadFull(f, version); err != nil {
		return nil, nil, fmt.Errorf("failed to read version: %w", err)
	}

	// Read header length
	var headerLen uint16
	if version[0] == 1 {
		if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
			return nil, nil, fmt.Errorf("failed to read header length: %w", err)
		}
	} else {
		var headerLen32 uint32
		if err := binary.Read(f, binary.LittleEndian, &headerLen32); err != nil {
			return nil, nil, fmt.Errorf("failed to read header length: %w", err)
		}
		headerLen = uint16(headerLen32)
	}

	// Read header
	header := make([]byte, headerLen)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, nil, fmt.Errorf("failed to read header: %w", err)
	}

	// Parse header to get dtype and shape
	headerStr := string(header)

	// Check dtype - we expect float32
	if !strings.Contains(headerStr, "'<f4'") && !strings.Contains(headerStr, "'>f4'") &&
		!strings.Contains(headerStr, "'float32'") {
		return nil, nil, fmt.Errorf("expected float32 dtype, got header: %s", headerStr)
	}

	// Parse shape
	shape, err := parseShape(headerStr)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse shape: %w", err)
	}

	// Calculate total elements
	numElements := 1
	for _, dim := range shape {
		numElements *= dim
	}

	// Read data
	data := make([]float32, numElements)
	if err := binary.Read(f, binary.LittleEndian, data); err != nil {
		return nil, nil, fmt.Errorf("failed to read data: %w", err)
	}

	return data, shape, nil
}

// parseShape extracts the shape tuple from a numpy header string.
func parseShape(header string) ([]int, error) {
	// Look for 'shape': (x, y, z) or 'shape': (x,)
	re := regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)
	matches := re.FindStringSubmatch(header)
	if len(matches) < 2 {
		return nil, fmt.Errorf("could not find shape in header: %s", header)
	}

	shapeStr := strings.TrimSpace(matches[1])
	if shapeStr == "" {
		return []int{}, nil // scalar
	}

	parts := strings.Split(shapeStr, ",")
	shape := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		dim, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid dimension %q: %w", p, err)
		}
		shape = append(shape, dim)
	}

	return shape, nil
}
