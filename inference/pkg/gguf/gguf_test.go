package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestTensorTypeString(t *testing.T) {
	tests := []struct {
		typ  TensorType
		want string
	}{
		{TensorTypeF32, "F32"},
		{TensorTypeF16, "F16"},
		{TensorTypeQ4_0, "Q4_0"},
		{TensorTypeQ8_0, "Q8_0"},
		{TensorTypeBF16, "BF16"},
		{TensorType(999), "Unknown(999)"},
	}

	for _, tt := range tests {
		if got := tt.typ.String(); got != tt.want {
			t.Errorf("TensorType(%d).String() = %q, want %q", tt.typ, got, tt.want)
		}
	}
}

func TestTensorTypeBlockSize(t *testing.T) {
	tests := []struct {
		typ  TensorType
		want int
	}{
		{TensorTypeF32, 1},
		{TensorTypeF16, 1},
		{TensorTypeQ4_0, 32},
		{TensorTypeQ8_0, 32},
		{TensorTypeQ4_K, 256},
	}

	for _, tt := range tests {
		if got := tt.typ.BlockSize(); got != tt.want {
			t.Errorf("TensorType(%d).BlockSize() = %d, want %d", tt.typ, got, tt.want)
		}
	}
}

func TestTensorTypeBytesPerBlock(t *testing.T) {
	tests := []struct {
		typ  TensorType
		want int
	}{
		{TensorTypeF32, 4},
		{TensorTypeF16, 2},
		{TensorTypeQ4_0, 18},  // 2 (scale) + 16 (32*4bits)
		{TensorTypeQ8_0, 34},  // 2 (scale) + 32 (32*8bits)
		{TensorTypeQ4_K, 144}, // K-quant
	}

	for _, tt := range tests {
		if got := tt.typ.BytesPerBlock(); got != tt.want {
			t.Errorf("TensorType(%d).BytesPerBlock() = %d, want %d", tt.typ, got, tt.want)
		}
	}
}

func TestTensorInfoNumElements(t *testing.T) {
	tests := []struct {
		dims []uint64
		want uint64
	}{
		{nil, 0},
		{[]uint64{}, 0},
		{[]uint64{10}, 10},
		{[]uint64{10, 20}, 200},
		{[]uint64{10, 20, 30}, 6000},
	}

	for _, tt := range tests {
		info := TensorInfo{Dimensions: tt.dims}
		if got := info.NumElements(); got != tt.want {
			t.Errorf("TensorInfo{Dimensions: %v}.NumElements() = %d, want %d", tt.dims, got, tt.want)
		}
	}
}

func TestMetadataValueAsUint32(t *testing.T) {
	tests := []struct {
		name string
		v    MetadataValue
		want uint32
	}{
		{"uint32", MetadataValue{Type: MetaTypeUint32, Uint32: 42}, 42},
		{"uint64", MetadataValue{Type: MetaTypeUint64, Uint64: 100}, 100},
		{"int32", MetadataValue{Type: MetaTypeInt32, Int32: 55}, 55},
		{"string", MetadataValue{Type: MetaTypeString, String: "test"}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.AsUint32(); got != tt.want {
				t.Errorf("AsUint32() = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestParseMinimalGGUF tests parsing a minimal synthetic GGUF file.
func TestParseMinimalGGUF(t *testing.T) {
	// Create a minimal valid GGUF file in memory
	buf := new(bytes.Buffer)

	// Magic
	binary.Write(buf, binary.LittleEndian, Magic)
	// Version
	binary.Write(buf, binary.LittleEndian, uint32(3))
	// Tensor count
	binary.Write(buf, binary.LittleEndian, uint64(1))
	// Metadata count
	binary.Write(buf, binary.LittleEndian, uint64(2))

	// Metadata 1: general.architecture = "llama"
	writeString(buf, "general.architecture")
	binary.Write(buf, binary.LittleEndian, uint32(MetaTypeString))
	writeString(buf, "llama")

	// Metadata 2: llama.block_count = 22
	writeString(buf, "llama.block_count")
	binary.Write(buf, binary.LittleEndian, uint32(MetaTypeUint32))
	binary.Write(buf, binary.LittleEndian, uint32(22))

	// Tensor info
	writeString(buf, "model.embed_tokens.weight")
	binary.Write(buf, binary.LittleEndian, uint32(2)) // nDims
	binary.Write(buf, binary.LittleEndian, uint64(32000))
	binary.Write(buf, binary.LittleEndian, uint64(2048))
	binary.Write(buf, binary.LittleEndian, uint32(TensorTypeF32))
	binary.Write(buf, binary.LittleEndian, uint64(0)) // offset

	// Parse
	f := &File{Metadata: make(map[string]MetadataValue)}
	r := bytes.NewReader(buf.Bytes())
	if err := f.parseHeader(r); err != nil {
		t.Fatalf("parseHeader() error = %v", err)
	}

	// Check metadata
	if f.Version != 3 {
		t.Errorf("Version = %d, want 3", f.Version)
	}
	if arch, ok := f.Metadata["general.architecture"]; !ok || arch.AsString() != "llama" {
		t.Errorf("Architecture = %v, want llama", arch)
	}
	if layers, ok := f.Metadata["llama.block_count"]; !ok || layers.AsUint32() != 22 {
		t.Errorf("block_count = %v, want 22", layers)
	}

	// Check tensors
	if len(f.Tensors) != 1 {
		t.Fatalf("len(Tensors) = %d, want 1", len(f.Tensors))
	}
	tensor := f.Tensors[0]
	if tensor.Name != "model.embed_tokens.weight" {
		t.Errorf("Tensor name = %q, want model.embed_tokens.weight", tensor.Name)
	}
	if tensor.NumElements() != 32000*2048 {
		t.Errorf("Tensor elements = %d, want %d", tensor.NumElements(), 32000*2048)
	}

	// Check extracted config
	f.extractModelConfig()
	if f.Architecture != "llama" {
		t.Errorf("Architecture = %q, want llama", f.Architecture)
	}
	if f.NumLayers != 22 {
		t.Errorf("NumLayers = %d, want 22", f.NumLayers)
	}
}

func writeString(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

func TestInvalidMagic(t *testing.T) {
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint32(0x12345678)) // Wrong magic

	f := &File{Metadata: make(map[string]MetadataValue)}
	r := bytes.NewReader(buf.Bytes())
	if err := f.parseHeader(r); err == nil {
		t.Error("Expected error for invalid magic")
	}
}

func TestUnsupportedVersion(t *testing.T) {
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, Magic)
	binary.Write(buf, binary.LittleEndian, uint32(1)) // Old version

	f := &File{Metadata: make(map[string]MetadataValue)}
	r := bytes.NewReader(buf.Bytes())
	if err := f.parseHeader(r); err == nil {
		t.Error("Expected error for unsupported version")
	}
}
