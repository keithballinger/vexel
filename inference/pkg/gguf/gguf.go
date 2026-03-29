// Package gguf implements parsing for the GGUF file format used by llama.cpp.
// GGUF (General GGML Universal Format) stores quantized LLM weights with metadata.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Magic number for GGUF files: "GGUF" in little-endian
const Magic uint32 = 0x46554747 // "GGUF" as uint32 little-endian

// Current GGUF version
const Version uint32 = 3

// MetadataType represents the type of a metadata value.
type MetadataType uint32

const (
	MetaTypeUint8   MetadataType = 0
	MetaTypeInt8    MetadataType = 1
	MetaTypeUint16  MetadataType = 2
	MetaTypeInt16   MetadataType = 3
	MetaTypeUint32  MetadataType = 4
	MetaTypeInt32   MetadataType = 5
	MetaTypeFloat32 MetadataType = 6
	MetaTypeBool    MetadataType = 7
	MetaTypeString  MetadataType = 8
	MetaTypeArray   MetadataType = 9
	MetaTypeUint64  MetadataType = 10
	MetaTypeInt64   MetadataType = 11
	MetaTypeFloat64 MetadataType = 12
)

// TensorType represents the quantization type of tensor data.
type TensorType uint32

const (
	TensorTypeF32  TensorType = 0
	TensorTypeF16  TensorType = 1
	TensorTypeQ4_0 TensorType = 2
	TensorTypeQ4_1 TensorType = 3
	TensorTypeQ5_0 TensorType = 6
	TensorTypeQ5_1 TensorType = 7
	TensorTypeQ8_0 TensorType = 8
	TensorTypeQ8_1 TensorType = 9
	TensorTypeQ2_K TensorType = 10
	TensorTypeQ3_K TensorType = 11
	TensorTypeQ4_K TensorType = 12
	TensorTypeQ5_K TensorType = 13
	TensorTypeQ6_K TensorType = 14
	TensorTypeQ8_K TensorType = 15
	TensorTypeIQ2  TensorType = 16
	TensorTypeIQ3  TensorType = 17
	TensorTypeIQ1  TensorType = 18
	TensorTypeBF16 TensorType = 30
)

// String returns the name of the tensor type.
func (t TensorType) String() string {
	switch t {
	case TensorTypeF32:
		return "F32"
	case TensorTypeF16:
		return "F16"
	case TensorTypeQ4_0:
		return "Q4_0"
	case TensorTypeQ4_1:
		return "Q4_1"
	case TensorTypeQ5_0:
		return "Q5_0"
	case TensorTypeQ5_1:
		return "Q5_1"
	case TensorTypeQ8_0:
		return "Q8_0"
	case TensorTypeQ8_1:
		return "Q8_1"
	case TensorTypeQ4_K:
		return "Q4_K"
	case TensorTypeQ5_K:
		return "Q5_K"
	case TensorTypeQ6_K:
		return "Q6_K"
	case TensorTypeBF16:
		return "BF16"
	default:
		return fmt.Sprintf("Unknown(%d)", t)
	}
}

// BlockSize returns the number of elements per quantization block.
func (t TensorType) BlockSize() int {
	switch t {
	case TensorTypeF32, TensorTypeF16, TensorTypeBF16:
		return 1
	case TensorTypeQ4_0, TensorTypeQ4_1, TensorTypeQ5_0, TensorTypeQ5_1, TensorTypeQ8_0, TensorTypeQ8_1:
		return 32
	case TensorTypeQ4_K, TensorTypeQ5_K, TensorTypeQ6_K:
		return 256
	default:
		return 32
	}
}

// BytesPerBlock returns the number of bytes per quantization block.
func (t TensorType) BytesPerBlock() int {
	switch t {
	case TensorTypeF32:
		return 4
	case TensorTypeF16, TensorTypeBF16:
		return 2
	case TensorTypeQ4_0:
		return 2 + 16 // scale (f16) + 32 * 4 bits
	case TensorTypeQ5_0:
		return 22 // scale (f16) + qh(4) + 32 * 4 bits
	case TensorTypeQ5_1:
		return 24 // scale (f16) + min (f16) + qh(4) + 32 * 4 bits
	case TensorTypeQ8_0:
		return 2 + 32 // scale (f16) + 32 * 8 bits
	case TensorTypeQ4_K:
		return 144 // Complex k-quant structure
	case TensorTypeQ5_K:
		return 176 // d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
	case TensorTypeQ6_K:
		return 210
	default:
		return 32
	}
}

// MetadataValue holds a parsed metadata value.
type MetadataValue struct {
	Type       MetadataType
	Uint8      uint8
	Int8       int8
	Uint16     uint16
	Int16      int16
	Uint32     uint32
	Int32      int32
	Uint64     uint64
	Int64      int64
	Float32    float32
	Float64    float64
	Bool       bool
	String     string
	Array      []MetadataValue
	ArrayType  MetadataType
}

// AsString returns the value as a string if it is one.
func (v MetadataValue) AsString() string {
	return v.String
}

// AsUint32 returns the value as uint32.
func (v MetadataValue) AsUint32() uint32 {
	switch v.Type {
	case MetaTypeUint32:
		return v.Uint32
	case MetaTypeUint64:
		return uint32(v.Uint64)
	case MetaTypeInt32:
		return uint32(v.Int32)
	default:
		return 0
	}
}

// AsUint64 returns the value as uint64.
func (v MetadataValue) AsUint64() uint64 {
	switch v.Type {
	case MetaTypeUint64:
		return v.Uint64
	case MetaTypeUint32:
		return uint64(v.Uint32)
	default:
		return 0
	}
}

// AsFloat32 returns the value as float32.
func (v MetadataValue) AsFloat32() float32 {
	switch v.Type {
	case MetaTypeFloat32:
		return v.Float32
	case MetaTypeFloat64:
		return float32(v.Float64)
	default:
		return 0
	}
}

// TensorInfo holds information about a tensor in the file.
type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       TensorType
	Offset     uint64 // Offset from start of tensor data section
}

// NumElements returns the total number of elements in the tensor.
func (t TensorInfo) NumElements() uint64 {
	if len(t.Dimensions) == 0 {
		return 0
	}
	n := uint64(1)
	for _, d := range t.Dimensions {
		n *= d
	}
	return n
}

// File represents a parsed GGUF file.
type File struct {
	Version    uint32
	Metadata   map[string]MetadataValue
	Tensors    []TensorInfo
	DataOffset int64 // Offset to tensor data section

	// Cached model config values
	Architecture    string
	NumLayers       int
	HiddenSize      int
	NumHeads        int
	NumKVHeads      int
	VocabSize       int
	IntermediateSize int
	ContextLength   int
	RoPETheta       float32
	RoPEDimCount    int // Dimensions for RoPE (0 = full headDim). For partial RoPE like Phi-2.
	SlidingWindow   int // Sliding window size
	ExpertCount     int // Total number of routed experts (MoE models, e.g., 64 or 256)
	ExpertUsedCount int // Number of experts selected per token (MoE models, e.g., 6 or 8)
	file            *os.File // Underlying file for mmap access
}

// Open opens and parses a GGUF file.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	gf := &File{
		Metadata: make(map[string]MetadataValue),
		file:     f,
	}

	if err := gf.parseHeader(f); err != nil {
		f.Close()
		return nil, err
	}

	gf.extractModelConfig()

	return gf, nil
}

// Close closes the underlying file.
func (f *File) Close() error {
	if f.file != nil {
		return f.file.Close()
	}
	return nil
}

func (f *File) parseHeader(r io.ReadSeeker) error {
	// Read magic
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return fmt.Errorf("failed to read magic: %w", err)
	}
	if magic != Magic {
		return fmt.Errorf("invalid magic: got 0x%08X, want 0x%08X", magic, Magic)
	}

	// Read version
	if err := binary.Read(r, binary.LittleEndian, &f.Version); err != nil {
		return fmt.Errorf("failed to read version: %w", err)
	}
	if f.Version < 2 || f.Version > 3 {
		return fmt.Errorf("unsupported version: %d", f.Version)
	}

	// Read counts
	var tensorCount, metadataCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return fmt.Errorf("failed to read tensor count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &metadataCount); err != nil {
		return fmt.Errorf("failed to read metadata count: %w", err)
	}

	// Read metadata
	for i := uint64(0); i < metadataCount; i++ {
		key, value, err := f.readMetadataKV(r)
		if err != nil {
			return fmt.Errorf("failed to read metadata %d: %w", i, err)
		}
		f.Metadata[key] = value
	}

	// Read tensor infos
	f.Tensors = make([]TensorInfo, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		info, err := f.readTensorInfo(r)
		if err != nil {
			return fmt.Errorf("failed to read tensor info %d: %w", i, err)
		}
		f.Tensors[i] = info
	}

	// Get alignment (default 32)
	alignment := uint64(32)
	if v, ok := f.Metadata["general.alignment"]; ok {
		alignment = v.AsUint64()
	}

	// Calculate data offset (aligned)
	currentPos, _ := r.Seek(0, io.SeekCurrent)
	f.DataOffset = ((currentPos + int64(alignment) - 1) / int64(alignment)) * int64(alignment)

	return nil
}

func (f *File) readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 65535 {
		return "", fmt.Errorf("string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func (f *File) readMetadataKV(r io.Reader) (string, MetadataValue, error) {
	key, err := f.readString(r)
	if err != nil {
		return "", MetadataValue{}, fmt.Errorf("failed to read key: %w", err)
	}

	value, err := f.readMetadataValue(r)
	if err != nil {
		return "", MetadataValue{}, fmt.Errorf("failed to read value for %s: %w", key, err)
	}

	return key, value, nil
}

func (f *File) readMetadataValue(r io.Reader) (MetadataValue, error) {
	var valueType uint32
	if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
		return MetadataValue{}, err
	}

	v := MetadataValue{Type: MetadataType(valueType)}

	switch MetadataType(valueType) {
	case MetaTypeUint8:
		binary.Read(r, binary.LittleEndian, &v.Uint8)
	case MetaTypeInt8:
		binary.Read(r, binary.LittleEndian, &v.Int8)
	case MetaTypeUint16:
		binary.Read(r, binary.LittleEndian, &v.Uint16)
	case MetaTypeInt16:
		binary.Read(r, binary.LittleEndian, &v.Int16)
	case MetaTypeUint32:
		binary.Read(r, binary.LittleEndian, &v.Uint32)
	case MetaTypeInt32:
		binary.Read(r, binary.LittleEndian, &v.Int32)
	case MetaTypeUint64:
		binary.Read(r, binary.LittleEndian, &v.Uint64)
	case MetaTypeInt64:
		binary.Read(r, binary.LittleEndian, &v.Int64)
	case MetaTypeFloat32:
		binary.Read(r, binary.LittleEndian, &v.Float32)
	case MetaTypeFloat64:
		binary.Read(r, binary.LittleEndian, &v.Float64)
	case MetaTypeBool:
		var b uint8
		binary.Read(r, binary.LittleEndian, &b)
		v.Bool = b != 0
	case MetaTypeString:
		s, err := f.readString(r)
		if err != nil {
			return v, err
		}
		v.String = s
	case MetaTypeArray:
		var arrayType uint32
		var arrayLen uint64
		binary.Read(r, binary.LittleEndian, &arrayType)
		binary.Read(r, binary.LittleEndian, &arrayLen)
		v.ArrayType = MetadataType(arrayType)
		v.Array = make([]MetadataValue, arrayLen)
		for i := uint64(0); i < arrayLen; i++ {
			elem, err := f.readArrayElement(r, MetadataType(arrayType))
			if err != nil {
				return v, err
			}
			v.Array[i] = elem
		}
	default:
		return v, fmt.Errorf("unknown metadata type: %d", valueType)
	}

	return v, nil
}

func (f *File) readArrayElement(r io.Reader, elemType MetadataType) (MetadataValue, error) {
	v := MetadataValue{Type: elemType}

	switch elemType {
	case MetaTypeUint8:
		binary.Read(r, binary.LittleEndian, &v.Uint8)
	case MetaTypeInt8:
		binary.Read(r, binary.LittleEndian, &v.Int8)
	case MetaTypeUint16:
		binary.Read(r, binary.LittleEndian, &v.Uint16)
	case MetaTypeInt16:
		binary.Read(r, binary.LittleEndian, &v.Int16)
	case MetaTypeUint32:
		binary.Read(r, binary.LittleEndian, &v.Uint32)
	case MetaTypeInt32:
		binary.Read(r, binary.LittleEndian, &v.Int32)
	case MetaTypeUint64:
		binary.Read(r, binary.LittleEndian, &v.Uint64)
	case MetaTypeInt64:
		binary.Read(r, binary.LittleEndian, &v.Int64)
	case MetaTypeFloat32:
		binary.Read(r, binary.LittleEndian, &v.Float32)
	case MetaTypeFloat64:
		binary.Read(r, binary.LittleEndian, &v.Float64)
	case MetaTypeBool:
		var b uint8
		binary.Read(r, binary.LittleEndian, &b)
		v.Bool = b != 0
	case MetaTypeString:
		s, err := f.readString(r)
		if err != nil {
			return v, err
		}
		v.String = s
	}

	return v, nil
}

func (f *File) readTensorInfo(r io.Reader) (TensorInfo, error) {
	var info TensorInfo

	// Read name
	name, err := f.readString(r)
	if err != nil {
		return info, fmt.Errorf("failed to read tensor name: %w", err)
	}
	info.Name = name

	// Read number of dimensions
	var nDims uint32
	if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
		return info, err
	}

	// Read dimensions
	info.Dimensions = make([]uint64, nDims)
	for i := uint32(0); i < nDims; i++ {
		if err := binary.Read(r, binary.LittleEndian, &info.Dimensions[i]); err != nil {
			return info, err
		}
	}

	// Read type
	var tensorType uint32
	if err := binary.Read(r, binary.LittleEndian, &tensorType); err != nil {
		return info, err
	}
	info.Type = TensorType(tensorType)

	// Read offset
	if err := binary.Read(r, binary.LittleEndian, &info.Offset); err != nil {
		return info, err
	}

	return info, nil
}

func (f *File) extractModelConfig() {
	// Architecture
	if v, ok := f.Metadata["general.architecture"]; ok {
		f.Architecture = v.AsString()
	}

	arch := f.Architecture
	if arch == "" {
		arch = "llama" // Default
	}

	// Common keys with architecture prefix
	prefix := arch + "."

	if v, ok := f.Metadata[prefix+"block_count"]; ok {
		f.NumLayers = int(v.AsUint32())
	}
	if v, ok := f.Metadata[prefix+"embedding_length"]; ok {
		f.HiddenSize = int(v.AsUint32())
	}
	if v, ok := f.Metadata[prefix+"attention.head_count"]; ok {
		f.NumHeads = int(v.AsUint32())
	}
	if v, ok := f.Metadata[prefix+"attention.head_count_kv"]; ok {
		f.NumKVHeads = int(v.AsUint32())
	} else {
		f.NumKVHeads = f.NumHeads // Default to MHA
	}
	if v, ok := f.Metadata[prefix+"feed_forward_length"]; ok {
		f.IntermediateSize = int(v.AsUint32())
	}
	if v, ok := f.Metadata[prefix+"context_length"]; ok {
		f.ContextLength = int(v.AsUint32())
	}
	if v, ok := f.Metadata[prefix+"rope.freq_base"]; ok {
		f.RoPETheta = v.AsFloat32()
	} else {
		f.RoPETheta = 10000.0 // Default
	}
	// RoPE dimension count (for partial rotary like Phi-2)
	// 0 means full head dimension (default for LLaMA-style)
	if v, ok := f.Metadata[prefix+"rope.dimension_count"]; ok {
		f.RoPEDimCount = int(v.AsUint32())
	}
	if v, ok := f.Metadata[prefix+"attention.sliding_window"]; ok {
		f.SlidingWindow = int(v.AsUint32())
	}

	// MoE (Mixture of Experts) parameters
	if v, ok := f.Metadata[prefix+"expert_count"]; ok {
		f.ExpertCount = int(v.AsUint32())
	}
	if v, ok := f.Metadata[prefix+"expert_used_count"]; ok {
		f.ExpertUsedCount = int(v.AsUint32())
	}

	// Vocab size from tokenizer
	if v, ok := f.Metadata["tokenizer.ggml.tokens"]; ok && v.Type == MetaTypeArray {
		f.VocabSize = len(v.Array)
	}
}

// GetTensor returns the tensor info by name.
func (f *File) GetTensor(name string) (TensorInfo, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// ReadTensorData reads the raw data for a tensor.
func (f *File) ReadTensorData(info TensorInfo) ([]byte, error) {
	if f.file == nil {
		return nil, fmt.Errorf("file not open")
	}

	// Calculate size in bytes
	numElements := info.NumElements()
	blockSize := uint64(info.Type.BlockSize())
	bytesPerBlock := uint64(info.Type.BytesPerBlock())
	numBlocks := (numElements + blockSize - 1) / blockSize
	dataSize := numBlocks * bytesPerBlock

	// Seek to tensor data
	offset := f.DataOffset + int64(info.Offset)
	if _, err := f.file.Seek(offset, io.SeekStart); err != nil {
		return nil, err
	}

	// Read data
	data := make([]byte, dataSize)
	if _, err := io.ReadFull(f.file, data); err != nil {
		return nil, err
	}

	return data, nil
}

// PrintSummary prints a summary of the GGUF file.
func (f *File) PrintSummary() {
	fmt.Printf("GGUF Version: %d\n", f.Version)
	fmt.Printf("Architecture: %s\n", f.Architecture)
	fmt.Printf("Layers: %d\n", f.NumLayers)
	fmt.Printf("Hidden Size: %d\n", f.HiddenSize)
	fmt.Printf("Heads: %d (KV: %d)\n", f.NumHeads, f.NumKVHeads)
	fmt.Printf("Intermediate: %d\n", f.IntermediateSize)
	fmt.Printf("Vocab Size: %d\n", f.VocabSize)
	fmt.Printf("Context Length: %d\n", f.ContextLength)
	fmt.Printf("RoPE Theta: %.1f\n", f.RoPETheta)
	fmt.Printf("Tensors: %d\n", len(f.Tensors))
	fmt.Printf("Data Offset: %d\n", f.DataOffset)
}

// ModelConfigValues returns the extracted model configuration values.
// This can be used to initialize a runtime.ModelConfig.
type ModelConfigValues struct {
	Architecture     string
	NumLayers        int
	HiddenSize       int
	IntermediateSize int
	NumHeads         int
	NumKVHeads       int
	VocabSize        int
	ContextLength    int
	RoPETheta        float32
	RoPEDimCount     int // Dimensions for RoPE (0 = full headDim)
	SlidingWindow    int // Sliding window size
	ExpertCount      int // Total number of routed experts (MoE models)
	ExpertUsedCount  int // Number of experts selected per token (MoE models)
}

// GetModelConfig returns the extracted model configuration.
func (f *File) GetModelConfig() ModelConfigValues {
	return ModelConfigValues{
		Architecture:     f.Architecture,
		NumLayers:        f.NumLayers,
		HiddenSize:       f.HiddenSize,
		IntermediateSize: f.IntermediateSize,
		NumHeads:         f.NumHeads,
		NumKVHeads:       f.NumKVHeads,
		VocabSize:        f.VocabSize,
		ContextLength:    f.ContextLength,
		RoPETheta:        f.RoPETheta,
		RoPEDimCount:     f.RoPEDimCount,
		SlidingWindow:    f.SlidingWindow,
		ExpertCount:      f.ExpertCount,
		ExpertUsedCount:  f.ExpertUsedCount,
	}
}
