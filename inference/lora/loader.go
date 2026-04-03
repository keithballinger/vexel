package lora

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
)

// tensorMeta holds the metadata for a single tensor in a safetensors file.
type tensorMeta struct {
	Dtype       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// LayerAdapter holds the LoRA A and B weight matrices for a single transformer
// layer's attention projections.
type LayerAdapter struct {
	// Q projection LoRA weights.
	QA      []float32
	QB      []float32
	QAShape [2]int64
	QBShape [2]int64

	// K projection LoRA weights.
	KA      []float32
	KB      []float32
	KAShape [2]int64
	KBShape [2]int64

	// V projection LoRA weights.
	VA      []float32
	VB      []float32
	VAShape [2]int64
	VBShape [2]int64

	// O (output) projection LoRA weights.
	OA      []float32
	OB      []float32
	OAShape [2]int64
	OBShape [2]int64
}

// HasQ returns true when this layer has Q-projection LoRA weights loaded.
func (l *LayerAdapter) HasQ() bool {
	return len(l.QA) > 0 && len(l.QB) > 0
}

// HasK returns true when this layer has K-projection LoRA weights loaded.
func (l *LayerAdapter) HasK() bool {
	return len(l.KA) > 0 && len(l.KB) > 0
}

// HasV returns true when this layer has V-projection LoRA weights loaded.
func (l *LayerAdapter) HasV() bool {
	return len(l.VA) > 0 && len(l.VB) > 0
}

// HasO returns true when this layer has O (output)-projection LoRA weights loaded.
func (l *LayerAdapter) HasO() bool {
	return len(l.OA) > 0 && len(l.OB) > 0
}

// Adapter is a fully-loaded LoRA adapter ready for inference.
type Adapter struct {
	Config AdapterConfig
	Scale  float32
	Layers []LayerAdapter
}

// layerTensorNames captures layer index and projection/matrix kind from a
// HuggingFace PEFT safetensors tensor name, e.g.:
//
//	base_model.model.layers.5.self_attn.q_proj.lora_A.weight  →  layer=5, proj=q, mat=A
var layerTensorRe = regexp.MustCompile(
	`layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.lora_([AB])\.weight`,
)

// LoadAdapter reads adapter_config.json and adapter_model.safetensors from
// dir and returns a populated Adapter ready for use at inference time.
func LoadAdapter(dir string) (*Adapter, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}

	weights, err := loadSafetensors(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("load safetensors: %w", err)
	}

	// Determine the maximum layer index present in the file.
	maxLayer := -1
	for name := range weights {
		m := layerTensorRe.FindStringSubmatch(name)
		if m == nil {
			continue
		}
		idx, _ := strconv.Atoi(m[1])
		if idx > maxLayer {
			maxLayer = idx
		}
	}
	if maxLayer < 0 {
		return nil, fmt.Errorf("no recognised LoRA tensors found in safetensors file")
	}

	layers := make([]LayerAdapter, maxLayer+1)

	for name, t := range weights {
		m := layerTensorRe.FindStringSubmatch(name)
		if m == nil {
			continue
		}
		layerIdx, _ := strconv.Atoi(m[1])
		proj := m[2] // "q", "k", "v", or "o"
		mat := m[3]  // "A" or "B"

		la := &layers[layerIdx]
		shape := [2]int64{t.shape[0], t.shape[1]}

		switch {
		case proj == "q" && mat == "A":
			la.QA = t.data
			la.QAShape = shape
		case proj == "q" && mat == "B":
			la.QB = t.data
			la.QBShape = shape
		case proj == "k" && mat == "A":
			la.KA = t.data
			la.KAShape = shape
		case proj == "k" && mat == "B":
			la.KB = t.data
			la.KBShape = shape
		case proj == "v" && mat == "A":
			la.VA = t.data
			la.VAShape = shape
		case proj == "v" && mat == "B":
			la.VB = t.data
			la.VBShape = shape
		case proj == "o" && mat == "A":
			la.OA = t.data
			la.OAShape = shape
		case proj == "o" && mat == "B":
			la.OB = t.data
			la.OBShape = shape
		}
	}

	return &Adapter{
		Config: cfg,
		Scale:  cfg.Scale(),
		Layers: layers,
	}, nil
}

// tensorData is an intermediate representation after parsing the safetensors
// header and converting all tensors to float32.
type tensorData struct {
	shape [2]int64
	data  []float32
}

// loadSafetensors parses a safetensors binary file and returns a map from
// tensor name to float32 data + shape. Only tensors with exactly 2 dimensions
// are retained (all LoRA weight matrices are 2-D).
func loadSafetensors(path string) (map[string]tensorData, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("file too short to be a valid safetensors file")
	}

	// First 8 bytes: little-endian uint64 header length.
	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if uint64(len(raw)) < 8+headerLen {
		return nil, fmt.Errorf("file truncated: header claims %d bytes but file has %d", headerLen, len(raw)-8)
	}

	headerJSON := raw[8 : 8+headerLen]
	dataBlob := raw[8+headerLen:]

	// Parse the JSON header.
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("parse header JSON: %w", err)
	}

	result := make(map[string]tensorData, len(header))

	for name, raw := range header {
		// The safetensors spec allows a top-level "__metadata__" key.
		if name == "__metadata__" {
			continue
		}

		var meta tensorMeta
		if err := json.Unmarshal(raw, &meta); err != nil {
			return nil, fmt.Errorf("tensor %q: parse metadata: %w", name, err)
		}
		if len(meta.Shape) != 2 {
			// Skip scalars, 1-D biases, etc.
			continue
		}

		start := meta.DataOffsets[0]
		end := meta.DataOffsets[1]
		if end > int64(len(dataBlob)) || start > end {
			return nil, fmt.Errorf("tensor %q: data_offsets [%d,%d] out of bounds (data blob is %d bytes)",
				name, start, end, len(dataBlob))
		}
		bytes := dataBlob[start:end]

		f32, err := convertToF32(bytes, meta.Dtype, meta.Shape)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}

		result[name] = tensorData{
			shape: [2]int64{meta.Shape[0], meta.Shape[1]},
			data:  f32,
		}
	}

	return result, nil
}

// convertToF32 converts a raw byte slice to a []float32 slice according to the
// safetensors dtype string. Supported dtypes: F32, F16, BF16.
func convertToF32(data []byte, dtype string, shape []int64) ([]float32, error) {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}

	switch dtype {
	case "F32":
		if int64(len(data)) != n*4 {
			return nil, fmt.Errorf("F32 dtype: expected %d bytes, got %d", n*4, len(data))
		}
		out := make([]float32, n)
		for i := range out {
			bits := binary.LittleEndian.Uint32(data[i*4:])
			out[i] = math.Float32frombits(bits)
		}
		return out, nil

	case "F16":
		if int64(len(data)) != n*2 {
			return nil, fmt.Errorf("F16 dtype: expected %d bytes, got %d", n*2, len(data))
		}
		out := make([]float32, n)
		for i := range out {
			bits := binary.LittleEndian.Uint16(data[i*2:])
			out[i] = float16ToF32(bits)
		}
		return out, nil

	case "BF16":
		if int64(len(data)) != n*2 {
			return nil, fmt.Errorf("BF16 dtype: expected %d bytes, got %d", n*2, len(data))
		}
		out := make([]float32, n)
		for i := range out {
			// BF16 is the upper 16 bits of an IEEE 754 float32; shift left 16.
			bits := uint32(binary.LittleEndian.Uint16(data[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
		return out, nil

	default:
		return nil, fmt.Errorf("unsupported dtype %q (only F32, F16, BF16 are supported)", dtype)
	}
}

// float16ToF32 converts an IEEE 754 half-precision float (uint16 bits) to a
// standard float32.
func float16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	var f uint32
	switch {
	case exp == 0 && mant == 0:
		// Signed zero.
		f = sign
	case exp == 0:
		// Subnormal: re-normalise.
		e := uint32(1)
		m := mant
		for m&0x400 == 0 {
			m <<= 1
			e++
		}
		m &= 0x3FF
		f = sign | ((127 - 15 - e + 1) << 23) | (m << 13)
	case exp == 0x1F:
		// Inf / NaN.
		f = sign | (0xFF << 23) | (mant << 13)
	default:
		// Normal.
		f = sign | ((exp + 112) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
}
