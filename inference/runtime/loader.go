package runtime

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"
	"vexel/inference/pkg/bf16"
	"vexel/inference/pkg/safetensors"
	"vexel/inference/tensor"
)

// LoadWeights loads model weights from a safetensors file.
func (m *ModelRuntime) LoadWeights(path string) error {
	mapped, err := safetensors.Mmap(path)
	if err != nil {
		return fmt.Errorf("failed to mmap weights file: %w", err)
	}
	m.mappedFile = mapped

	// Parse header
	data := mapped.Bytes()
	header, dataOffset, err := safetensors.ParseHeader(data)
	if err != nil {
		return fmt.Errorf("failed to parse safetensors header: %w", err)
	}

	// Base address of the data section
	baseAddr := uintptr(unsafe.Pointer(&data[0])) + uintptr(dataOffset)

	// Iterate tensors
	for name, meta := range header {
		// Meta is map[string]interface{}
		// We need "data_offsets" -> [start, end]
		// "shape" -> []int
		// "dtype" -> string
		
		info, ok := meta.(map[string]interface{})
		if !ok {
			continue // Should not happen for valid safetensors
		}
		
		offsets, _ := info["data_offsets"].([]interface{})
		if len(offsets) != 2 {
			continue
		}
		
		// Get shape
		shapeList, _ := info["shape"].([]interface{})
		dims := make([]int, len(shapeList))
		numElements := 1
		for i, v := range shapeList {
			dims[i] = int(v.(float64))
			numElements *= dims[i]
		}
		
		start := uint64(offsets[0].(float64)) // JSON numbers are float64

		// Calculate pointer
		ptr := baseAddr + uintptr(start)

		// Get DType (assuming matches config for now, or parse string)
		// TinyLlama is usually F16 or BF16.
		// For now using Config dtype.

		var tensorPtr tensor.DevicePtr
		dtype := m.config.DType

		// Hack: Force conversion to F32 if model is BF16/F16 but we run on CPU (Float32 kernel)
		// Safetensors dtype "BF16" -> we read 2 bytes
		// If our kernel expects F32, we must allocate and convert.

		stDType := info["dtype"].(string)
		
		if stDType == "BF16" && m.config.DType == tensor.Float32 {
			// Read raw bytes - start is offset within data section
			numElements := 1
			for _, d := range dims {
				numElements *= d
			}
			// Add dataOffset for absolute file position
			absStart := uint64(dataOffset) + start
			absEnd := absStart + uint64(numElements*2)
			rawBytes := data[absStart:absEnd]

			// Convert BF16 to FP32
			f32Data := bf16.ConvertToFP32(rawBytes)

			// Store pointer to converted data
			tensorPtr = tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&f32Data[0])))

			// Keep reference to prevent GC from collecting the converted data
			m.keepAlive = append(m.keepAlive, f32Data)
		} else {
			// Zero-copy (if types match)
			tensorPtr = tensor.NewDevicePtr(tensor.CPU, ptr)
		}
		
		t := tensor.NewTensor(
			tensor.NewShape(dims...),
			dtype,
			tensorPtr,
		)
		
		// Map to struct
		m.mapTensor(name, t)
	}

	return nil
}

func (m *ModelRuntime) mapTensor(name string, t tensor.Tensor) {
	// Global
	if name == "model.embed_tokens.weight" {
		m.Embedding = t
		return
	}
	if name == "model.norm.weight" {
		m.FinalNorm = t
		return
	}
	if name == "lm_head.weight" {
		m.OutputHead = t
		return
	}
	
	// Layers: model.layers.{i}.X
	if strings.HasPrefix(name, "model.layers.") {
		parts := strings.Split(name, ".")
		// parts[2] is index
		idx, err := strconv.Atoi(parts[2])
		if err != nil || idx >= len(m.layers) {
			return
		}
		
		layer := m.layers[idx]
		suffix := strings.Join(parts[3:], ".")
		
		switch suffix {
		case "self_attn.q_proj.weight":
			layer.Wq = t
		case "self_attn.k_proj.weight":
			layer.Wk = t
		case "self_attn.v_proj.weight":
			layer.Wv = t
		case "self_attn.o_proj.weight":
			layer.Wo = t
		case "mlp.gate_proj.weight":
			layer.W1 = t
		case "mlp.up_proj.weight":
			layer.W3 = t
		case "mlp.down_proj.weight":
			layer.W2 = t
		case "input_layernorm.weight":
			layer.AttnNorm = t
		case "post_attention_layernorm.weight":
			layer.FFNNorm = t
		}
	}
}