package runtime

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"unsafe"
	"vexel/inference/pkg/bf16"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/safetensors"
	"vexel/inference/tensor"
)

// LoadWeights loads model weights from a file (auto-detects format).
func (m *ModelRuntime) LoadWeights(path string) error {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".gguf" {
		return m.LoadWeightsGGUF(path)
	}
	return m.LoadWeightsSafetensors(path)
}

// LoadWeightsGGUF loads model weights from a GGUF file.
func (m *ModelRuntime) LoadWeightsGGUF(path string) error {
	loader, err := gguf.NewTensorLoader(path)
	if err != nil {
		return fmt.Errorf("failed to open GGUF file: %w", err)
	}
	// Keep loader open for potential future use
	m.ggufLoader = loader

	// Print stats
	loader.PrintTensorStats()

	// Load each required tensor
	tensorNames := m.requiredTensorNames()
	loadedCount := 0
	for _, hfName := range tensorNames {
		ggufName := gguf.GetLayerTensorName(hfName)

		data, dims, err := loader.LoadTensor(ggufName)
		if err != nil {
			// Try alternative naming patterns
			altNames := m.alternativeGGUFNames(ggufName)
			loaded := false
			for _, altName := range altNames {
				data, dims, err = loader.LoadTensor(altName)
				if err == nil {
					loaded = true
					break
				}
			}
			if !loaded {
				fmt.Printf("Warning: tensor %s (%s) not found\n", hfName, ggufName)
				continue
			}
		}

		// Create tensor from data
		// Note: GGUF loader already transposes 2D matrices and swaps dimensions
		t := tensor.NewTensor(
			tensor.NewShape(dims...),
			m.config.DType,
			tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&data[0]))),
		)

		// Keep reference to prevent GC
		m.keepAlive = append(m.keepAlive, data)

		// Map to struct
		m.mapTensor(hfName, t)
		loadedCount++
	}

	fmt.Printf("Loaded %d/%d tensors from GGUF\n", loadedCount, len(tensorNames))

	return nil
}

// requiredTensorNames returns the list of tensor names needed for the model.
func (m *ModelRuntime) requiredTensorNames() []string {
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight",
	}

	for i := 0; i < m.config.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		names = append(names,
			prefix+"self_attn.q_proj.weight",
			prefix+"self_attn.k_proj.weight",
			prefix+"self_attn.v_proj.weight",
			prefix+"self_attn.o_proj.weight",
			prefix+"mlp.gate_proj.weight",
			prefix+"mlp.up_proj.weight",
			prefix+"mlp.down_proj.weight",
			prefix+"input_layernorm.weight",
			prefix+"post_attention_layernorm.weight",
		)
	}

	return names
}

// alternativeGGUFNames returns alternative names to try for a tensor.
func (m *ModelRuntime) alternativeGGUFNames(name string) []string {
	// Some GGUF files use slightly different naming
	alts := []string{}

	// Try without .weight suffix
	if strings.HasSuffix(name, ".weight") {
		alts = append(alts, strings.TrimSuffix(name, ".weight"))
	}

	// Try with model. prefix
	if !strings.HasPrefix(name, "model.") {
		alts = append(alts, "model."+name)
	}

	return alts
}

// LoadWeightsSafetensors loads model weights from a safetensors file.
func (m *ModelRuntime) LoadWeightsSafetensors(path string) error {
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

// CopyWeightsToDevice copies all loaded weights to the backend device (GPU).
// This should be called after LoadWeights if using GPU execution.
// For CPU backend, this is a no-op since weights are already on CPU.
func (m *ModelRuntime) CopyWeightsToDevice() error {
	// Helper to copy a single tensor to device
	copyToDevice := func(t *tensor.Tensor) error {
		if t.DevicePtr().IsNil() {
			return nil
		}
		// If already on the right device, skip
		if t.DevicePtr().Location() == m.backend.Device().Location {
			return nil
		}

		// Get size in bytes
		numElements := t.Shape().NumElements()
		sizeBytes := numElements * 4 // Assuming float32

		// Allocate on device
		devicePtr := m.backend.Alloc(sizeBytes)
		if devicePtr.IsNil() {
			return fmt.Errorf("failed to allocate %d bytes on device", sizeBytes)
		}

		// Copy data from CPU to device
		cpuData := unsafe.Slice((*byte)(unsafe.Pointer(t.DevicePtr().Addr())), sizeBytes)
		m.backend.ToDevice(devicePtr, cpuData)

		// Update tensor with device pointer
		*t = tensor.NewTensor(t.Shape(), m.config.DType, devicePtr)
		return nil
	}

	// Copy global weights
	if err := copyToDevice(&m.Embedding); err != nil {
		return fmt.Errorf("embedding: %w", err)
	}
	if err := copyToDevice(&m.FinalNorm); err != nil {
		return fmt.Errorf("final_norm: %w", err)
	}
	if err := copyToDevice(&m.OutputHead); err != nil {
		return fmt.Errorf("output_head: %w", err)
	}

	// Copy layer weights
	for i, layer := range m.layers {
		if err := copyToDevice(&layer.AttnNorm); err != nil {
			return fmt.Errorf("layer %d attn_norm: %w", i, err)
		}
		if err := copyToDevice(&layer.Wq); err != nil {
			return fmt.Errorf("layer %d wq: %w", i, err)
		}
		if err := copyToDevice(&layer.Wk); err != nil {
			return fmt.Errorf("layer %d wk: %w", i, err)
		}
		if err := copyToDevice(&layer.Wv); err != nil {
			return fmt.Errorf("layer %d wv: %w", i, err)
		}
		if err := copyToDevice(&layer.Wo); err != nil {
			return fmt.Errorf("layer %d wo: %w", i, err)
		}
		if err := copyToDevice(&layer.FFNNorm); err != nil {
			return fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}
		if err := copyToDevice(&layer.W1); err != nil {
			return fmt.Errorf("layer %d w1: %w", i, err)
		}
		if err := copyToDevice(&layer.W2); err != nil {
			return fmt.Errorf("layer %d w2: %w", i, err)
		}
		if err := copyToDevice(&layer.W3); err != nil {
			return fmt.Errorf("layer %d w3: %w", i, err)
		}
	}

	// Sync to ensure all copies complete
	m.backend.Sync()

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