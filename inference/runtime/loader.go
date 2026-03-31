package runtime

import (
	"fmt"
	"log"
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
// For GPU backends with QuantizedMatMul support, Q4_0 tensors are kept in raw format.
// Otherwise, all tensors are dequantized to F32.
func (m *ModelRuntime) LoadWeightsGGUF(path string) error {
	loader, err := gguf.NewTensorLoader(path)
	if err != nil {
		return fmt.Errorf("failed to open GGUF file: %w", err)
	}
	// Keep loader open for potential future use
	m.ggufLoader = loader

	// Validate architecture compatibility
	if supported, warning := loader.ValidateArchitecture(); !supported {
		log.Printf("WARNING: %s", warning)
	} else if m.verbose {
		log.Printf("Architecture: %s", loader.Architecture())
	}

	// Print stats
	if m.verbose {
		loader.PrintTensorStats()
	}

	// Load each required tensor
	tensorNames := m.requiredTensorNames()
	loadedCount := 0
	q4Count := 0
	for _, hfName := range tensorNames {
		ggufName := gguf.GetLayerTensorName(hfName)

		// First, check the tensor type to decide loading strategy
		info, found := loader.GetTensorInfo(ggufName)
		// ...
		if !found {
			// Try alternative naming patterns
			altNames := m.alternativeGGUFNames(ggufName)
			for _, altName := range altNames {
				info, found = loader.GetTensorInfo(altName)
				if found {
					ggufName = altName
					break
				}
			}
		}
		if found && m.verbose {
			log.Printf("Loading %s (%s): Type=%v", hfName, ggufName, info.Type)
		}
		if !found {
			log.Printf("Warning: tensor %s (%s) not found", hfName, ggufName)
			continue
		}

		var t tensor.Tensor

		// For Q4_0/Q4_K weight matrices (not embeddings/norms), keep raw format
		// This enables GPU-native quantized inference
		if info.Type == gguf.TensorTypeQ4_0 && m.isWeightMatrix(hfName) {
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load raw tensor %s: %v", hfName, err)
				continue
			}

			// Create quantized tensor with raw Q4_0 data
			t = tensor.NewQuantTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&rawData[0]))),
				tensor.Q4_0,
			)
			m.keepAliveBytes = append(m.keepAliveBytes, rawData)
			q4Count++
		} else if info.Type == gguf.TensorTypeQ4_K && (m.isWeightMatrix(hfName) || hfName == "lm_head.weight") {
			// Q4_K native GPU kernel - uses get_scale_min_k4 format
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load raw Q4_K tensor %s: %v", hfName, err)
				continue
			}
			t = tensor.NewQuantTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&rawData[0]))),
				tensor.Q4_K,
			)
			m.keepAliveBytes = append(m.keepAliveBytes, rawData)
			q4Count++
		} else if info.Type == gguf.TensorTypeQ5_K && (m.isWeightMatrix(hfName) || hfName == "lm_head.weight") {
			// Q5_K native GPU kernel
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load raw Q5_K tensor %s: %v", hfName, err)
				continue
			}
			t = tensor.NewQuantTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&rawData[0]))),
				tensor.Q5_K,
			)
			m.keepAliveBytes = append(m.keepAliveBytes, rawData)
			q4Count++
		} else if info.Type == gguf.TensorTypeQ6_K && (m.isWeightMatrix(hfName) || hfName == "lm_head.weight") {
			// Keep as Q6_K for GPU-native quantized inference
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load raw tensor %s: %v", hfName, err)
				continue
			}
			t = tensor.NewQuantTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&rawData[0]))),
				tensor.Q6_K,
			)
			m.keepAliveBytes = append(m.keepAliveBytes, rawData)
			q4Count++
		} else if info.Type == gguf.TensorTypeQ5_0 && (m.isWeightMatrix(hfName) || hfName == "lm_head.weight") {
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load raw Q5_0 tensor %s: %v", hfName, err)
				continue
			}
			t = tensor.NewQuantTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&rawData[0]))),
				tensor.Q5_0,
			)
			m.keepAliveBytes = append(m.keepAliveBytes, rawData)
			q4Count++
		} else if info.Type == gguf.TensorTypeQ8_0 && (m.isWeightMatrix(hfName) || hfName == "lm_head.weight") {
			// Q8_0 native GPU kernel
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load raw Q8_0 tensor %s: %v", hfName, err)
				continue
			}
			t = tensor.NewQuantTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&rawData[0]))),
				tensor.Q8_0,
			)
			m.keepAliveBytes = append(m.keepAliveBytes, rawData)
			q4Count++
		} else if info.Type == gguf.TensorTypeBF16 && (m.isWeightMatrix(hfName) || hfName == "lm_head.weight") {
			// BF16 native GPU kernel - keep raw BF16 data on GPU
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load raw BF16 tensor %s: %v", hfName, err)
				continue
			}
			t = tensor.NewQuantTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&rawData[0]))),
				tensor.BF16,
			)
			m.keepAliveBytes = append(m.keepAliveBytes, rawData)
			q4Count++
		} else {
			// Dequantize to F32 for embeddings, norms, or non-Q4_0/Q6_K types
			data, dims, err := loader.LoadTensor(ggufName)
			if err != nil {
				log.Printf("Warning: failed to load tensor %s: %v", hfName, err)
				continue
			}

			t = tensor.NewTensor(
				tensor.NewShape(dims...),
				m.config.DType,
				tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&data[0]))),
			)
			m.keepAlive = append(m.keepAlive, data)
		}

		// Map to struct
		m.mapTensor(hfName, t)
		loadedCount++
	}

	if m.verbose {
		log.Printf("Loaded %d/%d tensors from GGUF (%d quantized raw)", loadedCount, len(tensorNames), q4Count)
	}

	return nil
}

// LoadWeightsF32 loads weights and forces F32 dequantization (no Q4_0 raw).
// Useful for debugging/comparing against quantized path.
func (m *ModelRuntime) LoadWeightsF32(path string) error {
	loader, err := gguf.NewTensorLoader(path)
	if err != nil {
		return fmt.Errorf("failed to open GGUF file: %w", err)
	}
	m.ggufLoader = loader

	if m.verbose {
		loader.PrintTensorStats()
	}

	tensorNames := m.requiredTensorNames()
	loadedCount := 0
	for _, hfName := range tensorNames {
		ggufName := gguf.GetLayerTensorName(hfName)

		info, found := loader.GetTensorInfo(ggufName)
		if !found {
			altNames := m.alternativeGGUFNames(ggufName)
			for _, altName := range altNames {
				info, found = loader.GetTensorInfo(altName)
				if found {
					ggufName = altName
					break
				}
			}
		}
		if !found {
			log.Printf("Warning: tensor %s (%s) not found", hfName, ggufName)
			continue
		}
		_ = info // Unused but needed for consistency

		// Always dequantize to F32
		data, dims, err := loader.LoadTensor(ggufName)
		if err != nil {
			log.Printf("Warning: failed to load tensor %s: %v", hfName, err)
			continue
		}

		t := tensor.NewTensor(
			tensor.NewShape(dims...),
			m.config.DType,
			tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&data[0]))),
		)
		m.keepAlive = append(m.keepAlive, data)

		m.mapTensor(hfName, t)
		loadedCount++
	}

	if m.verbose {
		log.Printf("Loaded %d/%d tensors from GGUF (all F32 dequantized)", loadedCount, len(tensorNames))
	}
	return nil
}

// isWeightMatrix returns true for tensors that can use quantized matmul.
// These are the projection matrices (Q/K/V/O, FFN), not embeddings or norms.
func (m *ModelRuntime) isWeightMatrix(name string) bool {
	// Embeddings and norms should be F32
	if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
		return false
	}
	if strings.HasSuffix(name, "_layernorm.weight") || strings.HasSuffix(name, ".norm.weight") {
		return false
	}
	// Layer weight matrices
	if strings.Contains(name, "self_attn.") || strings.Contains(name, "mlp.") {
		return true
	}
	return false
}

// requiredTensorNames returns the list of tensor names needed for the model.
func (m *ModelRuntime) requiredTensorNames() []string {
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight",
	}

	// Add final norm bias for LayerNorm architectures
	if m.config.NormType == NormLayerNorm {
		names = append(names, "model.norm.bias")
	}

	for i := 0; i < m.config.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		if m.config.MLPType == MLPGELU {
			// Phi-style architecture: combined QKV, fc1/fc2 MLP
			// Note: Phi uses parallel residual - no separate FFN norm (ffn_norm/post_attention_layernorm)
			// Both attn and MLP share the same attn_norm input
			names = append(names,
				prefix+"self_attn.qkv_proj.weight", // Combined QKV
				prefix+"self_attn.o_proj.weight",
				prefix+"mlp.fc1.weight", // Up/Gate combined
				prefix+"mlp.fc2.weight", // Down
				prefix+"input_layernorm.weight",
				// No post_attention_layernorm for parallel residual
			)
			// Add bias tensors
			if m.config.HasBias {
				names = append(names,
					prefix+"self_attn.qkv_proj.bias",
					prefix+"self_attn.o_proj.bias",
					prefix+"mlp.fc1.bias",
					prefix+"mlp.fc2.bias",
					prefix+"input_layernorm.bias",
					// No post_attention_layernorm.bias for parallel residual
				)
			}
		} else if m.config.MLPType == MLPSwiGLUFused {
			// Phi-3-style architecture: combined QKV, fused gate+up SwiGLU MLP, serial residual
			names = append(names,
				prefix+"self_attn.qkv_proj.weight", // Combined QKV
				prefix+"self_attn.o_proj.weight",
				prefix+"mlp.gate_up_proj.weight", // Fused gate+up [hidden, 2*intermediate]
				prefix+"mlp.down_proj.weight",    // Down projection
				prefix+"input_layernorm.weight",
				prefix+"post_attention_layernorm.weight",
			)
		} else if m.config.MLPType == MLPMoE {
			// DeepSeek MoE architecture: separate Q/K/V, MoE FFN with router + experts
			names = append(names,
				prefix+"self_attn.q_proj.weight",
				prefix+"self_attn.k_proj.weight",
				prefix+"self_attn.v_proj.weight",
				prefix+"self_attn.o_proj.weight",
				prefix+"mlp.gate.weight",              // Router/gate layer
				prefix+"mlp.experts.gate_proj.weight", // Concatenated expert gate projections
				prefix+"mlp.experts.up_proj.weight",   // Concatenated expert up projections
				prefix+"mlp.experts.down_proj.weight", // Concatenated expert down projections
				prefix+"input_layernorm.weight",
				prefix+"post_attention_layernorm.weight",
			)
		} else {
			// LLaMA-style architecture: separate Q/K/V, gate/up/down MLP
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
			// Gemma 2 post-norm weights (applied after attn/MLP, before residual)
			if m.config.HasPostNorms {
				names = append(names,
					prefix+"post_attention_norm.weight", // Post-attn RMSNorm
					prefix+"post_ffw_norm.weight",       // Post-FFN RMSNorm
				)
			}
			// Qwen2 and similar models have QKV bias but not output/MLP bias
			if m.config.HasQKVBias {
				names = append(names,
					prefix+"self_attn.q_proj.bias",
					prefix+"self_attn.k_proj.bias",
					prefix+"self_attn.v_proj.bias",
				)
			}
		}
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

		// Calculate size in bytes based on quantization profile
		var sizeBytes int
		numElements := t.Shape().NumElements()
		if t.IsQuantized() {
			switch t.QuantProfile() {
			case tensor.Q4_0:
				// Q4_0: 18 bytes per 32 elements (2 byte scale + 16 bytes nibbles)
				numBlocks := (numElements + 31) / 32
				sizeBytes = numBlocks * 18
			case tensor.Q4_K:
				// Q4_K: 144 bytes per 256 elements
				numBlocks := (numElements + 255) / 256
				sizeBytes = numBlocks * 144
			case tensor.Q5_K:
				// Q5_K: 176 bytes per 256 elements
				numBlocks := (numElements + 255) / 256
				sizeBytes = numBlocks * 176
			case tensor.Q6_K:
				// Q6_K: 210 bytes per 256 elements
				numBlocks := (numElements + 255) / 256
				sizeBytes = numBlocks * 210
			case tensor.Q5_0:
				// Q5_0: 22 bytes per 32 elements (2 byte f16 scale + 4 byte qh + 16 bytes qs)
				numBlocks := (numElements + 31) / 32
				sizeBytes = numBlocks * 22
			case tensor.Q8_0:
				// Q8_0: 34 bytes per 32 elements (2 byte f16 scale + 32 int8 values)
				numBlocks := (numElements + 31) / 32
				sizeBytes = numBlocks * 34
			case tensor.BF16:
				// BF16: 2 bytes per element
				sizeBytes = numElements * 2
			default:
				// Unknown quant, fall back to F32
				sizeBytes = numElements * 4
			}
		} else {
			// F32: 4 bytes per element
			sizeBytes = numElements * 4
		}

		// Allocate on device (use permanent allocation for weights)
		var devicePtr tensor.DevicePtr
		if allocPerm, ok := m.backend.(interface{ AllocPermanent(int) tensor.DevicePtr }); ok {
			devicePtr = allocPerm.AllocPermanent(sizeBytes)
		} else {
			devicePtr = m.backend.Alloc(sizeBytes)
		}
		if devicePtr.IsNil() {
			return fmt.Errorf("failed to allocate %d bytes on device", sizeBytes)
		}

		// Copy data from CPU to device
		cpuData := unsafe.Slice((*byte)(unsafe.Pointer(t.DevicePtr().Addr())), sizeBytes)
		m.backend.ToDevice(devicePtr, cpuData)

		// Update tensor with device pointer, preserving quant profile
		if t.IsQuantized() {
			*t = tensor.NewQuantTensor(t.Shape(), m.config.DType, devicePtr, t.QuantProfile())
		} else {
			*t = tensor.NewTensor(t.Shape(), m.config.DType, devicePtr)
		}
		return nil
	}

	// Copy global weights
	if err := copyToDevice(&m.Embedding); err != nil {
		return fmt.Errorf("embedding: %w", err)
	}
	if err := copyToDevice(&m.FinalNorm); err != nil {
		return fmt.Errorf("final_norm: %w", err)
	}
	if err := copyToDevice(&m.FinalNormBias); err != nil {
		return fmt.Errorf("final_norm_bias: %w", err)
	}
	if err := copyToDevice(&m.OutputHead); err != nil {
		return fmt.Errorf("output_head: %w", err)
	}
	// Weight tying: if output head is missing, use the embedding weights
	if m.OutputHead.DevicePtr().IsNil() && !m.Embedding.DevicePtr().IsNil() {
		m.OutputHead = m.Embedding
		log.Println("Using tied embedding weights for output head")
	}

	// Copy layer weights
	for i, layer := range m.layers {
		if err := copyToDevice(&layer.AttnNorm); err != nil {
			return fmt.Errorf("layer %d attn_norm: %w", i, err)
		}
		if err := copyToDevice(&layer.AttnNormBias); err != nil {
			return fmt.Errorf("layer %d attn_norm_bias: %w", i, err)
		}
		// For fused QKV (Phi-style): copy Wqkv to GPU, then derive Wq/Wk/Wv as sub-regions
		if !layer.Wqkv.DevicePtr().IsNil() && layer.Wqkv.DevicePtr().Location() != m.backend.Device().Location {
			if err := copyToDevice(&layer.Wqkv); err != nil {
				return fmt.Errorf("layer %d wqkv: %w", i, err)
			}
			m.splitQKVGPU(layer)
			if err := copyToDevice(&layer.WqkvBias); err != nil {
				return fmt.Errorf("layer %d wqkv_bias: %w", i, err)
			}
			if !layer.WqkvBias.DevicePtr().IsNil() && layer.WqkvBias.DevicePtr().Location() == m.backend.Device().Location {
				m.splitQKVBiasGPU(layer)
			}
		} else {
			if err := copyToDevice(&layer.Wq); err != nil {
				return fmt.Errorf("layer %d wq: %w", i, err)
			}
			if err := copyToDevice(&layer.Wk); err != nil {
				return fmt.Errorf("layer %d wk: %w", i, err)
			}
			if err := copyToDevice(&layer.Wv); err != nil {
				return fmt.Errorf("layer %d wv: %w", i, err)
			}
		}
		if err := copyToDevice(&layer.WqBias); err != nil {
			return fmt.Errorf("layer %d wq_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.WkBias); err != nil {
			return fmt.Errorf("layer %d wk_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.WvBias); err != nil {
			return fmt.Errorf("layer %d wv_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.Wo); err != nil {
			return fmt.Errorf("layer %d wo: %w", i, err)
		}
		if err := copyToDevice(&layer.WoBias); err != nil {
			return fmt.Errorf("layer %d wo_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.FFNNorm); err != nil {
			return fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}
		if err := copyToDevice(&layer.FFNNormBias); err != nil {
			return fmt.Errorf("layer %d ffn_norm_bias: %w", i, err)
		}
		// For fused gate+up (W1W3): copy to GPU, then derive W1/W3 as sub-regions
		// to avoid duplicating memory. W1W3 is set by Phi-3 loader (MLPSwiGLUFused).
		if !layer.W1W3.DevicePtr().IsNil() && layer.W1W3.DevicePtr().Location() != m.backend.Device().Location {
			if err := copyToDevice(&layer.W1W3); err != nil {
				return fmt.Errorf("layer %d w1w3: %w", i, err)
			}
			// Derive W1/W3 as sub-regions of the GPU W1W3 tensor
			m.splitGateUpGPU(layer)
		} else {
			if err := copyToDevice(&layer.W1); err != nil {
				return fmt.Errorf("layer %d w1: %w", i, err)
			}
			if err := copyToDevice(&layer.W3); err != nil {
				return fmt.Errorf("layer %d w3: %w", i, err)
			}
			if err := copyToDevice(&layer.W1W3); err != nil {
				return fmt.Errorf("layer %d w1w3: %w", i, err)
			}
		}
		if err := copyToDevice(&layer.W1Bias); err != nil {
			return fmt.Errorf("layer %d w1_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.W2); err != nil {
			return fmt.Errorf("layer %d w2: %w", i, err)
		}
		if err := copyToDevice(&layer.W2Bias); err != nil {
			return fmt.Errorf("layer %d w2_bias: %w", i, err)
		}
		// Post-norm weights (Gemma 2)
		if err := copyToDevice(&layer.PostAttnNorm); err != nil {
			return fmt.Errorf("layer %d post_attn_norm: %w", i, err)
		}
		if err := copyToDevice(&layer.PostFFNNorm); err != nil {
			return fmt.Errorf("layer %d post_ffn_norm: %w", i, err)
		}
		// MoE expert weights
		if err := copyToDevice(&layer.ExpertGate); err != nil {
			return fmt.Errorf("layer %d expert_gate: %w", i, err)
		}
		if err := copyToDevice(&layer.ExpertGateProj); err != nil {
			return fmt.Errorf("layer %d expert_gate_proj: %w", i, err)
		}
		if err := copyToDevice(&layer.ExpertUpProj); err != nil {
			return fmt.Errorf("layer %d expert_up_proj: %w", i, err)
		}
		if err := copyToDevice(&layer.ExpertDownProj); err != nil {
			return fmt.Errorf("layer %d expert_down_proj: %w", i, err)
		}
	}

	// Sync to ensure all copies complete
	m.backend.Sync()

	return nil
}

// FuseQKVWeights concatenates separate Wq, Wk, Wv weights into a single Wqkv tensor
// for each layer. This enables a single fused matmul during prefill instead of 3 separate ones,
// which improves GPU utilization at larger N dimensions (~50% GFLOPS gain from N=4096 → N=12288).
// Must be called after CopyWeightsToDevice. Separate Wq/Wk/Wv are preserved for decode.
func (m *ModelRuntime) FuseQKVWeights() error {
	// Need a buffer copier for GPU-to-GPU copy
	copier, ok := m.backend.(interface {
		CopyBuffer(src tensor.DevicePtr, srcOffset int, dst tensor.DevicePtr, dstOffset int, size int)
	})
	if !ok {
		return fmt.Errorf("backend does not support CopyBuffer")
	}

	allocPerm, hasPerm := m.backend.(interface{ AllocPermanent(int) tensor.DevicePtr })

	for i, layer := range m.layers {
		// Skip if already has fused QKV (e.g., Phi-2)
		if !layer.Wqkv.DevicePtr().IsNil() {
			continue
		}
		// Skip if missing any of Q/K/V
		if layer.Wq.DevicePtr().IsNil() || layer.Wk.DevicePtr().IsNil() || layer.Wv.DevicePtr().IsNil() {
			continue
		}
		// Only fuse Q4_0 weights (the only quant profile used for prefill GEMM)
		if !layer.Wq.IsQuantized() || layer.Wq.QuantProfile() != tensor.Q4_0 {
			continue
		}

		qDims := layer.Wq.Shape().Dims()
		kDims := layer.Wk.Shape().Dims()
		vDims := layer.Wv.Shape().Dims()
		if len(qDims) != 2 || len(kDims) != 2 || len(vDims) != 2 {
			continue
		}

		qRows, cols := qDims[0], qDims[1]
		kRows := kDims[0]
		vRows := vDims[0]
		totalRows := qRows + kRows + vRows

		// Q4_0: 32 elements per block, 18 bytes per block
		blockSize := 32
		bytesPerBlock := 18

		// Verify alignment
		qElements := qRows * cols
		kElements := kRows * cols
		vElements := vRows * cols
		if qElements%blockSize != 0 || kElements%blockSize != 0 || vElements%blockSize != 0 {
			log.Printf("Warning: layer %d QKV fusion skipped (not aligned to Q4_0 block size)", i)
			continue
		}

		qSizeBytes := (qElements / blockSize) * bytesPerBlock
		kSizeBytes := (kElements / blockSize) * bytesPerBlock
		vSizeBytes := (vElements / blockSize) * bytesPerBlock
		totalBytes := qSizeBytes + kSizeBytes + vSizeBytes

		// Allocate fused buffer
		var fusedPtr tensor.DevicePtr
		if hasPerm {
			fusedPtr = allocPerm.AllocPermanent(totalBytes)
		} else {
			fusedPtr = m.backend.Alloc(totalBytes)
		}
		if fusedPtr.IsNil() {
			return fmt.Errorf("layer %d: failed to allocate %d bytes for fused QKV", i, totalBytes)
		}

		// Copy Q, K, V data into fused buffer
		copier.CopyBuffer(layer.Wq.DevicePtr(), layer.Wq.DevicePtr().Offset(), fusedPtr, 0, qSizeBytes)
		copier.CopyBuffer(layer.Wk.DevicePtr(), layer.Wk.DevicePtr().Offset(), fusedPtr, qSizeBytes, kSizeBytes)
		copier.CopyBuffer(layer.Wv.DevicePtr(), layer.Wv.DevicePtr().Offset(), fusedPtr, qSizeBytes+kSizeBytes, vSizeBytes)

		// Create fused tensor
		layer.Wqkv = tensor.NewQuantTensor(tensor.NewShape(totalRows, cols), m.config.DType, fusedPtr, tensor.Q4_0)

		if i == 0 {
			log.Printf("QKV fusion: [%d,%d]+[%d,%d]+[%d,%d] → [%d,%d] (%.1f MB/layer, %d layers)",
				qRows, cols, kRows, cols, vRows, cols, totalRows, cols,
				float64(totalBytes)/(1024*1024), len(m.layers))
		}
	}

	// Sync to ensure all copies complete
	m.backend.Sync()
	return nil
}

// FuseGateUpWeights concatenates separate W1 (gate) and W3 (up) weights into a single
// W1W3 tensor for each layer. This enables a single fused matmul during prefill instead of 2
// separate ones, which improves GPU utilization at larger N dimensions.
// Must be called after CopyWeightsToDevice. Separate W1/W3 are preserved for decode (fused MLP).
func (m *ModelRuntime) FuseGateUpWeights() error {
	// Need a buffer copier for GPU-to-GPU copy
	copier, ok := m.backend.(interface {
		CopyBuffer(src tensor.DevicePtr, srcOffset int, dst tensor.DevicePtr, dstOffset int, size int)
	})
	if !ok {
		return fmt.Errorf("backend does not support CopyBuffer")
	}

	allocPerm, hasPerm := m.backend.(interface{ AllocPermanent(int) tensor.DevicePtr })

	for i, layer := range m.layers {
		// Skip if already has fused W1W3
		if !layer.W1W3.DevicePtr().IsNil() {
			continue
		}
		// Skip if missing W1 or W3 (GELU MLP models don't have W3)
		if layer.W1.DevicePtr().IsNil() || layer.W3.DevicePtr().IsNil() {
			continue
		}
		// Only fuse Q4_0 weights (the only quant profile used for prefill GEMM)
		if !layer.W1.IsQuantized() || layer.W1.QuantProfile() != tensor.Q4_0 {
			continue
		}

		w1Dims := layer.W1.Shape().Dims()
		w3Dims := layer.W3.Shape().Dims()
		if len(w1Dims) != 2 || len(w3Dims) != 2 {
			continue
		}

		w1Rows, cols := w1Dims[0], w1Dims[1]
		w3Rows := w3Dims[0]
		totalRows := w1Rows + w3Rows

		// Q4_0: 32 elements per block, 18 bytes per block
		blockSize := 32
		bytesPerBlock := 18

		// Verify alignment
		w1Elements := w1Rows * cols
		w3Elements := w3Rows * cols
		if w1Elements%blockSize != 0 || w3Elements%blockSize != 0 {
			log.Printf("Warning: layer %d gate_up fusion skipped (not aligned to Q4_0 block size)", i)
			continue
		}

		w1SizeBytes := (w1Elements / blockSize) * bytesPerBlock
		w3SizeBytes := (w3Elements / blockSize) * bytesPerBlock
		totalBytes := w1SizeBytes + w3SizeBytes

		// Allocate fused buffer
		var fusedPtr tensor.DevicePtr
		if hasPerm {
			fusedPtr = allocPerm.AllocPermanent(totalBytes)
		} else {
			fusedPtr = m.backend.Alloc(totalBytes)
		}
		if fusedPtr.IsNil() {
			return fmt.Errorf("layer %d: failed to allocate %d bytes for fused gate_up", i, totalBytes)
		}

		// Copy W1 (gate), W3 (up) data into fused buffer
		copier.CopyBuffer(layer.W1.DevicePtr(), layer.W1.DevicePtr().Offset(), fusedPtr, 0, w1SizeBytes)
		copier.CopyBuffer(layer.W3.DevicePtr(), layer.W3.DevicePtr().Offset(), fusedPtr, w1SizeBytes, w3SizeBytes)

		// Create fused tensor
		layer.W1W3 = tensor.NewQuantTensor(tensor.NewShape(totalRows, cols), m.config.DType, fusedPtr, tensor.Q4_0)

		if i == 0 {
			log.Printf("Gate_up fusion: [%d,%d]+[%d,%d] → [%d,%d] (%.1f MB/layer, %d layers)",
				w1Rows, cols, w3Rows, cols, totalRows, cols,
				float64(totalBytes)/(1024*1024), len(m.layers))
		}
	}

	// Sync to ensure all copies complete
	m.backend.Sync()
	return nil
}

func (m *ModelRuntime) mapTensor(name string, t tensor.Tensor) {
	// Global
	if name == "model.embed_tokens.weight" || name == "token_embd.weight" {
		m.Embedding = t
		return
	}
	if name == "model.norm.weight" || name == "output_norm.weight" {
		m.FinalNorm = t
		return
	}
	if name == "model.norm.bias" || name == "output_norm.bias" {
		m.FinalNormBias = t
		return
	}
	if name == "lm_head.weight" || name == "output.weight" {
		m.OutputHead = t
		return
	}

	// Layers: model.layers.{i}.X or blk.{i}.X
	var idx int
	var suffix string

	if strings.HasPrefix(name, "model.layers.") {
		parts := strings.Split(name, ".")
		// parts[2] is index
		var err error
		idx, err = strconv.Atoi(parts[2])
		if err != nil || idx >= len(m.layers) {
			return
		}
		suffix = strings.Join(parts[3:], ".")
	} else if strings.HasPrefix(name, "blk.") {
		parts := strings.Split(name, ".")
		// parts[1] is index
		var err error
		idx, err = strconv.Atoi(parts[1])
		if err != nil || idx >= len(m.layers) {
			return
		}
		suffix = strings.Join(parts[2:], ".")
	} else {
		return
	}

	layer := m.layers[idx]

	switch suffix {
	// LLaMA-style separate Q/K/V projections
	case "self_attn.q_proj.weight":
		layer.Wq = t
	case "self_attn.k_proj.weight":
		layer.Wk = t
	case "self_attn.v_proj.weight":
		layer.Wv = t
	// Separate Q/K/V bias (Qwen2)
	case "self_attn.q_proj.bias", "attn_q.bias":
		layer.WqBias = t
	case "self_attn.k_proj.bias", "attn_k.bias":
		layer.WkBias = t
	case "self_attn.v_proj.bias", "attn_v.bias":
		layer.WvBias = t

	// Phi-style combined QKV projection (needs splitting)
	// Also handling Phi-2 legacy names
	case "self_attn.qkv_proj.weight", "attn_qkv.weight":
		layer.Wqkv = t
		m.splitQKVWeight(layer, t)
	case "self_attn.qkv_proj.bias", "attn_qkv.bias":
		layer.WqkvBias = t
		m.splitQKVBias(layer, t)

	// Output projection
	case "self_attn.o_proj.weight", "attn_output.weight":
		layer.Wo = t
	case "self_attn.o_proj.bias", "attn_output.bias":
		layer.WoBias = t

	// LLaMA-style MLP (SwiGLU)
	case "mlp.gate_proj.weight":
		layer.W1 = t
	case "mlp.up_proj.weight":
		layer.W3 = t
	case "mlp.down_proj.weight":
		layer.W2 = t

	// MoE (Mixture of Experts) tensors
	case "mlp.gate.weight", "ffn_gate_inp.weight":
		layer.ExpertGate = t
	case "mlp.experts.gate_proj.weight", "ffn_gate_exps.weight":
		layer.ExpertGateProj = t
	case "mlp.experts.up_proj.weight", "ffn_up_exps.weight":
		layer.ExpertUpProj = t
	case "mlp.experts.down_proj.weight", "ffn_down_exps.weight":
		layer.ExpertDownProj = t

	// Phi-3-style fused gate+up projection (SwiGLU with pre-fused tensor)
	case "mlp.gate_up_proj.weight":
		layer.W1W3 = t
		m.splitGateUpWeight(layer, t)

	// Phi-style MLP (GELU)
	// fc1 is up/gate combined (or just up/gate depending on impl)
	// For Phi-2: fc1 is "ffn_up", fc2 is "ffn_down"
	case "mlp.fc1.weight", "ffn_up.weight":
		layer.W1 = t
	case "mlp.fc1.bias", "ffn_up.bias":
		layer.W1Bias = t
	case "mlp.fc2.weight", "ffn_down.weight":
		layer.W2 = t
	case "mlp.fc2.bias", "ffn_down.bias":
		layer.W2Bias = t

	// Normalization layers
	case "input_layernorm.weight", "attn_norm.weight":
		layer.AttnNorm = t
	case "input_layernorm.bias", "attn_norm.bias":
		layer.AttnNormBias = t
	case "post_attention_layernorm.weight", "ffn_norm.weight":
		layer.FFNNorm = t
	case "post_attention_layernorm.bias", "ffn_norm.bias":
		layer.FFNNormBias = t

	// Post-norm weights (Gemma 2): applied after attention/MLP, before residual
	// GGUF uses "post_attention_norm" and "post_ffw_norm" naming convention
	case "post_attention_norm.weight", "attn_post_norm.weight":
		layer.PostAttnNorm = t
	case "post_ffw_norm.weight", "ffn_post_norm.weight":
		layer.PostFFNNorm = t
	}
}

// splitQKVWeight splits a combined QKV weight matrix into separate Q, K, V tensors.
// Phi-2 has [3*hidden, hidden] shaped Wqkv that contains Q, K, V stacked.
func (m *ModelRuntime) splitQKVWeight(layer *BlockRuntime, combined tensor.Tensor) {
	dims := combined.Shape().Dims()
	if len(dims) != 2 {
		log.Printf("Warning: unexpected QKV weight shape: %v", dims)
		return
	}

	totalRows := dims[0] // 3 * hidden (or adjusted for GQA)
	cols := dims[1]      // hidden

	// Calculate Q, K, V sizes based on head configuration
	headDim := m.config.EffectiveHeadDim()
	qRows := m.config.NumAttentionHeads * headDim // Q uses all heads
	kvRows := m.config.NumKeyValueHeads * headDim // K, V may use fewer heads (GQA)

	// Verify total matches
	expectedRows := qRows + kvRows + kvRows
	if totalRows != expectedRows {
		log.Printf("Warning: QKV weight rows %d != expected %d (Q=%d, KV=%d each)",
			totalRows, expectedRows, qRows, kvRows)
		return
	}

	// Get raw data pointer
	srcPtr := combined.DevicePtr()
	if srcPtr.IsNil() {
		return
	}

	// For quantized tensors, we need different handling
	if combined.IsQuantized() {
		profile := combined.QuantProfile()
		var blockSize, bytesPerBlock int
		switch profile {
		case tensor.Q4_0:
			blockSize, bytesPerBlock = 32, 18
		case tensor.Q4_K:
			blockSize, bytesPerBlock = 256, 144
		case tensor.Q5_K:
			blockSize, bytesPerBlock = 256, 176
		case tensor.Q6_K:
			blockSize, bytesPerBlock = 256, 210
		case tensor.Q5_0:
			blockSize, bytesPerBlock = 32, 22
		case tensor.Q8_0:
			blockSize, bytesPerBlock = 32, 34
		case tensor.BF16:
			blockSize, bytesPerBlock = 1, 2
		default:
			log.Printf("Warning: splitting quantized profile %v not yet supported", profile)
			return
		}

		qElements := qRows * cols
		kvElements := kvRows * cols

		if qElements%blockSize != 0 || kvElements%blockSize != 0 {
			log.Printf("Warning: quantized split point not aligned with block size (%d)", blockSize)
			return
		}

		qSizeBytes := (qElements / blockSize) * bytesPerBlock
		kvSizeBytes := (kvElements / blockSize) * bytesPerBlock

		qPtr := srcPtr
		kPtr := tensor.DevicePtrOffset(srcPtr, uintptr(qSizeBytes))
		vPtr := tensor.DevicePtrOffset(srcPtr, uintptr(qSizeBytes+kvSizeBytes))

		layer.Wq = tensor.NewQuantTensor(tensor.NewShape(qRows, cols), m.config.DType, qPtr, profile)
		layer.Wk = tensor.NewQuantTensor(tensor.NewShape(kvRows, cols), m.config.DType, kPtr, profile)
		layer.Wv = tensor.NewQuantTensor(tensor.NewShape(kvRows, cols), m.config.DType, vPtr, profile)
		return
	}

	// F32 tensors: 4 bytes per element
	elemSize := 4

	// Create sub-tensors by offsetting into the combined buffer
	qSize := qRows * cols * elemSize
	kvSize := kvRows * cols * elemSize

	qPtr := srcPtr
	kPtr := tensor.DevicePtrOffset(srcPtr, uintptr(qSize))
	vPtr := tensor.DevicePtrOffset(srcPtr, uintptr(qSize+kvSize))

	layer.Wq = tensor.NewTensor(tensor.NewShape(qRows, cols), m.config.DType, qPtr)
	layer.Wk = tensor.NewTensor(tensor.NewShape(kvRows, cols), m.config.DType, kPtr)
	layer.Wv = tensor.NewTensor(tensor.NewShape(kvRows, cols), m.config.DType, vPtr)
}

// splitQKVBias splits a combined QKV bias vector into separate Q, K, V tensors.
func (m *ModelRuntime) splitQKVBias(layer *BlockRuntime, combined tensor.Tensor) {
	dims := combined.Shape().Dims()
	if len(dims) != 1 {
		log.Printf("Warning: unexpected QKV bias shape: %v", dims)
		return
	}

	totalSize := dims[0]

	// Calculate Q, K, V sizes based on head configuration
	headDim := m.config.EffectiveHeadDim()
	qSize := m.config.NumAttentionHeads * headDim
	kvSize := m.config.NumKeyValueHeads * headDim

	expectedSize := qSize + kvSize + kvSize
	if totalSize != expectedSize {
		log.Printf("Warning: QKV bias size %d != expected %d", totalSize, expectedSize)
		return
	}

	srcPtr := combined.DevicePtr()
	if srcPtr.IsNil() {
		return
	}

	// F32 bias: 4 bytes per element
	elemSize := 4

	qPtr := srcPtr
	kPtr := tensor.DevicePtrOffset(srcPtr, uintptr(qSize*elemSize))
	vPtr := tensor.DevicePtrOffset(srcPtr, uintptr((qSize+kvSize)*elemSize))

	layer.WqBias = tensor.NewTensor(tensor.NewShape(qSize), m.config.DType, qPtr)
	layer.WkBias = tensor.NewTensor(tensor.NewShape(kvSize), m.config.DType, kPtr)
	layer.WvBias = tensor.NewTensor(tensor.NewShape(kvSize), m.config.DType, vPtr)
}

// splitGateUpWeight splits a fused gate+up weight matrix into separate W1 (gate) and W3 (up) tensors.
// Phi-3 has [2*intermediate, hidden] shaped gate_up that contains gate and up stacked row-wise.
func (m *ModelRuntime) splitGateUpWeight(layer *BlockRuntime, combined tensor.Tensor) {
	dims := combined.Shape().Dims()
	if len(dims) != 2 {
		log.Printf("Warning: unexpected gate_up weight shape: %v", dims)
		return
	}

	totalRows := dims[0] // 2 * intermediate
	cols := dims[1]      // hidden

	intermediateSize := m.config.IntermediateSize
	if totalRows != 2*intermediateSize {
		log.Printf("Warning: gate_up weight rows %d != expected %d (2*%d)",
			totalRows, 2*intermediateSize, intermediateSize)
		return
	}

	srcPtr := combined.DevicePtr()
	if srcPtr.IsNil() {
		return
	}

	gateRows := intermediateSize
	upRows := intermediateSize

	// For quantized tensors, calculate byte offsets based on quantization block sizes
	if combined.IsQuantized() {
		profile := combined.QuantProfile()
		var blockSize, bytesPerBlock int
		switch profile {
		case tensor.Q4_0:
			blockSize, bytesPerBlock = 32, 18
		case tensor.Q4_K:
			blockSize, bytesPerBlock = 256, 144
		case tensor.Q5_K:
			blockSize, bytesPerBlock = 256, 176
		case tensor.Q6_K:
			blockSize, bytesPerBlock = 256, 210
		case tensor.Q5_0:
			blockSize, bytesPerBlock = 32, 22
		case tensor.Q8_0:
			blockSize, bytesPerBlock = 32, 34
		case tensor.BF16:
			blockSize, bytesPerBlock = 1, 2
		default:
			log.Printf("Warning: splitting quantized gate_up profile %v not yet supported", profile)
			return
		}

		gateElements := gateRows * cols
		if gateElements%blockSize != 0 {
			log.Printf("Warning: quantized gate_up split point not aligned with block size (%d)", blockSize)
			return
		}

		gateSizeBytes := (gateElements / blockSize) * bytesPerBlock

		gatePtr := srcPtr
		upPtr := tensor.DevicePtrOffset(srcPtr, uintptr(gateSizeBytes))

		layer.W1 = tensor.NewQuantTensor(tensor.NewShape(gateRows, cols), m.config.DType, gatePtr, profile)
		layer.W3 = tensor.NewQuantTensor(tensor.NewShape(upRows, cols), m.config.DType, upPtr, profile)
		return
	}

	// F32 tensors: 4 bytes per element
	elemSize := 4
	gateSizeBytes := gateRows * cols * elemSize

	gatePtr := srcPtr
	upPtr := tensor.DevicePtrOffset(srcPtr, uintptr(gateSizeBytes))

	layer.W1 = tensor.NewTensor(tensor.NewShape(gateRows, cols), m.config.DType, gatePtr)
	layer.W3 = tensor.NewTensor(tensor.NewShape(upRows, cols), m.config.DType, upPtr)
}

// splitGateUpGPU derives W1 (gate) and W3 (up) as sub-regions of the GPU W1W3 tensor.
// Called after W1W3 has been copied to GPU, to avoid duplicate GPU memory allocations.
func (m *ModelRuntime) splitGateUpGPU(layer *BlockRuntime) {
	dims := layer.W1W3.Shape().Dims()
	if len(dims) != 2 {
		return
	}

	totalRows := dims[0]
	cols := dims[1]
	intermediateSize := m.config.IntermediateSize

	if totalRows != 2*intermediateSize {
		return
	}

	gateRows := intermediateSize
	upRows := intermediateSize
	srcPtr := layer.W1W3.DevicePtr()

	if layer.W1W3.IsQuantized() {
		profile := layer.W1W3.QuantProfile()
		var blockSize, bytesPerBlock int
		switch profile {
		case tensor.Q4_0:
			blockSize, bytesPerBlock = 32, 18
		case tensor.Q4_K:
			blockSize, bytesPerBlock = 256, 144
		case tensor.Q5_K:
			blockSize, bytesPerBlock = 256, 176
		case tensor.Q6_K:
			blockSize, bytesPerBlock = 256, 210
		case tensor.Q5_0:
			blockSize, bytesPerBlock = 32, 22
		case tensor.Q8_0:
			blockSize, bytesPerBlock = 32, 34
		case tensor.BF16:
			blockSize, bytesPerBlock = 1, 2
		default:
			return
		}

		gateElements := gateRows * cols
		if gateElements%blockSize != 0 {
			return
		}
		gateSizeBytes := (gateElements / blockSize) * bytesPerBlock

		layer.W1 = tensor.NewQuantTensor(tensor.NewShape(gateRows, cols), m.config.DType, srcPtr, profile)
		layer.W3 = tensor.NewQuantTensor(tensor.NewShape(upRows, cols), m.config.DType,
			tensor.DevicePtrOffset(srcPtr, uintptr(gateSizeBytes)), profile)
		return
	}

	gateSizeBytes := gateRows * cols * 4
	layer.W1 = tensor.NewTensor(tensor.NewShape(gateRows, cols), m.config.DType, srcPtr)
	layer.W3 = tensor.NewTensor(tensor.NewShape(upRows, cols), m.config.DType,
		tensor.DevicePtrOffset(srcPtr, uintptr(gateSizeBytes)))
}

// splitQKVGPU derives Wq, Wk, Wv as sub-regions of the GPU Wqkv tensor.
// Called after Wqkv has been copied to GPU, to avoid duplicate GPU memory allocations.
func (m *ModelRuntime) splitQKVGPU(layer *BlockRuntime) {
	dims := layer.Wqkv.Shape().Dims()
	if len(dims) != 2 {
		return
	}

	totalRows := dims[0]
	cols := dims[1]

	headDim := m.config.EffectiveHeadDim()
	qRows := m.config.NumAttentionHeads * headDim
	kvRows := m.config.NumKeyValueHeads * headDim

	expectedRows := qRows + kvRows + kvRows
	if totalRows != expectedRows {
		return
	}

	srcPtr := layer.Wqkv.DevicePtr()

	if layer.Wqkv.IsQuantized() {
		profile := layer.Wqkv.QuantProfile()
		var blockSize, bytesPerBlock int
		switch profile {
		case tensor.Q4_0:
			blockSize, bytesPerBlock = 32, 18
		case tensor.Q4_K:
			blockSize, bytesPerBlock = 256, 144
		case tensor.Q5_K:
			blockSize, bytesPerBlock = 256, 176
		case tensor.Q6_K:
			blockSize, bytesPerBlock = 256, 210
		case tensor.Q5_0:
			blockSize, bytesPerBlock = 32, 22
		case tensor.Q8_0:
			blockSize, bytesPerBlock = 32, 34
		case tensor.BF16:
			blockSize, bytesPerBlock = 1, 2
		default:
			return
		}

		qElements := qRows * cols
		kvElements := kvRows * cols
		if qElements%blockSize != 0 || kvElements%blockSize != 0 {
			return
		}

		qSizeBytes := (qElements / blockSize) * bytesPerBlock
		kvSizeBytes := (kvElements / blockSize) * bytesPerBlock

		layer.Wq = tensor.NewQuantTensor(tensor.NewShape(qRows, cols), m.config.DType, srcPtr, profile)
		layer.Wk = tensor.NewQuantTensor(tensor.NewShape(kvRows, cols), m.config.DType,
			tensor.DevicePtrOffset(srcPtr, uintptr(qSizeBytes)), profile)
		layer.Wv = tensor.NewQuantTensor(tensor.NewShape(kvRows, cols), m.config.DType,
			tensor.DevicePtrOffset(srcPtr, uintptr(qSizeBytes+kvSizeBytes)), profile)
		return
	}

	elemSize := 4
	qSizeBytes := qRows * cols * elemSize
	kvSizeBytes := kvRows * cols * elemSize

	layer.Wq = tensor.NewTensor(tensor.NewShape(qRows, cols), m.config.DType, srcPtr)
	layer.Wk = tensor.NewTensor(tensor.NewShape(kvRows, cols), m.config.DType,
		tensor.DevicePtrOffset(srcPtr, uintptr(qSizeBytes)))
	layer.Wv = tensor.NewTensor(tensor.NewShape(kvRows, cols), m.config.DType,
		tensor.DevicePtrOffset(srcPtr, uintptr(qSizeBytes+kvSizeBytes)))
}

// splitQKVBiasGPU derives WqBias, WkBias, WvBias as sub-regions of the GPU WqkvBias tensor.
func (m *ModelRuntime) splitQKVBiasGPU(layer *BlockRuntime) {
	dims := layer.WqkvBias.Shape().Dims()
	if len(dims) != 1 {
		return
	}

	headDim := m.config.EffectiveHeadDim()
	qSize := m.config.NumAttentionHeads * headDim
	kvSize := m.config.NumKeyValueHeads * headDim

	srcPtr := layer.WqkvBias.DevicePtr()
	elemSize := 4

	layer.WqBias = tensor.NewTensor(tensor.NewShape(qSize), m.config.DType, srcPtr)
	layer.WkBias = tensor.NewTensor(tensor.NewShape(kvSize), m.config.DType,
		tensor.DevicePtrOffset(srcPtr, uintptr(qSize*elemSize)))
	layer.WvBias = tensor.NewTensor(tensor.NewShape(kvSize), m.config.DType,
		tensor.DevicePtrOffset(srcPtr, uintptr((qSize+kvSize)*elemSize)))
}
