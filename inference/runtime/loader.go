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
		fmt.Printf("\n⚠️  WARNING: %s\n\n", warning)
	} else {
		fmt.Printf("Architecture: %s\n", loader.Architecture())
	}

	// Print stats
	loader.PrintTensorStats()

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
		if found {
			fmt.Printf("Loading %s (%s): Type=%v\n", hfName, ggufName, info.Type)
		}
		if !found {
			fmt.Printf("Warning: tensor %s (%s) not found\n", hfName, ggufName)
			continue
		}


		var t tensor.Tensor

		// For Q4_0/Q4_K weight matrices (not embeddings/norms), keep raw format
		// This enables GPU-native quantized inference
		if info.Type == gguf.TensorTypeQ4_0 && m.isWeightMatrix(hfName) {
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				fmt.Printf("Warning: failed to load raw tensor %s: %v\n", hfName, err)
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
		} else if info.Type == gguf.TensorTypeQ4_K && m.isWeightMatrix(hfName) {
			// Q4_K native GPU kernel - uses get_scale_min_k4 format
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				fmt.Printf("Warning: failed to load raw Q4_K tensor %s: %v\n", hfName, err)
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
		} else if info.Type == gguf.TensorTypeQ5_K && m.isWeightMatrix(hfName) {
			// Q5_K native GPU kernel
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				fmt.Printf("Warning: failed to load raw Q5_K tensor %s: %v\n", hfName, err)
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
				fmt.Printf("Warning: failed to load raw tensor %s: %v\n", hfName, err)
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
		} else if info.Type == gguf.TensorTypeQ8_0 && m.isWeightMatrix(hfName) {
			// Q8_0 native GPU kernel
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				fmt.Printf("Warning: failed to load raw Q8_0 tensor %s: %v\n", hfName, err)
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
		} else if info.Type == gguf.TensorTypeBF16 && m.isWeightMatrix(hfName) {
			// BF16 native GPU kernel - keep raw BF16 data on GPU
			rawData, dims, _, err := loader.LoadTensorRaw(ggufName)
			if err != nil {
				fmt.Printf("Warning: failed to load raw BF16 tensor %s: %v\n", hfName, err)
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
				fmt.Printf("Warning: failed to load tensor %s: %v\n", hfName, err)
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

	fmt.Printf("Loaded %d/%d tensors from GGUF (%d quantized raw)\n", loadedCount, len(tensorNames), q4Count)

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

	loader.PrintTensorStats()

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
			fmt.Printf("Warning: tensor %s (%s) not found\n", hfName, ggufName)
			continue
		}
		_ = info // Unused but needed for consistency

		// Always dequantize to F32
		data, dims, err := loader.LoadTensor(ggufName)
		if err != nil {
			fmt.Printf("Warning: failed to load tensor %s: %v\n", hfName, err)
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

	fmt.Printf("Loaded %d/%d tensors from GGUF (all F32 dequantized)\n", loadedCount, len(tensorNames))
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
			case tensor.Q6_K:
				// Q6_K: 210 bytes per 256 elements
				numBlocks := (numElements + 255) / 256
				sizeBytes = numBlocks * 210
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

	// Copy layer weights
	for i, layer := range m.layers {
		if err := copyToDevice(&layer.AttnNorm); err != nil {
			return fmt.Errorf("layer %d attn_norm: %w", i, err)
		}
		if err := copyToDevice(&layer.AttnNormBias); err != nil {
			return fmt.Errorf("layer %d attn_norm_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.Wq); err != nil {
			return fmt.Errorf("layer %d wq: %w", i, err)
		}
		if err := copyToDevice(&layer.WqBias); err != nil {
			return fmt.Errorf("layer %d wq_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.Wk); err != nil {
			return fmt.Errorf("layer %d wk: %w", i, err)
		}
		if err := copyToDevice(&layer.WkBias); err != nil {
			return fmt.Errorf("layer %d wk_bias: %w", i, err)
		}
		if err := copyToDevice(&layer.Wv); err != nil {
			return fmt.Errorf("layer %d wv: %w", i, err)
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
		if err := copyToDevice(&layer.W1); err != nil {
			return fmt.Errorf("layer %d w1: %w", i, err)
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
	}
}

// splitQKVWeight splits a combined QKV weight matrix into separate Q, K, V tensors.
// Phi-2 has [3*hidden, hidden] shaped Wqkv that contains Q, K, V stacked.
func (m *ModelRuntime) splitQKVWeight(layer *BlockRuntime, combined tensor.Tensor) {
	dims := combined.Shape().Dims()
	if len(dims) != 2 {
		fmt.Printf("Warning: unexpected QKV weight shape: %v\n", dims)
		return
	}

	totalRows := dims[0] // 3 * hidden (or adjusted for GQA)
	cols := dims[1]      // hidden

	// Calculate Q, K, V sizes based on head configuration
	headDim := m.config.HiddenSize / m.config.NumAttentionHeads
	qRows := m.config.NumAttentionHeads * headDim    // Q uses all heads
	kvRows := m.config.NumKeyValueHeads * headDim   // K, V may use fewer heads (GQA)

	// Verify total matches
	expectedRows := qRows + kvRows + kvRows
	if totalRows != expectedRows {
		fmt.Printf("Warning: QKV weight rows %d != expected %d (Q=%d, KV=%d each)\n",
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
		case tensor.Q8_0:
			blockSize, bytesPerBlock = 32, 34
		case tensor.BF16:
			blockSize, bytesPerBlock = 1, 2
		default:
			fmt.Printf("Warning: splitting quantized profile %v not yet supported\n", profile)
			return
		}

		qElements := qRows * cols
		kvElements := kvRows * cols

		if qElements%blockSize != 0 || kvElements%blockSize != 0 {
			fmt.Printf("Warning: quantized split point not aligned with block size (%d)\n", blockSize)
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
		fmt.Printf("Warning: unexpected QKV bias shape: %v\n", dims)
		return
	}

	totalSize := dims[0]

	// Calculate Q, K, V sizes based on head configuration
	headDim := m.config.HiddenSize / m.config.NumAttentionHeads
	qSize := m.config.NumAttentionHeads * headDim
	kvSize := m.config.NumKeyValueHeads * headDim

	expectedSize := qSize + kvSize + kvSize
	if totalSize != expectedSize {
		fmt.Printf("Warning: QKV bias size %d != expected %d\n", totalSize, expectedSize)
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