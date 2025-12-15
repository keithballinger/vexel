package gguf

import (
	"fmt"
	"strings"
)

// TensorLoader provides a high-level interface for loading tensors from GGUF files.
type TensorLoader struct {
	file *File
}

// NewTensorLoader creates a new tensor loader from a GGUF file.
func NewTensorLoader(path string) (*TensorLoader, error) {
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	return &TensorLoader{file: f}, nil
}

// Close closes the underlying file.
func (l *TensorLoader) Close() error {
	return l.file.Close()
}

// File returns the underlying GGUF file for metadata access.
func (l *TensorLoader) File() *File {
	return l.file
}

// Architecture returns the model architecture (e.g., "llama", "phi", "falcon").
func (l *TensorLoader) Architecture() string {
	return l.file.Architecture
}

// SupportedArchitectures lists architectures that Vexel fully supports.
var SupportedArchitectures = map[string]bool{
	"llama":   true, // LLaMA 2/3, TinyLlama, Mistral, Codestral
	"mistral": true, // Some Mistral models use this
	"qwen2":   true, // Qwen 2 is LLaMA-compatible
}

// ValidateArchitecture checks if the architecture is supported and returns a warning if not.
func (l *TensorLoader) ValidateArchitecture() (supported bool, warning string) {
	arch := l.Architecture()
	if arch == "" {
		return true, "" // Unknown architecture, assume compatible
	}

	if SupportedArchitectures[arch] {
		return true, ""
	}

	return false, fmt.Sprintf(
		"Architecture '%s' may not be fully supported. "+
			"Vexel is optimized for LLaMA-family models (llama, mistral, qwen2). "+
			"Other architectures may produce incorrect results due to differences in "+
			"normalization (LayerNorm vs RMSNorm), FFN structure, or position encoding.",
		arch)
}

// LoadTensor loads and dequantizes a tensor by name.
// GGUF stores dimensions in "ne" order where ne[0] is the innermost/fastest dimension.
// This function converts to row-major (C) convention by reversing dimension order.
func (l *TensorLoader) LoadTensor(name string) ([]float32, []int, error) {
	info, ok := l.file.GetTensor(name)
	if !ok {
		return nil, nil, fmt.Errorf("tensor not found: %s", name)
	}

	data, err := l.file.ReadTensorData(info)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read tensor data: %w", err)
	}

	numElements := int(info.NumElements())
	f32Data := Dequantize(data, info.Type, numElements)

	// Convert dimensions to []int, reversing order from GGUF's ne order to row-major
	// GGUF: ne[0] is fastest dimension (innermost)
	// Row-major: last dimension is fastest (rightmost)
	// So [ne[0], ne[1], ...] becomes [..., ne[1], ne[0]]
	dims := make([]int, len(info.Dimensions))
	for i, d := range info.Dimensions {
		dims[len(dims)-1-i] = int(d)
	}

	return f32Data, dims, nil
}

// LoadTensorRaw loads a tensor without dequantizing - returns raw bytes and type.
// This is used for GPU quantized inference where dequantization happens on GPU.
func (l *TensorLoader) LoadTensorRaw(name string) ([]byte, []int, TensorType, error) {
	info, ok := l.file.GetTensor(name)
	if !ok {
		return nil, nil, 0, fmt.Errorf("tensor not found: %s", name)
	}

	data, err := l.file.ReadTensorData(info)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("failed to read tensor data: %w", err)
	}

	// Convert dimensions to []int, reversing order from GGUF's ne order to row-major
	dims := make([]int, len(info.Dimensions))
	for i, d := range info.Dimensions {
		dims[len(dims)-1-i] = int(d)
	}

	return data, dims, info.Type, nil
}

// TensorNameMapping maps HuggingFace-style names to GGUF-style names.
// GGUF uses a different naming convention based on llama.cpp.
var TensorNameMapping = map[string]string{
	// Embeddings
	"model.embed_tokens.weight": "token_embd.weight",
	"lm_head.weight":            "output.weight",
	"model.norm.weight":         "output_norm.weight",
}

// GetLayerTensorName converts a layer tensor name to GGUF format.
// Input: model.layers.0.self_attn.q_proj.weight
// Output: blk.0.attn_q.weight
func GetLayerTensorName(hfName string) string {
	if !strings.HasPrefix(hfName, "model.layers.") {
		if mapped, ok := TensorNameMapping[hfName]; ok {
			return mapped
		}
		return hfName
	}

	// Parse layer index and suffix
	parts := strings.Split(hfName, ".")
	if len(parts) < 4 {
		return hfName
	}
	layerIdx := parts[2]
	suffix := strings.Join(parts[3:], ".")

	// Map suffix to GGUF style
	var ggufSuffix string
	switch suffix {
	case "self_attn.q_proj.weight":
		ggufSuffix = "attn_q.weight"
	case "self_attn.k_proj.weight":
		ggufSuffix = "attn_k.weight"
	case "self_attn.v_proj.weight":
		ggufSuffix = "attn_v.weight"
	case "self_attn.o_proj.weight":
		ggufSuffix = "attn_output.weight"
	case "mlp.gate_proj.weight":
		ggufSuffix = "ffn_gate.weight"
	case "mlp.up_proj.weight":
		ggufSuffix = "ffn_up.weight"
	case "mlp.down_proj.weight":
		ggufSuffix = "ffn_down.weight"
	case "input_layernorm.weight":
		ggufSuffix = "attn_norm.weight"
	case "post_attention_layernorm.weight":
		ggufSuffix = "ffn_norm.weight"
	default:
		return hfName
	}

	return fmt.Sprintf("blk.%s.%s", layerIdx, ggufSuffix)
}

// ListTensors returns all tensor names in the file.
func (l *TensorLoader) ListTensors() []string {
	names := make([]string, len(l.file.Tensors))
	for i, t := range l.file.Tensors {
		names[i] = t.Name
	}
	return names
}

// GetTensorInfo returns information about a tensor.
func (l *TensorLoader) GetTensorInfo(name string) (TensorInfo, bool) {
	return l.file.GetTensor(name)
}

// TensorStats returns statistics about tensors by type.
func (l *TensorLoader) TensorStats() map[TensorType]int {
	stats := make(map[TensorType]int)
	for _, t := range l.file.Tensors {
		stats[t.Type]++
	}
	return stats
}

// PrintTensorStats prints a summary of tensor types.
func (l *TensorLoader) PrintTensorStats() {
	stats := l.TensorStats()
	fmt.Println("Tensor quantization types:")
	for typ, count := range stats {
		fmt.Printf("  %s: %d tensors\n", typ.String(), count)
	}
}
