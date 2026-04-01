package lora

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// adapterConfigFile is the on-disk representation written by SaveAdapter.
// It includes all fields required by the HuggingFace PEFT format.
type adapterConfigFile struct {
	PeftType             string   `json:"peft_type"`
	TaskType             string   `json:"task_type"`
	Rank                 int      `json:"r"`
	Alpha                float32  `json:"lora_alpha"`
	TargetModules        []string `json:"target_modules"`
	BaseModel            string   `json:"base_model_name_or_path"`
	Bias                 string   `json:"bias"`
	FanInFanOut          bool     `json:"fan_in_fan_out"`
}

// tensorEntry is one entry in the safetensors JSON header.
type tensorEntry struct {
	Dtype       string   `json:"dtype"`
	Shape       [2]int64 `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// SaveAdapter writes a LoRA adapter to dir in HuggingFace PEFT format:
//   - adapter_config.json
//   - adapter_model.safetensors
//
// Only layers that have at least one projection (Q or V) are emitted.
// Layers with neither are skipped silently. Tensors are written as F32.
func SaveAdapter(adapter *Adapter, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create adapter dir: %w", err)
	}

	if err := saveAdapterConfig(adapter, dir); err != nil {
		return err
	}
	if err := saveAdapterWeights(adapter, dir); err != nil {
		return err
	}
	return nil
}

// saveAdapterConfig writes adapter_config.json.
func saveAdapterConfig(adapter *Adapter, dir string) error {
	cfg := adapterConfigFile{
		PeftType:      "LORA",
		TaskType:      "CAUSAL_LM",
		Rank:          adapter.Config.Rank,
		Alpha:         adapter.Config.Alpha,
		TargetModules: adapter.Config.TargetModules,
		BaseModel:     adapter.Config.BaseModel,
		Bias:          "none",
		FanInFanOut:   false,
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal adapter_config.json: %w", err)
	}
	path := filepath.Join(dir, "adapter_config.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write adapter_config.json: %w", err)
	}
	return nil
}

// saveAdapterWeights writes adapter_model.safetensors.
func saveAdapterWeights(adapter *Adapter, dir string) error {
	// Collect tensors in a deterministic order: layer index ascending, then
	// Q before V, then A before B within each projection.
	type namedTensor struct {
		name  string
		shape [2]int64
		data  []float32
	}

	var tensors []namedTensor

	for i, la := range adapter.Layers {
		if la.HasQ() {
			tensors = append(tensors,
				namedTensor{
					name:  tensorName(i, "q", "A"),
					shape: la.QAShape,
					data:  la.QA,
				},
				namedTensor{
					name:  tensorName(i, "q", "B"),
					shape: la.QBShape,
					data:  la.QB,
				},
			)
		}
		if la.HasV() {
			tensors = append(tensors,
				namedTensor{
					name:  tensorName(i, "v", "A"),
					shape: la.VAShape,
					data:  la.VA,
				},
				namedTensor{
					name:  tensorName(i, "v", "B"),
					shape: la.VBShape,
					data:  la.VB,
				},
			)
		}
	}

	// Build the header map and data blob simultaneously.
	header := make(map[string]tensorEntry, len(tensors))
	var dataBlob []byte
	var offset int64

	for _, t := range tensors {
		byteLen := int64(len(t.data)) * 4
		header[t.name] = tensorEntry{
			Dtype:       "F32",
			Shape:       t.shape,
			DataOffsets: [2]int64{offset, offset + byteLen},
		}
		dataBlob = append(dataBlob, f32Bytes(t.data)...)
		offset += byteLen
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("marshal safetensors header: %w", err)
	}

	// Assemble the binary: 8-byte LE header length, header JSON, data blob.
	out := make([]byte, 0, 8+len(headerJSON)+len(dataBlob))

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	out = append(out, lenBuf[:]...)
	out = append(out, headerJSON...)
	out = append(out, dataBlob...)

	path := filepath.Join(dir, "adapter_model.safetensors")
	if err := os.WriteFile(path, out, 0o644); err != nil {
		return fmt.Errorf("write adapter_model.safetensors: %w", err)
	}
	return nil
}

// tensorName returns the HuggingFace PEFT tensor name for a LoRA weight.
// proj is "q" or "v"; mat is "A" or "B".
func tensorName(layerIdx int, proj, mat string) string {
	return fmt.Sprintf(
		"base_model.model.model.layers.%d.self_attn.%s_proj.lora_%s.weight",
		layerIdx, proj, mat,
	)
}

// f32Bytes serialises a []float32 to little-endian raw bytes.
func f32Bytes(vals []float32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
