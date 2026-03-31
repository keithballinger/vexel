//go:build metal && darwin && cgo

package lora_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"vexel/inference/lora"
)

func TestZeroLoRALoadAndUpload(t *testing.T) {
	// Create synthetic adapter: rank=4, 1 layer, Q+V, B=0 (no effect)
	dir := t.TempDir()

	cfgJSON := `{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}`
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(cfgJSON), 0o644); err != nil {
		t.Fatalf("write adapter_config.json: %v", err)
	}

	rank := 4
	hidden := 64 // small for testing
	qDim := 32
	vDim := 16

	qa := make([]float32, rank*hidden)
	qb := make([]float32, qDim*rank) // zeros — no effect
	va := make([]float32, rank*hidden)
	vb := make([]float32, vDim*rank) // zeros — no effect

	for i := range qa {
		qa[i] = 0.01 * float32(i)
	}
	for i := range va {
		va[i] = 0.02 * float32(i)
	}

	st := buildSafetensors(map[string]stSpec{
		"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {[2]int64{int64(rank), int64(hidden)}, qa},
		"base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {[2]int64{int64(qDim), int64(rank)}, qb},
		"base_model.model.layers.0.self_attn.v_proj.lora_A.weight": {[2]int64{int64(rank), int64(hidden)}, va},
		"base_model.model.layers.0.self_attn.v_proj.lora_B.weight": {[2]int64{int64(vDim), int64(rank)}, vb},
	})
	if err := os.WriteFile(filepath.Join(dir, "adapter_model.safetensors"), st, 0o644); err != nil {
		t.Fatalf("write adapter_model.safetensors: %v", err)
	}

	adapter, err := lora.LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	// Verify config
	if adapter.Config.Rank != 4 {
		t.Errorf("rank=%d, want 4", adapter.Config.Rank)
	}
	if adapter.Config.Alpha != 8.0 {
		t.Errorf("alpha=%g, want 8.0", adapter.Config.Alpha)
	}
	if adapter.Scale != 2.0 {
		t.Errorf("scale=%f, want 2.0", adapter.Scale)
	}

	// Verify layers
	if len(adapter.Layers) < 1 {
		t.Fatalf("no layers")
	}
	l0 := adapter.Layers[0]
	if !l0.HasQ() {
		t.Error("layer 0 should have Q LoRA")
	}
	if !l0.HasV() {
		t.Error("layer 0 should have V LoRA")
	}

	// Verify shapes
	if l0.QAShape != [2]int64{4, 64} {
		t.Errorf("QAShape=%v, want [4,64]", l0.QAShape)
	}
	if l0.QBShape != [2]int64{32, 4} {
		t.Errorf("QBShape=%v, want [32,4]", l0.QBShape)
	}
	if l0.VAShape != [2]int64{4, 64} {
		t.Errorf("VAShape=%v, want [4,64]", l0.VAShape)
	}
	if l0.VBShape != [2]int64{16, 4} {
		t.Errorf("VBShape=%v, want [16,4]", l0.VBShape)
	}

	// Verify that QB and VB are all zeros (no effect on output)
	for i, v := range l0.QB {
		if v != 0.0 {
			t.Errorf("QB[%d]=%g, want 0.0 (zero B matrix means no effect)", i, v)
		}
	}
	for i, v := range l0.VB {
		if v != 0.0 {
			t.Errorf("VB[%d]=%g, want 0.0 (zero B matrix means no effect)", i, v)
		}
	}

	// Verify QA data is populated
	for i, v := range l0.QA {
		want := 0.01 * float32(i)
		if math.Abs(float64(v-want)) > 1e-6 {
			t.Errorf("QA[%d]=%g, want %g", i, v, want)
		}
	}

	// Verify VA data is populated
	for i, v := range l0.VA {
		want := 0.02 * float32(i)
		if math.Abs(float64(v-want)) > 1e-6 {
			t.Errorf("VA[%d]=%g, want %g", i, v, want)
		}
	}

	t.Logf("Adapter: rank=%d, alpha=%.0f, scale=%.2f, layers=%d, Q=%v, V=%v",
		adapter.Config.Rank, adapter.Config.Alpha, adapter.Scale,
		len(adapter.Layers), l0.HasQ(), l0.HasV())
}

type stSpec struct {
	Shape [2]int64
	Data  []float32
}

func buildSafetensors(tensors map[string]stSpec) []byte {
	header := make(map[string]interface{})
	var allData []byte
	offset := 0

	// Build deterministic order by collecting entries first
	type entry struct {
		name string
		spec stSpec
	}
	entries := make([]entry, 0, len(tensors))
	for name, spec := range tensors {
		entries = append(entries, entry{name, spec})
	}

	// Process in sorted order for determinism (Go's iteration is random)
	// For this test, we process in the order they appear in the map
	for name, spec := range tensors {
		data := toBytes(spec.Data)
		header[name] = map[string]interface{}{
			"dtype":        "F32",
			"shape":        spec.Shape,
			"data_offsets": [2]int{offset, offset + len(data)},
		}
		allData = append(allData, data...)
		offset += len(data)
	}

	hdr, err := json.Marshal(header)
	if err != nil {
		panic("buildSafetensors: marshal header: " + err.Error())
	}

	out := make([]byte, 8+len(hdr)+len(allData))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(hdr)))
	copy(out[8:], hdr)
	copy(out[8+len(hdr):], allData)
	return out
}

func toBytes(data []float32) []byte {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}
