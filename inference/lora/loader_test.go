package lora

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// --- helpers -----------------------------------------------------------------

// f32ToBytes serialises a []float32 to little-endian raw bytes.
func f32ToBytes(vals []float32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// tensorSpec describes one tensor for buildSafetensors.
type tensorSpec struct {
	shape [2]int64
	data  []float32
}

// buildSafetensors produces a valid safetensors binary from the provided map
// of tensor name → tensorSpec.  All tensors are written as F32.
func buildSafetensors(tensors map[string]tensorSpec) []byte {
	// Build contiguous data blob and compute per-tensor offsets.
	type entry struct {
		name  string
		spec  tensorSpec
		start int64
		end   int64
	}
	entries := make([]entry, 0, len(tensors))
	var offset int64
	for name, spec := range tensors {
		byteLen := int64(len(spec.data)) * 4
		entries = append(entries, entry{
			name:  name,
			spec:  spec,
			start: offset,
			end:   offset + byteLen,
		})
		offset += byteLen
	}

	// Build header JSON.
	headerMap := make(map[string]interface{}, len(entries))
	for _, e := range entries {
		headerMap[e.name] = map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int64{e.spec.shape[0], e.spec.shape[1]},
			"data_offsets": [2]int64{e.start, e.end},
		}
	}
	headerBytes, err := json.Marshal(headerMap)
	if err != nil {
		panic("buildSafetensors: marshal header: " + err.Error())
	}

	// Assemble: 8-byte LE header length + header JSON + data blob.
	out := make([]byte, 0, 8+len(headerBytes)+int(offset))

	lenBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenBuf, uint64(len(headerBytes)))
	out = append(out, lenBuf...)
	out = append(out, headerBytes...)

	for _, e := range entries {
		out = append(out, f32ToBytes(e.spec.data)...)
	}
	return out
}

// writeTempAdapter writes a minimal adapter directory to a temp dir and
// returns the dir path.
func writeTempAdapter(t *testing.T, tensors map[string]tensorSpec) string {
	t.Helper()
	dir := t.TempDir()

	// Write adapter_config.json.
	cfg := `{"r":4,"lora_alpha":8.0,"target_modules":["q_proj","v_proj"],"base_model_name_or_path":"meta-llama/Llama-2-7b-hf"}`
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatalf("write adapter_config.json: %v", err)
	}

	// Write adapter_model.safetensors.
	st := buildSafetensors(tensors)
	if err := os.WriteFile(filepath.Join(dir, "adapter_model.safetensors"), st, 0o644); err != nil {
		t.Fatalf("write adapter_model.safetensors: %v", err)
	}

	return dir
}

// --- tests -------------------------------------------------------------------

func TestLoadAdapter(t *testing.T) {
	// rank=4, hidden=8 → A is (rank × hidden) = (4,8), B is (hidden × rank) = (8,4)
	const (
		rank   = 4
		hidden = 8
	)

	qaData := make([]float32, rank*hidden) // shape [4,8]
	qbData := make([]float32, hidden*rank) // shape [8,4]
	for i := range qaData {
		qaData[i] = float32(i) * 0.1
	}
	for i := range qbData {
		qbData[i] = float32(i) * 0.2
	}

	tensors := map[string]tensorSpec{
		"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {
			shape: [2]int64{rank, hidden},
			data:  qaData,
		},
		"base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {
			shape: [2]int64{hidden, rank},
			data:  qbData,
		},
	}

	dir := writeTempAdapter(t, tensors)

	adapter, err := LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	// Config sanity.
	if adapter.Config.Rank != 4 {
		t.Errorf("Config.Rank: got %d, want 4", adapter.Config.Rank)
	}
	if adapter.Config.Alpha != 8.0 {
		t.Errorf("Config.Alpha: got %g, want 8.0", adapter.Config.Alpha)
	}
	wantScale := float32(8.0) / float32(4)
	if adapter.Scale != wantScale {
		t.Errorf("Scale: got %g, want %g", adapter.Scale, wantScale)
	}

	// Layer count.
	if len(adapter.Layers) != 1 {
		t.Fatalf("Layers: got %d layers, want 1", len(adapter.Layers))
	}

	la := adapter.Layers[0]

	// HasQ / HasV.
	if !la.HasQ() {
		t.Error("HasQ() should be true")
	}
	if la.HasV() {
		t.Error("HasV() should be false (no V tensors in this adapter)")
	}

	// QA shape.
	if la.QAShape != ([2]int64{rank, hidden}) {
		t.Errorf("QAShape: got %v, want [%d,%d]", la.QAShape, rank, hidden)
	}
	// QB shape.
	if la.QBShape != ([2]int64{hidden, rank}) {
		t.Errorf("QBShape: got %v, want [%d,%d]", la.QBShape, hidden, rank)
	}

	// QA data round-trip.
	if len(la.QA) != rank*hidden {
		t.Errorf("QA len: got %d, want %d", len(la.QA), rank*hidden)
	}
	for i, v := range la.QA {
		if want := float32(i) * 0.1; math.Abs(float64(v-want)) > 1e-6 {
			t.Errorf("QA[%d]: got %g, want %g", i, v, want)
		}
	}

	// QB data round-trip.
	if len(la.QB) != hidden*rank {
		t.Errorf("QB len: got %d, want %d", len(la.QB), hidden*rank)
	}
	for i, v := range la.QB {
		if want := float32(i) * 0.2; math.Abs(float64(v-want)) > 1e-6 {
			t.Errorf("QB[%d]: got %g, want %g", i, v, want)
		}
	}
}

func TestLoadAdapterMultiLayer(t *testing.T) {
	// Two layers (0 and 3) with both Q and V projections.
	mkQA := func(layer, rank, hidden int) []float32 {
		d := make([]float32, rank*hidden)
		for i := range d {
			d[i] = float32(layer*1000+i) * 0.01
		}
		return d
	}
	mkQB := func(layer, rank, hidden int) []float32 {
		d := make([]float32, hidden*rank)
		for i := range d {
			d[i] = float32(layer*1000+i) * 0.02
		}
		return d
	}

	const rank, hidden = 8, 16

	tensors := map[string]tensorSpec{
		"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {shape: [2]int64{rank, hidden}, data: mkQA(0, rank, hidden)},
		"base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {shape: [2]int64{hidden, rank}, data: mkQB(0, rank, hidden)},
		"base_model.model.layers.0.self_attn.v_proj.lora_A.weight": {shape: [2]int64{rank, hidden}, data: mkQA(10, rank, hidden)},
		"base_model.model.layers.0.self_attn.v_proj.lora_B.weight": {shape: [2]int64{hidden, rank}, data: mkQB(10, rank, hidden)},
		"base_model.model.layers.3.self_attn.q_proj.lora_A.weight": {shape: [2]int64{rank, hidden}, data: mkQA(3, rank, hidden)},
		"base_model.model.layers.3.self_attn.q_proj.lora_B.weight": {shape: [2]int64{hidden, rank}, data: mkQB(3, rank, hidden)},
	}

	dir := writeTempAdapter(t, tensors)

	adapter, err := LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	if len(adapter.Layers) != 4 {
		t.Fatalf("Layers: got %d, want 4 (indices 0..3)", len(adapter.Layers))
	}

	if !adapter.Layers[0].HasQ() {
		t.Error("layer 0: HasQ() should be true")
	}
	if !adapter.Layers[0].HasV() {
		t.Error("layer 0: HasV() should be true")
	}
	if !adapter.Layers[3].HasQ() {
		t.Error("layer 3: HasQ() should be true")
	}
	if adapter.Layers[3].HasV() {
		t.Error("layer 3: HasV() should be false")
	}
	// Layers 1 and 2 should be empty.
	for _, idx := range []int{1, 2} {
		if adapter.Layers[idx].HasQ() || adapter.Layers[idx].HasV() {
			t.Errorf("layer %d: expected empty, got HasQ=%v HasV=%v",
				idx, adapter.Layers[idx].HasQ(), adapter.Layers[idx].HasV())
		}
	}
}

func TestLoadAdapterMissingFile(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"r":4,"lora_alpha":8.0,"target_modules":["q_proj"],"base_model_name_or_path":"x"}`
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	// No safetensors file written.
	_, err := LoadAdapter(dir)
	if err == nil {
		t.Fatal("expected error when adapter_model.safetensors is missing, got nil")
	}
}

func TestLoadAdapterNoTensors(t *testing.T) {
	// Valid safetensors but no recognisable LoRA tensor names.
	tensors := map[string]tensorSpec{
		"some_unrelated_weight": {shape: [2]int64{4, 8}, data: make([]float32, 32)},
	}
	dir := writeTempAdapter(t, tensors)
	_, err := LoadAdapter(dir)
	if err == nil {
		t.Fatal("expected error when no LoRA tensors are present, got nil")
	}
}
