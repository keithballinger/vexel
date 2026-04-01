package lora

import (
	"math"
	"testing"
)

// TestSaveAdapterRoundTrip creates an Adapter with known weights, saves it to
// a temp directory via SaveAdapter, reloads it with LoadAdapter, and verifies
// that all config fields and weight values are preserved exactly.
func TestSaveAdapterRoundTrip(t *testing.T) {
	const (
		rank   = 4
		hidden = 8
	)

	// Build known weight slices.
	qaData := make([]float32, rank*hidden)
	qbData := make([]float32, hidden*rank)
	vaData := make([]float32, rank*hidden)
	vbData := make([]float32, hidden*rank)
	for i := range qaData {
		qaData[i] = float32(i) * 0.1
	}
	for i := range qbData {
		qbData[i] = float32(i) * 0.2
	}
	for i := range vaData {
		vaData[i] = float32(i) * 0.3
	}
	for i := range vbData {
		vbData[i] = float32(i) * 0.4
	}

	// Layer 0: both Q and V.
	// Layer 1: Q only.
	// Layer 2: empty (should survive the round-trip as empty).
	// Layer 3: V only.
	original := &Adapter{
		Config: AdapterConfig{
			Rank:          rank,
			Alpha:         8.0,
			TargetModules: []string{"q_proj", "v_proj"},
			BaseModel:     "meta-llama/Llama-2-7b-hf",
		},
		Scale: 8.0 / float32(rank),
		Layers: []LayerAdapter{
			// layer 0: Q + V
			{
				QA: qaData, QB: qbData,
				QAShape: [2]int64{rank, hidden}, QBShape: [2]int64{hidden, rank},
				VA: vaData, VB: vbData,
				VAShape: [2]int64{rank, hidden}, VBShape: [2]int64{hidden, rank},
			},
			// layer 1: Q only
			{
				QA: qaData, QB: qbData,
				QAShape: [2]int64{rank, hidden}, QBShape: [2]int64{hidden, rank},
			},
			// layer 2: empty
			{},
			// layer 3: V only
			{
				VA: vaData, VB: vbData,
				VAShape: [2]int64{rank, hidden}, VBShape: [2]int64{hidden, rank},
			},
		},
	}

	dir := t.TempDir()
	if err := SaveAdapter(original, dir); err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	loaded, err := LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	// --- Config ---
	if loaded.Config.Rank != rank {
		t.Errorf("Config.Rank: got %d, want %d", loaded.Config.Rank, rank)
	}
	if loaded.Config.Alpha != 8.0 {
		t.Errorf("Config.Alpha: got %g, want 8.0", loaded.Config.Alpha)
	}
	if loaded.Config.BaseModel != "meta-llama/Llama-2-7b-hf" {
		t.Errorf("Config.BaseModel: got %q, want %q", loaded.Config.BaseModel, "meta-llama/Llama-2-7b-hf")
	}
	wantScale := float32(8.0) / float32(rank)
	if loaded.Scale != wantScale {
		t.Errorf("Scale: got %g, want %g", loaded.Scale, wantScale)
	}

	// --- Layer count ---
	// The writer only emits layers that HasQ or HasV, so the loaded adapter
	// will have indices up to the highest written layer (3).  Empty layers in
	// between will be zero-value LayerAdapters.
	if len(loaded.Layers) != 4 {
		t.Fatalf("len(Layers): got %d, want 4", len(loaded.Layers))
	}

	// --- Layer 0: Q + V ---
	l0 := loaded.Layers[0]
	if !l0.HasQ() {
		t.Error("layer 0: HasQ() should be true")
	}
	if !l0.HasV() {
		t.Error("layer 0: HasV() should be true")
	}
	checkShape(t, "layer0 QAShape", l0.QAShape, [2]int64{rank, hidden})
	checkShape(t, "layer0 QBShape", l0.QBShape, [2]int64{hidden, rank})
	checkShape(t, "layer0 VAShape", l0.VAShape, [2]int64{rank, hidden})
	checkShape(t, "layer0 VBShape", l0.VBShape, [2]int64{hidden, rank})
	checkWeights(t, "layer0 QA", l0.QA, qaData)
	checkWeights(t, "layer0 QB", l0.QB, qbData)
	checkWeights(t, "layer0 VA", l0.VA, vaData)
	checkWeights(t, "layer0 VB", l0.VB, vbData)

	// --- Layer 1: Q only ---
	l1 := loaded.Layers[1]
	if !l1.HasQ() {
		t.Error("layer 1: HasQ() should be true")
	}
	if l1.HasV() {
		t.Error("layer 1: HasV() should be false")
	}
	checkWeights(t, "layer1 QA", l1.QA, qaData)
	checkWeights(t, "layer1 QB", l1.QB, qbData)

	// --- Layer 2: empty ---
	l2 := loaded.Layers[2]
	if l2.HasQ() || l2.HasV() {
		t.Errorf("layer 2: expected empty, got HasQ=%v HasV=%v", l2.HasQ(), l2.HasV())
	}

	// --- Layer 3: V only ---
	l3 := loaded.Layers[3]
	if l3.HasQ() {
		t.Error("layer 3: HasQ() should be false")
	}
	if !l3.HasV() {
		t.Error("layer 3: HasV() should be true")
	}
	checkWeights(t, "layer3 VA", l3.VA, vaData)
	checkWeights(t, "layer3 VB", l3.VB, vbData)
}

// TestSaveAdapterSingleLayer verifies the minimal single-layer case so that
// any regression in the tensor naming or binary layout is caught early.
func TestSaveAdapterSingleLayer(t *testing.T) {
	const (
		rank   = 2
		hidden = 4
	)

	qaData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} // rank*hidden = 8
	qbData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0} // hidden*rank = 8

	adapter := &Adapter{
		Config: AdapterConfig{
			Rank:          rank,
			Alpha:         4.0,
			TargetModules: []string{"q_proj"},
			BaseModel:     "test/model",
		},
		Scale: 4.0 / float32(rank),
		Layers: []LayerAdapter{
			{
				QA: qaData, QB: qbData,
				QAShape: [2]int64{rank, hidden}, QBShape: [2]int64{hidden, rank},
			},
		},
	}

	dir := t.TempDir()
	if err := SaveAdapter(adapter, dir); err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	loaded, err := LoadAdapter(dir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	if len(loaded.Layers) != 1 {
		t.Fatalf("len(Layers): got %d, want 1", len(loaded.Layers))
	}
	checkWeights(t, "QA", loaded.Layers[0].QA, qaData)
	checkWeights(t, "QB", loaded.Layers[0].QB, qbData)
}

// TestSaveAdapterCreatesDir ensures SaveAdapter creates the target directory
// when it does not already exist.
func TestSaveAdapterCreatesDir(t *testing.T) {
	base := t.TempDir()
	dir := base + "/nested/adapter"

	adapter := &Adapter{
		Config: AdapterConfig{
			Rank:          2,
			Alpha:         2.0,
			TargetModules: []string{"q_proj"},
			BaseModel:     "test/model",
		},
		Scale: 1.0,
		Layers: []LayerAdapter{
			{
				QA: []float32{1, 2, 3, 4}, QB: []float32{0, 0, 0, 0},
				QAShape: [2]int64{2, 2}, QBShape: [2]int64{2, 2},
			},
		},
	}

	if err := SaveAdapter(adapter, dir); err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	if _, err := LoadAdapter(dir); err != nil {
		t.Fatalf("LoadAdapter after auto-mkdir: %v", err)
	}
}

// --- helpers -----------------------------------------------------------------

func checkShape(t *testing.T, label string, got, want [2]int64) {
	t.Helper()
	if got != want {
		t.Errorf("%s: got %v, want %v", label, got, want)
	}
}

func checkWeights(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: len got %d, want %d", label, len(got), len(want))
		return
	}
	for i, v := range got {
		if math.Abs(float64(v-want[i])) > 1e-6 {
			t.Errorf("%s[%d]: got %g, want %g", label, i, v, want[i])
		}
	}
}
