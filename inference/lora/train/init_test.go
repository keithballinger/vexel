package train

import (
	"math"
	"testing"

	"vexel/inference/lora"
)

func TestInitAdapter(t *testing.T) {
	const (
		rank       = 8
		alpha      = float32(16)
		numLayers  = 4
		hiddenSize = 512
		qDim       = 512
		vDim       = 512
	)

	cfg := lora.AdapterConfig{
		Rank:  rank,
		Alpha: alpha,
	}

	adapter := InitAdapter(cfg, numLayers, hiddenSize, qDim, vDim)

	// --- top-level fields ---

	if adapter.Scale != cfg.Scale() {
		t.Errorf("Scale: got %v, want %v", adapter.Scale, cfg.Scale())
	}
	if len(adapter.Layers) != numLayers {
		t.Fatalf("Layers: got %d, want %d", len(adapter.Layers), numLayers)
	}

	bound := float32(1.0 / math.Sqrt(float64(hiddenSize)))

	for i, la := range adapter.Layers {
		// --- shapes ---

		if la.QAShape != [2]int64{rank, hiddenSize} {
			t.Errorf("layer %d QAShape: got %v, want [%d %d]", i, la.QAShape, rank, hiddenSize)
		}
		if la.QBShape != [2]int64{qDim, rank} {
			t.Errorf("layer %d QBShape: got %v, want [%d %d]", i, la.QBShape, qDim, rank)
		}
		if la.VAShape != [2]int64{rank, hiddenSize} {
			t.Errorf("layer %d VAShape: got %v, want [%d %d]", i, la.VAShape, rank, hiddenSize)
		}
		if la.VBShape != [2]int64{vDim, rank} {
			t.Errorf("layer %d VBShape: got %v, want [%d %d]", i, la.VBShape, vDim, rank)
		}

		// --- slice lengths ---

		wantALen := rank * hiddenSize
		wantQBLen := qDim * rank
		wantVBLen := vDim * rank

		if len(la.QA) != wantALen {
			t.Errorf("layer %d QA length: got %d, want %d", i, len(la.QA), wantALen)
		}
		if len(la.QB) != wantQBLen {
			t.Errorf("layer %d QB length: got %d, want %d", i, len(la.QB), wantQBLen)
		}
		if len(la.VA) != wantALen {
			t.Errorf("layer %d VA length: got %d, want %d", i, len(la.VA), wantALen)
		}
		if len(la.VB) != wantVBLen {
			t.Errorf("layer %d VB length: got %d, want %d", i, len(la.VB), wantVBLen)
		}

		// --- B matrices must be all zeros ---

		for j, v := range la.QB {
			if v != 0 {
				t.Errorf("layer %d QB[%d] = %v; want 0 (zero init)", i, j, v)
				break
			}
		}
		for j, v := range la.VB {
			if v != 0 {
				t.Errorf("layer %d VB[%d] = %v; want 0 (zero init)", i, j, v)
				break
			}
		}

		// --- A matrices must be non-zero and within Kaiming bounds ---

		allZeroQA := true
		for _, v := range la.QA {
			if v != 0 {
				allZeroQA = false
			}
			if v < -bound || v > bound {
				t.Errorf("layer %d QA value %v out of Kaiming bound [%v, %v]", i, v, -bound, bound)
				break
			}
		}
		if allZeroQA {
			t.Errorf("layer %d QA is all zeros; expected Kaiming-uniform noise", i)
		}

		allZeroVA := true
		for _, v := range la.VA {
			if v != 0 {
				allZeroVA = false
			}
			if v < -bound || v > bound {
				t.Errorf("layer %d VA value %v out of Kaiming bound [%v, %v]", i, v, -bound, bound)
				break
			}
		}
		if allZeroVA {
			t.Errorf("layer %d VA is all zeros; expected Kaiming-uniform noise", i)
		}

		// --- HasQ / HasV helpers ---

		if !la.HasQ() {
			t.Errorf("layer %d HasQ() returned false", i)
		}
		if !la.HasV() {
			t.Errorf("layer %d HasV() returned false", i)
		}
	}
}

// TestInitAdapter_RankScale verifies that Scale = alpha/rank is computed correctly.
func TestInitAdapter_RankScale(t *testing.T) {
	cfg := lora.AdapterConfig{Rank: 4, Alpha: 8}
	adapter := InitAdapter(cfg, 1, 64, 64, 64)

	want := float32(2.0) // 8/4
	if adapter.Scale != want {
		t.Errorf("Scale: got %v, want %v", adapter.Scale, want)
	}
}

// TestKaimingUniform checks statistical properties of the helper directly.
func TestKaimingUniform(t *testing.T) {
	const rows, cols = 32, 128
	data := kaimingUniform(rows, cols)

	if len(data) != rows*cols {
		t.Fatalf("length: got %d, want %d", len(data), rows*cols)
	}

	bound := float32(1.0 / math.Sqrt(float64(cols)))
	hasNonZero := false
	for _, v := range data {
		if v < -bound || v > bound {
			t.Errorf("value %v outside [-%.6f, %.6f]", v, bound, bound)
			break
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	if !hasNonZero {
		t.Error("all values are zero; expected random noise")
	}
}
