package cpu_test

import (
	"testing"
	"vexel/inference/backend/cpu"
)

func TestEmbedding(t *testing.T) {
	// Table: 3 tokens, dim 2
	// ID 0: [1, 1]
	// ID 1: [2, 2]
	// ID 2: [3, 3]
	table := []float32{
		1, 1,
		2, 2,
		3, 3,
	}

	// Input IDs: [0, 2]
	ids := []int{0, 2}

	// Output: [2, 2] -> [2 vectors * dim 2] = 4 floats
	out := make([]float32, 4)

	b := cpu.NewBackend()
	ops, ok := b.(interface {
		Embedding(ids []int, table []float32, out []float32, dim int)
	})
	if !ok {
		t.Fatal("Backend does not expose Embedding")
	}

	ops.Embedding(ids, table, out, 2)

	expected := []float32{
		1, 1, // ID 0
		3, 3, // ID 2
	}

	for i, v := range out {
		if v != expected[i] {
			t.Errorf("Index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}
