package memory_test

import (
	"testing"
	"vexel/inference/memory"
)

func TestArenaKind(t *testing.T) {
	tests := []struct {
		name string
		kind memory.ArenaKind
	}{
		{"Weights", memory.Weights},
		{"KV", memory.KV},
		{"Scratch", memory.Scratch},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.kind.String() == "" {
				t.Error("ArenaKind.String() returned empty string")
			}
		})
	}
}
