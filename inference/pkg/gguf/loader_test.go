package gguf

import (
	"testing"
)

func TestGetLayerTensorName(t *testing.T) {
	tests := []struct {
		hfName   string
		ggufName string
	}{
		// Global tensors
		{"model.embed_tokens.weight", "token_embd.weight"},
		{"lm_head.weight", "output.weight"},
		{"model.norm.weight", "output_norm.weight"},
		// Layer tensors
		{"model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"},
		{"model.layers.5.self_attn.k_proj.weight", "blk.5.attn_k.weight"},
		{"model.layers.21.self_attn.v_proj.weight", "blk.21.attn_v.weight"},
		{"model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"},
		{"model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"},
		{"model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"},
		{"model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"},
		{"model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"},
		{"model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight"},
		// Unknown patterns pass through
		{"unknown.tensor.name", "unknown.tensor.name"},
	}

	for _, tt := range tests {
		t.Run(tt.hfName, func(t *testing.T) {
			got := GetLayerTensorName(tt.hfName)
			if got != tt.ggufName {
				t.Errorf("GetLayerTensorName(%q) = %q, want %q", tt.hfName, got, tt.ggufName)
			}
		})
	}
}

func TestTensorNameMapping(t *testing.T) {
	// Verify the global tensor mapping
	if TensorNameMapping["model.embed_tokens.weight"] != "token_embd.weight" {
		t.Error("Expected embed_tokens -> token_embd mapping")
	}
	if TensorNameMapping["lm_head.weight"] != "output.weight" {
		t.Error("Expected lm_head -> output mapping")
	}
	if TensorNameMapping["model.norm.weight"] != "output_norm.weight" {
		t.Error("Expected model.norm -> output_norm mapping")
	}
}
