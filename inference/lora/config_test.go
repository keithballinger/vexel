package lora

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	j := `{"r": 16, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": "test"}`
	os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(j), 0644)

	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Rank != 16 {
		t.Errorf("rank=%d, want 16", cfg.Rank)
	}
	if cfg.Scale() != 1.0 {
		t.Errorf("scale=%f, want 1.0", cfg.Scale())
	}
	if len(cfg.TargetModules) != 2 {
		t.Errorf("target_modules len=%d, want 2", len(cfg.TargetModules))
	}
}

func TestLoadConfigMissing(t *testing.T) {
	_, err := LoadConfig(t.TempDir())
	if err == nil {
		t.Error("expected error for missing config")
	}
}
