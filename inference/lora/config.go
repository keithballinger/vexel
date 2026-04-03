package lora

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type AdapterConfig struct {
	Rank          int      `json:"r"`
	Alpha         float32  `json:"lora_alpha"`
	TargetModules []string `json:"target_modules"`
	BaseModel     string   `json:"base_model_name_or_path"`
}

func (c AdapterConfig) Scale() float32 {
	if c.Rank == 0 {
		return 0
	}
	return c.Alpha / float32(c.Rank)
}

// HasTargetModule returns true when module (e.g. "k_proj") is listed in
// TargetModules. This controls which projections get LoRA weights.
func (c AdapterConfig) HasTargetModule(module string) bool {
	for _, m := range c.TargetModules {
		if m == module {
			return true
		}
	}
	return false
}

func LoadConfig(dir string) (AdapterConfig, error) {
	data, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		return AdapterConfig{}, fmt.Errorf("read adapter_config.json: %w", err)
	}
	var cfg AdapterConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return AdapterConfig{}, fmt.Errorf("parse adapter_config.json: %w", err)
	}
	if cfg.Rank <= 0 {
		return AdapterConfig{}, fmt.Errorf("invalid LoRA rank: %d", cfg.Rank)
	}
	return cfg, nil
}
