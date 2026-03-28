//go:build metal && darwin && cgo

package main

import (
	"os"
	"testing"

	"vexel/inference/scheduler"
)

func TestMedusaSchedulerCreation(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		t.Skip("VEXEL_TEST_MODEL not set; skipping E2E test")
	}

	rt, tok, gpuBackend, err := initModel(modelPath, 128, 0, false, false)
	if err != nil {
		t.Fatalf("initModel failed: %v", err)
	}
	defer gpuBackend.Close()

	cfg := scheduler.Config{
		MaxBatchSize: 1,
		MaxSequences: 1,
		MaxTokens:    32,
	}
	medusaCfg := scheduler.DefaultMedusaConfig()

	ms, err := scheduler.NewMedusaScheduler(rt, tok, cfg, medusaCfg)
	if err != nil {
		t.Fatalf("NewMedusaScheduler failed: %v", err)
	}

	if ms.BaseScheduler() == nil {
		t.Fatal("BaseScheduler() returned nil")
	}
}
