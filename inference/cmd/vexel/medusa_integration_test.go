package main

import (
	"testing"
)

func TestParseMedusaFlag(t *testing.T) {
	args := []string{"vexel", "--model", "model.gguf", "--medusa", "generate"}
	cmd, globals, err := parseArgs(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cmd != "generate" {
		t.Errorf("expected subcommand %q, got %q", "generate", cmd)
	}
	if !globals.Medusa {
		t.Error("expected Medusa to be true")
	}
	if globals.MedusaHeadsPath != "" {
		t.Errorf("expected empty MedusaHeadsPath, got %q", globals.MedusaHeadsPath)
	}
}

func TestParseMedusaHeadsFlag(t *testing.T) {
	args := []string{"vexel", "--model", "model.gguf", "--medusa-heads", "/path/to/heads.bin", "serve"}
	cmd, globals, err := parseArgs(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cmd != "serve" {
		t.Errorf("expected subcommand %q, got %q", "serve", cmd)
	}
	if globals.MedusaHeadsPath != "/path/to/heads.bin" {
		t.Errorf("expected MedusaHeadsPath %q, got %q", "/path/to/heads.bin", globals.MedusaHeadsPath)
	}
	if !globals.Medusa {
		t.Error("expected Medusa to be true (implied by --medusa-heads)")
	}
}

func TestParseMedusaHeadsMissingValue(t *testing.T) {
	args := []string{"vexel", "--medusa-heads"}
	_, _, err := parseArgs(args)
	if err == nil {
		t.Fatal("expected error for --medusa-heads without value")
	}
}

func TestParseMedusaWithOtherFlags(t *testing.T) {
	args := []string{"vexel", "--verbose", "--model", "m.gguf", "--medusa", "--medusa-heads", "heads.bin", "chat"}
	cmd, globals, err := parseArgs(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cmd != "chat" {
		t.Errorf("expected subcommand %q, got %q", "chat", cmd)
	}
	if !globals.Verbose {
		t.Error("expected Verbose to be true")
	}
	if globals.Model != "m.gguf" {
		t.Errorf("expected Model %q, got %q", "m.gguf", globals.Model)
	}
	if !globals.Medusa {
		t.Error("expected Medusa to be true")
	}
	if globals.MedusaHeadsPath != "heads.bin" {
		t.Errorf("expected MedusaHeadsPath %q, got %q", "heads.bin", globals.MedusaHeadsPath)
	}
}
