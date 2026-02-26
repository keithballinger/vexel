//go:build metal && darwin && cgo

package metal

import (
	"bytes"
	"fmt"
	"testing"
)

// TestFusionCandidateAnalysis documents the forward pass kernel sequence for
// LLaMA-style models and identifies fusion opportunities.
//
// This is the reference analysis for Track 2: Kernel Fusion & GPU Optimization.
// Updated after Phase 3 to reflect FusedMLP + AddRMSNorm wiring.
func TestFusionCandidateAnalysis(t *testing.T) {
	// Simulate a single decode step for a 32-layer LLaMA model.
	// Each layer dispatches kernels in a fixed sequence.
	p := NewDispatchProfiler()
	p.Enable()
	p.BeginPass()

	// Pre-layer: Embedding lookup
	p.RecordDispatch("Embedding")

	numLayers := 32

	for layer := 0; layer < numLayers; layer++ {
		_ = layer
		// === Attention Phase ===
		// Fused path (Q4_0 weights, decode mode, seqLen=1):
		// 3x FusedRMSNorm+MatMul for Q, K, V projections
		p.RecordDispatch("FusedRMSNorm+MatMul") // Q projection
		p.RecordDispatch("FusedRMSNorm+MatMul") // K projection
		p.RecordDispatch("FusedRMSNorm+MatMul") // V projection

		// RoPE: rotary position encoding on Q and K
		p.RecordDispatch("RoPE")

		// KV Cache: scatter K and V into head-major cache
		p.RecordDispatch("ScatterKV") // K scatter
		p.RecordDispatch("ScatterKV") // V scatter

		// SDPA: scaled dot-product attention
		p.RecordDispatch("SDPA")

		// Output projection: Wo matmul
		p.RecordDispatch("MatMulQ4_0")

		// === MLP Phase ===
		// Fused Add1+RMSNorm: x += attn_output, normOut = RMSNorm(x)
		// (replaces separate Add1 + RMSNorm2 = 2 dispatches → 1)
		p.RecordDispatch("AddRMSNorm")

		// FusedMLP: SiLU(x @ W1) * (x @ W3) in single kernel
		// (replaces FusedRMSNorm+MatMul(W1) + FusedRMSNorm+MatMul(W3) + SiLUMul = 3 dispatches → 1)
		p.RecordDispatch("FusedMLP")

		// Down projection: W2 matmul
		p.RecordDispatch("MatMulQ4_0")

		// Residual add
		p.RecordDispatch("Add")
	}

	// Post-layers: final norm + lm_head matmul
	p.RecordDispatch("RMSNorm")     // Final norm
	p.RecordDispatch("MatMulQ4_0") // LM head (or Q6_K)

	profile := p.EndPass()

	// === Analysis Output ===
	var buf bytes.Buffer
	profile.PrintReport(&buf)
	fmt.Print(buf.String())

	// Verify expected total dispatches for fused decode path
	// Per layer: 3 fused QKV + 1 RoPE + 2 scatter + 1 SDPA + 1 Wo +
	//            1 AddRMSNorm + 1 FusedMLP + 1 W2 + 1 Add2 = 12
	// Pre/post: 1 embedding + 1 final norm + 1 lm_head = 3
	// Total: 12 * 32 + 3 = 387
	expectedPerLayer := 12
	expectedTotal := expectedPerLayer*numLayers + 3
	if profile.TotalDispatches != expectedTotal {
		t.Errorf("total dispatches: got %d, want %d", profile.TotalDispatches, expectedTotal)
	}

	// === Fusion Candidate Analysis ===
	fmt.Println("\n[FUSION ANALYSIS] Remaining Optimization Targets")
	fmt.Println("=================================================")
	fmt.Println()
	fmt.Printf("Current: %d dispatches per layer × 32 layers + 3 = %d total\n", expectedPerLayer, expectedTotal)
	fmt.Println()
	fmt.Println("COMPLETED fusions:")
	fmt.Println("  ✓ FusedRMSNorm+MatMul for QKV projections (already existed)")
	fmt.Println("  ✓ AddRMSNorm: fused Add1+RMSNorm2 (saved 1 dispatch/layer = 32 total)")
	fmt.Println("  ✓ FusedMLP: fused SiLU(x@W1)*(x@W3) (saved 2 dispatches/layer = 64 total)")
	fmt.Println()
	fmt.Println("Target 1: Fuse RoPE + ScatterKV (saves 2 per layer)")
	fmt.Printf("  Current: RoPE(Q,K) → ScatterKV(K) → ScatterKV(V) = 3 dispatches\n")
	fmt.Printf("  Fused:   RoPE+ScatterKV = 1 dispatch\n")
	fmt.Printf("  Savings: 2 × 32 = 64 dispatches (17%%)\n")
	fmt.Println()
	fmt.Println("Target 2: Fuse W2 + Add2 (saves 1 per layer)")
	fmt.Printf("  Current: MatMulQ4_0(W2) → Add(residual) = 2 dispatches\n")
	fmt.Printf("  Fused:   MatMul+Add = 1 dispatch\n")
	fmt.Printf("  Savings: 1 × 32 = 32 dispatches (8%%)\n")
	fmt.Println()
	fmt.Println("Target 3: Fuse Wo + AddRMSNorm (saves 1 per layer)")
	fmt.Printf("  Current: MatMulQ4_0(Wo) → AddRMSNorm = 2 dispatches\n")
	fmt.Printf("  Fused:   MatMul+Add+RMSNorm = 1 dispatch\n")
	fmt.Printf("  Savings: 1 × 32 = 32 dispatches (8%%)\n")
	fmt.Println()

	// Combined future savings from remaining targets
	fmt.Println("Combined Future Reduction:")
	fusedPerLayer := expectedPerLayer - 2 - 1 - 1 // = 8
	fusedTotal := fusedPerLayer*numLayers + 3
	reduction := float64(expectedTotal-fusedTotal) / float64(expectedTotal) * 100
	fmt.Printf("  From: %d per layer (%d total)\n", expectedPerLayer, expectedTotal)
	fmt.Printf("  To:   %d per layer (%d total)\n", fusedPerLayer, fusedTotal)
	fmt.Printf("  Reduction: %.0f%% fewer dispatches\n", reduction)
	fmt.Println("=================================================")

	// Verify fusion target math
	if fusedPerLayer != 8 {
		t.Errorf("fused per layer: got %d, want 8", fusedPerLayer)
	}
}

// TestPrefillKernelSequence documents the prefill path (seqLen > 1).
// Prefill uses standard matmuls (no fused QKV), but still benefits from AddRMSNorm.
func TestPrefillKernelSequence(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()
	p.BeginPass()

	p.RecordDispatch("Embedding")

	numLayers := 32
	for layer := 0; layer < numLayers; layer++ {
		_ = layer
		// Standard path (no FusedRMSNorm+MatMul or FusedMLP for prefill)
		p.RecordDispatch("RMSNorm")         // Attention norm
		p.RecordDispatch("MatMulQ4_0")      // Q projection
		p.RecordDispatch("MatMulQ4_0")      // K projection
		p.RecordDispatch("MatMulQ4_0")      // V projection
		p.RecordDispatch("RoPE")            // Rotary encoding
		p.RecordDispatch("ScatterKV")       // K cache scatter
		p.RecordDispatch("ScatterKV")       // V cache scatter
		p.RecordDispatch("FlashAttention2") // FA2 prefill
		p.RecordDispatch("MatMulQ4_0")      // Wo output projection
		// AddRMSNorm: fused Add1+RMSNorm2 (applies to prefill too)
		p.RecordDispatch("AddRMSNorm")      // x += attn_output, normOut = RMSNorm(x)
		p.RecordDispatch("MatMulQ4_0")      // W1 (gate)
		p.RecordDispatch("MatMulQ4_0")      // W3 (up)
		p.RecordDispatch("SiLUMul")         // SiLU activation
		p.RecordDispatch("MatMulQ4_0")      // W2 (down)
		p.RecordDispatch("Add")             // Residual add 2
	}

	p.RecordDispatch("RMSNorm")     // Final norm
	p.RecordDispatch("MatMulQ4_0") // LM head

	profile := p.EndPass()

	var buf bytes.Buffer
	profile.PrintReport(&buf)
	fmt.Print(buf.String())

	// Per layer: 1 norm + 7 matmuls + 1 RoPE + 2 scatter + 1 FA2 +
	//            1 AddRMSNorm + 1 SiLUMul + 1 add = 15
	expectedPerLayer := 15
	expectedTotal := expectedPerLayer*numLayers + 3
	if profile.TotalDispatches != expectedTotal {
		t.Errorf("prefill total dispatches: got %d, want %d", profile.TotalDispatches, expectedTotal)
	}

	fmt.Printf("\nPrefill path: %d dispatches per layer × %d layers + 3 = %d total\n",
		expectedPerLayer, numLayers, expectedTotal)
}
