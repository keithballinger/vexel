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

		// Residual add
		p.RecordDispatch("Add")

		// === MLP Phase ===
		// Fused path: FusedRMSNorm+MatMul for gate (W1) and up (W3)
		p.RecordDispatch("FusedRMSNorm+MatMul") // W1 (gate)
		p.RecordDispatch("FusedRMSNorm+MatMul") // W3 (up)

		// SiLU activation * up
		p.RecordDispatch("SiLUMul")

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
	// Per layer: 3 fused QKV + 1 RoPE + 2 scatter + 1 SDPA + 1 Wo + 1 Add1 +
	//            2 fused gate/up + 1 SiLUMul + 1 W2 + 1 Add2 = 14
	// Pre/post: 1 embedding + 1 final norm + 1 lm_head = 3
	// Total: 14 * 32 + 3 = 451
	expectedPerLayer := 14
	expectedTotal := expectedPerLayer*numLayers + 3
	if profile.TotalDispatches != expectedTotal {
		t.Errorf("total dispatches: got %d, want %d", profile.TotalDispatches, expectedTotal)
	}

	// === Fusion Candidate Analysis ===
	fmt.Println("\n[FUSION ANALYSIS] Optimization Targets")
	fmt.Println("=======================================")
	fmt.Println()
	fmt.Println("Current: 14 dispatches per layer × 32 layers + 3 = 451 total")
	fmt.Println()
	fmt.Println("Target 1: Fuse RoPE + ScatterKV (saves 2 per layer)")
	fmt.Printf("  Current: RoPE(Q,K) → ScatterKV(K) → ScatterKV(V) = 3 dispatches\n")
	fmt.Printf("  Fused:   RoPE+ScatterKV = 1 dispatch\n")
	fmt.Printf("  Savings: 2 × 32 = 64 dispatches (14%%)\n")
	fmt.Println()
	fmt.Println("Target 2: Fuse SiLUMul + W2 matmul (saves 1 per layer)")
	fmt.Printf("  Current: SiLUMul → MatMulQ4_0(W2) = 2 dispatches\n")
	fmt.Printf("  Fused:   FusedSiLUMul+MatMul = 1 dispatch\n")
	fmt.Printf("  Savings: 1 × 32 = 32 dispatches (7%%)\n")
	fmt.Println()
	fmt.Println("Target 3: Fuse Wo + Add1 (saves 1 per layer)")
	fmt.Printf("  Current: MatMulQ4_0(Wo) → Add(residual) = 2 dispatches\n")
	fmt.Printf("  Fused:   MatMul+Add = 1 dispatch\n")
	fmt.Printf("  Savings: 1 × 32 = 32 dispatches (7%%)\n")
	fmt.Println()
	fmt.Println("Target 4: Fuse W2 + Add2 (saves 1 per layer)")
	fmt.Printf("  Current: MatMulQ4_0(W2) → Add(residual) = 2 dispatches\n")
	fmt.Printf("  Fused:   MatMul+Add = 1 dispatch\n")
	fmt.Printf("  Savings: 1 × 32 = 32 dispatches (7%%)\n")
	fmt.Println()

	// Combined savings
	fmt.Println("Combined Reduction:")
	fusedPerLayer := 14 - 2 - 1 - 1 - 1 // = 9
	fusedTotal := fusedPerLayer*numLayers + 3
	reduction := float64(expectedTotal-fusedTotal) / float64(expectedTotal) * 100
	fmt.Printf("  From: %d per layer (%d total)\n", expectedPerLayer, expectedTotal)
	fmt.Printf("  To:   %d per layer (%d total)\n", fusedPerLayer, fusedTotal)
	fmt.Printf("  Reduction: %.0f%% fewer dispatches\n", reduction)
	fmt.Println("=======================================")

	// Verify fusion target math
	if fusedPerLayer != 9 {
		t.Errorf("fused per layer: got %d, want 9", fusedPerLayer)
	}
}

// TestPrefillKernelSequence documents the non-fused prefill path (seqLen > 1).
func TestPrefillKernelSequence(t *testing.T) {
	p := NewDispatchProfiler()
	p.Enable()
	p.BeginPass()

	p.RecordDispatch("Embedding")

	numLayers := 32
	for layer := 0; layer < numLayers; layer++ {
		_ = layer
		// Standard path (no fusion for prefill)
		p.RecordDispatch("RMSNorm")       // Attention norm
		p.RecordDispatch("MatMulQ4_0")    // Q projection
		p.RecordDispatch("MatMulQ4_0")    // K projection
		p.RecordDispatch("MatMulQ4_0")    // V projection
		p.RecordDispatch("RoPE")          // Rotary encoding
		p.RecordDispatch("ScatterKV")     // K cache scatter
		p.RecordDispatch("ScatterKV")     // V cache scatter
		p.RecordDispatch("FlashAttention2") // FA2 prefill
		p.RecordDispatch("MatMulQ4_0")    // Wo output projection
		p.RecordDispatch("Add")           // Residual add 1
		p.RecordDispatch("RMSNorm")       // FFN norm
		p.RecordDispatch("MatMulQ4_0")    // W1 (gate)
		p.RecordDispatch("MatMulQ4_0")    // W3 (up)
		p.RecordDispatch("SiLUMul")       // SiLU activation
		p.RecordDispatch("MatMulQ4_0")    // W2 (down)
		p.RecordDispatch("Add")           // Residual add 2
	}

	p.RecordDispatch("RMSNorm")     // Final norm
	p.RecordDispatch("MatMulQ4_0") // LM head

	profile := p.EndPass()

	var buf bytes.Buffer
	profile.PrintReport(&buf)
	fmt.Print(buf.String())

	// Per layer: 2 norms + 7 matmuls + 1 RoPE + 2 scatter + 1 FA2 + 1 SiLUMul + 2 adds = 16
	expectedPerLayer := 16
	expectedTotal := expectedPerLayer*numLayers + 3
	if profile.TotalDispatches != expectedTotal {
		t.Errorf("prefill total dispatches: got %d, want %d", profile.TotalDispatches, expectedTotal)
	}

	fmt.Printf("\nPrefill path: %d dispatches per layer × %d layers + 3 = %d total\n",
		expectedPerLayer, numLayers, expectedTotal)
}
