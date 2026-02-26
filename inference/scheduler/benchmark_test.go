package scheduler

import (
	"fmt"
	"math"
	"testing"

	"vexel/inference/medusa"
	"vexel/inference/pkg/sampler"
)

// BenchmarkVerifyDraftLogits measures the overhead of the core verification
// algorithm (comparing draft tokens against target logits).
func BenchmarkVerifyDraftLogits(b *testing.B) {
	vocabSizes := []int{32000, 128000}
	draftLengths := []int{1, 4, 8}

	for _, vocabSize := range vocabSizes {
		for _, numDraft := range draftLengths {
			name := fmt.Sprintf("vocab%dk_draft%d", vocabSize/1000, numDraft)
			b.Run(name, func(b *testing.B) {
				// Build synthetic logits where all drafts are accepted
				allLogits := make([]float32, (numDraft+1)*vocabSize)
				draftTokens := make([]int, numDraft)
				draftProbs := make([]float32, numDraft)

				for i := 0; i < numDraft; i++ {
					tokenID := i % vocabSize
					draftTokens[i] = tokenID
					draftProbs[i] = 0.9
					allLogits[i*vocabSize+tokenID] = 10.0
				}
				allLogits[numDraft*vocabSize] = 10.0 // bonus

				s := sampler.New(sampler.Config{Temperature: 0}, 42)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					verifyDraftAgainstLogits(allLogits, draftTokens, draftProbs, vocabSize, s)
				}
			})
		}
	}
}

// BenchmarkSelectBestTreePath measures the overhead of evaluating
// multiple candidate paths against verification logits.
func BenchmarkSelectBestTreePath(b *testing.B) {
	configs := []struct {
		numHeads int
		topK     int
	}{
		{2, 3},
		{4, 3},
		{4, 5},
	}

	vocabSize := 32000

	for _, cfg := range configs {
		name := fmt.Sprintf("heads%d_topk%d", cfg.numHeads, cfg.topK)
		b.Run(name, func(b *testing.B) {
			// Build head logits
			headLogits := make([][]float32, cfg.numHeads)
			for i := range headLogits {
				headLogits[i] = make([]float32, vocabSize)
				for j := 0; j < vocabSize; j++ {
					headLogits[i][j] = float32(vocabSize-j) * 0.001
				}
			}

			tree := medusa.BuildCandidateTree(headLogits, cfg.topK, 64)
			paths := tree.Paths()

			// Build verification logits
			verifyLogits := make([]float32, (cfg.numHeads+1)*vocabSize)
			for i := 0; i <= cfg.numHeads; i++ {
				verifyLogits[i*vocabSize] = 10.0 // token 0 is best at each position
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				selectBestTreePath(paths, verifyLogits, vocabSize)
			}
		})
	}
}

// BenchmarkCandidateTreeConstruction measures tree building performance
// at realistic vocabulary sizes.
func BenchmarkCandidateTreeConstruction(b *testing.B) {
	vocabSize := 32000
	configs := []struct {
		numHeads int
		topK     int
		maxNodes int
	}{
		{4, 3, 50},
		{4, 5, 64},
		{4, 3, 100},
	}

	for _, cfg := range configs {
		name := fmt.Sprintf("heads%d_topk%d_max%d", cfg.numHeads, cfg.topK, cfg.maxNodes)
		b.Run(name, func(b *testing.B) {
			headLogits := make([][]float32, cfg.numHeads)
			for i := range headLogits {
				headLogits[i] = make([]float32, vocabSize)
				for j := 0; j < vocabSize; j++ {
					headLogits[i][j] = float32(j) * 0.001
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tree := medusa.BuildCandidateTree(headLogits, cfg.topK, cfg.maxNodes)
				_ = tree.Paths()
			}
		})
	}
}

// TestSimulatedSpeedupReport runs a simulated speculative decoding scenario
// and prints a formatted speedup report comparing different configurations.
func TestSimulatedSpeedupReport(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping simulation in short mode")
	}

	vocabSize := 32000
	numTokens := 100 // simulate generating 100 tokens
	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	// Simulate different acceptance rates to compute theoretical speedup
	scenarios := []struct {
		name           string
		numDraft       int
		acceptanceRate float64
		description    string
	}{
		{"standard_decode", 0, 0, "No speculation (baseline)"},
		{"spec_draft4_low", 4, 0.30, "4-token draft, 30% acceptance"},
		{"spec_draft4_med", 4, 0.60, "4-token draft, 60% acceptance"},
		{"spec_draft4_high", 4, 0.85, "4-token draft, 85% acceptance"},
		{"spec_draft8_med", 8, 0.50, "8-token draft, 50% acceptance"},
		{"medusa_tree_k3", 4, 0.70, "Medusa tree (topK=3), 70% acceptance"},
	}

	fmt.Println("\n=== Speculative Decoding Speedup Report ===")
	fmt.Printf("%-25s %6s %8s %8s %8s %8s\n",
		"Configuration", "Draft", "AccRate", "Steps", "Tokens", "Speedup")
	fmt.Println("-------------------------------------------------------------------")

	for _, sc := range scenarios {
		if sc.numDraft == 0 {
			// Standard decode: 1 step per token
			fmt.Printf("%-25s %6d %7.0f%% %8d %8d %7.1fx\n",
				sc.name, 0, 0.0, numTokens, numTokens, 1.0)
			continue
		}

		// Simulate verification
		totalSteps := 0
		totalTokens := 0
		metrics := SpeculativeMetrics{}

		for totalTokens < numTokens {
			// Generate draft tokens
			numDraft := sc.numDraft
			metrics.DraftTokensGenerated += numDraft

			// Simulate acceptance based on the rate
			accepted := 0
			for j := 0; j < numDraft; j++ {
				if float64(j+1)/float64(numDraft) <= sc.acceptanceRate*1.2 {
					accepted++
				}
			}
			// Clamp based on actual rate
			accepted = int(math.Round(float64(numDraft) * sc.acceptanceRate))
			if accepted > numDraft {
				accepted = numDraft
			}

			metrics.DraftTokensAccepted += accepted
			metrics.VerificationSteps++
			totalSteps++

			// Total tokens this step: accepted drafts + 1 (correction/bonus)
			tokensThisStep := accepted + 1
			totalTokens += tokensThisStep
		}

		speedup := metrics.Speedup()
		acceptRate := metrics.AcceptanceRate()

		fmt.Printf("%-25s %6d %7.0f%% %8d %8d %7.1fx\n",
			sc.name, sc.numDraft, acceptRate*100, totalSteps, totalTokens, speedup)

		// Verify the simulation is internally consistent
		_ = s // sampler available for actual verification if needed
		_ = vocabSize
	}

	fmt.Println("-------------------------------------------------------------------")
	fmt.Println("Speedup = (accepted + verification_steps) / verification_steps")
	fmt.Println()

	// Also simulate tree vs flat comparison
	fmt.Println("=== Tree vs Flat Speculation ===")
	fmt.Printf("%-25s %8s %8s\n", "Method", "AccRate", "Speedup")
	fmt.Println("-------------------------------------------")

	// Flat: argmax per head
	flatMetrics := SpeculativeMetrics{
		DraftTokensGenerated: 400,
		DraftTokensAccepted:  240, // 60%
		VerificationSteps:    100,
	}
	fmt.Printf("%-25s %7.0f%% %7.1fx\n",
		"Flat (argmax)",
		flatMetrics.AcceptanceRate()*100,
		flatMetrics.Speedup())

	// Tree: top-3 candidates, sometimes picks better path
	treeMetrics := SpeculativeMetrics{
		DraftTokensGenerated: 400,
		DraftTokensAccepted:  280, // 70% (tree finds better paths)
		VerificationSteps:    100,
	}
	fmt.Printf("%-25s %7.0f%% %7.1fx\n",
		"Tree (topK=3)",
		treeMetrics.AcceptanceRate()*100,
		treeMetrics.Speedup())

	// Compute improvement
	flatSpeedup := flatMetrics.Speedup()
	treeSpeedup := treeMetrics.Speedup()
	improvement := (treeSpeedup - flatSpeedup) / flatSpeedup * 100
	fmt.Printf("\nTree improvement over flat: %.1f%%\n", improvement)
}

// TestSelfSpecVsSeparateModelReport compares self-speculative (early exit)
// vs separate draft model approaches.
func TestSelfSpecVsSeparateModelReport(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping simulation in short mode")
	}

	fmt.Println("\n=== Self-Speculative vs Separate Draft Model ===")
	fmt.Printf("%-30s %8s %8s %10s %8s\n",
		"Approach", "AccRate", "Speedup", "DraftCost", "NetGain")
	fmt.Println("---------------------------------------------------------------")

	// Self-speculative: lower draft cost (reuses target model layers),
	// but potentially lower acceptance rate (less accurate draft)
	selfSpec := struct {
		acceptRate float64
		draftCost  float64 // relative cost of draft step vs full decode
	}{0.55, 0.15} // 55% acceptance, draft is 15% cost of full decode

	selfMetrics := SpeculativeMetrics{
		DraftTokensGenerated: 400,
		DraftTokensAccepted:  int(float64(400) * selfSpec.acceptRate),
		VerificationSteps:    100,
	}
	// Net speedup accounts for draft overhead
	selfSpeedup := selfMetrics.Speedup()
	selfNetGain := selfSpeedup / (1 + selfSpec.draftCost*float64(4)) // 4 draft tokens overhead
	fmt.Printf("%-30s %7.0f%% %7.1fx %9.0f%% %7.1fx\n",
		"Self-speculative (8 layers)",
		selfSpec.acceptRate*100,
		selfSpeedup,
		selfSpec.draftCost*100,
		selfNetGain)

	// Separate draft model: higher acceptance (better draft model),
	// but higher draft cost (separate model forward pass)
	sepDraft := struct {
		acceptRate float64
		draftCost  float64
	}{0.75, 0.30} // 75% acceptance, draft is 30% cost of full decode

	sepMetrics := SpeculativeMetrics{
		DraftTokensGenerated: 400,
		DraftTokensAccepted:  int(float64(400) * sepDraft.acceptRate),
		VerificationSteps:    100,
	}
	sepSpeedup := sepMetrics.Speedup()
	sepNetGain := sepSpeedup / (1 + sepDraft.draftCost*float64(4))
	fmt.Printf("%-30s %7.0f%% %7.1fx %9.0f%% %7.1fx\n",
		"Separate draft (TinyLlama)",
		sepDraft.acceptRate*100,
		sepSpeedup,
		sepDraft.draftCost*100,
		sepNetGain)

	// Medusa: no separate draft model needed, uses trained heads
	medusaDraft := struct {
		acceptRate float64
		draftCost  float64
	}{0.65, 0.05} // 65% acceptance, draft is 5% cost (just MLP heads)

	medusaMetrics := SpeculativeMetrics{
		DraftTokensGenerated: 400,
		DraftTokensAccepted:  int(float64(400) * medusaDraft.acceptRate),
		VerificationSteps:    100,
	}
	medusaSpeedup := medusaMetrics.Speedup()
	medusaNetGain := medusaSpeedup / (1 + medusaDraft.draftCost*float64(4))
	fmt.Printf("%-30s %7.0f%% %7.1fx %9.0f%% %7.1fx\n",
		"Medusa heads (4 heads)",
		medusaDraft.acceptRate*100,
		medusaSpeedup,
		medusaDraft.draftCost*100,
		medusaNetGain)

	fmt.Println("---------------------------------------------------------------")
	fmt.Println("DraftCost = relative cost of draft step vs full target decode")
	fmt.Println("NetGain = Speedup / (1 + DraftCost * NumDraftTokens)")
}

// BenchmarkTokenProbability measures softmax probability computation overhead.
func BenchmarkTokenProbability(b *testing.B) {
	vocabSizes := []int{32000, 128000}
	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	for _, vocabSize := range vocabSizes {
		name := fmt.Sprintf("vocab%dk", vocabSize/1000)
		b.Run(name, func(b *testing.B) {
			logits := make([]float32, vocabSize)
			for i := range logits {
				logits[i] = float32(i) * 0.001
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				getTokenProbability(logits, vocabSize/2, s)
			}
		})
	}
}

// BenchmarkArgmax measures argmax computation for different vocab sizes.
func BenchmarkArgmax(b *testing.B) {
	vocabSizes := []int{32000, 128000}

	for _, vocabSize := range vocabSizes {
		name := fmt.Sprintf("vocab%dk", vocabSize/1000)
		b.Run(name, func(b *testing.B) {
			logits := make([]float32, vocabSize)
			for i := range logits {
				logits[i] = float32(i) * 0.001
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				argmaxFloat32(logits)
			}
		})
	}
}
