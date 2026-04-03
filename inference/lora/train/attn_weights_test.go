//go:build metal && darwin && cgo

package train

import (
	"math"
	"os"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
	"vexel/inference/lora"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// TestAttnWeightsMatch verifies that computeAttnWeights produces attention
// weights that, when used to compute attention output, match the forward pass.
func TestAttnWeightsMatch(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		candidates := []string{
			"/Users/qeetbastudio/projects/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
		}
		for _, p := range candidates {
			if _, err := os.Stat(p); err == nil {
				modelPath = p
				break
			}
		}
	}
	if modelPath == "" {
		t.Skip("No test model")
	}

	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()

	gf, _ := gguf.Open(modelPath)
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	memCtx := memory.NewInferenceContext(tensor.Metal)
	totalScratch := modelCfg.TotalArenaBytes(256)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	model, _ := runtime.NewModelRuntime(gpuBackend, memCtx, nil, modelCfg)
	model.LoadWeightsF32(modelPath)
	model.CopyWeightsToDevice()

	mc := model.Config()
	numHeads := mc.NumAttentionHeads
	numKVHeads := mc.NumKeyValueHeads
	headDim := mc.EffectiveHeadDim()
	hiddenSize := mc.HiddenSize

	rank := 2
	adapterCfg := lora.AdapterConfig{Rank: rank, Alpha: float32(rank), TargetModules: []string{"q_proj", "v_proj"}}
	cpuAdapter := InitAdapter(adapterCfg, mc.NumHiddenLayers, hiddenSize, numHeads*headDim, numKVHeads*headDim)
	gpu, _ := lora.UploadToGPU(cpuAdapter, gpuBackend)
	model.AttachLoRA(gpu)

	tokens := []int{1, 2, 3, 4, 5}
	seqLen := len(tokens)

	_, savedPerLayer, _, err := model.TrainingForward(tokens)
	if err != nil {
		t.Fatalf("TrainingForward: %v", err)
	}
	defer func() {
		for _, sa := range savedPerLayer {
			sa.Free(gpuBackend)
		}
	}()

	// Check layer 0
	saved := savedPerLayer[0]

	// Recompute attention weights via GPU kernel
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnWeights := gpuBackend.Alloc(numHeads * seqLen * seqLen * 4)
	training := backend.TrainingOps(gpuBackend)
	training.ComputeAttnWeights(saved.Q, saved.K, attnWeights, seqLen, headDim, numHeads, numKVHeads, scale)
	gpuBackend.Sync()

	// Use attention weights to compute attention output: out = attnWeights @ V
	// attnWeights[numHeads, seqLen, seqLen], V[seqLen, numKVHeads, headDim]
	// out[seqLen, numHeads, headDim]
	awData := downloadF32(gpuBackend, attnWeights, numHeads*seqLen*seqLen)
	vData := downloadF32(gpuBackend, saved.V, seqLen*numKVHeads*headDim)

	headsPerGroup := numHeads / numKVHeads
	recomputedOut := make([]float32, seqLen*numHeads*headDim)
	for h := 0; h < numHeads; h++ {
		kvh := h / headsPerGroup
		for i := 0; i < seqLen; i++ {
			awOff := h*seqLen*seqLen + i*seqLen
			for d := 0; d < headDim; d++ {
				var sum float32
				for j := 0; j < seqLen; j++ {
					vIdx := j*numKVHeads*headDim + kvh*headDim + d
					sum += awData[awOff+j] * vData[vIdx]
				}
				outIdx := i*numHeads*headDim + h*headDim + d
				recomputedOut[outIdx] = sum
			}
		}
	}

	// Compare against the actual attention output from forward
	actualAttnOut := downloadF32(gpuBackend, saved.AttnOut, seqLen*numHeads*headDim)

	var maxErr float64
	var recompNorm, actualNorm float64
	for i := range recomputedOut {
		diff := math.Abs(float64(recomputedOut[i] - actualAttnOut[i]))
		if diff > maxErr {
			maxErr = diff
		}
		recompNorm += float64(recomputedOut[i]) * float64(recomputedOut[i])
		actualNorm += float64(actualAttnOut[i]) * float64(actualAttnOut[i])
	}
	recompNorm = math.Sqrt(recompNorm)
	actualNorm = math.Sqrt(actualNorm)
	relErr := math.Abs(recompNorm-actualNorm) / (actualNorm + 1e-8)

	t.Logf("Attention output: recompNorm=%.6f actualNorm=%.6f relErr=%.6f maxErr=%.8f",
		recompNorm, actualNorm, relErr, maxErr)

	if relErr > 0.01 {
		t.Errorf("Attention output mismatch: relErr=%.4f", relErr)
		for i := 0; i < min(10, len(recomputedOut)); i++ {
			t.Logf("  [%d] recomp=%.6f actual=%.6f", i, recomputedOut[i], actualAttnOut[i])
		}
	}
}
