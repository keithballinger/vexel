package train

import (
	"math"
	"math/rand"

	"vexel/inference/lora"
)

// kaimingUniform fills a [rows x cols] matrix with values drawn from a uniform
// distribution in [-bound, bound] where bound = 1/sqrt(cols).  This matches
// the PyTorch default for LoRA A-matrix initialisation.
func kaimingUniform(rows, cols int) []float32 {
	bound := float32(1.0 / math.Sqrt(float64(cols)))
	out := make([]float32, rows*cols)
	for i := range out {
		// rand.Float32() returns [0, 1); scale to [-bound, bound].
		out[i] = (rand.Float32()*2 - 1) * bound
	}
	return out
}

// InitAdapter constructs a new *lora.Adapter with randomly-initialised weights
// suitable as the starting point for LoRA fine-tuning.
//
// For each layer the target modules listed in cfg.TargetModules are initialised:
//   - A matrices are initialised with Kaiming-uniform noise.
//   - B matrices are zero-initialised so the adapter has no effect at the start.
//
// Parameters:
//
//	cfg        – adapter hyper-parameters (rank, alpha, target_modules, …)
//	numLayers  – number of transformer layers to create adapters for
//	hiddenSize – input dimension of Q/K/V projections (fan-in for A)
//	qDim       – output dimension of the Q projection (numHeads * headDim)
//	vDim       – output dimension of the V projection (numKVHeads * headDim)
//
// Dimension notes:
//   - K projection: A is [rank, hiddenSize], B is [kDim, rank] where kDim == vDim (numKVHeads * headDim)
//   - O projection: A is [rank, qDim], B is [hiddenSize, rank] (maps qDim → hiddenSize)
func InitAdapter(cfg lora.AdapterConfig, numLayers, hiddenSize, qDim, vDim int) *lora.Adapter {
	rank := cfg.Rank
	kDim := vDim // K and V have the same dimension (numKVHeads * headDim)

	layers := make([]lora.LayerAdapter, numLayers)
	for i := range layers {
		la := &layers[i]

		if cfg.HasTargetModule("q_proj") {
			la.QA = kaimingUniform(rank, hiddenSize)
			la.QAShape = [2]int64{int64(rank), int64(hiddenSize)}
			la.QB = make([]float32, qDim*rank)
			la.QBShape = [2]int64{int64(qDim), int64(rank)}
		}

		if cfg.HasTargetModule("k_proj") {
			la.KA = kaimingUniform(rank, hiddenSize)
			la.KAShape = [2]int64{int64(rank), int64(hiddenSize)}
			la.KB = make([]float32, kDim*rank)
			la.KBShape = [2]int64{int64(kDim), int64(rank)}
		}

		if cfg.HasTargetModule("v_proj") {
			la.VA = kaimingUniform(rank, hiddenSize)
			la.VAShape = [2]int64{int64(rank), int64(hiddenSize)}
			la.VB = make([]float32, vDim*rank)
			la.VBShape = [2]int64{int64(vDim), int64(rank)}
		}

		if cfg.HasTargetModule("o_proj") {
			// O maps from qDim to hiddenSize: A is [rank, qDim], B is [hiddenSize, rank]
			la.OA = kaimingUniform(rank, qDim)
			la.OAShape = [2]int64{int64(rank), int64(qDim)}
			la.OB = make([]float32, hiddenSize*rank)
			la.OBShape = [2]int64{int64(hiddenSize), int64(rank)}
		}
	}

	return &lora.Adapter{
		Config: cfg,
		Scale:  cfg.Scale(),
		Layers: layers,
	}
}
