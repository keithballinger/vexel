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
// For each layer:
//   - QA / VA  (A matrices) are initialised with Kaiming-uniform noise.
//   - QB / VB  (B matrices) are zero-initialised so that the adapter has no
//     effect at the start of training.
//
// Parameters:
//
//	cfg        – adapter hyper-parameters (rank, alpha, …)
//	numLayers  – number of transformer layers to create adapters for
//	hiddenSize – input dimension of the Q and V projections (fan-in for A)
//	qDim       – output dimension of the Q projection (rows of QB)
//	vDim       – output dimension of the V projection (rows of VB)
func InitAdapter(cfg lora.AdapterConfig, numLayers, hiddenSize, qDim, vDim int) *lora.Adapter {
	rank := cfg.Rank

	layers := make([]lora.LayerAdapter, numLayers)
	for i := range layers {
		la := &layers[i]

		// A matrices: [rank, hiddenSize]  (Kaiming uniform)
		la.QA = kaimingUniform(rank, hiddenSize)
		la.QAShape = [2]int64{int64(rank), int64(hiddenSize)}

		la.VA = kaimingUniform(rank, hiddenSize)
		la.VAShape = [2]int64{int64(rank), int64(hiddenSize)}

		// B matrices: [qDim, rank] and [vDim, rank]  (zero init)
		la.QB = make([]float32, qDim*rank)
		la.QBShape = [2]int64{int64(qDim), int64(rank)}

		la.VB = make([]float32, vDim*rank)
		la.VBShape = [2]int64{int64(vDim), int64(rank)}
	}

	return &lora.Adapter{
		Config: cfg,
		Scale:  cfg.Scale(),
		Layers: layers,
	}
}
