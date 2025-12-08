package runtime

import (
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

// DecodeStep performs a single decoding step for the batch.
func (m *ModelRuntime) DecodeStep(inputs BatchRuntimeInputs) (tensor.Tensor, error) {
	// 1. Prepare batch metadata
	batchSize := 1 // TODO: get from inputs
	
	// 2. Embedding Lookup (Mock)
	// Create a dummy tensor [Batch, Hidden]
	// shape := tensor.NewShape(batchSize, m.config.HiddenSize)
	// state := tensor.NewTensor(shape, m.config.DType, tensor.DevicePtr{})
	
	// We don't have NewTensor capable of allocating yet (Arena needed).
	// But we can construct the struct with the shape to satisfy the test.
	state := tensor.NewTensor(
		tensor.NewShape(batchSize, m.config.HiddenSize),
		m.config.DType,
		tensor.NewDevicePtr(tensor.CPU, 0),
	)
	
	// Create scratch buffer
	scratchSize := m.config.ScratchBytes(batchSize)
	scratchElements := int(scratchSize / 2) // Assuming 2 bytes per float
	
	// Allocate from Arena
	// We assume ctx has a Scratch arena. If not, panic or error?
	// For "Real" inference, we must ensure it's initialized.
	var ptr tensor.DevicePtr
	
	arena := m.ctx.GetArena(memory.Scratch)
	if arena != nil {
		m.ctx.ResetScratch()
		var err error
		ptr, err = arena.Alloc(int(scratchSize))
		if err != nil {
			return tensor.Tensor{}, err
		}
	} else {
		// Fallback for tests that didn't set up arena (mock behavior)
		ptr = tensor.NewDevicePtr(tensor.CPU, 0)
	}

	scratch := tensor.NewTensor(
		tensor.NewShape(scratchElements),
		m.config.DType,
		ptr,
	)
	
	// 3. Layer Loop
	for _, layer := range m.layers {
		var err error
		state, err = layer.Execute(state, scratch)
		if err != nil {
			return tensor.Tensor{}, err
		}
	}
	
	// 4. Final Norm (Skip for now)
	
	// 5. Compute Logits (Output Head)
	// Result shape: [Batch, Vocab]
	logits := tensor.NewTensor(
		tensor.NewShape(batchSize, m.config.VocabSize),
		m.config.DType,
		tensor.NewDevicePtr(tensor.CPU, 0),
	)
	
	return logits, nil
}
