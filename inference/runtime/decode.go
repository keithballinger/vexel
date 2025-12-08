package runtime

import (
	"fmt"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

// DecodeStep performs a single decoding step for the batch.
func (m *ModelRuntime) DecodeStep(inputs BatchRuntimeInputs) (tensor.Tensor, error) {
	// 1. Prepare batch metadata
	tokens := inputs.Tokens()
	batchSize := len(tokens)
	if batchSize == 0 {
		return tensor.Tensor{}, nil
	}
	
	hiddenSize := m.config.HiddenSize
	
	// Create scratch buffer
	scratchSize := m.config.ScratchBytes(batchSize)
	// Add space for State [Batch, Hidden] and Logits [Batch, Vocab] and FinalNorm [Batch, Hidden]
	// Ideally Arena handles this cumulatively.
	
	// Allocate from Arena
	// We assume ctx has a Scratch arena.
	if m.ctx == nil {
		fmt.Println("[DEBUG] Context is NIL!")
		// Mock/Test fallback for uninitialized runtime
		return tensor.NewTensor(
			tensor.NewShape(batchSize, m.config.VocabSize),
			m.config.DType,
			tensor.NewDevicePtr(tensor.CPU, 0),
		), nil
	}
	arena := m.ctx.GetArena(memory.Scratch)
	if arena == nil {
		fmt.Println("[DEBUG] Arena is NIL!")
		// Mock/Test fallback
		return tensor.NewTensor(
			tensor.NewShape(batchSize, m.config.VocabSize),
			m.config.DType,
			tensor.NewDevicePtr(tensor.CPU, 0),
		), nil
	}
	
	m.ctx.ResetScratch()
	
	// Helper to alloc tensor
	allocTensor := func(shape []int) (tensor.Tensor, []float32, error) {
		numElements := 1
		for _, d := range shape {
			numElements *= d
		}
		sizeBytes := numElements * 4 // Float32
		ptr, err := arena.Alloc(sizeBytes)
		if err != nil {
			return tensor.Tensor{}, nil, err
		}
		t := tensor.NewTensor(tensor.NewShape(shape...), m.config.DType, ptr)
		data := tensor.ToFloat32Slice(t)
		return t, data, nil
	}
	
	// 2. Embedding Lookup
	// Allocate State [Batch, Hidden]
	state, stateData, err := allocTensor([]int{batchSize, hiddenSize})
	if err != nil {
		return tensor.Tensor{}, err
	}
	
	// Perform Lookup
	if !m.Embedding.DevicePtr().IsNil() {
		table := tensor.ToFloat32Slice(m.Embedding)
		m.backend.Embedding(tokens, table, stateData, hiddenSize)
	}
	
	// Allocate Scratch for Layers
	// Note: Layers expect 'scratch' tensor which they sub-allocate/view manually?
	// BlockRuntime.Execute splits scratch into Q,K,V etc.
	// We need to pass a large enough buffer.
	scratchBytes := scratchSize // Bytes
	scratchPtr, err := arena.Alloc(int(scratchBytes))
	if err != nil {
		return tensor.Tensor{}, err
	}
	scratch := tensor.NewTensor(
		tensor.NewShape(int(scratchBytes/4)), // float32 elements
		m.config.DType,
		scratchPtr,
	)
	
	// Get position from inputs (default to 0 for backwards compatibility)
	pos := 0
	if positions := inputs.Positions(); len(positions) > 0 {
		// For now, use position of first token in batch
		// TODO: Support variable positions per sequence in batch
		pos = positions[0]
	}

	// 3. Layer Loop
	for i, layer := range m.layers {
		state, err = layer.Execute(state, scratch, m.cache, i, pos)
		if err != nil {
			return tensor.Tensor{}, err
		}
	}
	
	// 4. Final Norm
	// Allocate Output for Norm? Or in-place?
	// RMSNorm can be in-place if x != out?
	// stateData is reused.
	// We need weights.
	if !m.FinalNorm.DevicePtr().IsNil() {
		normWeights := tensor.ToFloat32Slice(m.FinalNorm)
		// In-place update of state
		m.backend.RMSNorm(stateData, normWeights, stateData, batchSize, hiddenSize, float32(m.config.RMSNormEPS))
	}
	
	// 5. Compute Logits (Output Head)
	// Result shape: [Batch, Vocab]
	logits, logitsData, err := allocTensor([]int{batchSize, m.config.VocabSize})
	if err != nil {
		fmt.Printf("DecodeStep Alloc Logits Failed: %v\n", err)
		return tensor.Tensor{}, err
	}
	// fmt.Printf("DecodeStep Logits Ptr: %v, Len: %d\n", logits.DevicePtr(), len(logitsData))
	
	if !m.OutputHead.DevicePtr().IsNil() {
		headWeights := tensor.ToFloat32Slice(m.OutputHead)
		// State [Batch, Hidden] * Head^T [Hidden, Vocab] -> [Batch, Vocab]
		// Head weights are [Vocab, Hidden] usually (Out, In).
		// So we use MatmulTransposeB.
		m.backend.MatmulTransposeB(stateData, headWeights, logitsData, batchSize, m.config.VocabSize, hiddenSize)
	}

	return logits, nil
}

// DecodeStepWithPagedKV performs a single decoding step using paged KV cache.
// This is the production path that properly utilizes KV caching for autoregressive generation.
func (m *ModelRuntime) DecodeStepWithPagedKV(inputs BatchRuntimeInputs) (tensor.Tensor, error) {
	tokens := inputs.Tokens()
	batchSize := len(tokens)
	if batchSize == 0 {
		return tensor.Tensor{}, nil
	}

	// Currently only support batch size 1
	if batchSize != 1 {
		return tensor.Tensor{}, fmt.Errorf("paged KV cache currently only supports batch size 1, got %d", batchSize)
	}

	hiddenSize := m.config.HiddenSize

	// Verify we have paged cache
	if m.pagedCache == nil {
		return tensor.Tensor{}, fmt.Errorf("paged KV cache not initialized")
	}

	// Get sequence ID
	seqIDs := inputs.SeqIDs()
	if len(seqIDs) == 0 {
		return tensor.Tensor{}, fmt.Errorf("sequence IDs required for paged KV cache")
	}
	seqID := seqIDs[0]

	// Get position
	pos := 0
	if positions := inputs.Positions(); len(positions) > 0 {
		pos = positions[0]
	}

	if m.ctx == nil {
		return tensor.Tensor{}, fmt.Errorf("inference context not initialized")
	}

	arena := m.ctx.GetArena(memory.Scratch)
	if arena == nil {
		return tensor.Tensor{}, fmt.Errorf("scratch arena not initialized")
	}

	m.ctx.ResetScratch()

	// Helper to alloc tensor
	allocTensor := func(shape []int) (tensor.Tensor, []float32, error) {
		numElements := 1
		for _, d := range shape {
			numElements *= d
		}
		sizeBytes := numElements * 4 // Float32
		ptr, err := arena.Alloc(sizeBytes)
		if err != nil {
			return tensor.Tensor{}, nil, err
		}
		t := tensor.NewTensor(tensor.NewShape(shape...), m.config.DType, ptr)
		data := tensor.ToFloat32Slice(t)
		return t, data, nil
	}

	// 1. Embedding Lookup
	state, stateData, err := allocTensor([]int{batchSize, hiddenSize})
	if err != nil {
		return tensor.Tensor{}, err
	}

	if !m.Embedding.DevicePtr().IsNil() {
		table := tensor.ToFloat32Slice(m.Embedding)
		m.backend.Embedding(tokens, table, stateData, hiddenSize)
	}

	// Allocate Scratch for Layers
	scratchSize := m.config.ScratchBytes(batchSize)
	scratchPtr, err := arena.Alloc(int(scratchSize))
	if err != nil {
		return tensor.Tensor{}, err
	}
	scratch := tensor.NewTensor(
		tensor.NewShape(int(scratchSize/4)),
		m.config.DType,
		scratchPtr,
	)

	// 2. Layer Loop with Paged KV Cache
	for i, layer := range m.layers {
		state, err = layer.ExecuteWithPagedKV(state, scratch, m.pagedCache, seqID, i, pos)
		if err != nil {
			return tensor.Tensor{}, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	// 3. Final Norm
	if !m.FinalNorm.DevicePtr().IsNil() {
		normWeights := tensor.ToFloat32Slice(m.FinalNorm)
		m.backend.RMSNorm(stateData, normWeights, stateData, batchSize, hiddenSize, float32(m.config.RMSNormEPS))
	}

	// 4. Compute Logits
	logits, logitsData, err := allocTensor([]int{batchSize, m.config.VocabSize})
	if err != nil {
		return tensor.Tensor{}, err
	}

	if !m.OutputHead.DevicePtr().IsNil() {
		headWeights := tensor.ToFloat32Slice(m.OutputHead)
		m.backend.MatmulTransposeB(stateData, headWeights, logitsData, batchSize, m.config.VocabSize, hiddenSize)
	}

	return logits, nil
}
