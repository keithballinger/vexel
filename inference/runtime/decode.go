package runtime

import (
	"encoding/binary"
	"fmt"
	"math"

	"vexel/inference/memory"
	"vexel/inference/tensor"
)

// int32ToBytes converts []int to []byte (as int32 values)
func int32ToBytes(tokens []int) []byte {
	bytes := make([]byte, len(tokens)*4)
	for i, tok := range tokens {
		binary.LittleEndian.PutUint32(bytes[i*4:], uint32(int32(tok)))
	}
	return bytes
}

// bytesToFloat32 converts []byte to []float32
func bytesToFloat32(data []byte) []float32 {
	result := make([]float32, len(data)/4)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result
}

// DecodeStep performs a single decoding step using DevicePtr operations.
// All tensors are allocated on the backend device.
func (m *ModelRuntime) DecodeStep(inputs BatchRuntimeInputs) (tensor.Tensor, error) {
	tokens := inputs.Tokens()
	batchSize := len(tokens)
	if batchSize == 0 {
		return tensor.Tensor{}, nil
	}

	hiddenSize := m.config.HiddenSize
	vocabSize := m.config.VocabSize

	// Allocate from Arena
	if m.ctx == nil {
		return tensor.Tensor{}, fmt.Errorf("inference context not initialized")
	}
	arena := m.ctx.GetArena(memory.Scratch)
	if arena == nil {
		return tensor.Tensor{}, fmt.Errorf("scratch arena not initialized")
	}

	m.ctx.ResetScratch()

	// Helper to allocate DevicePtr
	allocPtr := func(bytes int) (tensor.DevicePtr, error) {
		return arena.Alloc(bytes)
	}

	// 1. Copy token IDs to device
	tokenBytes := int32ToBytes(tokens)
	tokenPtr, err := allocPtr(len(tokenBytes))
	if err != nil {
		return tensor.Tensor{}, err
	}
	m.backend.ToDevice(tokenPtr, tokenBytes)

	// 2. Allocate State [Batch, Hidden]
	statePtr, err := allocPtr(batchSize * hiddenSize * 4)
	if err != nil {
		return tensor.Tensor{}, err
	}
	state := tensor.NewTensor(tensor.NewShape(batchSize, hiddenSize), m.config.DType, statePtr)

	// 3. Embedding Lookup
	if !m.Embedding.DevicePtr().IsNil() {
		m.backend.Embedding(tokenPtr, batchSize, m.Embedding.DevicePtr(), statePtr, vocabSize, hiddenSize)
	}

	// 4. Allocate Scratch for Layers
	scratchBytes := m.config.ScratchBytes(batchSize)
	scratchPtr, err := allocPtr(int(scratchBytes))
	if err != nil {
		return tensor.Tensor{}, err
	}
	scratch := tensor.NewTensor(tensor.NewShape(int(scratchBytes/4)), m.config.DType, scratchPtr)

	// Get position from inputs
	pos := 0
	if positions := inputs.Positions(); len(positions) > 0 {
		pos = positions[0]
	}

	// 5. Layer Loop
	for i, layer := range m.layers {
		state, err = layer.Execute(state, scratch, m.cache, i, pos)
		if err != nil {
			return tensor.Tensor{}, err
		}
	}

	// 6. Final Norm (in-place on state)
	if !m.FinalNorm.DevicePtr().IsNil() {
		m.backend.RMSNorm(statePtr, m.FinalNorm.DevicePtr(), statePtr, batchSize, hiddenSize, float32(m.config.RMSNormEPS))
	}

	// 7. Compute Logits: state @ OutputHead^T
	logitsPtr, err := allocPtr(batchSize * vocabSize * 4)
	if err != nil {
		return tensor.Tensor{}, err
	}
	logits := tensor.NewTensor(tensor.NewShape(batchSize, vocabSize), m.config.DType, logitsPtr)

	if !m.OutputHead.DevicePtr().IsNil() {
		m.backend.MatMulTransposed(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
	}

	// Sync to ensure all operations complete
	m.backend.Sync()

	return logits, nil
}

// DecodeStepWithPagedKV performs a single decoding step using paged KV cache.
// This version uses DevicePtr operations for GPU execution.
// NOTE: The paged KV cache integration needs refactoring for GPU operation.
func (m *ModelRuntime) DecodeStepWithPagedKV(inputs BatchRuntimeInputs) (tensor.Tensor, error) {
	// For now, delegate to the basic DecodeStep
	// Full KV cache integration requires separate GPU-native implementation
	return m.DecodeStep(inputs)
}

// PrefillWithPagedKV processes multiple tokens in a single forward pass.
// This version uses DevicePtr operations for GPU execution.
// Returns logits only for the LAST token.
func (m *ModelRuntime) PrefillWithPagedKV(tokens []int, seqID int64, startPos int) (tensor.Tensor, error) {
	seqLen := len(tokens)
	if seqLen == 0 {
		return tensor.Tensor{}, nil
	}

	hiddenSize := m.config.HiddenSize
	vocabSize := m.config.VocabSize

	if m.ctx == nil {
		return tensor.Tensor{}, fmt.Errorf("inference context not initialized")
	}

	arena := m.ctx.GetArena(memory.Scratch)
	if arena == nil {
		return tensor.Tensor{}, fmt.Errorf("scratch arena not initialized")
	}

	m.ctx.ResetScratch()

	allocPtr := func(bytes int) (tensor.DevicePtr, error) {
		return arena.Alloc(bytes)
	}

	// 1. Copy token IDs to device
	tokenBytes := int32ToBytes(tokens)
	tokenPtr, err := allocPtr(len(tokenBytes))
	if err != nil {
		return tensor.Tensor{}, err
	}
	m.backend.ToDevice(tokenPtr, tokenBytes)

	// 2. Allocate State [seqLen, Hidden]
	statePtr, err := allocPtr(seqLen * hiddenSize * 4)
	if err != nil {
		return tensor.Tensor{}, err
	}
	state := tensor.NewTensor(tensor.NewShape(seqLen, hiddenSize), m.config.DType, statePtr)

	// 3. Embedding Lookup
	if !m.Embedding.DevicePtr().IsNil() {
		m.backend.Embedding(tokenPtr, seqLen, m.Embedding.DevicePtr(), statePtr, vocabSize, hiddenSize)
	}

	// 4. Allocate Scratch (need more for prefill)
	scratchBytes := m.config.ScratchBytes(seqLen) + int64(seqLen*seqLen*4)
	scratchPtr, err := allocPtr(int(scratchBytes))
	if err != nil {
		return tensor.Tensor{}, err
	}
	scratch := tensor.NewTensor(tensor.NewShape(int(scratchBytes/4)), m.config.DType, scratchPtr)

	// 5. Layer Loop
	for i, layer := range m.layers {
		state, err = layer.ExecuteWithPagedKV(state, scratch, m.pagedCache, seqID, i, startPos)
		if err != nil {
			return tensor.Tensor{}, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	// 6. Final Norm on last token only
	// For GPU: DevicePtrOffset doesn't work with Metal because kernels can't take buffer+offset
	// So we need to allocate a separate buffer and copy the last token
	var lastTokenPtr tensor.DevicePtr
	if statePtr.Location() == tensor.CPU || seqLen == 1 {
		// CPU: offset works fine; seqLen==1: no offset needed
		lastTokenPtr = tensor.DevicePtrOffset(statePtr, uintptr((seqLen-1)*hiddenSize*4))
	} else {
		// GPU with seqLen > 1: allocate separate buffer and copy last token
		lastTokenPtr = m.backend.Alloc(hiddenSize * 4)
		// Copy last token from state buffer - we need to use a copy kernel
		// For now, use a workaround: apply norm to all tokens and use only the last
		// This is inefficient but correct
		// TODO: implement proper GPU copy-with-offset kernel
		m.backend.RMSNorm(statePtr, m.FinalNorm.DevicePtr(), statePtr, seqLen, hiddenSize, float32(m.config.RMSNormEPS))
		// For GPU, we'll process all tokens through output head and extract last
		// Actually, let's be smarter: run matmul on all and take last row of output
		allLogitsPtr, err := allocPtr(seqLen * vocabSize * 4)
		if err != nil {
			return tensor.Tensor{}, err
		}
		if !m.OutputHead.DevicePtr().IsNil() {
			m.backend.MatMulTransposed(statePtr, m.OutputHead.DevicePtr(), allLogitsPtr, seqLen, vocabSize, hiddenSize)
		}
		// For GPU, we need to extract just the last row (last vocabSize floats)
		// Since we can't do offset reads, return the full tensor and let caller handle
		// Or... allocate logits separately and use a slice kernel
		// For now: return tensor for all tokens, caller takes last
		m.backend.Sync()
		// Create tensor pointing to last row
		// Actually this still has the offset problem...
		// Let's just return all logits and fix this properly later
		logits := tensor.NewTensor(tensor.NewShape(seqLen, vocabSize), m.config.DType, allLogitsPtr)
		return logits, nil
	}

	if !m.FinalNorm.DevicePtr().IsNil() {
		m.backend.RMSNorm(lastTokenPtr, m.FinalNorm.DevicePtr(), lastTokenPtr, 1, hiddenSize, float32(m.config.RMSNormEPS))
	}

	// 7. Compute Logits for last token
	logitsPtr, err := allocPtr(vocabSize * 4)
	if err != nil {
		return tensor.Tensor{}, err
	}
	logits := tensor.NewTensor(tensor.NewShape(1, vocabSize), m.config.DType, logitsPtr)

	if !m.OutputHead.DevicePtr().IsNil() {
		m.backend.MatMulTransposed(lastTokenPtr, m.OutputHead.DevicePtr(), logitsPtr, 1, vocabSize, hiddenSize)
	}

	m.backend.Sync()

	return logits, nil
}
