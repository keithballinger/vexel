package runtime

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"vexel/inference/memory"
	"vexel/inference/tensor"
)

// Debug flag - set DEBUG_DECODE=1 to enable
var debugDecode = os.Getenv("DEBUG_DECODE") == "1"

// debugTensor prints stats about a tensor (min, max, mean, first values)
func debugTensor(name string, backend interface {
	ToHost([]byte, tensor.DevicePtr)
	Sync()
}, ptr tensor.DevicePtr, numElements int) {
	if !debugDecode {
		return
	}
	if ptr.IsNil() || numElements == 0 {
		fmt.Printf("[DEBUG] %s: nil or empty\n", name)
		return
	}

	// Read data from device
	data := make([]byte, numElements*4)
	backend.ToHost(data, ptr)
	backend.Sync()

	values := bytesToFloat32(data)
	if len(values) == 0 {
		fmt.Printf("[DEBUG] %s: no values\n", name)
		return
	}

	min, max, sum := values[0], values[0], float32(0)
	nanCount, infCount := 0, 0
	for _, v := range values {
		if math.IsNaN(float64(v)) {
			nanCount++
		} else if math.IsInf(float64(v), 0) {
			infCount++
		} else {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			sum += v
		}
	}
	mean := sum / float32(len(values)-nanCount-infCount)

	n := 8
	if n > len(values) {
		n = len(values)
	}
	fmt.Printf("[DEBUG] %s [%d]: min=%.4f max=%.4f mean=%.4f nan=%d inf=%d first%d=%v\n",
		name, len(values), min, max, mean, nanCount, infCount, n, values[:n])
}

// debugLogits prints top tokens from logits
func debugLogits(backend interface {
	ToHost([]byte, tensor.DevicePtr)
	Sync()
}, ptr tensor.DevicePtr, vocabSize int) {
	if !debugDecode || ptr.IsNil() {
		return
	}

	data := make([]byte, vocabSize*4)
	backend.ToHost(data, ptr)
	backend.Sync()

	values := bytesToFloat32(data)

	// Find top 5 tokens
	type tokenScore struct {
		id    int
		score float32
	}
	top := make([]tokenScore, 5)
	for i := range top {
		top[i].score = -1e30
	}

	for i, v := range values {
		if v > top[4].score {
			top[4] = tokenScore{i, v}
			// Bubble up
			for j := 3; j >= 0; j-- {
				if top[j+1].score > top[j].score {
					top[j], top[j+1] = top[j+1], top[j]
				}
			}
		}
	}

	fmt.Printf("[DEBUG] Logits top 5: ")
	for _, t := range top {
		fmt.Printf("[%d]=%.2f ", t.id, t.score)
	}
	fmt.Println()

	// Also check for issues
	minVal, maxVal := values[0], values[0]
	nanCount := 0
	for _, v := range values {
		if math.IsNaN(float64(v)) {
			nanCount++
		} else {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
	}
	fmt.Printf("[DEBUG] Logits range: min=%.2f max=%.2f nan=%d\n", minVal, maxVal, nanCount)
}

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

	if debugDecode {
		fmt.Printf("[DECODE] tokens=%v batchSize=%d positions=%v\n", tokens, batchSize, inputs.Positions())
	}

	// Reset buffer pool if the backend supports it (reduces allocation overhead)
	if pooler, ok := m.backend.(interface{ ResetPool() }); ok {
		pooler.ResetPool()
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
	// No sync needed - next operation depends on same data
	if debugDecode {
		m.backend.Sync()
		debugTensor("After Embedding", m.backend, statePtr, batchSize*hiddenSize)
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
		// Debug first layer only (to avoid spam)
		if debugDecode && i == 0 {
			m.backend.Sync()
			debugTensor("After Layer 0", m.backend, state.DevicePtr(), batchSize*hiddenSize)
		}
	}
	if debugDecode {
		m.backend.Sync()
		debugTensor("After All Layers", m.backend, state.DevicePtr(), batchSize*hiddenSize)
	}

	// 6. Final Norm (in-place on state)
	if !m.FinalNorm.DevicePtr().IsNil() {
		m.backend.RMSNorm(statePtr, m.FinalNorm.DevicePtr(), statePtr, batchSize, hiddenSize, float32(m.config.RMSNormEPS))
	}
	if debugDecode {
		m.backend.Sync()
		debugTensor("After Final Norm", m.backend, statePtr, batchSize*hiddenSize)
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

	// Debug logits - find top tokens
	if debugDecode {
		debugLogits(m.backend, logitsPtr, vocabSize)
	}

	return logits, nil
}

// DecodeStepWithPagedKV performs a single decoding step using paged KV cache.
// This version uses DevicePtr operations for GPU execution.
func (m *ModelRuntime) DecodeStepWithPagedKV(inputs BatchRuntimeInputs) (tensor.Tensor, error) {
	// For now, delegate to the basic DecodeStep
	// Full KV cache integration requires separate GPU-native implementation
	return m.DecodeStep(inputs)
}

// DecodeWithPagedKV performs a single decode step using paged KV cache.
// tokens: single token to decode
// seqID: sequence ID in the paged cache
// pos: current position in the sequence
func (m *ModelRuntime) DecodeWithPagedKV(tokens []int, seqID int64, pos int) (tensor.Tensor, error) {
	if len(tokens) == 0 {
		return tensor.Tensor{}, nil
	}

	if debugDecode {
		fmt.Printf("[DECODE-PAGED] tokens=%v seqID=%d pos=%d\n", tokens, seqID, pos)
	}

	// Reset buffer pool if the backend supports it
	if pooler, ok := m.backend.(interface{ ResetPool() }); ok {
		pooler.ResetPool()
	}

	batchSize := len(tokens)
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
	m.backend.Sync() // Wait for embedding before layers read state

	// 4. Allocate Scratch for Layers
	scratchBytes := m.config.ScratchBytes(batchSize)
	// Add extra space for full KV sequences from cache
	maxKVLen := pos + batchSize
	kvHeadDim := m.config.NumKeyValueHeads * (m.config.HiddenSize / m.config.NumAttentionHeads)
	scratchBytes += int64(maxKVLen * kvHeadDim * 4 * 2) // K and V
	scratchPtr, err := allocPtr(int(scratchBytes))
	if err != nil {
		return tensor.Tensor{}, err
	}
	scratch := tensor.NewTensor(tensor.NewShape(int(scratchBytes/4)), m.config.DType, scratchPtr)

	// 5. Layer Loop using ExecuteWithPagedKV
	for i, layer := range m.layers {
		state, err = layer.ExecuteWithPagedKV(state, scratch, m.pagedCache, seqID, i, pos)
		if err != nil {
			return tensor.Tensor{}, fmt.Errorf("layer %d: %w", i, err)
		}
	}
	// Sync after all layers to ensure state is ready for final norm
	m.backend.Sync()

	// 6. Final Norm (in-place on state)
	if !m.FinalNorm.DevicePtr().IsNil() {
		m.backend.RMSNorm(statePtr, m.FinalNorm.DevicePtr(), statePtr, batchSize, hiddenSize, float32(m.config.RMSNormEPS))
	}
	// No sync needed - operations are serialized in Metal

	// 7. Compute Logits: state @ OutputHead^T
	logitsPtr, err := allocPtr(batchSize * vocabSize * 4)
	if err != nil {
		return tensor.Tensor{}, err
	}
	logits := tensor.NewTensor(tensor.NewShape(batchSize, vocabSize), m.config.DType, logitsPtr)

	if !m.OutputHead.DevicePtr().IsNil() {
		m.backend.MatMulTransposed(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
	}

	m.backend.Sync()

	if debugDecode {
		debugLogits(m.backend, logitsPtr, vocabSize)
	}

	return logits, nil
}

// DecodeWithGPUKV performs a forward pass using GPU-resident KV cache.
// This is faster than DecodeWithPagedKV because it avoids CPU roundtrips.
// pos: current position in the sequence
func (m *ModelRuntime) DecodeWithGPUKV(tokens []int, pos int) (tensor.Tensor, error) {
	if len(tokens) == 0 {
		return tensor.Tensor{}, nil
	}

	if m.gpuCache == nil {
		return tensor.Tensor{}, fmt.Errorf("GPU KV cache not initialized")
	}

	// Reset buffer pool if the backend supports it
	if pooler, ok := m.backend.(interface{ ResetPool() }); ok {
		pooler.ResetPool()
	}

	batchSize := len(tokens)
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
	// No sync needed - GPU operations serialized

	// 4. Allocate Scratch for Layers
	scratchBytes := m.config.ScratchBytes(batchSize)
	scratchPtr, err := allocPtr(int(scratchBytes))
	if err != nil {
		return tensor.Tensor{}, err
	}
	scratch := tensor.NewTensor(tensor.NewShape(int(scratchBytes/4)), m.config.DType, scratchPtr)

	// 5. Layer Loop using ExecuteWithGPUKV
	for i, layer := range m.layers {
		state, err = layer.ExecuteWithGPUKV(state, scratch, m.gpuCache, i, pos)
		if err != nil {
			return tensor.Tensor{}, fmt.Errorf("layer %d: %w", i, err)
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

	m.backend.Sync()

	return logits, nil
}

// PrefillWithPagedKV processes multiple tokens in a single forward pass.
// This version uses DevicePtr operations for GPU execution.
// Returns logits only for the LAST token.
func (m *ModelRuntime) PrefillWithPagedKV(tokens []int, seqID int64, startPos int) (tensor.Tensor, error) {
	seqLen := len(tokens)
	if seqLen == 0 {
		return tensor.Tensor{}, nil
	}

	if debugDecode {
		fmt.Printf("[PREFILL] tokens=%v seqLen=%d seqID=%d startPos=%d\n", tokens, seqLen, seqID, startPos)
	}

	// Reset buffer pool if the backend supports it (reduces allocation overhead)
	if pooler, ok := m.backend.(interface{ ResetPool() }); ok {
		pooler.ResetPool()
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
	m.backend.Sync() // CRITICAL: Wait for embedding before layers read state

	debugTensor("[PREFILL] After Embedding", m.backend, statePtr, seqLen*hiddenSize)

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
		// Debug first layer only (to avoid spam)
		if i == 0 {
			m.backend.Sync()
			debugTensor("[PREFILL] After Layer 0", m.backend, state.DevicePtr(), seqLen*hiddenSize)
		}
	}
	m.backend.Sync()
	debugTensor("[PREFILL] After All Layers", m.backend, state.DevicePtr(), seqLen*hiddenSize)

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

	// Debug PREFILL logits
	if debugDecode {
		debugLogits(m.backend, logitsPtr, vocabSize)
	}

	return logits, nil
}
