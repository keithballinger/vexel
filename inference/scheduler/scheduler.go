package scheduler

import (
	"context"
	"fmt"
	"math"
	"os"
	goruntime "runtime"
	"sync"
	"time"
	"unsafe"
	"vexel/inference/backend"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

var debugDecode = os.Getenv("DEBUG_DECODE") == "1"

// Config holds configuration for the scheduler.
type Config struct {
	MaxBatchSize  int
	MaxSequences  int
	MaxTokens     int            // Max tokens to generate per sequence (0 = unlimited)
	SamplerConfig sampler.Config // Sampling parameters
}

// SchedulerMetrics holds performance indicators.
type SchedulerMetrics struct {
	ActiveSequences    int
	CompletedSequences int
	TotalTokens        int
	PrefillTokens      int           // Tokens processed during prefill
	DecodeTokens       int           // Tokens generated during decode
	PrefillTime        time.Duration // Total time spent in prefill
	DecodeTime         time.Duration // Total time spent in decode
}

// TokensPerSecond returns the decode tok/s rate.
func (m SchedulerMetrics) TokensPerSecond() float64 {
	if m.DecodeTime == 0 {
		return 0
	}
	return float64(m.DecodeTokens) / m.DecodeTime.Seconds()
}

// PrefillTokensPerSecond returns the prefill tok/s rate.
func (m SchedulerMetrics) PrefillTokensPerSecond() float64 {
	if m.PrefillTime == 0 {
		return 0
	}
	return float64(m.PrefillTokens) / m.PrefillTime.Seconds()
}

// Scheduler manages the execution of sequences.
// It handles:
//   - Batching of sequences for efficient GPU execution.
//   - Prefilling of prompt tokens.
//   - Decoding steps for token generation.
//   - KV cache management (via PagedKVCache or simple GPU cache).
//   - Metrics tracking (throughput, latency).
type Scheduler struct {
	mu        sync.RWMutex
	runtime   *runtime.ModelRuntime
	tokenizer *tokenizer.Tokenizer
	sampler   *sampler.Sampler
	config    Config
	sequences map[SequenceID]*Sequence
	metrics   SchedulerMetrics
	cond      *sync.Cond  // Condition variable for reliable concurrent wakeup
	readyBuf  []*Sequence // Reused buffer for collectReady to avoid per-step allocation
}

// NewScheduler creates a new Scheduler instance.
//
// Parameters:
//   - rt: The initialized model runtime (must not be nil).
//   - tok: The tokenizer for encoding/decoding text.
//   - config: Configuration for batch size, limits, and sampling.
//
// Returns:
//   - A new *Scheduler instance, or error if runtime is nil.
func NewScheduler(rt *runtime.ModelRuntime, tok *tokenizer.Tokenizer, config Config) (*Scheduler, error) {
	if rt == nil {
		return nil, fmt.Errorf("runtime cannot be nil")
	}

	// Create sampler with config (use time-based seed for randomness)
	s := sampler.New(config.SamplerConfig, time.Now().UnixNano())

	return &Scheduler{
		runtime:   rt,
		tokenizer: tok,
		sampler:   s,
		config:    config,
		sequences: make(map[SequenceID]*Sequence),
		cond:      sync.NewCond(&sync.Mutex{}),
	}, nil
}

// wakeUp triggers the scheduler loop to run a step immediately.
func (s *Scheduler) wakeUp() {
	s.cond.Broadcast()
}

// isStopToken returns true if tokenID should end generation for the given sequence.
// It checks the primary EOS token from the tokenizer as well as any per-sequence
// extra stop tokens (e.g. <|end|> for Phi-3/3.5 in chat mode).
func (s *Scheduler) isStopToken(tokenID int, seq *Sequence) bool {
	eosToken := 2
	if s.tokenizer != nil {
		eosToken = s.tokenizer.EOS()
	}
	return tokenID == eosToken || seq.IsExtraStopToken(tokenID)
}

// Run starts the scheduler's main loop.
// It continuously polls for work (sequences in Pending/Decoding/Prefill states) and executes steps.
// It blocks until the context is canceled or a fatal error occurs.
//
// The loop runs a "step" which:
//  1. Collects ready sequences.
//  2. Forms a batch up to MaxBatchSize.
//  3. Runs a decode step (prefill or generate token).
func (s *Scheduler) Run(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		if err := s.step(ctx); err != nil {
			return err
		}

		// Brief yield after each step to allow token delivery to consumers.
		// CGO calls in Metal GPU operations don't trigger Go scheduler preemption,
		// so without an explicit sleep the scheduler goroutine can monopolize the
		// OS thread and prevent the main goroutine from reading generated tokens.
		time.Sleep(50 * time.Microsecond)

		// If sequences remain, loop immediately
		if s.SequenceCount() > 0 {
			continue
		}

		// No sequences — wait for wakeup from AddSequence
		done := make(chan struct{})
		go func() {
			s.cond.L.Lock()
			s.cond.Wait()
			s.cond.L.Unlock()
			close(done)
		}()

		select {
		case <-ctx.Done():
			s.cond.Broadcast()
			return nil
		case <-done:
		}
	}
}

// step performs a single scheduling iteration.
func (s *Scheduler) step(ctx context.Context) error {
	// 1. Collect ready sequences
	ready := s.collectReady()

	// Update Active Sequences metric
	s.mu.Lock()
	s.metrics.ActiveSequences = len(s.sequences)
	s.mu.Unlock()

	// 2. Form batch
	batch := s.formBatches(ready)
	if len(batch) == 0 {
		// No ready sequences — clean up finished ones and yield to avoid
		// busy-looping. Without this, the scheduler goroutine spins indefinitely
		// on finished sequences, starving the main goroutine from receiving the
		// channel close notification.
		s.mu.Lock()
		for id, seq := range s.sequences {
			if seq.State() == StateFinished {
				delete(s.sequences, id)
			}
		}
		s.mu.Unlock()
		goruntime.Gosched()
		return nil
	}

	// 3. Run DecodeStep
	if err := s.runDecodeStep(ctx, batch); err != nil {
		return err
	}

	return nil
}

// collectReady identifies sequences that are eligible for execution.
// Reuses a pre-allocated buffer to avoid per-step allocation.
func (s *Scheduler) collectReady() []*Sequence {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.sequences) == 0 {
		return nil
	}

	s.readyBuf = s.readyBuf[:0]
	for _, seq := range s.sequences {
		switch seq.State() {
		case StatePending, StatePrefill, StateDecoding:
			s.readyBuf = append(s.readyBuf, seq)
		}
	}
	if len(s.readyBuf) == 0 {
		return nil
	}
	return s.readyBuf
}

// formBatches selects a subset of ready sequences to run in the next step.
func (s *Scheduler) formBatches(ready []*Sequence) []*Sequence {
	if len(ready) == 0 {
		return nil
	}

	// Prioritize:
	// 1. Sequences that need prefill (StatePending with prompt)
	// 2. Decoding sequences

	// Simple FIFO/Round-robin for now
	if len(ready) <= s.config.MaxBatchSize {
		return ready
	}

	return ready[:s.config.MaxBatchSize]
}

// runDecodeStep orchestrates the model execution for a batch of sequences.
func (s *Scheduler) runDecodeStep(ctx context.Context, batch []*Sequence) error {
	if len(batch) == 0 {
		return nil
	}

	// Check which KV cache we have
	usePagedCache := s.runtime.PagedKVCache() != nil
	useGPUCache := s.runtime.GPUKVCache() != nil

	// Process each sequence - check if any need batched prefill
	for _, seq := range batch {
		if seq.State() == StatePending && len(seq.PromptTokens()) > 0 && !seq.IsPrefillComplete() {
			// This sequence needs prefill - do it all at once
			if useGPUCache {
				if err := s.runGPUPrefill(seq); err != nil {
					return err
				}
				// Sync GPU after prefill to flush command queue before decode.
				// Without this, the transition from many prefill dispatches (~300+)
				// to decode can deadlock the Metal command queue.
				s.runtime.Backend().Sync()
				continue
			} else if usePagedCache {
				if err := s.runBatchedPrefill(seq); err != nil {
					return err
				}
				s.runtime.Backend().Sync()
				continue
			}
		}
	}

	// Filter to sequences that need decode (already prefilled or no prompt)
	decodeSeqs := make([]*Sequence, 0, len(batch))
	for _, seq := range batch {
		if seq.State() == StateDecoding || (seq.State() == StatePending && seq.IsPrefillComplete()) {
			decodeSeqs = append(decodeSeqs, seq)
		}
	}

	if len(decodeSeqs) == 0 {
		return nil
	}

	// Prepare inputs for decode
	tokens := make([]int, len(decodeSeqs))
	positions := make([]int, len(decodeSeqs))
	seqIDs := make([]int64, len(decodeSeqs))

	for i, seq := range decodeSeqs {
		token, pos, hasMore := seq.NextInputToken()
		if !hasMore {
			token = 1 // BOS fallback
			pos = 0
		}
		tokens[i] = token
		positions[i] = pos
		seqIDs[i] = seq.KVSeqID()
	}

	var logits tensor.Tensor
	var err error

	startTime := time.Now()

	if useGPUCache {
		// Use GPU KV cache for single-sequence decode
		// GPU KV cache only supports single sequence
		if os.Getenv("DEBUG_DECODE") == "1" {
			fmt.Printf("[SCHEDULER] Calling DecodeWithGPUKV tokens=%v pos=%d\n", tokens, positions[0])
		}
		if len(decodeSeqs) == 1 {
			logits, err = s.runtime.DecodeWithGPUKV(tokens, positions[0])
		} else {
			// Process one at a time for multi-sequence
			for i, seq := range decodeSeqs {
				singleLogits, singleErr := s.runtime.DecodeWithGPUKV([]int{tokens[i]}, positions[i])
				if singleErr != nil {
					err = singleErr
					break
				}
				if i == 0 {
					logits = singleLogits
				}
				// Sample immediately for this sequence
				vocabSize := s.runtime.Config().VocabSize
				singleLogitsData := s.getLogitsOnCPU(singleLogits, vocabSize)
				if singleLogitsData != nil {
					seq.AdvancePosition()
					if seq.State() == StatePending {
						seq.SetState(StateDecoding)
					}
					tokenID := s.sampler.Sample(singleLogitsData)
					seq.AddGeneratedToken(tokenID)
					if s.isStopToken(tokenID, seq) {
						seq.SetState(StateFinished)
						seq.Close()
					} else if seq.ReachedMaxTokens(s.config.MaxTokens) {
						seq.SetState(StateFinished)
						seq.Close()
					} else {
						var text string
						if s.tokenizer != nil {
							text, _ = s.tokenizer.Decode([]int{tokenID})
						} else {
							text = fmt.Sprintf(" %d", tokenID)
						}
						seq.PushToken(text)
					}
				}
			}
			// Fall through to update metrics
		}
	} else if usePagedCache {
		// Use DecodeWithPagedKV which properly uses the KV cache
		// For now, process one sequence at a time (batching TBD)
		if len(decodeSeqs) == 1 {
			logits, err = s.runtime.DecodeWithPagedKV(tokens, seqIDs[0], positions[0])
		} else {
			logits, err = s.runtime.DecodeWithPagedKVBatched(tokens, seqIDs, positions)
		}
	} else {
		inputs := runtime.NewBatchRuntimeInputsWithPos(tokens, positions, nil)
		logits, err = s.runtime.DecodeStep(inputs)
	}

	if err != nil {
		return err
	}

	// The multi-sequence GPU KV cache path does inline sampling above.
	// All other paths (single-seq GPU, paged, BatchRuntimeInputs) produce
	// logits that need sampling here.
	alreadySampled := useGPUCache && len(decodeSeqs) > 1

	if !alreadySampled {
		// Sample and Decode
		vocabSize := s.runtime.Config().VocabSize

		// Single-sequence greedy: use GPU argmax to avoid 128KB transfer.
		// Skip greedy path if the sequence has per-request temperature > 0.
		seqHasTemp := len(decodeSeqs) == 1 && decodeSeqs[0].HasSamplingParams() && decodeSeqs[0].SamplingTemperature() > 0
		if len(decodeSeqs) == 1 && s.config.SamplerConfig.Temperature == 0 && !seqHasTemp {
			seq := decodeSeqs[0]
			seq.AdvancePosition()
			if seq.State() == StatePending {
				seq.SetState(StateDecoding)
			}
			tokenID := s.sampleToken(logits, vocabSize)
			seq.AddGeneratedToken(tokenID)

			// Check for EOS or extra stop tokens
			if s.isStopToken(tokenID, seq) {
				seq.SetState(StateFinished)
				seq.Close()
			} else if seq.ReachedMaxTokens(s.config.MaxTokens) {
				seq.SetState(StateFinished)
				seq.Close()
			} else {
				var text string
				if s.tokenizer != nil {
					text, _ = s.tokenizer.Decode([]int{tokenID})
				} else {
					text = fmt.Sprintf(" %d", tokenID)
				}
				seq.PushToken(text)
			}
		} else {
			// Multi-sequence or non-greedy: copy logits to CPU
			logitsData := s.getLogitsOnCPU(logits, len(decodeSeqs)*vocabSize)

			for i, seq := range decodeSeqs {
				seq.AdvancePosition()

				if seq.State() == StatePending {
					seq.SetState(StateDecoding)
				}

				// Extract logits for this sequence
				start := i * vocabSize
				end := start + vocabSize

				if logitsData == nil || end > len(logitsData) {
					seq.PushToken("?")
					continue
				}

				seqLogits := logitsData[start:end]

				// Sample — use per-sequence sampler if custom params set
				var tokenID int
				if seq.HasSamplingParams() {
					seqSampler := sampler.New(sampler.Config{
						Temperature: seq.SamplingTemperature(),
						TopK:        seq.SamplingTopK(),
						TopP:        seq.SamplingTopP(),
					}, int64(seq.ID()))
					tokenID = seqSampler.Sample(seqLogits)
				} else {
					tokenID = s.sampler.Sample(seqLogits)
				}
				seq.AddGeneratedToken(tokenID)

				// Check for EOS or extra stop tokens
				if s.isStopToken(tokenID, seq) {
					seq.SetState(StateFinished)
					seq.Close()
					continue
				}

				// Check max tokens
				if seq.ReachedMaxTokens(s.config.MaxTokens) {
					seq.SetState(StateFinished)
					seq.Close()
					continue
				}

				// Decode token to text
				var text string
				if s.tokenizer != nil {
					text, _ = s.tokenizer.Decode([]int{tokenID})
				} else {
					text = fmt.Sprintf(" %d", tokenID)
				}

				seq.PushToken(text)
			}
		}
	}

	// Update metrics at the very end
	s.mu.Lock()
	s.metrics.TotalTokens += len(decodeSeqs)
	s.metrics.DecodeTokens += len(decodeSeqs)
	s.metrics.DecodeTime += time.Since(startTime)
	s.mu.Unlock()

	return nil
}

// runBatchedPrefill processes all prompt tokens for a sequence in one forward pass.
func (s *Scheduler) runBatchedPrefill(seq *Sequence) error {
	promptTokens := seq.PromptTokens()
	if len(promptTokens) == 0 {
		seq.SetState(StateDecoding)
		return nil
	}

	// Run prefill with all tokens at once
	startTime := time.Now()
	logits, err := s.runtime.PrefillWithPagedKV(promptTokens, seq.KVSeqID(), 0)

	if err != nil {
		return fmt.Errorf("prefill failed: %w", err)
	}

	// Mark all prompt tokens as processed
	seq.SetPrefillComplete(len(promptTokens))

	// Sample from the logits (which are for the last prompt token)
	// Copy logits from GPU to CPU if needed
	// For multi-token prefill, logits may have shape [seqLen, vocabSize]
	// We need to extract the LAST row (last token's logits)
	vocabSize := s.runtime.Config().VocabSize
	numLogitElements := logits.NumElements()
	logitsData := s.getLogitsOnCPU(logits, numLogitElements)

	if logitsData == nil || len(logitsData) < vocabSize {
		seq.SetState(StateDecoding)
		return nil
	}

	// Extract last row of logits (for multi-token prefill, logits shape is [seqLen, vocabSize])
	seqLogits := logitsData[len(logitsData)-vocabSize:]

	// Sample first generated token
	tokenID := s.sampler.Sample(seqLogits)
	seq.AddGeneratedToken(tokenID)

	// Update metrics (including sampling/sync time)
	prefillTime := time.Since(startTime)
	s.mu.Lock()
	s.metrics.TotalTokens += len(promptTokens)
	s.metrics.PrefillTokens += len(promptTokens)
	s.metrics.PrefillTime += prefillTime
	s.mu.Unlock()

	// Check for EOS or extra stop tokens
	if s.isStopToken(tokenID, seq) {
		seq.SetState(StateFinished)
		seq.Close()
		return nil
	}

	// Decode and push token
	var text string
	if s.tokenizer != nil {
		text, _ = s.tokenizer.Decode([]int{tokenID})
	} else {
		text = fmt.Sprintf(" %d", tokenID)
	}
	if debugDecode {
		fmt.Printf("[SAMPLE-PREFILL] token %d -> %q\n", tokenID, text)
	}
	seq.PushToken(text)

	// Transition to decode state
	seq.SetState(StateDecoding)
	return nil
}

// runGPUPrefill processes all prompt tokens for a sequence using GPU KV cache.
func (s *Scheduler) runGPUPrefill(seq *Sequence) error {
	promptTokens := seq.PromptTokens()
	if len(promptTokens) == 0 {
		seq.SetState(StateDecoding)
		return nil
	}

	// Reset GPU KV cache for new sequence
	if cache := s.runtime.GPUKVCache(); cache != nil {
		cache.Reset()
	}

	// Run prefill: either batched (all tokens at once) or sequential (one at a time).
	// Sequential mode ensures the SDPA decode kernel is used consistently,
	// avoiding FP32 precision divergence between SDPAPrefill and SDPA decode
	// that compounds for models with large QKV bias (e.g., Qwen).
	startTime := time.Now()
	var logits tensor.Tensor
	var err error

	useSequentialPrefill := os.Getenv("VEXEL_SEQUENTIAL_PREFILL") == "1"
	if useSequentialPrefill {
		// Process each token individually using the decode (M=1) path
		for i, tok := range promptTokens {
			logits, err = s.runtime.DecodeWithGPUKV([]int{tok}, i)
			if err != nil {
				return fmt.Errorf("sequential prefill token %d failed: %w", i, err)
			}
		}
	} else {
		logits, err = s.runtime.DecodeWithGPUKV(promptTokens, 0)
	}

	if err != nil {
		return fmt.Errorf("GPU prefill failed: %w", err)
	}

	// Mark all prompt tokens as processed
	seq.SetPrefillComplete(len(promptTokens))

	// Sample from the logits (which are for the last prompt token)
	vocabSize := s.runtime.Config().VocabSize
	numLogitElements := logits.NumElements()
	logitsData := s.getLogitsOnCPU(logits, numLogitElements)

	if logitsData == nil || len(logitsData) < vocabSize {
		seq.SetState(StateDecoding)
		return nil
	}

	// Extract last row of logits (for multi-token prefill, logits shape is [seqLen, vocabSize])
	seqLogits := logitsData[len(logitsData)-vocabSize:]

	// Sample first generated token
	tokenID := s.sampler.Sample(seqLogits)
	seq.AddGeneratedToken(tokenID)

	// Update metrics (including sampling/sync time)
	prefillTime := time.Since(startTime)
	s.mu.Lock()
	s.metrics.TotalTokens += len(promptTokens)
	s.metrics.PrefillTokens += len(promptTokens)
	s.metrics.PrefillTime += prefillTime
	s.mu.Unlock()

	// Check for EOS or extra stop tokens
	if s.isStopToken(tokenID, seq) {
		seq.SetState(StateFinished)
		seq.Close()
		return nil
	}

	// Decode and push token
	var text string
	if s.tokenizer != nil {
		text, _ = s.tokenizer.Decode([]int{tokenID})
	} else {
		text = fmt.Sprintf(" %d", tokenID)
	}
	seq.PushToken(text)

	// Transition to decode state
	seq.SetState(StateDecoding)
	return nil
}

// AddSequence registers a new sequence with the scheduler.
// If a tokenizer is available, encodes the prompt into tokens.
func (s *Scheduler) AddSequence(seq *Sequence) {
	// Encode prompt if tokenizer available and prompt not yet encoded
	if s.tokenizer != nil && len(seq.PromptTokens()) == 0 && seq.prompt != "" {
		tokens, err := s.tokenizer.Encode(seq.prompt)
		if err == nil {
			// Prepend BOS token only for models that expect it (Llama, not Phi-2)
			if s.tokenizer.AddBOS() {
				bos := s.tokenizer.BOS()
				tokens = append([]int{bos}, tokens...)
			}
			seq.SetPromptTokens(tokens)
		}
	}

	// Create sequence in PagedKVCache if available
	if cache := s.runtime.PagedKVCache(); cache != nil {
		kvSeqID := cache.CreateSequence()
		seq.SetKVSeqID(kvSeqID)
	}

	// Create sequence in GPU block pool if available
	if pool := s.runtime.GetGPUBlockPool(); pool != nil && seq.KVSeqID() != 0 {
		pool.CreateSequence(seq.KVSeqID())
	}

	s.mu.Lock()
	s.sequences[seq.ID()] = seq
	s.mu.Unlock()

	// Notify scheduler loop
	s.wakeUp()
}

// RemoveSequence removes a sequence and cleans up its KV cache.
func (s *Scheduler) RemoveSequence(id SequenceID) {
	s.mu.Lock()
	seq, ok := s.sequences[id]
	if !ok {
		s.mu.Unlock()
		return
	}
	delete(s.sequences, id)
	s.metrics.CompletedSequences++
	s.mu.Unlock()

	// Clean up GPU block pool (must come before PagedKVCache cleanup)
	if pool := s.runtime.GetGPUBlockPool(); pool != nil && seq.KVSeqID() != 0 {
		pool.DeleteSequence(seq.KVSeqID())
	}

	// Clean up KV cache
	if cache := s.runtime.PagedKVCache(); cache != nil && seq.KVSeqID() != 0 {
		cache.DeleteSequence(seq.KVSeqID())
	}
}

// SequenceCount returns the number of active sequences.
func (s *Scheduler) SequenceCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.sequences)
}

// GetSequences returns all active sequences. Used for testing.
func (s *Scheduler) GetSequences() []*Sequence {
	s.mu.RLock()
	defer s.mu.RUnlock()
	seqs := make([]*Sequence, 0, len(s.sequences))
	for _, seq := range s.sequences {
		seqs = append(seqs, seq)
	}
	return seqs
}

// Metrics returns the current performance metrics.
func (s *Scheduler) Metrics() SchedulerMetrics {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.metrics
}

// GPUMemoryStats returns GPU block pool memory usage in MB.
// Returns (0, 0, 0) if no GPU block pool is available.
func (s *Scheduler) GPUMemoryStats() (totalMB, usedMB, freeMB float64) {
	if pool := s.runtime.GetGPUBlockPool(); pool != nil {
		return pool.MemoryStats()
	}
	return 0, 0, 0
}

// ModelConfig returns the loaded model's configuration.
// Returns a zero-value config if no model is loaded.
func (s *Scheduler) ModelConfig() runtime.ModelConfig {
	if s.runtime == nil {
		return runtime.ModelConfig{}
	}
	return s.runtime.Config()
}

// getLogitsOnCPU returns logits as a float32 slice on CPU.
// If the logits are on GPU, it copies them to host first.
// NOTE: Caller must ensure GPU work is complete (e.g., Decode functions call Sync).
func (s *Scheduler) getLogitsOnCPU(logits tensor.Tensor, numElements int) []float32 {
	ptr := logits.DevicePtr()
	if ptr.IsNil() {
		return nil
	}

	// If already on CPU, use unsafe slice directly
	if ptr.Location() == tensor.CPU {
		return tensor.ToFloat32Slice(logits)
	}

	// GPU: need to copy to host
	// Get backend with ToHost capability
	backend := s.runtime.Backend()
	if backend == nil {
		return nil
	}

	// Allocate host buffer as float32 directly to avoid byte-to-float conversion
	result := make([]float32, numElements)
	hostData := unsafe.Slice((*byte)(unsafe.Pointer(&result[0])), numElements*4)
	// NOTE: Sync is NOT needed here - the Decode functions already call Sync()
	// before returning, ensuring GPU work is complete.
	backend.ToHost(hostData, ptr)

	// Apply final logit soft-capping (Gemma 2: cap * tanh(logits / cap), cap=30.0)
	s.runtime.ApplyFinalLogitSoftCap(result)

	return result
}

// sampleToken samples a token from logits, using GPU argmax for greedy (temp=0)
// or falling back to CPU sampling for other temperatures.
// logits: the logits tensor on GPU
// vocabSize: number of vocabulary entries
// Returns the sampled token ID.
func (s *Scheduler) sampleToken(logits tensor.Tensor, vocabSize int) int {
	// Sync GPU before sampling to ensure logits are ready
	s.runtime.Backend().Sync()

	// For greedy sampling (temp=0), try GPU argmax to avoid 128KB transfer
	if s.config.SamplerConfig.Temperature == 0 {
		if argmax, ok := s.runtime.Backend().(backend.ArgmaxOps); ok {
			ptr := logits.DevicePtr()
			if !ptr.IsNil() && ptr.Location() != tensor.CPU {
				// Get offset to last vocab-size elements (for multi-token prefill)
				numElements := logits.NumElements()
				if numElements >= vocabSize {
					// Create a DevicePtr that points to the last vocabSize elements
					offset := uintptr((numElements - vocabSize) * 4) // 4 bytes per float32
					lastRowPtr := tensor.DevicePtrOffset(ptr, offset)
					tokenID := argmax.Argmax(lastRowPtr, vocabSize)
					if debugDecode {
						text := ""
						if s.tokenizer != nil {
							text, _ = s.tokenizer.Decode([]int{tokenID})
						}
						fmt.Printf("[SAMPLE] GPU argmax -> %d (%q)\n", tokenID, text)
					}
					return tokenID
				}
			}
		}
	}

	// Fall back to CPU sampling
	numElements := logits.NumElements()
	logitsData := s.getLogitsOnCPU(logits, numElements)
	if logitsData == nil || len(logitsData) < vocabSize {
		return 0
	}
	seqLogits := logitsData[len(logitsData)-vocabSize:]
	return s.sampler.Sample(seqLogits)
}

// float32FromBits converts uint32 bits to float32.
func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}
