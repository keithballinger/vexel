package scheduler

import (
	"context"
	"fmt"
	"math"
	"time"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

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
type Scheduler struct {
	runtime   *runtime.ModelRuntime
	tokenizer *tokenizer.Tokenizer
	sampler   *sampler.Sampler
	config    Config
	sequences map[SequenceID]*Sequence
	metrics   SchedulerMetrics
}

// NewScheduler creates a new Scheduler instance.
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
	}, nil
}

// Run starts the scheduler's main loop.
// It blocks until the context is canceled or a fatal error occurs.
func (s *Scheduler) Run(ctx context.Context) error {
	// Simple ticker for now, maybe event-driven later
	ticker := time.NewTicker(1 * time.Millisecond) // aggressive poll
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := s.step(ctx); err != nil {
				return err
			}
		}
	}
}

// step performs a single scheduling iteration.
func (s *Scheduler) step(ctx context.Context) error {
	// 1. Collect ready sequences
	ready := s.collectReady()

	// Update Active Sequences metric
	s.metrics.ActiveSequences = len(s.sequences)

	// 2. Form batch
	batch := s.formBatches(ready)
	if len(batch) == 0 {
		return nil
	}

	// 3. Run DecodeStep
	if err := s.runDecodeStep(ctx, batch); err != nil {
		return err
	}

	return nil
}

// collectReady identifies sequences that are eligible for execution.
func (s *Scheduler) collectReady() []*Sequence {
	ready := make([]*Sequence, 0)
	for _, seq := range s.sequences {
		switch seq.State() {
		case StatePending, StatePrefill, StateDecoding:
			ready = append(ready, seq)
		}
	}
	return ready
}

// formBatches selects a subset of ready sequences to run in the next step.
func (s *Scheduler) formBatches(ready []*Sequence) []*Sequence {
	if len(ready) == 0 {
		return nil
	}
	
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
				continue
			} else if usePagedCache {
				if err := s.runBatchedPrefill(seq); err != nil {
					return err
				}
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
					eosToken := 2
					if s.tokenizer != nil {
						eosToken = s.tokenizer.EOS()
					}
					if tokenID == eosToken {
						seq.SetState(StateFinished)
						seq.Close()
					} else if s.config.MaxTokens > 0 && len(seq.GeneratedTokens()) >= s.config.MaxTokens {
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
					eosToken := 2
					if s.tokenizer != nil {
						eosToken = s.tokenizer.EOS()
					}
					if tokenID == eosToken {
						seq.SetState(StateFinished)
						seq.Close()
					} else if s.config.MaxTokens > 0 && len(seq.GeneratedTokens()) >= s.config.MaxTokens {
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
			// Fallback for batches - process one at a time
			// TODO: batch decode with paged KV
			for i, seq := range decodeSeqs {
				singleLogits, singleErr := s.runtime.DecodeWithPagedKV([]int{tokens[i]}, seqIDs[i], positions[i])
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
					eosToken := 2
					if s.tokenizer != nil {
						eosToken = s.tokenizer.EOS()
					}
					if tokenID == eosToken {
						seq.SetState(StateFinished)
						seq.Close()
					} else if s.config.MaxTokens > 0 && len(seq.GeneratedTokens()) >= s.config.MaxTokens {
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
			// Fall through
		}
	} else {
		inputs := runtime.NewBatchRuntimeInputsWithPos(tokens, positions, nil)
		logits, err = s.runtime.DecodeStep(inputs)
	}

	if err != nil {
		return err
	}

	// For batched GPU/Paged paths, sampling is already done above.
	// We only need to sample here if we took the single-sequence path OR the BatchRuntimeInputs path.
	// But wait, the single-sequence GPU path (len=1) falls through here and needs sampling.
	// The multi-sequence GPU path (loop) did sampling inside loop.
	// This structure is messy.
	// Let's rely on `logits` being set. If `logits` is set, we sample.
	// But in loop cases, `logits` is set to first result?
	// And we already sampled.
	// We need to avoid double sampling.

	// Refactor: Only sample here if we didn't sample in loop.
	alreadySampled := (useGPUCache && len(decodeSeqs) > 1) || (usePagedCache && len(decodeSeqs) > 1)

	if !alreadySampled {
		// Sample and Decode
		// Copy logits from GPU to CPU if needed
		vocabSize := s.runtime.Config().VocabSize
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

			// Sample
			tokenID := s.sampler.Sample(seqLogits)
			seq.AddGeneratedToken(tokenID)

			// Check for EOS
			eosToken := 2
			if s.tokenizer != nil {
				eosToken = s.tokenizer.EOS()
			}
			if tokenID == eosToken {
				seq.SetState(StateFinished)
				seq.Close()
				continue
			}

			// Check max tokens
			if s.config.MaxTokens > 0 && len(seq.GeneratedTokens()) >= s.config.MaxTokens {
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

	// Update metrics at the very end
	s.metrics.TotalTokens += len(decodeSeqs)
	s.metrics.DecodeTokens += len(decodeSeqs)
	s.metrics.DecodeTime += time.Since(startTime)

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
	s.metrics.TotalTokens += len(promptTokens)
	s.metrics.PrefillTokens += len(promptTokens)
	s.metrics.PrefillTime += prefillTime

	// Check for EOS
	eosToken := 2
	if s.tokenizer != nil {
		eosToken = s.tokenizer.EOS()
	}
	if tokenID == eosToken {
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

	// Run prefill with all tokens at once using GPU KV cache
	startTime := time.Now()
	logits, err := s.runtime.DecodeWithGPUKV(promptTokens, 0)

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
	s.metrics.TotalTokens += len(promptTokens)
	s.metrics.PrefillTokens += len(promptTokens)
	s.metrics.PrefillTime += prefillTime

	// Check for EOS
	eosToken := 2
	if s.tokenizer != nil {
		eosToken = s.tokenizer.EOS()
	}
	if tokenID == eosToken {
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
			// Prepend BOS token - LLMs expect BOS at start of sequence
			bos := s.tokenizer.BOS()
			tokens = append([]int{bos}, tokens...)
			seq.SetPromptTokens(tokens)
		}
	}

	// Create sequence in PagedKVCache if available
	if cache := s.runtime.PagedKVCache(); cache != nil {
		kvSeqID := cache.CreateSequence()
		seq.SetKVSeqID(kvSeqID)
	}

	s.sequences[seq.ID()] = seq
}

// RemoveSequence removes a sequence and cleans up its KV cache.
func (s *Scheduler) RemoveSequence(id SequenceID) {
	seq, ok := s.sequences[id]
	if !ok {
		return
	}

	// Clean up KV cache
	if cache := s.runtime.PagedKVCache(); cache != nil && seq.KVSeqID() != 0 {
		cache.DeleteSequence(seq.KVSeqID())
	}

	delete(s.sequences, id)
	s.metrics.CompletedSequences++
}

// SequenceCount returns the number of active sequences.
func (s *Scheduler) SequenceCount() int {
	return len(s.sequences)
}

// Metrics returns the current performance metrics.
func (s *Scheduler) Metrics() SchedulerMetrics {
	return s.metrics
}

// getLogitsOnCPU returns logits as a float32 slice on CPU.
// If the logits are on GPU, it copies them to host first.
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

	// Allocate host buffer
	hostData := make([]byte, numElements*4)
	backend.Sync() // Wait for GPU before reading shared memory
	backend.ToHost(hostData, ptr)

	// Convert bytes to float32
	result := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		bits := uint32(hostData[i*4]) | uint32(hostData[i*4+1])<<8 | uint32(hostData[i*4+2])<<16 | uint32(hostData[i*4+3])<<24
		result[i] = float32FromBits(bits)
	}
	return result
}

// float32FromBits converts uint32 bits to float32.
func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}