// Package scheduler provides an event-driven inference scheduler for managing
// concurrent LLM generation requests with continuous batching.
//
// The scheduler handles the full lifecycle of generation requests (sequences):
// prompt tokenization, batched prefill, autoregressive decode, sampling, and
// streaming tokens back to callers.
//
// # Architecture
//
// The scheduler operates as an event loop driven by incoming sequences:
//
//  1. A caller creates a [Sequence] with [NewSequence] and submits it via [Scheduler.AddSequence].
//  2. The scheduler tokenizes the prompt, runs batched prefill on the GPU,
//     and transitions the sequence to decode state.
//  3. During decode, the scheduler generates tokens one at a time, applies
//     sampling (temperature, top-k, top-p), and pushes decoded text to the
//     sequence's token channel.
//  4. The caller reads tokens from [Sequence.TokenChan] until the channel closes.
//
// # Usage
//
// Create a scheduler with a model runtime and tokenizer:
//
//	sched, err := scheduler.NewScheduler(model, tok, scheduler.Config{
//	    MaxBatchSize:  1,
//	    MaxSequences:  64,
//	    MaxTokens:     256,
//	    SamplerConfig: sampler.DefaultConfig(),
//	})
//
//	// Start the scheduler loop (blocks until context is canceled)
//	go sched.Run(ctx)
//
//	// Submit a generation request
//	seq := scheduler.NewSequence(id, "Hello, world!")
//	sched.AddSequence(seq)
//
//	// Stream tokens as they are generated
//	for token := range seq.TokenChan() {
//	    fmt.Print(token)
//	}
//
// # Metrics
//
// The scheduler tracks performance metrics accessible via [Scheduler.Metrics]:
//
//	m := sched.Metrics()
//	fmt.Printf("Decode: %.1f tok/s\n", m.TokensPerSecond())
//	fmt.Printf("Prefill: %.1f tok/s\n", m.PrefillTokensPerSecond())
//
// # Sequence Lifecycle
//
// Each [Sequence] progresses through states: Pending -> Prefill -> Decoding -> Finished.
// The scheduler manages these transitions automatically. Sequences can be removed
// early via [Scheduler.RemoveSequence].
package scheduler
