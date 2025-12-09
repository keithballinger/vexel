package internal

import (
	"bufio"
	"fmt"
	"io"
	"strings"
	"time"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/scheduler"
)

// SchedulerInterface defines what we need from the scheduler.
type SchedulerInterface interface {
	AddSequence(seq *scheduler.Sequence)
	Metrics() scheduler.SchedulerMetrics
}

// REPLConfig configures the chat REPL.
type REPLConfig struct {
	ChatMode     bool   // Use chat template formatting
	SystemPrompt string // System prompt for chat mode
}

// DefaultREPLConfig returns default configuration.
func DefaultREPLConfig() REPLConfig {
	return REPLConfig{
		ChatMode:     true,
		SystemPrompt: "You are a helpful assistant.",
	}
}

// RunChatLoop starts the REPL with default config.
func RunChatLoop(r io.Reader, w io.Writer, sched SchedulerInterface) error {
	return RunChatLoopWithConfig(r, w, sched, DefaultREPLConfig())
}

// RunChatLoopWithConfig starts the REPL with custom config.
func RunChatLoopWithConfig(r io.Reader, w io.Writer, sched SchedulerInterface, config REPLConfig) error {
	scanner := bufio.NewScanner(r)
	template := tokenizer.TinyLlamaChatTemplate()

	if config.ChatMode {
		fmt.Fprintln(w, "Chat mode enabled. Type 'exit' to quit.")
	}

	for {
		fmt.Fprint(w, ">> ")
		if !scanner.Scan() {
			return nil
		}

		text := strings.TrimSpace(scanner.Text())
		if text == "Exit" || text == "exit" || text == "" {
			return nil
		}

		// Format prompt based on mode
		var prompt string
		if config.ChatMode {
			prompt = template.FormatChat(config.SystemPrompt, text)
		} else {
			prompt = text
		}

		if sched != nil {
			// Get metrics before
			metricsBefore := sched.Metrics()

			// Submit
			seqID := scheduler.SequenceID(time.Now().UnixNano())
			seq := scheduler.NewSequence(seqID, prompt)
			sched.AddSequence(seq)

			// Stream response
			tokenCount := 0
			for token := range seq.TokenChan() {
				fmt.Fprint(w, token)
				tokenCount++
			}
			fmt.Fprintln(w)

			// Get metrics after and compute tok/s for this generation
			metricsAfter := sched.Metrics()
			newDecodeTokens := metricsAfter.DecodeTokens - metricsBefore.DecodeTokens
			newDecodeTime := metricsAfter.DecodeTime - metricsBefore.DecodeTime
			newPrefillTokens := metricsAfter.PrefillTokens - metricsBefore.PrefillTokens
			newPrefillTime := metricsAfter.PrefillTime - metricsBefore.PrefillTime

			// Print performance stats
			if newDecodeTime > 0 && newDecodeTokens > 0 {
				decodeTokS := float64(newDecodeTokens) / newDecodeTime.Seconds()
				prefillTokS := float64(0)
				if newPrefillTime > 0 {
					prefillTokS = float64(newPrefillTokens) / newPrefillTime.Seconds()
				}
				fmt.Fprintf(w, "[%d tokens | prefill: %.1f tok/s | decode: %.1f tok/s]\n",
					tokenCount, prefillTokS, decodeTokS)
			}
		} else {
			fmt.Fprintln(w, "Error: Scheduler not available")
			return fmt.Errorf("scheduler nil")
		}
	}
}
