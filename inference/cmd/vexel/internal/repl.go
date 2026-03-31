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
	ModelPath    string // Model file path for template detection
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

	// Select chat template based on model path, or use default
	var template tokenizer.ChatTemplate
	if config.ModelPath != "" {
		template = tokenizer.DetectChatTemplate(config.ModelPath)
	} else {
		template = tokenizer.DefaultChatTemplate()
	}

	// Initialize conversation history
	var history []tokenizer.ChatMessage
	if config.SystemPrompt != "" && config.ChatMode {
		history = append(history, tokenizer.ChatMessage{Role: "system", Content: config.SystemPrompt})
	}

	if config.ChatMode {
		fmt.Fprintf(w, "Chat mode enabled (template: %s). Type /clear to reset, exit to quit.\n", template.Name)
	}

	for {
		fmt.Fprint(w, ">>> ")
		if !scanner.Scan() {
			return nil
		}

		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		if text == "exit" || text == "quit" {
			return nil
		}

		// Handle /clear command
		if text == "/clear" {
			history = nil
			if config.SystemPrompt != "" && config.ChatMode {
				history = append(history, tokenizer.ChatMessage{Role: "system", Content: config.SystemPrompt})
			}
			fmt.Fprintln(w, "Conversation history cleared.")
			continue
		}

		// Format prompt based on mode
		var prompt string
		if config.ChatMode {
			// Append user message to history
			history = append(history, tokenizer.ChatMessage{Role: "user", Content: text})
			// Format full conversation
			prompt = template.FormatConversation(history)
		} else {
			prompt = text
		}

		if sched != nil {
			// Get metrics before
			metricsBefore := sched.Metrics()

			// Submit
			seqID := scheduler.SequenceID(time.Now().UnixNano())
			seq := scheduler.NewSequence(seqID, prompt)
			if config.ChatMode && len(template.ExtraStopTokenIDs) > 0 {
				seq.SetStopTokens(template.ExtraStopTokenIDs)
			}
			sched.AddSequence(seq)

			// Stream response and collect full text
			var responseBuf strings.Builder
			tokenCount := 0
			for token := range seq.TokenChan() {
				fmt.Fprint(w, token)
				responseBuf.WriteString(token)
				tokenCount++
			}
			fmt.Fprintln(w)

			// Append assistant response to history
			if config.ChatMode {
				response := strings.TrimSpace(responseBuf.String())
				history = append(history, tokenizer.ChatMessage{Role: "assistant", Content: response})
			}

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
