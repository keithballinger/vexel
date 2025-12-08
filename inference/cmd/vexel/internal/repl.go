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
			// Submit
			seqID := scheduler.SequenceID(time.Now().UnixNano())
			seq := scheduler.NewSequence(seqID, prompt)
			sched.AddSequence(seq)

			// Stream response
			for token := range seq.TokenChan() {
				fmt.Fprint(w, token)
			}
			fmt.Fprintln(w)
		} else {
			fmt.Fprintln(w, "Error: Scheduler not available")
			return fmt.Errorf("scheduler nil")
		}
	}
}
