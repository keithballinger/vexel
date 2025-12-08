package internal

import (
	"bufio"
	"fmt"
	"io"
	"strings"
	"time"
	"vexel/inference/scheduler"
)

// SchedulerInterface defines what we need from the scheduler.
type SchedulerInterface interface {
	AddSequence(seq *scheduler.Sequence)
}

// RunChatLoop starts the REPL.
func RunChatLoop(r io.Reader, w io.Writer, sched SchedulerInterface) error {
	scanner := bufio.NewScanner(r)
	
	for {
		fmt.Fprint(w, ">> ")
		if !scanner.Scan() {
			return nil
		}
		
		text := strings.TrimSpace(scanner.Text())
		if text == "Exit" || text == "exit" || text == "" {
			return nil
		}
		
		if sched != nil {
			// Submit
			seqID := scheduler.SequenceID(time.Now().UnixNano())
			seq := scheduler.NewSequence(seqID, text)
			sched.AddSequence(seq)
			
			// Stream response
			for token := range seq.TokenChan() {
				fmt.Fprint(w, token)
				// For mock, we break manually or wait for close?
				// Our scheduler currently Pushes forever.
				// We need a break condition for the demo.
				// Let's print 10 tokens and stop.
				// Or assume Sequence handles EOS.
			}
			fmt.Fprintln(w)
		} else {
			fmt.Fprintln(w, "Error: Scheduler not available")
			return fmt.Errorf("scheduler nil")
		}
	}
}
