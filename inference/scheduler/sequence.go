package scheduler

// SequenceID uniquely identifies a sequence.
type SequenceID int64

// SequenceState represents the current lifecycle state of a sequence.
type SequenceState int

const (
	StatePending SequenceState = iota
	StatePrefill
	StateDecoding
	StateFinished
)

func (s SequenceState) String() string {
	switch s {
	case StatePending:
		return "Pending"
	case StatePrefill:
		return "Prefill"
	case StateDecoding:
		return "Decoding"
	case StateFinished:
		return "Finished"
	default:
		return "Unknown"
	}
}

// Sequence represents a single generation request.
type Sequence struct {
	id     SequenceID
	prompt string
	state  SequenceState
	
	// tokenChan creates a stream of generated tokens back to the caller.
	tokenChan chan string
}

// NewSequence creates a new sequence in the Pending state.
func NewSequence(id SequenceID, prompt string) *Sequence {
	return &Sequence{
		id:        id,
		prompt:    prompt,
		state:     StatePending,
		tokenChan: make(chan string, 100), // Buffer to prevent blocking scheduler
	}
}

// ID returns the sequence ID.
func (s *Sequence) ID() SequenceID {
	return s.id
}

// State returns the current state.
func (s *Sequence) State() SequenceState {
	return s.state
}

// SetState updates the sequence state.
func (s *Sequence) SetState(newState SequenceState) {
	s.state = newState
}

// TokenChan returns the channel for receiving generated tokens.
func (s *Sequence) TokenChan() <-chan string {
	return s.tokenChan
}

// PushToken adds a token to the stream.
func (s *Sequence) PushToken(token string) {
	select {
	case s.tokenChan <- token:
	default:
		// Drop token if channel full? Or block?
		// Blocking scheduler is bad. Drop or expand buffer.
		// For now, drop implies user is too slow.
	}
}

// Close closes the token stream.
func (s *Sequence) Close() {
	close(s.tokenChan)
}