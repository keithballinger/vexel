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
}

// NewSequence creates a new sequence in the Pending state.
func NewSequence(id SequenceID, prompt string) *Sequence {
	return &Sequence{
		id:     id,
		prompt: prompt,
		state:  StatePending,
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
	// Simple transition logic for now.
	// In the future, we might enforce valid transitions (e.g., Finished -> Pending is invalid).
	s.state = newState
}
