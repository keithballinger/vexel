package scheduler

import "fmt"

// SequenceID uniquely identifies a sequence.
type SequenceID int64

func (id SequenceID) String() string {
	return fmt.Sprintf("%d", id)
}

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

	// promptTokens is the encoded prompt (set by scheduler when tokenizer available).
	promptTokens []int
	// promptPos tracks how many prompt tokens have been processed.
	promptPos int
	// position is the current sequence position (for RoPE).
	// After processing N tokens, position = N.
	position int
	// generatedTokens stores the IDs of tokens we've generated.
	generatedTokens []int

	// kvSeqID is the sequence ID in the PagedKVCache.
	kvSeqID int64

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

// SetPromptTokens sets the encoded prompt tokens.
func (s *Sequence) SetPromptTokens(tokens []int) {
	s.promptTokens = tokens
}

// PromptTokens returns the encoded prompt tokens.
func (s *Sequence) PromptTokens() []int {
	return s.promptTokens
}

// Position returns the current sequence position (for RoPE).
func (s *Sequence) Position() int {
	return s.position
}

// NextInputToken returns the next token to feed to the model.
// For prefill, returns prompt tokens one at a time.
// For decode, returns the last generated token.
// Returns (token, position, hasMore).
func (s *Sequence) NextInputToken() (token int, pos int, hasMore bool) {
	if s.promptPos < len(s.promptTokens) {
		// Still processing prompt
		token = s.promptTokens[s.promptPos]
		pos = s.promptPos
		hasMore = true
		return
	}
	if len(s.generatedTokens) > 0 {
		// Return last generated token for continuation
		token = s.generatedTokens[len(s.generatedTokens)-1]
		pos = s.position
		hasMore = true
		return
	}
	// No tokens available
	return 0, 0, false
}

// AdvancePosition increments position and promptPos after processing a token.
func (s *Sequence) AdvancePosition() {
	if s.promptPos < len(s.promptTokens) {
		s.promptPos++
	}
	s.position++
}

// AddGeneratedToken records a newly generated token.
func (s *Sequence) AddGeneratedToken(tokenID int) {
	s.generatedTokens = append(s.generatedTokens, tokenID)
}

// GeneratedTokens returns all generated token IDs.
func (s *Sequence) GeneratedTokens() []int {
	return s.generatedTokens
}

// IsPrefillComplete returns true if all prompt tokens have been processed.
func (s *Sequence) IsPrefillComplete() bool {
	return s.promptPos >= len(s.promptTokens)
}

// SetPrefillComplete marks all prompt tokens as processed and sets position.
// Used by batched prefill to skip token-by-token processing.
func (s *Sequence) SetPrefillComplete(numTokens int) {
	s.promptPos = len(s.promptTokens)
	s.position = numTokens
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

// KVSeqID returns the sequence ID in the PagedKVCache.
func (s *Sequence) KVSeqID() int64 {
	return s.kvSeqID
}

// SetKVSeqID sets the sequence ID in the PagedKVCache.
func (s *Sequence) SetKVSeqID(id int64) {
	s.kvSeqID = id
}