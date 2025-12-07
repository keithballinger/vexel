package memory

// ArenaKind represents the type of memory arena.
// Different kinds have different lifecycles and allocation strategies.
type ArenaKind int

const (
	// Weights represents static model weights.
	Weights ArenaKind = iota
	// KV represents the Key-Value cache for attention.
	KV
	// Scratch represents temporary workspace for activations.
	Scratch
)

func (k ArenaKind) String() string {
	switch k {
	case Weights:
		return "Weights"
	case KV:
		return "KV"
	case Scratch:
		return "Scratch"
	default:
		return "Unknown"
	}
}
