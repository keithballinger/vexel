package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Tokenizer handles text <-> id conversion.
type Tokenizer struct {
	vocab         map[string]int
	ids           map[int]string
	bos           int      // Beginning of sequence token
	eos           int      // End of sequence token
	specialTokens []string // Special tokens to match before BPE (longest first)
}

// Load loads a tokenizer from a file (tokenizer.json).
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Parse simplified structure.
	// Real tokenizer.json is complex (normalizers, pre-tokenizers, model).
	// We look for model.vocab.
	
	var data struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}

	if err := json.NewDecoder(f).Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode tokenizer.json: %w", err)
	}

	ids := make(map[int]string)
	for k, v := range data.Model.Vocab {
		ids[v] = k
	}

	// Default special token IDs (Llama convention)
	bos := 1
	eos := 2

	// Collect special tokens and find BOS/EOS
	var specialTokens []string
	for _, tok := range data.AddedTokens {
		switch tok.Content {
		case "<s>":
			bos = tok.ID
		case "</s>":
			eos = tok.ID
		}
		// Add all added_tokens to vocab and special tokens list
		if _, exists := data.Model.Vocab[tok.Content]; !exists {
			data.Model.Vocab[tok.Content] = tok.ID
			ids[tok.ID] = tok.Content
		}
		specialTokens = append(specialTokens, tok.Content)
	}

	// Sort special tokens by length (longest first) for greedy matching
	sort.Slice(specialTokens, func(i, j int) bool {
		return len(specialTokens[i]) > len(specialTokens[j])
	})

	return &Tokenizer{
		vocab:         data.Model.Vocab,
		ids:           ids,
		bos:           bos,
		eos:           eos,
		specialTokens: specialTokens,
	}, nil
}

// Encode converts text to token IDs.
// This is a simplified implementation that tries:
// 1. Special token matching (exact match for special tokens)
// 2. Exact match for the whole string
// 3. Greedy longest-match tokenization
// 4. Byte-level fallback for unknown characters
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return nil, nil
	}

	// Split on special tokens first, then encode each segment
	segments := t.splitOnSpecialTokens(text)
	var tokens []int

	for i, seg := range segments {
		if seg.isSpecial {
			if id, ok := t.vocab[seg.text]; ok {
				tokens = append(tokens, id)
			}
		} else {
			// Each segment after a special token starts at a word boundary
			atWordBoundary := i == 0 || (i > 0 && segments[i-1].isSpecial)
			// isAbsoluteStart is true only for the very first segment (no prior tokens)
			isAbsoluteStart := i == 0 && len(tokens) == 0
			segTokens, _ := t.encodeSegment(seg.text, atWordBoundary, isAbsoluteStart)
			tokens = append(tokens, segTokens...)
		}
	}

	if len(tokens) == 0 {
		return []int{t.bos}, nil // Fallback to BOS
	}

	return tokens, nil
}

// segment represents a piece of text, either a special token or regular text
type segment struct {
	text      string
	isSpecial bool
}

// splitOnSpecialTokens splits text around special tokens
func (t *Tokenizer) splitOnSpecialTokens(text string) []segment {
	if len(t.specialTokens) == 0 {
		return []segment{{text: text, isSpecial: false}}
	}

	var segments []segment
	remaining := text

	for len(remaining) > 0 {
		// Find the earliest special token
		earliestIdx := -1
		earliestPos := len(remaining)
		var earliestToken string

		for _, st := range t.specialTokens {
			pos := strings.Index(remaining, st)
			if pos >= 0 && pos < earliestPos {
				earliestPos = pos
				earliestIdx = pos
				earliestToken = st
			}
		}

		if earliestIdx < 0 {
			// No more special tokens, add remaining text
			if len(remaining) > 0 {
				segments = append(segments, segment{text: remaining, isSpecial: false})
			}
			break
		}

		// Add text before special token
		if earliestPos > 0 {
			segments = append(segments, segment{text: remaining[:earliestPos], isSpecial: false})
		}

		// Add the special token
		segments = append(segments, segment{text: earliestToken, isSpecial: true})
		remaining = remaining[earliestPos+len(earliestToken):]
	}

	return segments
}

// encodeSegment encodes a segment of text (not containing special tokens)
// atWordBoundary indicates if this segment starts at a word boundary (start of text or after special token)
// isAbsoluteStart indicates if this is the very beginning of the input (no prior tokens)
func (t *Tokenizer) encodeSegment(text string, atWordBoundary bool, isAbsoluteStart bool) ([]int, error) {
	if text == "" {
		return nil, nil
	}

	// Try exact match first
	if id, ok := t.vocab[text]; ok {
		return []int{id}, nil
	}

	// Greedy longest-match tokenization
	var tokens []int
	remaining := text

	for len(remaining) > 0 {
		// Find longest matching token
		bestLen := 0
		bestID := -1

		// SentencePiece uses ▁ (U+2581) as word boundary marker.
		// At word boundaries (start of segment, after space), prefer ▁ prefixed tokens.
		const spUnderscore = "▁"
		spLen := len(spUnderscore) // 3 bytes in UTF-8

		isFirstToken := len(tokens) == 0
		isStart := isFirstToken && atWordBoundary
		hasLeadingSpace := len(remaining) > 0 && remaining[0] == ' '

		// At absolute start of input with special chars like <, try plain text match FIRST (no ▁ prefix)
		// This is important for chat templates like <|system|> which shouldn't have leading space
		startsWithSpecialChar := len(remaining) > 0 && (remaining[0] == '<' || remaining[0] == '[' || remaining[0] == '{')
		if isFirstToken && isAbsoluteStart && !hasLeadingSpace && startsWithSpecialChar {
			for l := min(len(remaining), 20); l > 0; l-- {
				candidate := remaining[:l]
				if id, ok := t.vocab[candidate]; ok {
					bestLen = l
					bestID = id
					break
				}
			}
		}

		// At word boundary, try ▁ prefix (preferred for SentencePiece)
		// BUT: don't add ▁ prefix if starting with newline or special chars at absolute start
		startsWithNewline := len(remaining) > 0 && remaining[0] == '\n'
		skipSpacePrefix := startsWithNewline || (isAbsoluteStart && startsWithSpecialChar)
		if bestID < 0 && isStart && !hasLeadingSpace && !skipSpacePrefix {
			prefixed := spUnderscore + remaining
			foundPrefixed := false
			for l := min(len(prefixed), 20+spLen); l > spLen; l-- {
				candidate := prefixed[:l]
				if id, ok := t.vocab[candidate]; ok {
					bestLen = l - spLen
					bestID = id
					foundPrefixed = true
					break
				}
			}
			// If no ▁X token found at word boundary, emit just ▁
			if !foundPrefixed {
				if id, ok := t.vocab[spUnderscore]; ok {
					bestLen = 0 // don't consume any characters
					bestID = id
				}
			}
		}

		if hasLeadingSpace {
			textAfterSpace := remaining[1:]
			prefixed := spUnderscore + textAfterSpace
			foundPrefixed := false
			for l := min(len(prefixed), 20+spLen); l > spLen; l-- {
				candidate := prefixed[:l]
				if id, ok := t.vocab[candidate]; ok {
					actualLen := 1 + (l - spLen)
					if actualLen > bestLen {
						bestLen = actualLen
						bestID = id
						foundPrefixed = true
					}
					break
				}
			}
			// If no ▁X token found, emit just ▁ (the space marker)
			if !foundPrefixed && bestID < 0 {
				if id, ok := t.vocab[spUnderscore]; ok {
					bestLen = 1 // consume just the space
					bestID = id
				}
			}
		}

		// Fall back to plain text match if no ▁ match found, or for mid-word
		if bestID < 0 {
			for l := min(len(remaining), 20); l > 0; l-- {
				candidate := remaining[:l]
				if id, ok := t.vocab[candidate]; ok {
					bestLen = l
					bestID = id
					break
				}
			}
		}

		if bestID >= 0 {
			tokens = append(tokens, bestID)
			// Advance remaining by bestLen (can be 0 when emitting just ▁)
			if bestLen > 0 {
				if bestLen <= len(remaining) {
					remaining = remaining[bestLen:]
				} else {
					remaining = ""
				}
			}
			// If bestLen == 0, we just emitted ▁ but don't consume input
			// The next iteration will handle the actual content
		} else {
			// Byte-level fallback: encode as <0xNN>
			b := remaining[0]
			byteToken := fmt.Sprintf("<0x%02X>", b)
			if id, ok := t.vocab[byteToken]; ok {
				tokens = append(tokens, id)
			} else {
				// Skip unknown byte (shouldn't happen with proper tokenizer)
			}
			remaining = remaining[1:]
		}
	}

	return tokens, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// byteTokenRegex matches byte tokens like <0x0A>, <0x20>, etc.
var byteTokenRegex = regexp.MustCompile(`<0x([0-9A-Fa-f]{2})>`)

// Decode converts token IDs to text.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var out strings.Builder
	for _, id := range ids {
		if s, ok := t.ids[id]; ok {
			out.WriteString(s)
		}
	}
	return decodeSpecialChars(out.String()), nil
}

// decodeSpecialChars converts SentencePiece and byte tokens to readable text.
func decodeSpecialChars(s string) string {
	// Replace SentencePiece underscore (U+2581) with space
	s = strings.ReplaceAll(s, "▁", " ")

	// Decode byte tokens like <0x0A> -> newline, <0x20> -> space
	s = byteTokenRegex.ReplaceAllStringFunc(s, func(match string) string {
		// Extract hex value
		hexStr := match[3:5] // "0A" from "<0x0A>"
		if b, err := strconv.ParseUint(hexStr, 16, 8); err == nil {
			return string(rune(b))
		}
		return match // Return original if parse fails
	})

	// Don't trim leading space - it's part of the token meaning
	return s
}

// BOS returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOS() int {
	return t.bos
}

// EOS returns the end-of-sequence token ID.
func (t *Tokenizer) EOS() int {
	return t.eos
}

// ChatTemplate defines how to format messages for chat models.
type ChatTemplate struct {
	SystemPrefix    string
	SystemSuffix    string
	UserPrefix      string
	UserSuffix      string
	AssistantPrefix string
	AssistantSuffix string
}

// TinyLlamaChatTemplate returns the chat template for TinyLlama models.
// Note: AssistantPrefix has trailing newline to match llama.cpp format.
func TinyLlamaChatTemplate() ChatTemplate {
	return ChatTemplate{
		SystemPrefix:    "<|system|>\n",
		SystemSuffix:    "</s>\n",
		UserPrefix:      "<|user|>\n",
		UserSuffix:      "</s>\n",
		AssistantPrefix: "<|assistant|>\n", // Trailing newline to match llama.cpp
		AssistantSuffix: "</s>\n",
	}
}

// Llama2ChatTemplate returns the chat template for Llama-2 chat models.
func Llama2ChatTemplate() ChatTemplate {
	return ChatTemplate{
		SystemPrefix:    "[INST] <<SYS>>\n",
		SystemSuffix:    "\n<</SYS>>\n\n",
		UserPrefix:      "",
		UserSuffix:      " [/INST] ",
		AssistantPrefix: "",
		AssistantSuffix: " </s><s>[INST] ",
	}
}

// FormatChat formats a conversation using the chat template.
// messages is a slice of role/content pairs: [{"user", "Hello"}, {"assistant", "Hi!"}]
func (ct ChatTemplate) FormatChat(systemPrompt string, userMessage string) string {
	var sb strings.Builder

	// Add system prompt if provided
	if systemPrompt != "" {
		sb.WriteString(ct.SystemPrefix)
		sb.WriteString(systemPrompt)
		sb.WriteString(ct.SystemSuffix)
	}

	// Add user message
	sb.WriteString(ct.UserPrefix)
	sb.WriteString(userMessage)
	sb.WriteString(ct.UserSuffix)

	// Add assistant prefix (model will generate the response)
	sb.WriteString(ct.AssistantPrefix)

	return sb.String()
}
