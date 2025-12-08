package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Tokenizer handles text <-> id conversion.
type Tokenizer struct {
	vocab map[string]int
	ids   map[int]string
	bos   int // Beginning of sequence token
	eos   int // End of sequence token
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

	// Try to find special tokens in added_tokens
	for _, tok := range data.AddedTokens {
		switch tok.Content {
		case "<s>":
			bos = tok.ID
		case "</s>":
			eos = tok.ID
		}
	}

	return &Tokenizer{
		vocab: data.Model.Vocab,
		ids:   ids,
		bos:   bos,
		eos:   eos,
	}, nil
}

// Encode converts text to token IDs.
// This is a simplified implementation that tries:
// 1. Exact match for the whole string
// 2. Greedy longest-match tokenization
// 3. Byte-level fallback for unknown characters
func (t *Tokenizer) Encode(text string) ([]int, error) {
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

		// Try with SentencePiece underscore prefix for word starts
		tryTexts := []string{remaining}
		if len(tokens) == 0 || (len(remaining) > 0 && remaining[0] == ' ') {
			// At start or after space, try with underscore
			trimmed := strings.TrimPrefix(remaining, " ")
			if trimmed != remaining {
				tryTexts = append(tryTexts, "▁"+trimmed)
			} else {
				tryTexts = append(tryTexts, "▁"+remaining)
			}
		}

		for _, tryText := range tryTexts {
			for l := min(len(tryText), 20); l > 0; l-- {
				candidate := tryText[:l]
				if id, ok := t.vocab[candidate]; ok {
					// Adjust length for underscore prefix
					actualLen := l
					if strings.HasPrefix(candidate, "▁") && !strings.HasPrefix(remaining, "▁") {
						// We added the underscore, so consume the space
						if strings.HasPrefix(remaining, " ") {
							actualLen = l // Consume space + matched chars - 1 (underscore)
						} else {
							actualLen = l - 1 // Just the matched chars without underscore
						}
					}
					if actualLen > bestLen {
						bestLen = actualLen
						bestID = id
					}
					break
				}
			}
		}

		if bestID >= 0 {
			tokens = append(tokens, bestID)
			// Handle space consumption
			if strings.HasPrefix(remaining, " ") && bestLen > 0 {
				remaining = remaining[1:] // consume space
				bestLen--                 // adjust for consumed space
			}
			remaining = remaining[bestLen:]
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

	if len(tokens) == 0 {
		return []int{t.bos}, nil // Fallback to BOS
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
func TinyLlamaChatTemplate() ChatTemplate {
	return ChatTemplate{
		SystemPrefix:    "<|system|>\n",
		SystemSuffix:    "</s>\n",
		UserPrefix:      "<|user|>\n",
		UserSuffix:      "</s>\n",
		AssistantPrefix: "<|assistant|>\n",
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
