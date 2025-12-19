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
	addBos        bool     // Whether to add BOS token to prompts (false for Phi-2, GPT-2)
	specialTokens []string // Special tokens to match before BPE (longest first)
	useByteLevel  bool     // Whether to use ByteLevel BPE (spaces -> Ġ)
}

// Load loads a tokenizer from a file (tokenizer.json).
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

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
	addBos := true // Default: add BOS for Llama-style models

	// Collect special tokens and find BOS/EOS
	var specialTokens []string
	for _, tok := range data.AddedTokens {
		switch tok.Content {
		case "<s>":
			bos = tok.ID
		case "</s>":
			eos = tok.ID
		case "<|endoftext|>":
			// Phi-2 and similar models use <|endoftext|> as both BOS and EOS
			bos = tok.ID
			eos = tok.ID
			addBos = false // GPT-2/Phi style: don't add BOS
		}
		// Add all added_tokens to vocab and special tokens list
		if _, exists := data.Model.Vocab[tok.Content]; !exists {
			data.Model.Vocab[tok.Content] = tok.ID
			ids[tok.ID] = tok.Content
		}
		specialTokens = append(specialTokens, tok.Content)
	}

	// Check for ByteLevel BPE marker Ġ (U+0120)
	useByteLevel := false
	if _, ok := data.Model.Vocab["Ġ"]; ok {
		useByteLevel = true
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
		addBos:        addBos,
		specialTokens: specialTokens,
		useByteLevel:  useByteLevel,
	}, nil
}

// Encode converts text to token IDs.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return nil, nil
	}

	segments := t.splitOnSpecialTokens(text)
	var tokens []int

	for i, seg := range segments {
		if seg.isSpecial {
			if id, ok := t.vocab[seg.text]; ok {
				tokens = append(tokens, id)
			}
		} else {
			atWordBoundary := i == 0 || (i > 0 && segments[i-1].isSpecial)
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

type segment struct {
	text      string
	isSpecial bool
}

func (t *Tokenizer) splitOnSpecialTokens(text string) []segment {
	if len(t.specialTokens) == 0 {
		return []segment{{text: text, isSpecial: false}}
	}

	var segments []segment
	remaining := text

	for len(remaining) > 0 {
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
			if len(remaining) > 0 {
				segments = append(segments, segment{text: remaining, isSpecial: false})
			}
			break
		}

		if earliestPos > 0 {
			segments = append(segments, segment{text: remaining[:earliestPos], isSpecial: false})
		}

		segments = append(segments, segment{text: earliestToken, isSpecial: true})
		remaining = remaining[earliestPos+len(earliestToken):]
	}

	return segments
}

func (t *Tokenizer) encodeSegment(text string, atWordBoundary bool, isAbsoluteStart bool) ([]int, error) {
	if text == "" {
		return nil, nil
	}

	// Try exact match first
	if id, ok := t.vocab[text]; ok {
		return []int{id}, nil
	}

	// Define space token based on tokenizer type
	spaceChar := "▁" // SentencePiece default
	if t.useByteLevel {
		spaceChar = "Ġ"
	}
	spLen := len(spaceChar)

	var tokens []int
	remaining := text

	for len(remaining) > 0 {
		bestLen := 0
		bestID := -1

		isFirstToken := len(tokens) == 0
		isStart := isFirstToken && atWordBoundary
		hasLeadingSpace := len(remaining) > 0 && remaining[0] == ' '

		startsWithSpecialChar := len(remaining) > 0 && (remaining[0] == '<' || remaining[0] == '[' || remaining[0] == '{')
		
		if t.useByteLevel {
			// ByteLevel Logic: Treat space as Ġ
			// If has leading space, prefix with Ġ
			if hasLeadingSpace {
				textAfterSpace := remaining[1:]
				prefixed := spaceChar + textAfterSpace
				
				// Try finding longest match with Ġ prefix
				foundPrefixed := false
				for l := min(len(prefixed), 40); l > spLen; l-- {
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
				
				// If no match with suffix, try just Ġ
				if !foundPrefixed && bestID < 0 {
					if id, ok := t.vocab[spaceChar]; ok {
						bestLen = 1 // consume space
						bestID = id
					}
				}
			} else {
				// No leading space, standard match
				for l := min(len(remaining), 40); l > 0; l-- {
					candidate := remaining[:l]
					if id, ok := t.vocab[candidate]; ok {
						bestLen = l
						bestID = id
						break
					}
				}
			}
		} else {
			// SentencePiece Logic (Original)
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

			startsWithNewline := len(remaining) > 0 && remaining[0] == '\n'
			skipSpacePrefix := startsWithNewline || (isAbsoluteStart && startsWithSpecialChar)
			if bestID < 0 && isStart && !hasLeadingSpace && !skipSpacePrefix {
				prefixed := spaceChar + remaining
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
				if !foundPrefixed {
					if id, ok := t.vocab[spaceChar]; ok {
						bestLen = 0
						bestID = id
					}
				}
			}

			if hasLeadingSpace {
				textAfterSpace := remaining[1:]
				prefixed := spaceChar + textAfterSpace
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
				if !foundPrefixed && bestID < 0 {
					if id, ok := t.vocab[spaceChar]; ok {
						bestLen = 1
						bestID = id
					}
				}
			}

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
		}

		if bestID >= 0 {
			tokens = append(tokens, bestID)
			if bestLen > 0 {
				if bestLen <= len(remaining) {
					remaining = remaining[bestLen:]
				} else {
					remaining = ""
				}
			}
		} else {
			b := remaining[0]
			byteToken := fmt.Sprintf("<0x%02X>", b)
			if id, ok := t.vocab[byteToken]; ok {
				tokens = append(tokens, id)
			} else {
				// Fallback for ByteLevel BPE: if <0xNN> not found,
				// check if single char is in vocab (most ASCII are)
				if t.useByteLevel {
					charStr := string(b)
					if id, ok := t.vocab[charStr]; ok {
						tokens = append(tokens, id)
					}
				}
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

var byteTokenRegex = regexp.MustCompile(`<0x([0-9A-Fa-f]{2})>`) 

func (t *Tokenizer) Decode(ids []int) (string, error) {
	var out strings.Builder
	for _, id := range ids {
		if s, ok := t.ids[id]; ok {
			out.WriteString(s)
		}
	}
	s := out.String()
	// Replace Ġ with space for ByteLevel
	s = strings.ReplaceAll(s, "Ġ", " ")
	return decodeSpecialChars(s), nil
}

func decodeSpecialChars(s string) string {
	s = strings.ReplaceAll(s, "▁", " ")
	s = byteTokenRegex.ReplaceAllStringFunc(s, func(match string) string {
		hexStr := match[3:5]
		if b, err := strconv.ParseUint(hexStr, 16, 8); err == nil {
			return string(rune(b))
		}
		return match
	})
	return s
}

func (t *Tokenizer) BOS() int { return t.bos }
func (t *Tokenizer) EOS() int { return t.eos }
func (t *Tokenizer) AddBOS() bool { return t.addBos }
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

type ChatTemplate struct {
	SystemPrefix    string
	SystemSuffix    string
	UserPrefix      string
	UserSuffix      string
	AssistantPrefix string
	AssistantSuffix string
}

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

func (ct ChatTemplate) FormatChat(systemPrompt string, userMessage string) string {
	var sb strings.Builder
	if systemPrompt != "" {
		sb.WriteString(ct.SystemPrefix)
		sb.WriteString(systemPrompt)
		sb.WriteString(ct.SystemSuffix)
	}
	sb.WriteString(ct.UserPrefix)
	sb.WriteString(userMessage)
	sb.WriteString(ct.UserSuffix)
	sb.WriteString(ct.AssistantPrefix)
	return sb.String()
}