package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"vexel/inference/pkg/gguf"
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

	// BPE merge rules: maps "tokenA tokenB" -> merge priority (lower = higher priority).
	mergeRanks map[string]int
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
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"`
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

	// Parse BPE merge rules (supports both string "a b" and array ["a","b"] formats)
	mergeRanks := make(map[string]int)
	if len(data.Model.Merges) > 0 {
		// Try array-of-strings first: ["a b", "c d", ...]
		var stringMerges []string
		if err := json.Unmarshal(data.Model.Merges, &stringMerges); err == nil {
			for i, m := range stringMerges {
				mergeRanks[m] = i
			}
		} else {
			// Try array-of-arrays: [["a","b"], ["c","d"], ...]
			var arrayMerges [][]string
			if err := json.Unmarshal(data.Model.Merges, &arrayMerges); err == nil {
				for i, pair := range arrayMerges {
					if len(pair) == 2 {
						mergeRanks[pair[0]+" "+pair[1]] = i
					}
				}
			}
		}
	}

	return &Tokenizer{
		vocab:         data.Model.Vocab,
		ids:           ids,
		bos:           bos,
		eos:           eos,
		addBos:        addBos,
		specialTokens: specialTokens,
		useByteLevel:  useByteLevel,
		mergeRanks:    mergeRanks,
	}, nil
}

// LoadFromGGUF loads a tokenizer from the vocabulary embedded in a GGUF file.
// This is the preferred method since it guarantees the tokenizer matches the model.
func LoadFromGGUF(path string) (*Tokenizer, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}
	defer gf.Close()

	// Read token strings from tokenizer.ggml.tokens
	tokensVal, ok := gf.Metadata["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("GGUF file missing tokenizer.ggml.tokens")
	}

	vocab := make(map[string]int, len(tokensVal.Array))
	ids := make(map[int]string, len(tokensVal.Array))
	for i, tv := range tokensVal.Array {
		token := tv.AsString()
		vocab[token] = i
		ids[i] = token
	}

	// Read scores (used for SentencePiece models)
	var scores []float32
	if scoresVal, ok := gf.Metadata["tokenizer.ggml.scores"]; ok {
		scores = make([]float32, len(scoresVal.Array))
		for i, sv := range scoresVal.Array {
			scores[i] = sv.AsFloat32()
		}
	}

	// Read token types to identify special tokens
	var tokenTypes []int
	if typesVal, ok := gf.Metadata["tokenizer.ggml.token_type"]; ok {
		tokenTypes = make([]int, len(typesVal.Array))
		for i, tv := range typesVal.Array {
			tokenTypes[i] = int(tv.AsUint32())
		}
	}

	// Determine BOS/EOS from metadata
	bos := 1
	eos := 2
	addBos := true
	if v, ok := gf.Metadata["tokenizer.ggml.bos_token_id"]; ok {
		bos = int(v.AsUint32())
	}
	if v, ok := gf.Metadata["tokenizer.ggml.eos_token_id"]; ok {
		eos = int(v.AsUint32())
	}

	// Detect tokenizer type
	tokenizerModel := "llama" // default to SentencePiece
	if v, ok := gf.Metadata["tokenizer.ggml.model"]; ok {
		tokenizerModel = v.AsString()
	}
	useByteLevel := tokenizerModel == "gpt2"

	// Use explicit add_bos_token from GGUF metadata if available.
	// This is the authoritative source: e.g., Qwen2 has bos!=eos but add_bos_token=false.
	if v, ok := gf.Metadata["tokenizer.ggml.add_bos_token"]; ok {
		addBos = v.AsBool()
	} else {
		// Fallback heuristic: disable BOS when BOS and EOS share the same token (GPT-2, Phi-2).
		// LLaMA 3 uses GPT-2 BPE format but has a distinct BOS (128000) — it still needs BOS.
		if bos == eos {
			addBos = false
		}
	}

	// Collect special tokens (types 3=control, 6=byte)
	var specialTokens []string
	for i, tok := range tokensVal.Array {
		token := tok.AsString()
		if len(tokenTypes) > i {
			tt := tokenTypes[i]
			if tt == 3 { // control token
				specialTokens = append(specialTokens, token)
			}
		}
	}
	sort.Slice(specialTokens, func(i, j int) bool {
		return len(specialTokens[i]) > len(specialTokens[j])
	})

	// Build BPE merge rules from scores (SentencePiece uses scores as merge priority).
	// For SentencePiece, we don't have explicit merges — the vocab+scores define the model.
	// We build pseudo-merges from the vocabulary: if token "AB" exists and both "A" and "B"
	// exist, then "A B" is a merge with priority based on the token's score.
	mergeRanks := make(map[string]int)

	// Read explicit merges if available (some GGUF files have them)
	if mergesVal, ok := gf.Metadata["tokenizer.ggml.merges"]; ok && len(mergesVal.Array) > 0 {
		for i, mv := range mergesVal.Array {
			mergeRanks[mv.AsString()] = i
		}
	}

	// If no explicit merges but we have scores, build merge rules from vocabulary.
	// This handles SentencePiece models where merges are implicit in the vocab scores.
	if len(mergeRanks) == 0 && len(scores) > 0 && !useByteLevel {
		type scoredToken struct {
			token string
			id    int
			score float32
		}
		var sortedTokens []scoredToken
		for i, tv := range tokensVal.Array {
			token := tv.AsString()
			if len(tokenTypes) > i && tokenTypes[i] >= 3 {
				continue // skip control/byte tokens
			}
			if len([]rune(token)) > 1 {
				sortedTokens = append(sortedTokens, scoredToken{token, i, scores[i]})
			}
		}
		// Sort by score descending (higher score = higher priority merge)
		sort.Slice(sortedTokens, func(i, j int) bool {
			return sortedTokens[i].score > sortedTokens[j].score
		})
		// Build merge pairs: for each multi-char token, register ALL valid splits as merges.
		// A token like "▁Hello" can be split as "▁"+"Hello" or "▁H"+"ello" etc.
		// All valid splits must be registered because BPE may encounter any of these
		// symbol pairs depending on the order of earlier merges applied.
		for rank, st := range sortedTokens {
			runes := []rune(st.token)
			for splitPos := 1; splitPos < len(runes); splitPos++ {
				left := string(runes[:splitPos])
				right := string(runes[splitPos:])
				if _, leftOk := vocab[left]; leftOk {
					if _, rightOk := vocab[right]; rightOk {
						key := left + " " + right
						if _, exists := mergeRanks[key]; !exists {
							mergeRanks[key] = rank
						}
					}
				}
			}
		}
	}

	return &Tokenizer{
		vocab:         vocab,
		ids:           ids,
		bos:           bos,
		eos:           eos,
		addBos:        addBos,
		specialTokens: specialTokens,
		useByteLevel:  useByteLevel,
		mergeRanks:    mergeRanks,
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

	// If we have BPE merges and SentencePiece mode, use proper BPE encoding.
	// Must be done before greedy matching because SentencePiece needs to
	// preprocess (add ▁ prefix, replace spaces) before matching vocab.
	if len(t.mergeRanks) > 0 && !t.useByteLevel {
		return t.encodeBPE(text, atWordBoundary, isAbsoluteStart)
	}

	// Try exact match first (for non-BPE or ByteLevel paths)
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
			// SentencePiece Logic (greedy fallback when no merges available)
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

// encodeBPE encodes a text segment using proper BPE merge rules (SentencePiece style).
// SentencePiece replaces all spaces with ▁ and optionally prepends ▁ at the start,
// then applies BPE merging on the entire sequence.
func (t *Tokenizer) encodeBPE(text string, atWordBoundary bool, isAbsoluteStart bool) ([]int, error) {
	spaceChar := "▁"

	// Convert text to SentencePiece form: replace spaces with ▁ and optionally add leading ▁.
	// When text starts with a space at a word boundary, that space becomes the ▁ prefix
	// (no extra ▁ added). When text doesn't start with a space, ▁ is prepended.
	spText := strings.ReplaceAll(text, " ", spaceChar)

	if atWordBoundary {
		startsWithNewline := len(text) > 0 && text[0] == '\n'
		startsWithSpecialChar := len(text) > 0 && (text[0] == '<' || text[0] == '[' || text[0] == '{')
		startsWithSpace := len(text) > 0 && text[0] == ' '
		skipPrefix := startsWithNewline || (isAbsoluteStart && startsWithSpecialChar) || startsWithSpace
		if !skipPrefix {
			spText = spaceChar + spText
		}
	}

	return t.bpeEncodeWord(spText, spaceChar), nil
}

// bpeEncodeWord encodes a string (already in SentencePiece form with ▁) using BPE merge rules.
func (t *Tokenizer) bpeEncodeWord(word string, spaceChar string) []int {
	// Build initial symbol sequence: split into individual UTF-8 characters.
	var symbols []string
	runes := []rune(word)
	for _, r := range runes {
		symbols = append(symbols, string(r))
	}

	if len(symbols) == 0 {
		return nil
	}

	// Check if any initial symbols are not in vocab; use byte fallback
	for i, sym := range symbols {
		if _, ok := t.vocab[sym]; !ok {
			// Try byte fallback for each byte in the symbol
			var byteFallbacks []string
			valid := true
			for _, b := range []byte(sym) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				if _, ok := t.vocab[byteToken]; ok {
					byteFallbacks = append(byteFallbacks, byteToken)
				} else {
					valid = false
					break
				}
			}
			if valid && len(byteFallbacks) > 0 {
				// Replace symbol with byte fallback tokens
				newSymbols := make([]string, 0, len(symbols)+len(byteFallbacks)-1)
				newSymbols = append(newSymbols, symbols[:i]...)
				newSymbols = append(newSymbols, byteFallbacks...)
				newSymbols = append(newSymbols, symbols[i+1:]...)
				symbols = newSymbols
			}
		}
	}

	// Iteratively apply the highest-priority merge
	for len(symbols) > 1 {
		// Find the pair with the lowest merge rank (highest priority)
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			key := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.mergeRanks[key]; ok {
				if bestIdx == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break // No more merges possible
		}

		// Apply the merge: combine symbols[bestIdx] and symbols[bestIdx+1]
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	// Convert symbols to token IDs
	var ids []int
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			ids = append(ids, id)
		} else {
			// Byte fallback for unknown symbols
			for _, b := range []byte(sym) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				if id, ok := t.vocab[byteToken]; ok {
					ids = append(ids, id)
				}
			}
		}
	}
	return ids
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

var byteTokenRegex = regexp.MustCompile(`<0x([0-9A-Fa-f]{2})>`)

// gpt2UnicodeToByte is the reverse of GPT-2's bytes_to_unicode() mapping.
// GPT-2 BPE maps each byte (0-255) to a printable Unicode character.
// This table maps those Unicode characters back to byte values for decoding.
var gpt2UnicodeToByte = func() map[rune]byte {
	b2u := make(map[byte]rune)
	next := rune(256)
	for b := 0; b < 256; b++ {
		r := rune(b)
		if (r >= '!' && r <= '~') || (r >= 0xA1 && r <= 0xAC) || (r >= 0xAE && r <= 0xFF) {
			b2u[byte(b)] = r
		} else {
			b2u[byte(b)] = next
			next++
		}
	}
	u2b := make(map[rune]byte, 256)
	for b, u := range b2u {
		u2b[u] = b
	}
	return u2b
}()

func (t *Tokenizer) Decode(ids []int) (string, error) {
	if t.useByteLevel {
		return t.decodeByteLevelBPE(ids)
	}
	var out strings.Builder
	for _, id := range ids {
		if s, ok := t.ids[id]; ok {
			out.WriteString(s)
		}
	}
	s := out.String()
	return decodeSpecialChars(s), nil
}

// decodeByteLevelBPE decodes GPT-2/LLaMA 3 byte-level BPE tokens.
// Each token character is mapped back to its byte value, then the byte
// sequence is interpreted as UTF-8.
func (t *Tokenizer) decodeByteLevelBPE(ids []int) (string, error) {
	var buf []byte
	for _, id := range ids {
		s, ok := t.ids[id]
		if !ok {
			continue
		}
		if len(s) == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>' {
			if b, err := strconv.ParseUint(s[3:5], 16, 8); err == nil {
				buf = append(buf, byte(b))
				continue
			}
		}
		for _, r := range s {
			if b, ok := gpt2UnicodeToByte[r]; ok {
				buf = append(buf, b)
			} else {
				buf = append(buf, []byte(string(r))...)
			}
		}
	}
	return string(buf), nil
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

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string // "system", "user", or "assistant"
	Content string
}

// ChatTemplate defines prefix/suffix tokens for formatting chat conversations.
type ChatTemplate struct {
	Name            string // Template name for identification
	SystemPrefix    string
	SystemSuffix    string
	UserPrefix      string
	UserSuffix      string
	AssistantPrefix string
	AssistantSuffix string
	BOS             string // Optional beginning-of-sequence token
}

// DefaultChatTemplate returns the default template (TinyLlama/Zephyr style).
func DefaultChatTemplate() ChatTemplate {
	return TinyLlamaChatTemplate()
}

// TinyLlamaChatTemplate returns the TinyLlama/Zephyr chat template.
func TinyLlamaChatTemplate() ChatTemplate {
	return ChatTemplate{
		Name:            "tinyllama",
		SystemPrefix:    "<|system|>\n",
		SystemSuffix:    "</s>\n",
		UserPrefix:      "<|user|>\n",
		UserSuffix:      "</s>\n",
		AssistantPrefix: "<|assistant|>\n",
		AssistantSuffix: "</s>\n",
	}
}

// Llama2ChatTemplate returns the Llama 2 chat template.
func Llama2ChatTemplate() ChatTemplate {
	return ChatTemplate{
		Name:            "llama2",
		SystemPrefix:    "[INST] <<SYS>>\n",
		SystemSuffix:    "\n<</SYS>>\n\n",
		UserPrefix:      "",
		UserSuffix:      " [/INST] ",
		AssistantPrefix: "",
		AssistantSuffix: " </s><s>[INST] ",
	}
}

// Llama3ChatTemplate returns the Llama 3 / Llama 3.1 chat template.
func Llama3ChatTemplate() ChatTemplate {
	return ChatTemplate{
		Name:            "llama3",
		BOS:             "<|begin_of_text|>",
		SystemPrefix:    "<|start_header_id|>system<|end_header_id|>\n\n",
		SystemSuffix:    "<|eot_id|>",
		UserPrefix:      "<|start_header_id|>user<|end_header_id|>\n\n",
		UserSuffix:      "<|eot_id|>",
		AssistantPrefix: "<|start_header_id|>assistant<|end_header_id|>\n\n",
		AssistantSuffix: "<|eot_id|>",
	}
}

// ChatMLTemplate returns the ChatML template (used by Mistral, Phi-3, Qwen, etc.).
func ChatMLTemplate() ChatTemplate {
	return ChatTemplate{
		Name:            "chatml",
		SystemPrefix:    "<|im_start|>system\n",
		SystemSuffix:    "<|im_end|>\n",
		UserPrefix:      "<|im_start|>user\n",
		UserSuffix:      "<|im_end|>\n",
		AssistantPrefix: "<|im_start|>assistant\n",
		AssistantSuffix: "<|im_end|>\n",
	}
}

// DetectChatTemplate picks a chat template based on the model file path.
// It inspects the filename for known model family patterns.
func DetectChatTemplate(modelPath string) ChatTemplate {
	lower := strings.ToLower(modelPath)

	switch {
	case strings.Contains(lower, "llama-3") ||
		strings.Contains(lower, "llama3") ||
		strings.Contains(lower, "llama_3"):
		return Llama3ChatTemplate()

	case strings.Contains(lower, "llama-2") ||
		strings.Contains(lower, "llama2") ||
		strings.Contains(lower, "llama_2"):
		return Llama2ChatTemplate()

	case strings.Contains(lower, "mistral") ||
		strings.Contains(lower, "phi-3") ||
		strings.Contains(lower, "phi3") ||
		strings.Contains(lower, "qwen") ||
		strings.Contains(lower, "chatml"):
		return ChatMLTemplate()

	case strings.Contains(lower, "tinyllama") ||
		strings.Contains(lower, "zephyr"):
		return TinyLlamaChatTemplate()

	default:
		return DefaultChatTemplate()
	}
}

// FormatChat formats a single user message with an optional system prompt.
// Kept for backward compatibility.
func (ct ChatTemplate) FormatChat(systemPrompt string, userMessage string) string {
	msgs := []ChatMessage{{Role: "user", Content: userMessage}}
	if systemPrompt != "" {
		msgs = append([]ChatMessage{{Role: "system", Content: systemPrompt}}, msgs...)
	}
	return ct.FormatConversation(msgs)
}

// FormatConversation formats a full multi-turn conversation into a single prompt string.
func (ct ChatTemplate) FormatConversation(messages []ChatMessage) string {
	var sb strings.Builder

	if ct.BOS != "" {
		sb.WriteString(ct.BOS)
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			sb.WriteString(ct.SystemPrefix)
			sb.WriteString(msg.Content)
			sb.WriteString(ct.SystemSuffix)
		case "user":
			sb.WriteString(ct.UserPrefix)
			sb.WriteString(msg.Content)
			sb.WriteString(ct.UserSuffix)
		case "assistant":
			sb.WriteString(ct.AssistantPrefix)
			sb.WriteString(msg.Content)
			sb.WriteString(ct.AssistantSuffix)
		}
	}

	// End with assistant prefix to prompt the model to generate
	sb.WriteString(ct.AssistantPrefix)
	return sb.String()
}