package train

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

type DataFormat int

const (
	FormatText             DataFormat = iota
	FormatPromptCompletion
)

type Example struct {
	Format     DataFormat
	Text       string
	Prompt     string
	Completion string
}

func LoadData(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open data file: %w", err)
	}
	defer f.Close()

	var examples []Example
	var detectedFormat DataFormat
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	lineNum := 0

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		lineNum++

		var raw map[string]string
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return nil, fmt.Errorf("line %d: invalid JSON: %w", lineNum, err)
		}

		var ex Example
		_, hasText := raw["text"]
		_, hasPrompt := raw["prompt"]
		_, hasCompletion := raw["completion"]

		if hasText {
			ex.Format = FormatText
			ex.Text = raw["text"]
		} else if hasPrompt && hasCompletion {
			ex.Format = FormatPromptCompletion
			ex.Prompt = raw["prompt"]
			ex.Completion = raw["completion"]
		} else {
			return nil, fmt.Errorf("line %d: must have \"text\" or \"prompt\"+\"completion\" fields", lineNum)
		}

		if lineNum == 1 {
			detectedFormat = ex.Format
		} else if ex.Format != detectedFormat {
			return nil, fmt.Errorf("line %d: mixed formats (first line was %v, this line is %v)", lineNum, detectedFormat, ex.Format)
		}

		examples = append(examples, ex)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read data file: %w", err)
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("data file is empty")
	}

	return examples, nil
}

func BuildLossMask(tokens []int32, format DataFormat, promptLen int) []float32 {
	n := len(tokens)
	mask := make([]float32, n)

	switch format {
	case FormatText:
		for i := 0; i < n-1; i++ {
			mask[i] = 1
		}
	case FormatPromptCompletion:
		for i := promptLen; i < n-1; i++ {
			mask[i] = 1
		}
	}

	return mask
}
