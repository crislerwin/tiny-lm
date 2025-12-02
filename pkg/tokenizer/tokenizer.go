package tokenizer

import (
	"fmt"
	"strings"
)

// Tokenizer handles text to token conversion
type Tokenizer struct {
	Vocab     []string
	VocabMap  map[string]int
	UnknownID int
}

// NewTokenizer creates a new tokenizer with the given vocabulary
func NewTokenizer(vocab []string) *Tokenizer {
	vocabMap := make(map[string]int)
	for i, word := range vocab {
		vocabMap[word] = i
	}

	return &Tokenizer{
		Vocab:     vocab,
		VocabMap:  vocabMap,
		UnknownID: -1,
	}
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string) ([]int, error) {
	words := strings.Fields(text)
	tokens := make([]int, 0, len(words))

	for _, word := range words {
		if id, exists := t.VocabMap[word]; exists {
			tokens = append(tokens, id)
		} else {
			return nil, fmt.Errorf("word not in vocabulary: %s", word)
		}
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("no valid tokens found in text: %s", text)
	}

	return tokens, nil
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(tokens []int) (string, error) {
	words := make([]string, 0, len(tokens))

	for _, id := range tokens {
		if id < 0 || id >= len(t.Vocab) {
			return "", fmt.Errorf("invalid token ID: %d (vocab size: %d)", id, len(t.Vocab))
		}
		words = append(words, t.Vocab[id])
	}

	return strings.Join(words, " "), nil
}

// VocabSize returns the size of the vocabulary
func (t *Tokenizer) VocabSize() int {
	return len(t.Vocab)
}
