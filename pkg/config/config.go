package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// ModelConfig holds the model configuration
type ModelConfig struct {
	DModel    int      `json:"d_model"`
	NLayers   int      `json:"n_layers"`
	NHeads    int      `json:"n_heads"`
	VocabSize int      `json:"vocab_size"`
	MaxSeqLen int      `json:"max_seq_len"`
	Vocab     []string `json:"vocab"`
	Eps       float64  `json:"eps"`
}

// Validate checks if the configuration is valid
func (c *ModelConfig) Validate() error {
	if c.DModel <= 0 {
		return fmt.Errorf("d_model must be positive, got %d", c.DModel)
	}
	if c.NLayers <= 0 {
		return fmt.Errorf("n_layers must be positive, got %d", c.NLayers)
	}
	if c.NHeads <= 0 {
		return fmt.Errorf("n_heads must be positive, got %d", c.NHeads)
	}
	if c.DModel%c.NHeads != 0 {
		return fmt.Errorf("d_model (%d) must be divisible by n_heads (%d)", c.DModel, c.NHeads)
	}
	if c.MaxSeqLen <= 0 {
		return fmt.Errorf("max_seq_len must be positive, got %d", c.MaxSeqLen)
	}
	if len(c.Vocab) == 0 {
		return fmt.Errorf("vocab cannot be empty")
	}
	if c.Eps <= 0 {
		c.Eps = 1e-5 // Default epsilon
	}
	if c.VocabSize == 0 {
		c.VocabSize = len(c.Vocab)
	}
	return nil
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(filename string) (*ModelConfig, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config ModelConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config JSON: %w", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &config, nil
}

// DefaultConfig returns a default configuration
func DefaultConfig() *ModelConfig {
	return &ModelConfig{
		DModel:    128,
		NLayers:   4,
		NHeads:    4,
		VocabSize: 1000,
		MaxSeqLen: 512,
		Vocab:     []string{},
		Eps:       1e-5,
	}
}
