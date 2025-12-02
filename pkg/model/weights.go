package model

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/crislerwin/tiny-lm/pkg/config"
	tmath "github.com/crislerwin/tiny-lm/pkg/math"
)

// Weights holds all model parameters
type Weights struct {
	Config *config.ModelConfig
	Data   map[string]interface{}
}

// NewWeights creates a new Weights instance
func NewWeights(cfg *config.ModelConfig) *Weights {
	return &Weights{
		Config: cfg,
		Data:   make(map[string]interface{}),
	}
}

// GetMatrix retrieves a matrix weight by key
func (w *Weights) GetMatrix(key string) (tmath.Matrix, error) {
	val, exists := w.Data[key]
	if !exists {
		return nil, fmt.Errorf("weight not found: %s", key)
	}

	matrix, ok := val.(tmath.Matrix)
	if !ok {
		return nil, fmt.Errorf("weight %s is not a matrix", key)
	}

	return matrix, nil
}

// GetVector retrieves a vector weight by key
func (w *Weights) GetVector(key string) (tmath.Vector, error) {
	val, exists := w.Data[key]
	if !exists {
		return nil, fmt.Errorf("weight not found: %s", key)
	}

	vector, ok := val.(tmath.Vector)
	if !ok {
		return nil, fmt.Errorf("weight %s is not a vector", key)
	}

	return vector, nil
}

// LoadFromJSON loads weights from a JSON file
func LoadFromJSON(filename string) (*Weights, error) {
	file, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read weights file: %w", err)
	}

	var raw struct {
		DModel  int                    `json:"d_model"`
		NLayers int                    `json:"n_layers"`
		NHeads  int                    `json:"n_heads"`
		Vocab   []string               `json:"vocab"`
		Weights map[string]interface{} `json:"weights"`
	}

	if err := json.Unmarshal(file, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse weights JSON: %w", err)
	}

	cfg := &config.ModelConfig{
		DModel:    raw.DModel,
		NLayers:   raw.NLayers,
		NHeads:    raw.NHeads,
		Vocab:     raw.Vocab,
		VocabSize: len(raw.Vocab),
		MaxSeqLen: 512,
		Eps:       1e-5,
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid model config in weights file: %w", err)
	}

	w := NewWeights(cfg)

	// Convert raw weights to typed matrices/vectors
	for key, val := range raw.Weights {
		converted, err := convertWeight(val)
		if err != nil {
			return nil, fmt.Errorf("failed to convert weight %s: %w", key, err)
		}
		w.Data[key] = converted
	}

	return w, nil
}

// convertWeight converts interface{} to Matrix or Vector
func convertWeight(v interface{}) (interface{}, error) {
	list, ok := v.([]interface{})
	if !ok {
		return nil, fmt.Errorf("weight is not a list")
	}

	if len(list) == 0 {
		return tmath.Vector{}, nil
	}

	// Check if it's a matrix (nested list) or vector (flat list)
	if nested, isNested := list[0].([]interface{}); isNested {
		// It's a matrix
		rows := len(list)
		cols := len(nested)
		matrix := tmath.NewMatrix(rows, cols)

		for i := range list {
			row := list[i].([]interface{})
			if len(row) != cols {
				return nil, fmt.Errorf("inconsistent matrix columns")
			}
			for j := range row {
				val, ok := row[j].(float64)
				if !ok {
					return nil, fmt.Errorf("non-numeric value in matrix")
				}
				matrix[i][j] = val
			}
		}
		return matrix, nil
	}

	// It's a vector
	vector := tmath.NewVector(len(list))
	for i, val := range list {
		num, ok := val.(float64)
		if !ok {
			return nil, fmt.Errorf("non-numeric value in vector")
		}
		vector[i] = num
	}
	return vector, nil
}
