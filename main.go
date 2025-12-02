package main

import (
	"fmt"
	"log"

	"github.com/crislerwin/tiny-lm/pkg/model"
	"github.com/crislerwin/tiny-lm/pkg/tokenizer"
)

func main() {
	// Load weights from JSON file
	weights, err := model.LoadFromJSON("tiny_gpt_weights.json")
	if err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// Create transformer model
	transformer := model.NewTransformer(weights)

	// Create tokenizer
	tok := tokenizer.NewTokenizer(weights.Config.Vocab)

	// Run inference examples
	prompts := []string{
		"the cat",
		"the dog",
	}

	for _, prompt := range prompts {
		if err := generate(prompt, transformer, tok); err != nil {
			log.Printf("Generation failed for '%s': %v", prompt, err)
		}
	}
}

func generate(prompt string, transformer *model.Transformer, tok *tokenizer.Tokenizer) error {
	// Encode the prompt
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return fmt.Errorf("failed to encode prompt: %w", err)
	}

	fmt.Printf("\nInput: '%s'\n", prompt)
	fmt.Printf("Tokens: %v\n", tokens)

	// Generate continuation (3 new tokens)
	generated, err := transformer.Generate(tokens, 3)
	if err != nil {
		return fmt.Errorf("failed to generate: %w", err)
	}

	// Decode the result
	result, err := tok.Decode(generated)
	if err != nil {
		return fmt.Errorf("failed to decode: %w", err)
	}

	fmt.Printf("Generated: %s\n", result)
	fmt.Println("---")

	return nil
}
