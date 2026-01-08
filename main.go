package main

import (
	"fmt"
	"log"

	"github.com/crislerwin/tiny-lm/pkg/model"
	"github.com/crislerwin/tiny-lm/pkg/tokenizer"
)

func main() {
	fmt.Println("Starting TinyLM...")
	// Load weights from JSON file
	weights, err := model.LoadFromJSON("tiny_gpt_weights.json")
	if err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// Create transformer model
	transformer := model.NewTransformer(weights)

	// Create tokenizer
	tok := tokenizer.NewTokenizer(weights.Config.Vocab)

	fmt.Println("Starting training...")

	// Training data: "the cat sat on mat" -> predict next token
	// tokens: [the, cat, sat, on, mat]
	// input: [the, cat, sat, on]
	// target: [cat, sat, on, mat]

	text := "the cat sat on mat END"
	tokens, err := tok.Encode(text)
	if err != nil {
		log.Fatalf("Failed to encode training text: %v", err)
	}

	inputs := tokens[:len(tokens)-1]
	targets := tokens[1:]

	lr := 0.01
	for epoch := 0; epoch < 50; epoch++ {
		loss, err := transformer.TrainStep(inputs, targets, lr)
		if err != nil {
			log.Fatalf("Training failed at epoch %d: %v", epoch, err)
		}
		if epoch%5 == 0 {
			fmt.Printf("Epoch %d: Loss = %.4f\n", epoch, loss)
		}
	}

	fmt.Println("\nTraining complete! Testing generation:")

	// Run inference examples
	prompts := []string{
		"the cat",
		"the dog", // OOV or rare? "dog" is in vocab
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

	// Generate continuation (5 new tokens)
	generated, err := transformer.Generate(tokens, 5)
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
