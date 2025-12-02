package model

import (
	"fmt"

	tmath "github.com/crislerwin/tiny-lm/pkg/math"
)

// Transformer represents the full transformer model
type Transformer struct {
	Weights *Weights
}

// NewTransformer creates a new transformer model
func NewTransformer(weights *Weights) *Transformer {
	return &Transformer{
		Weights: weights,
	}
}

// Forward performs a forward pass through the transformer
func (t *Transformer) Forward(tokens []int) (tmath.Matrix, error) {
	seqLen := len(tokens)
	if seqLen == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}

	// Get embeddings
	tokenEmbed, err := t.Weights.GetMatrix("token_embed.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to get token embeddings: %w", err)
	}
	posEmbed, err := t.Weights.GetMatrix("pos_embed.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to get position embeddings: %w", err)
	}

	// Validate sequence length
	if seqLen > len(posEmbed) {
		return nil, fmt.Errorf("sequence length %d exceeds max position embeddings %d", seqLen, len(posEmbed))
	}

	// Create initial embeddings
	x := tmath.NewMatrix(seqLen, t.Weights.Config.DModel)
	for i, tokenID := range tokens {
		if tokenID < 0 || tokenID >= len(tokenEmbed) {
			return nil, fmt.Errorf("invalid token ID %d at position %d", tokenID, i)
		}
		for j := 0; j < t.Weights.Config.DModel; j++ {
			x[i][j] = tokenEmbed[tokenID][j] + posEmbed[i][j]
		}
	}

	// Create causal mask (lower triangular)
	mask := tmath.NewMatrix(seqLen, seqLen)
	for i := range seqLen {
		for j := 0; j <= i; j++ {
			mask[i][j] = 1.0
		}
	}

	// Process through transformer blocks
	for i := 0; i < t.Weights.Config.NLayers; i++ {
		x, err = TransformerBlock(x, i, t.Weights, mask)
		if err != nil {
			return nil, fmt.Errorf("block %d failed: %w", i, err)
		}
	}

	// Final layer normalization
	lnFW, err := t.Weights.GetVector("ln_f.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to get final norm weights: %w", err)
	}
	lnFB, err := t.Weights.GetVector("ln_f.bias")
	if err != nil {
		return nil, fmt.Errorf("failed to get final norm bias: %w", err)
	}
	x = tmath.LayerNorm(x, lnFW, lnFB, t.Weights.Config.Eps)

	// Language model head
	lmHead, err := t.Weights.GetMatrix("lm_head.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to get language model head: %w", err)
	}
	logits, err := tmath.MatMul(x, tmath.Transpose(lmHead))
	if err != nil {
		return nil, fmt.Errorf("failed to compute logits: %w", err)
	}

	return logits, nil
}

// Generate generates text continuation
func (t *Transformer) Generate(tokens []int, maxTokens int) ([]int, error) {
	if maxTokens <= 0 {
		return tokens, nil
	}

	generated := make([]int, len(tokens))
	copy(generated, tokens)

	for i := range maxTokens {
		// Forward pass
		logits, err := t.Forward(generated)
		if err != nil {
			return nil, fmt.Errorf("generation step %d failed: %w", i, err)
		}

		// Get logits for last token
		lastLogits := logits[len(logits)-1]
		probs := tmath.Softmax(lastLogits)

		// Greedy selection (argmax)
		maxIdx := 0
		maxProb := -1.0
		for idx, prob := range probs {
			if prob > maxProb {
				maxProb = prob
				maxIdx = idx
			}
		}

		// Append new token
		generated = append(generated, maxIdx)

		// Stop if we generate END token
		if maxIdx < len(t.Weights.Config.Vocab) && t.Weights.Config.Vocab[maxIdx] == "END" {
			break
		}
	}

	return generated, nil
}
