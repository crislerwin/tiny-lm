package model

import (
	"fmt"
	"math"

	tmath "github.com/crislerwin/tiny-lm/pkg/math"
)

// MultiHeadAttention performs multi-head self-attention
func MultiHeadAttention(x tmath.Matrix, prefix string, w *Weights, mask tmath.Matrix) (tmath.Matrix, error) {
	seqLen := len(x)
	dModel := w.Config.DModel
	nHeads := w.Config.NHeads
	dK := dModel / nHeads

	// Get weight matrices
	Wq, err := w.GetMatrix(prefix + ".attn.W_q.weight")
	if err != nil {
		return nil, err
	}
	Wk, err := w.GetMatrix(prefix + ".attn.W_k.weight")
	if err != nil {
		return nil, err
	}
	Wv, err := w.GetMatrix(prefix + ".attn.W_v.weight")
	if err != nil {
		return nil, err
	}
	Wo, err := w.GetMatrix(prefix + ".attn.W_o.weight")
	if err != nil {
		return nil, err
	}

	// Projections
	Q, err := tmath.MatMul(x, Wq)
	if err != nil {
		return nil, fmt.Errorf("Q projection failed: %w", err)
	}
	K, err := tmath.MatMul(x, Wk)
	if err != nil {
		return nil, fmt.Errorf("K projection failed: %w", err)
	}
	V, err := tmath.MatMul(x, Wv)
	if err != nil {
		return nil, fmt.Errorf("V projection failed: %w", err)
	}

	// Process each attention head
	headOuts := make([]tmath.Matrix, nHeads)

	for h := 0; h < nHeads; h++ {
		// Extract head slices
		qHead := tmath.NewMatrix(seqLen, dK)
		kHead := tmath.NewMatrix(seqLen, dK)
		vHead := tmath.NewMatrix(seqLen, dK)

		start := h * dK
		end := (h + 1) * dK

		for i := 0; i < seqLen; i++ {
			copy(qHead[i], Q[i][start:end])
			copy(kHead[i], K[i][start:end])
			copy(vHead[i], V[i][start:end])
		}

		// Attention scores: Q @ K.T
		scores, err := tmath.MatMul(qHead, tmath.Transpose(kHead))
		if err != nil {
			return nil, fmt.Errorf("attention scores failed: %w", err)
		}

		// Scale and apply mask
		scale := math.Sqrt(float64(dK))
		for r := 0; r < seqLen; r++ {
			for c := 0; c < seqLen; c++ {
				scores[r][c] /= scale
				if mask != nil && mask[r][c] == 0 {
					scores[r][c] = -1e9
				}
			}
			scores[r] = tmath.Softmax(scores[r])
		}

		// Attention output
		headOuts[h], err = tmath.MatMul(scores, vHead)
		if err != nil {
			return nil, fmt.Errorf("attention output failed: %w", err)
		}
	}

	// Concatenate heads
	concat := tmath.NewMatrix(seqLen, dModel)
	for i := 0; i < seqLen; i++ {
		for h := 0; h < nHeads; h++ {
			copy(concat[i][h*dK:(h+1)*dK], headOuts[h][i])
		}
	}

	// Final linear projection
	output, err := tmath.MatMul(concat, Wo)
	if err != nil {
		return nil, fmt.Errorf("output projection failed: %w", err)
	}

	return output, nil
}

// FeedForward performs the feed-forward network
func FeedForward(x tmath.Matrix, prefix string, w *Weights) (tmath.Matrix, error) {
	W1, err := w.GetMatrix(prefix + ".ff.linear1.weight")
	if err != nil {
		return nil, err
	}
	b1, err := w.GetVector(prefix + ".ff.linear1.bias")
	if err != nil {
		return nil, err
	}
	W2, err := w.GetMatrix(prefix + ".ff.linear2.weight")
	if err != nil {
		return nil, err
	}
	b2, err := w.GetVector(prefix + ".ff.linear2.bias")
	if err != nil {
		return nil, err
	}

	// First linear layer
	curr, err := tmath.MatMul(x, W1)
	if err != nil {
		return nil, fmt.Errorf("first linear layer failed: %w", err)
	}

	// Add bias
	for i := range curr {
		for j := range curr[i] {
			curr[i][j] += b1[j]
		}
	}

	// GELU activation
	curr = tmath.ApplyGelu(curr)

	// Second linear layer
	curr, err = tmath.MatMul(curr, W2)
	if err != nil {
		return nil, fmt.Errorf("second linear layer failed: %w", err)
	}

	// Add bias
	for i := range curr {
		for j := range curr[i] {
			curr[i][j] += b2[j]
		}
	}

	return curr, nil
}

// TransformerBlock processes one transformer block
func TransformerBlock(x tmath.Matrix, idx int, w *Weights, mask tmath.Matrix) (tmath.Matrix, error) {
	prefix := fmt.Sprintf("blocks.%d", idx)

	// Layer Norm 1
	ln1W, err := w.GetVector(prefix + ".norm1.weight")
	if err != nil {
		return nil, err
	}
	ln1B, err := w.GetVector(prefix + ".norm1.bias")
	if err != nil {
		return nil, err
	}
	norm1 := tmath.LayerNorm(x, ln1W, ln1B, w.Config.Eps)

	// Multi-head attention
	attn, err := MultiHeadAttention(norm1, prefix, w, mask)
	if err != nil {
		return nil, fmt.Errorf("attention failed in block %d: %w", idx, err)
	}

	// Residual connection
	x, err = tmath.Add(x, attn)
	if err != nil {
		return nil, fmt.Errorf("residual connection failed in block %d: %w", idx, err)
	}

	// Layer Norm 2
	ln2W, err := w.GetVector(prefix + ".norm2.weight")
	if err != nil {
		return nil, err
	}
	ln2B, err := w.GetVector(prefix + ".norm2.bias")
	if err != nil {
		return nil, err
	}
	norm2 := tmath.LayerNorm(x, ln2W, ln2B, w.Config.Eps)

	// Feed-forward
	ff, err := FeedForward(norm2, prefix, w)
	if err != nil {
		return nil, fmt.Errorf("feed-forward failed in block %d: %w", idx, err)
	}

	// Residual connection
	x, err = tmath.Add(x, ff)
	if err != nil {
		return nil, fmt.Errorf("residual connection failed in block %d: %w", idx, err)
	}

	return x, nil
}
