package model

import (
	"fmt"

	tmath "github.com/crislerwin/tiny-lm/pkg/math"
)

// Backpropagate performs the full backward pass
func (t *Transformer) Backpropagate(gradLogits tmath.Matrix, cache *TransformerCache) error {
	w := t.Weights

	// 5. LM Head
	lmHead, _ := w.GetMatrix("lm_head.weight")

	// dL/d(LN_f_out) = gradLogits @ lmHead
	// dL/dLmHead = gradLogits.T @ LN_f_out

	dLdLmHead, _ := tmath.MatMul(tmath.Transpose(gradLogits), cache.FinalNormIn) // Wait, cache.FinalNormIn = LN output?
	// In ForwardTrain:
	// finalNormIn := x
	// x = LayerNorm(x)
	// logits = x @ lmHead^T

	// cache.FinalNormIn is input to LN.
	// We need x (output of LN) to compute dLmHead.
	// We can recompute x = LayerNorm(cache.FinalNormIn)

	lnFW, _ := w.GetVector("ln_f.weight")
	lnFB, _ := w.GetVector("ln_f.bias")
	lnOut := tmath.LayerNorm(cache.FinalNormIn, lnFW, lnFB, w.Config.Eps)

	// dLmHead = gradLogits.T @ lnOut
	// (B, V).T @ (B, D) -> (V, D) matches (V, D)
	dLdLmHead, _ = tmath.MatMul(tmath.Transpose(gradLogits), lnOut)
	if err := w.AddGradient("lm_head.weight", dLdLmHead); err != nil {
		return err
	}

	// dL/dLnOut = gradLogits @ lmHead
	dLdLnOut, _ := tmath.MatMul(gradLogits, lmHead)

	// 4. Final Layer Norm
	dLdFinalNormIn, dLnFW, dLnFB := tmath.LayerNormBackward(dLdLnOut, cache.FinalNormIn, lnFW, w.Config.Eps)
	if err := w.AddGradient("ln_f.weight", dLnFW); err != nil {
		return err
	}
	if err := w.AddGradient("ln_f.bias", dLnFB); err != nil {
		return err
	}

	// 3. Blocks Backward
	gradX := dLdFinalNormIn

	for i := w.Config.NLayers - 1; i >= 0; i-- {
		var err error
		gradX, err = TransformerBlockBackward(gradX, cache.BlockCaches[i], i, w)
		if err != nil {
			return fmt.Errorf("block %d backward failed: %w", i, err)
		}
	}

	// 2. Embeddings Backward
	// gradX is now dL/d(Inputs). Inputs = TokenEmbed + PosEmbed
	// So dL/dToken = gradX, dL/dPos = gradX
	// We need to scatter gradX back to embeddings.

	// Token Embeddings
	// dTokenEmbed is matrix (Vocab, D).
	// We iterate over the sequence and add gradX[t] to dTokenEmbed[token_id]

	// We need the input tokens! cache doesn't have them?
	// cache.Inputs is the sum matrix. We didn't cache the raw tokens.
	// We need to pass tokens to Backpropagate.

	// Oops.
	return fmt.Errorf("missing tokens for embedding gradient")
}

// TrainStep performs one training step
func (t *Transformer) TrainStep(tokens []int, targets []int, lr float64) (float64, error) {
	// 1. Forward
	logits, cache, err := t.ForwardTrain(tokens)
	if err != nil {
		return 0, err
	}

	// 2. Loss
	loss, gradLogits, err := tmath.CrossEntropyLoss(logits, targets)
	if err != nil {
		return 0, err
	}

	// 3. Zero Gradients (optional if we don't accumulate across batches, here we clear first)
	// Actually we should create a new Grads map or clear it.
	t.Weights.Grads = make(map[string]interface{})

	// 4. Backward
	if err := t.BackpropagateWithTokens(tokens, gradLogits, cache); err != nil {
		return 0, err
	}

	// 5. Update Weights
	t.UpdateWeights(lr)

	return loss, nil
}

func (t *Transformer) BackpropagateWithTokens(tokens []int, gradLogits tmath.Matrix, cache *TransformerCache) error {
	// Re-implemented logic from above but with tokens

	w := t.Weights

	lmHead, _ := w.GetMatrix("lm_head.weight")
	lnFW, _ := w.GetVector("ln_f.weight")
	lnFB, _ := w.GetVector("ln_f.bias")
	lnOut := tmath.LayerNorm(cache.FinalNormIn, lnFW, lnFB, w.Config.Eps)

	dLdLmHead, _ := tmath.MatMul(tmath.Transpose(gradLogits), lnOut)
	if err := w.AddGradient("lm_head.weight", dLdLmHead); err != nil {
		return err
	}

	dLdLnOut, _ := tmath.MatMul(gradLogits, lmHead)

	dLdFinalNormIn, dLnFW, dLnFB := tmath.LayerNormBackward(dLdLnOut, cache.FinalNormIn, lnFW, w.Config.Eps)
	if err := w.AddGradient("ln_f.weight", dLnFW); err != nil {
		return err
	}
	if err := w.AddGradient("ln_f.bias", dLnFB); err != nil {
		return err
	}

	gradX := dLdFinalNormIn

	for i := w.Config.NLayers - 1; i >= 0; i-- {
		var err error
		gradX, err = TransformerBlockBackward(gradX, cache.BlockCaches[i], i, w)
		if err != nil {
			return err
		}
	}

	// Embedding Gradients
	dTokenEmbed, _ := w.GetMatrix("token_embed.weight") // Shape only
	// Actually we need to initialize a zero gradient matrix of that shape
	rowsT, colsT := dTokenEmbed.Shape()
	gradTokenEmbed := tmath.NewMatrix(rowsT, colsT)

	// Position Embeddings Gradients
	dPosEmbed, _ := w.GetMatrix("pos_embed.weight") // Shape only
	rowsP, colsP := dPosEmbed.Shape()
	gradPosEmbed := tmath.NewMatrix(rowsP, colsP)

	// Accumulate
	for i, tokenID := range tokens {
		// gradient for token embedding
		for j := 0; j < colsT; j++ {
			gradTokenEmbed[tokenID][j] += gradX[i][j]
		}
		// gradient for positional embedding
		if i < rowsP {
			for j := 0; j < colsP; j++ {
				gradPosEmbed[i][j] += gradX[i][j]
			}
		}
	}

	if err := w.AddGradient("token_embed.weight", gradTokenEmbed); err != nil {
		return err
	}
	if err := w.AddGradient("pos_embed.weight", gradPosEmbed); err != nil {
		return err
	}

	return nil
}

func (t *Transformer) UpdateWeights(lr float64) {
	for key, grad := range t.Weights.Grads {
		val, _ := t.Weights.Data[key]

		switch g := grad.(type) {
		case tmath.Matrix:
			v := val.(tmath.Matrix)
			rows, cols := v.Shape()
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					v[r][c] -= lr * g[r][c]
				}
			}
		case tmath.Vector:
			v := val.(tmath.Vector)
			for i := range v {
				v[i] -= lr * g[i]
			}
		}
	}
}
