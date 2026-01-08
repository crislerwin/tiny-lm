package model

import (
	"fmt"
	"math"

	tmath "github.com/crislerwin/tiny-lm/pkg/math"
)

// AddGradient accumulates gradient into w.Grads
func (w *Weights) AddGradient(key string, grad interface{}) error {
	if w.Grads == nil {
		w.Grads = make(map[string]interface{})
	}

	// If it doesn't exist, just set it
	if _, exists := w.Grads[key]; !exists {
		w.Grads[key] = grad
		return nil
	}

	// Accumulate
	current := w.Grads[key]

	switch g := grad.(type) {
	case tmath.Matrix:
		cMatrix, ok := current.(tmath.Matrix)
		if !ok {
			return fmt.Errorf("gradient type mismatch for %s", key)
		}
		sum, err := tmath.Add(cMatrix, g)
		if err != nil {
			return err
		}
		w.Grads[key] = sum
	case tmath.Vector:
		cVector, ok := current.(tmath.Vector)
		if !ok {
			return fmt.Errorf("gradient type mismatch for %s", key)
		}
		// Implement Vector Add manually as it's not in math package explicitly or Add works on Matrices
		// Let's assume we need a Vector Add or cast to matrix?
		// Actually Vector is []float64.
		if len(g) != len(cVector) {
			return fmt.Errorf("vector length mismatch")
		}
		newVec := make(tmath.Vector, len(g))
		for i := range g {
			newVec[i] = cVector[i] + g[i]
		}
		w.Grads[key] = newVec
	default:
		return fmt.Errorf("unsupported gradient type")
	}
	return nil
}

// FeedForwardBackward computes gradients for FeedForward layer
func FeedForwardBackward(gradOutput tmath.Matrix, cache *FeedForwardCache, prefix string, w *Weights) (tmath.Matrix, error) {
	// Re-get weights
	W2, _ := w.GetMatrix(prefix + ".ff.linear2.weight")

	// 1. Second Linear Layer Backward
	// output = linear2 + bias
	// dL/d(linear2) = gradOutput
	// dL/dW2 = geluOut.T @ gradOutput
	// dL/db2 = sum(gradOutput, axis=0)
	// dL/d(geluOut) = gradOutput @ W2.T

	dLdLinear2 := gradOutput

	// Gradient w.r.t W2
	dLinear2W, err := tmath.MatMul(tmath.Transpose(cache.GeluOut), dLdLinear2)
	if err != nil {
		return nil, err
	}
	if err := w.AddGradient(prefix+".ff.linear2.weight", dLinear2W); err != nil {
		return nil, err
	}

	// Gradient w.r.t b2
	dLinear2B := tmath.NewVector(len(dLdLinear2[0]))
	for i := range dLdLinear2 {
		for j := range dLdLinear2[i] {
			dLinear2B[j] += dLdLinear2[i][j]
		}
	}
	if err := w.AddGradient(prefix+".ff.linear2.bias", dLinear2B); err != nil {
		return nil, err
	}

	// Gradient w.r.t GeluOut
	dLdGeluOut, err := tmath.MatMul(dLdLinear2, tmath.Transpose(W2))
	if err != nil {
		return nil, err
	}

	// 2. GELU Backward
	dLdLinear1Out := tmath.GeluBackward(cache.Linear1Out, dLdGeluOut)

	// 3. First Linear Layer Backward
	// dL/dW1 = input.T @ dLdLinear1Out
	dLinear1W, err := tmath.MatMul(tmath.Transpose(cache.Input), dLdLinear1Out)
	if err != nil {
		return nil, err
	}
	if err := w.AddGradient(prefix+".ff.linear1.weight", dLinear1W); err != nil {
		return nil, err
	}

	// Gradient w.r.t b1
	dLinear1B := tmath.NewVector(len(dLdLinear1Out[0]))
	for i := range dLdLinear1Out {
		for j := range dLdLinear1Out[i] {
			dLinear1B[j] += dLdLinear1Out[i][j]
		}
	}
	if err := w.AddGradient(prefix+".ff.linear1.bias", dLinear1B); err != nil {
		return nil, err
	}

	// Gradient w.r.t Input
	W1, _ := w.GetMatrix(prefix + ".ff.linear1.weight")
	gradInput, err := tmath.MatMul(dLdLinear1Out, tmath.Transpose(W1))
	if err != nil {
		return nil, err
	}

	return gradInput, nil
}

// MultiHeadAttentionBackward computes gradients for MHA
func MultiHeadAttentionBackward(gradOutput tmath.Matrix, cache *AttentionCache, prefix string, w *Weights) (tmath.Matrix, error) {
	seqLen := len(gradOutput)
	dModel := w.Config.DModel
	nHeads := w.Config.NHeads
	dK := dModel / nHeads

	// 1. Output Projection
	Wo, _ := w.GetMatrix(prefix + ".attn.W_o.weight")

	// Note: cache.Input is 'x' input to MHA.
	// To get dWo, we need input *to* Wo, which is 'concat'.
	// We didn't cache 'concat'. We can regenerate everything since we have Q, K, V.
	// Or we can assume we only need dL/dConcat to propagate back.
	// But we need dWo. dWo = Concat.T @ gradOutput.
	// So we MUST regenerate 'Concat'.

	// Regenerate head outputs and concat
	// This duplicates forward logic, but is safe.
	headOuts := make([]tmath.Matrix, nHeads)
	mask := tmath.NewMatrix(seqLen, seqLen) // Dummy mask? No, mask matters for scores.
	// Wait, we need the mask used in forward! It wasn't cached!
	// But mask is deterministic (causal).
	for i := range seqLen {
		for j := 0; j <= i; j++ {
			mask[i][j] = 1.0
		}
	}

	// Wait, in forward we stored Q, K, V.

	for h := 0; h < nHeads; h++ {
		start := h * dK
		end := (h + 1) * dK

		qHead := tmath.NewMatrix(seqLen, dK)
		kHead := tmath.NewMatrix(seqLen, dK)
		vHead := tmath.NewMatrix(seqLen, dK)

		for i := 0; i < seqLen; i++ {
			copy(qHead[i], cache.Q[i][start:end])
			copy(kHead[i], cache.K[i][start:end])
			copy(vHead[i], cache.V[i][start:end])
		}

		scores, _ := tmath.MatMul(qHead, tmath.Transpose(kHead))
		scale := math.Sqrt(float64(dK))
		for r := 0; r < seqLen; r++ {
			for c := 0; c < seqLen; c++ {
				scores[r][c] /= scale
				if mask[r][c] == 0 {
					scores[r][c] = -1e9
				}
			}
			scores[r] = tmath.Softmax(scores[r])
		}

		headOuts[h], _ = tmath.MatMul(scores, vHead)
	}

	concat := tmath.NewMatrix(seqLen, dModel)
	for i := 0; i < seqLen; i++ {
		for h := 0; h < nHeads; h++ {
			copy(concat[i][h*dK:(h+1)*dK], headOuts[h][i])
		}
	}

	// Now we have concat.
	dWo, _ := tmath.MatMul(tmath.Transpose(concat), gradOutput)
	if err := w.AddGradient(prefix+".attn.W_o.weight", dWo); err != nil {
		return nil, err
	}

	dLdConcat, _ := tmath.MatMul(gradOutput, tmath.Transpose(Wo))

	// Backprop through heads
	dLdQ := tmath.NewMatrix(seqLen, dModel)
	dLdK := tmath.NewMatrix(seqLen, dModel)
	dLdV := tmath.NewMatrix(seqLen, dModel)

	Wq, _ := w.GetMatrix(prefix + ".attn.W_q.weight")
	Wk, _ := w.GetMatrix(prefix + ".attn.W_k.weight")
	Wv, _ := w.GetMatrix(prefix + ".attn.W_v.weight")

	for h := 0; h < nHeads; h++ {
		start := h * dK
		end := (h + 1) * dK

		// Grads for this head from concat
		dLdHeadOut := tmath.NewMatrix(seqLen, dK)
		for i := 0; i < seqLen; i++ {
			copy(dLdHeadOut[i], dLdConcat[i][start:end])
		}

		// Regenerate intermediates for this head
		qHead := tmath.NewMatrix(seqLen, dK)
		kHead := tmath.NewMatrix(seqLen, dK)
		vHead := tmath.NewMatrix(seqLen, dK)
		for i := 0; i < seqLen; i++ {
			copy(qHead[i], cache.Q[i][start:end])
			copy(kHead[i], cache.K[i][start:end])
			copy(vHead[i], cache.V[i][start:end])
		}

		scores, _ := tmath.MatMul(qHead, tmath.Transpose(kHead))
		scale := math.Sqrt(float64(dK))
		for r := 0; r < seqLen; r++ {
			for c := 0; c < seqLen; c++ {
				scores[r][c] /= scale
				if mask[r][c] == 0 {
					scores[r][c] = -1e9
				}
			}
			scores[r] = tmath.Softmax(scores[r])
		}

		// dL/dV = Scores.T @ dLdHeadOut
		dLdVHead, _ := tmath.MatMul(tmath.Transpose(scores), dLdHeadOut)

		// dL/dScores = dLdHeadOut @ V.T
		dLdScores, _ := tmath.MatMul(dLdHeadOut, tmath.Transpose(vHead))

		// Backprop Softmax
		// We have scores (probabilities).
		// dSoftmax = scores * (grad - sum(grad*scores)) - this is for dense, but row-wise:
		// S_ij (dL_ij - sum_k S_ik dL_ik)
		for r := 0; r < seqLen; r++ {
			// Compute sum_k S_ik dL_ik
			sum := 0.0
			for c := 0; c < seqLen; c++ {
				sum += scores[r][c] * dLdScores[r][c]
			}
			for c := 0; c < seqLen; c++ {
				dLdScores[r][c] = scores[r][c] * (dLdScores[r][c] - sum)
			}
		}

		// Backprop Mask and Scale
		// Mask: gradient doesn't pass through masked values (they were -1e9, softmaxed to 0)
		for r := 0; r < seqLen; r++ {
			for c := 0; c < seqLen; c++ {
				dLdScores[r][c] /= scale
				if mask[r][c] == 0 {
					dLdScores[r][c] = 0 // Technically it was already ~0 contribution, effectively cutting graph
				}
			}
		}

		// dL/dQ = dLdScores @ K
		dLdQHead, _ := tmath.MatMul(dLdScores, kHead)

		// dL/dK = dLdScores.T @ Q
		dLdKHead, _ := tmath.MatMul(tmath.Transpose(dLdScores), qHead)

		// Accumulate into dLdQ, dLdK, dLdV
		for i := 0; i < seqLen; i++ {
			copy(dLdQ[i][start:end], dLdQHead[i])
			copy(dLdK[i][start:end], dLdKHead[i])
			copy(dLdV[i][start:end], dLdVHead[i])
		}
	}

	// Gradients for Wq, Wk, Wv
	dWq, _ := tmath.MatMul(tmath.Transpose(cache.Input), dLdQ)
	if err := w.AddGradient(prefix+".attn.W_q.weight", dWq); err != nil {
		return nil, err
	}

	dWk, _ := tmath.MatMul(tmath.Transpose(cache.Input), dLdK)
	if err := w.AddGradient(prefix+".attn.W_k.weight", dWk); err != nil {
		return nil, err
	}

	dWv, _ := tmath.MatMul(tmath.Transpose(cache.Input), dLdV)
	if err := w.AddGradient(prefix+".attn.W_v.weight", dWv); err != nil {
		return nil, err
	}

	// dL/dX = dLdQ @ Wq.T + dLdK @ Wk.T + dLdV @ Wv.T
	dx1, _ := tmath.MatMul(dLdQ, tmath.Transpose(Wq))
	dx2, _ := tmath.MatMul(dLdK, tmath.Transpose(Wk))
	dx3, _ := tmath.MatMul(dLdV, tmath.Transpose(Wv))

	dx, _ := tmath.Add(dx1, dx2)
	dx, _ = tmath.Add(dx, dx3)

	return dx, nil
}

func TransformerBlockBackward(gradOutput tmath.Matrix, cache *TransformerBlockCache, idx int, w *Weights) (tmath.Matrix, error) {
	prefix := fmt.Sprintf("blocks.%d", idx)

	// Residual 2: grad splits to FF and shortcut
	gradFF := gradOutput   // dL/dFF
	gradRes2 := gradOutput // dL/dRes1 (shortcut)

	// Feed Forward Backward
	dLdNorm2, err := FeedForwardBackward(gradFF, cache.FFCache, prefix, w)
	if err != nil {
		return nil, err
	}

	// Add grad from residual
	dLdNorm2, _ = tmath.Add(dLdNorm2, gradRes2)

	// Norm 2 Backward
	ln2W, _ := w.GetVector(prefix + ".norm2.weight")
	dLdRes1, dLn2W, dLn2B := tmath.LayerNormBackward(dLdNorm2, cache.Norm2In, ln2W, w.Config.Eps)

	if err := w.AddGradient(prefix+".norm2.weight", dLn2W); err != nil {
		return nil, err
	}
	if err := w.AddGradient(prefix+".norm2.bias", dLn2B); err != nil {
		return nil, err
	}

	// Residual 1: grad splits to Attn and shortcut
	gradAttn := dLdRes1
	gradRes1 := dLdRes1

	// Attention Backward
	dLdNorm1, err := MultiHeadAttentionBackward(gradAttn, cache.AttnCache, prefix, w)
	if err != nil {
		return nil, err
	}

	// Add grad from residual
	dLdNorm1, _ = tmath.Add(dLdNorm1, gradRes1)

	// Norm 1 Backward
	ln1W, _ := w.GetVector(prefix + ".norm1.weight")
	dLdInput, dLn1W, dLn1B := tmath.LayerNormBackward(dLdNorm1, cache.Norm1In, ln1W, w.Config.Eps)

	if err := w.AddGradient(prefix+".norm1.weight", dLn1W); err != nil {
		return nil, err
	}
	if err := w.AddGradient(prefix+".norm1.bias", dLn1B); err != nil {
		return nil, err
	}

	return dLdInput, nil
}
