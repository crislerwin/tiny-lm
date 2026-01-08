package math

import (
	"fmt"
	"math"
)

// MatMulBackward computes gradients for matrix multiplication C = A @ B
// Returns dL/dA and dL/dB given dL/dC (gradOutput)
func MatMulBackward(gradOutput, a, b Matrix) (Matrix, Matrix, error) {
	// dL/dA = dL/dC @ B^T
	bT := Transpose(b)
	gradA, err := MatMul(gradOutput, bT)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compute gradA: %w", err)
	}

	// dL/dB = A^T @ dL/dC
	aT := Transpose(a)
	gradB, err := MatMul(aT, gradOutput)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compute gradB: %w", err)
	}

	return gradA, gradB, nil
}

// AddBackward computes gradients for addition C = A + B
// Since dC/dA = 1 and dC/dB = 1, gradA = gradOutput and gradB = gradOutput
func AddBackward(gradOutput Matrix) (Matrix, Matrix) {
	// Deep copy gradOutput for both gradients to ensure safety
	rows, cols := gradOutput.Shape()
	gradA := NewMatrix(rows, cols)
	gradB := NewMatrix(rows, cols)

	for i := 0; i < rows; i++ {
		copy(gradA[i], gradOutput[i])
		copy(gradB[i], gradOutput[i])
	}

	return gradA, gradB
}

// GeluBackward computes gradient for GELU activation
// approx: 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * (sqrt(2/pi) * (1 + 3 * 0.044715 * x^2))
// z = sqrt(2/pi) * (x + 0.044715 * x^3)
func GeluBackward(x Matrix, gradOutput Matrix) Matrix {
	rows, cols := x.Shape()
	gradInput := NewMatrix(rows, cols)

	const c1 = 0.044715
	sqrt2pi := math.Sqrt(2.0 / math.Pi)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := x[i][j]
			val3 := val * val * val

			inner := sqrt2pi * (val + c1*val3)
			tanhInner := math.Tanh(inner)

			// Derivative calculation
			// d/dx(GELU) = 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh^2(inner)) * d/dx(inner)
			// d/dx(inner) = sqrt2pi * (1 + 3 * c1 * x^2)

			dInner := sqrt2pi * (1.0 + 3.0*c1*val*val)
			sech2 := 1.0 - tanhInner*tanhInner

			dGelu := 0.5*(1.0+tanhInner) + 0.5*val*sech2*dInner

			gradInput[i][j] = gradOutput[i][j] * dGelu
		}
	}
	return gradInput
}

// LayerNormBackward computes gradients for Layer Normalization
// Returns gradX, gradGamma (weight), gradBeta (bias)
func LayerNormBackward(gradOutput, x Matrix, gamma Vector, eps float64) (Matrix, Vector, Vector) {
	rows, cols := x.Shape()
	gradX := NewMatrix(rows, cols)
	gradGamma := NewVector(cols)
	gradBeta := NewVector(cols)

	for i := 0; i < rows; i++ {
		// Recompute mean and std (needed for gradient)
		mean := 0.0
		for _, v := range x[i] {
			mean += v
		}
		mean /= float64(cols)

		varSum := 0.0
		for _, v := range x[i] {
			varSum += (v - mean) * (v - mean)
		}
		std := math.Sqrt((varSum / float64(cols)) + eps)
		invStd := 1.0 / std

		// Gradients for this row
		dStd := 0.0
		dMean := 0.0

		for j := 0; j < cols; j++ {
			xNorm := (x[i][j] - mean) * invStd

			// Accumulate gradients for gamma and beta
			gradGamma[j] += gradOutput[i][j] * xNorm
			gradBeta[j] += gradOutput[i][j]

			// Backprop through affine transformation
			dXNorm := gradOutput[i][j] * gamma[j]

			// Backprop through normalization
			// dL/dv = -0.5 * Sum(dL/dx_norm * (x - mu) * v^(-1.5))
			dStd += dXNorm * (x[i][j] - mean) * (-invStd * invStd)

			// dL/dmu = Sum(dL/dx_norm * -1/std) + dL/dv * dv/dmu
			dMean += dXNorm * (-invStd)
		}

		// Term from dStd to dVar
		// dv/ds = 0.5 * (var + eps)^(-0.5) -- wait std is sqrt(var)
		// already handled std = sqrt(var), so dVar part is included implicitly?
		// No, let's follow standard LN backprop form strictly

		// Let's use the explicit gradients for LN:
		// dx_hat = gradOutput * gamma
		// dx = 1/N * invStd * (N*dx_hat - Sum(dx_hat) - x_hat * Sum(dx_hat * x_hat))

		// First pass: compute sums
		sumDxHat := 0.0
		sumDxHatXHat := 0.0

		for j := 0; j < cols; j++ {
			dxHat := gradOutput[i][j] * gamma[j]
			xHat := (x[i][j] - mean) * invStd
			sumDxHat += dxHat
			sumDxHatXHat += dxHat * xHat
		}

		// Second pass: compute dx

		// Correct formula:
		// dL/dx_i = (1 / sigma) * (dL/dx_hat_i - (1/N)*sum(dL/dx_hat) - (x_hat_i/N)*sum(dL/dx_hat * x_hat))

		for j := 0; j < cols; j++ {
			dxHat := gradOutput[i][j] * gamma[j]
			xHat := (x[i][j] - mean) * invStd

			term1 := dxHat
			term2 := sumDxHat / float64(cols)
			term3 := xHat * sumDxHatXHat / float64(cols)

			gradX[i][j] = invStd * (term1 - term2 - term3)
		}
	}

	return gradX, gradGamma, gradBeta
}

// CrossEntropyLoss computes the loss and gradients for Softmax + CrossEntropy
// logits: (batch_size, vocab_size)
// targets: (batch_size) - token indices
// Returns: loss (scalar), gradLogits (batch_size, vocab_size)
func CrossEntropyLoss(logits Matrix, targets []int) (float64, Matrix, error) {
	rows, cols := logits.Shape()
	if len(targets) != rows {
		return 0, nil, fmt.Errorf("batch size mismatch: logits %d vs targets %d", rows, len(targets))
	}

	gradLogits := NewMatrix(rows, cols)
	totalLoss := 0.0

	for i := 0; i < rows; i++ {
		// Softmax for this row
		probs := Softmax(logits[i])

		targetIdx := targets[i]
		if targetIdx < 0 || targetIdx >= cols {
			return 0, nil, fmt.Errorf("target index out of bounds: %d", targetIdx)
		}

		// Negative Log Likelihood
		if probs[targetIdx] > 0 {
			totalLoss -= math.Log(probs[targetIdx])
		} else {
			// Avoid log(0)
			totalLoss -= math.Log(1e-10)
		}

		// Gradient of CrossEntropy + Softmax is (p - y)
		// where y is 1-hot vector of target
		for j := 0; j < cols; j++ {
			if j == targetIdx {
				gradLogits[i][j] = probs[j] - 1.0
			} else {
				gradLogits[i][j] = probs[j]
			}
			// Divide by batch size here? typically done at optimization step or loss sum
			// Let's keep it as sum gradient, normalize by batch size later if needed.
			// Or standard: average loss over batch?
		}
	}

	return totalLoss / float64(rows), gradLogits, nil
}
