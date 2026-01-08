package math

import (
	"math"
	"testing"
)

func TestMatMulBackward(t *testing.T) {
	// Simple scalar case disguised as 1x1 matrix
	// C = A * B
	// Let A = 2, B = 3 -> C = 6
	// dL/dC = 1
	// dL/dA = B * dL/dC = 3 * 1 = 3
	// dL/dB = A * dL/dC = 2 * 1 = 2

	a := Matrix{{2.0}}
	b := Matrix{{3.0}}
	gradOutput := Matrix{{1.0}}

	dA, dB, err := MatMulBackward(gradOutput, a, b)
	if err != nil {
		t.Fatalf("MatMulBackward failed: %v", err)
	}

	if math.Abs(dA[0][0]-3.0) > 1e-6 {
		t.Errorf("Expected gradA 3.0, got %f", dA[0][0])
	}
	if math.Abs(dB[0][0]-2.0) > 1e-6 {
		t.Errorf("Expected gradB 2.0, got %f", dB[0][0])
	}
}

func TestGeluBackward(t *testing.T) {
	// x = 0
	// GELU(0) = 0
	// d(GELU)/dx at 0 = 0.5

	x := Matrix{{0.0}}
	gradOutput := Matrix{{1.0}}

	dx := GeluBackward(x, gradOutput)

	if math.Abs(dx[0][0]-0.5) > 1e-6 {
		t.Errorf("Expected gradX at 0 to be 0.5, got %f", dx[0][0])
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	// Logits: [0, 0] -> Softmax: [0.5, 0.5]
	// Target: 0
	// Loss: -log(0.5) = 0.693147
	// Grad: [0.5 - 1, 0.5] = [-0.5, 0.5]

	logits := Matrix{{0.0, 0.0}}
	targets := []int{0}

	loss, grad, err := CrossEntropyLoss(logits, targets)
	if err != nil {
		t.Fatalf("CrossEntropyLoss failed: %v", err)
	}

	expectedLoss := -math.Log(0.5)
	if math.Abs(loss-expectedLoss) > 1e-6 {
		t.Errorf("Expected loss %f, got %f", expectedLoss, loss)
	}

	if math.Abs(grad[0][0]-(-0.5)) > 1e-6 {
		t.Errorf("Expected grad[0][0] -0.5, got %f", grad[0][0])
	}
	if math.Abs(grad[0][1]-0.5) > 1e-6 {
		t.Errorf("Expected grad[0][1] 0.5, got %f", grad[0][1])
	}
}
