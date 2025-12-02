package math

import (
	"fmt"
	"math"
)

// Matrix represents a 2D matrix
type Matrix [][]float64

// Vector represents a 1D vector
type Vector []float64

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make(Vector, cols)
	}
	return m
}

// NewVector creates a new vector with given size
func NewVector(size int) Vector {
	return make(Vector, size)
}

// Shape returns the dimensions of the matrix
func (m Matrix) Shape() (int, int) {
	if len(m) == 0 {
		return 0, 0
	}
	return len(m), len(m[0])
}

// MatMul performs matrix multiplication (A @ B)
func MatMul(a, b Matrix) (Matrix, error) {
	rowsA, colsA := a.Shape()
	rowsB, colsB := b.Shape()

	if colsA != rowsB {
		return nil, fmt.Errorf("shape mismatch in MatMul: (%d,%d) x (%d,%d)", rowsA, colsA, rowsB, colsB)
	}

	result := NewMatrix(rowsA, colsB)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			sum := 0.0
			for k := 0; k < colsA; k++ {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result, nil
}

// Add performs element-wise addition
func Add(a, b Matrix) (Matrix, error) {
	rowsA, colsA := a.Shape()
	rowsB, colsB := b.Shape()

	if rowsA != rowsB || colsA != colsB {
		return nil, fmt.Errorf("shape mismatch in Add: (%d,%d) vs (%d,%d)", rowsA, colsA, rowsB, colsB)
	}

	result := NewMatrix(rowsA, colsA)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result, nil
}

// Transpose swaps matrix dimensions
func Transpose(m Matrix) Matrix {
	rows, cols := m.Shape()
	result := NewMatrix(cols, rows)
	for i := 0; i < cols; i++ {
		for j := 0; j < rows; j++ {
			result[i][j] = m[j][i]
		}
	}
	return result
}

// Gelu applies the GELU activation function
func Gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// ApplyGelu applies GELU to a matrix element-wise
func ApplyGelu(m Matrix) Matrix {
	rows, cols := m.Shape()
	result := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = Gelu(m[i][j])
		}
	}
	return result
}

// Softmax applies the softmax function to a vector
func Softmax(v Vector) Vector {
	if len(v) == 0 {
		return Vector{}
	}

	maxVal := -math.MaxFloat64
	for _, val := range v {
		if val > maxVal {
			maxVal = val
		}
	}

	sumExp := 0.0
	result := make(Vector, len(v))
	for i, val := range v {
		result[i] = math.Exp(val - maxVal)
		sumExp += result[i]
	}

	for i := range result {
		result[i] /= sumExp
	}
	return result
}

// LayerNorm applies layer normalization
func LayerNorm(x Matrix, weight, bias Vector, eps float64) Matrix {
	rows, cols := x.Shape()
	result := NewMatrix(rows, cols)

	for i := 0; i < rows; i++ {
		// Calculate mean
		sum := 0.0
		for _, val := range x[i] {
			sum += val
		}
		mean := sum / float64(cols)

		// Calculate variance
		variance := 0.0
		for _, val := range x[i] {
			variance += (val - mean) * (val - mean)
		}
		variance /= float64(cols)

		// Normalize
		stdDev := math.Sqrt(variance + eps)
		for j := 0; j < cols; j++ {
			norm := (x[i][j] - mean) / stdDev
			result[i][j] = weight[j]*norm + bias[j]
		}
	}
	return result
}
