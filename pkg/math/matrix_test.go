package math

import (
	"math"
	"testing"
)

func TestMatMul(t *testing.T) {
	tests := []struct {
		name    string
		a       Matrix
		b       Matrix
		want    Matrix
		wantErr bool
	}{
		{
			name: "basic 2x2 multiplication",
			a:    Matrix{{1, 2}, {3, 4}},
			b:    Matrix{{5, 6}, {7, 8}},
			want: Matrix{{19, 22}, {43, 50}},
		},
		{
			name:    "dimension mismatch",
			a:       Matrix{{1, 2}},
			b:       Matrix{{1}, {2}, {3}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MatMul(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("MatMul() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !matrixEqual(got, tt.want) {
				t.Errorf("MatMul() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	tests := []struct {
		name    string
		a       Matrix
		b       Matrix
		want    Matrix
		wantErr bool
	}{
		{
			name: "basic addition",
			a:    Matrix{{1, 2}, {3, 4}},
			b:    Matrix{{5, 6}, {7, 8}},
			want: Matrix{{6, 8}, {10, 12}},
		},
		{
			name:    "dimension mismatch",
			a:       Matrix{{1, 2}},
			b:       Matrix{{1, 2, 3}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Add(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !matrixEqual(got, tt.want) {
				t.Errorf("Add() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	m := Matrix{{1, 2, 3}, {4, 5, 6}}
	want := Matrix{{1, 4}, {2, 5}, {3, 6}}
	got := Transpose(m)

	if !matrixEqual(got, want) {
		t.Errorf("Transpose() = %v, want %v", got, want)
	}
}

func TestSoftmax(t *testing.T) {
	v := Vector{1.0, 2.0, 3.0}
	result := Softmax(v)

	// Check sum equals 1
	sum := 0.0
	for _, val := range result {
		sum += val
	}

	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax sum = %v, want 1.0", sum)
	}

	// Check values are positive
	for i, val := range result {
		if val <= 0 {
			t.Errorf("Softmax[%d] = %v, want positive value", i, val)
		}
	}
}

func TestGelu(t *testing.T) {
	tests := []struct {
		input float64
		want  float64
	}{
		{0.0, 0.0},
		{1.0, 0.8411},   // approximate
		{-1.0, -0.1588}, // approximate
	}

	for _, tt := range tests {
		got := Gelu(tt.input)
		if math.Abs(got-tt.want) > 0.01 {
			t.Errorf("Gelu(%v) = %v, want ~%v", tt.input, got, tt.want)
		}
	}
}

// Helper function to compare matrices
func matrixEqual(a, b Matrix) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > 1e-9 {
				return false
			}
		}
	}
	return true
}
