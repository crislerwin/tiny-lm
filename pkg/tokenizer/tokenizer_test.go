package tokenizer

import (
	"reflect"
	"testing"
)

func TestTokenizer_Encode(t *testing.T) {
	vocab := []string{"the", "cat", "sat", "on", "mat"}
	tok := NewTokenizer(vocab)

	tests := []struct {
		name    string
		text    string
		want    []int
		wantErr bool
	}{
		{
			name: "valid text",
			text: "the cat sat",
			want: []int{0, 1, 2},
		},
		{
			name:    "unknown word",
			text:    "the dog",
			wantErr: true,
		},
		{
			name:    "empty text",
			text:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.Encode(tt.text)
			if (err != nil) != tt.wantErr {
				t.Errorf("Encode() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Encode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTokenizer_Decode(t *testing.T) {
	vocab := []string{"the", "cat", "sat", "on", "mat"}
	tok := NewTokenizer(vocab)

	tests := []struct {
		name    string
		tokens  []int
		want    string
		wantErr bool
	}{
		{
			name:   "valid tokens",
			tokens: []int{0, 1, 2},
			want:   "the cat sat",
		},
		{
			name:    "invalid token ID",
			tokens:  []int{0, 99},
			wantErr: true,
		},
		{
			name:    "negative token ID",
			tokens:  []int{0, -1},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.Decode(tt.tokens)
			if (err != nil) != tt.wantErr {
				t.Errorf("Decode() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("Decode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTokenizer_VocabSize(t *testing.T) {
	vocab := []string{"the", "cat", "sat", "on", "mat"}
	tok := NewTokenizer(vocab)

	if got := tok.VocabSize(); got != 5 {
		t.Errorf("VocabSize() = %v, want 5", got)
	}
}
