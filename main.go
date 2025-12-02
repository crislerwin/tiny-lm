package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

func LoadJSONWeights(filename string) (*Weights, error) {
	file, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// Temporary struct for parsing
	var raw struct {
		DModel  int                    `json:"d_model"`
		NLayers int                    `json:"n_layers"`
		NHeads  int                    `json:"n_heads"`
		Vocab   []string               `json:"vocab"`
		Weights map[string]interface{} `json:"weights"`
	}

	if err := json.Unmarshal(file, &raw); err != nil {
		return nil, err
	}

	w := &Weights{
		DModel:  raw.DModel,
		NLayers: raw.NLayers,
		NHeads:  raw.NHeads,
		Vocab:   raw.Vocab,
		Data:    make(map[string]interface{}),
	}

	// Helper to cast []interface{} -> Matrix ([][]float64)
	toMatrix := func(v interface{}) Matrix {
		rawRows := v.([]interface{})
		rows := len(rawRows)
		if rows == 0 {
			return Matrix{}
		}

		// Check if it's 1D (Vector) or 2D (Matrix)
		// Note: JSON unmarshal might make vectors look like [x, y, z]
		// You might need distinct logic if your JSON treats Vectors as flat lists
		// But assuming your Python dump made them standard lists:

		_, isList := rawRows[0].([]interface{})
		if !isList {
			// It's a Vector, but we return a Matrix for consistency or panic?
			// Actually, your code distinguishes Vector vs Matrix types.
			panic("Expected Matrix, got Vector data")
		}

		cols := len(rawRows[0].([]interface{}))
		out := make(Matrix, rows)
		for i := range rawRows {
			row := rawRows[i].([]interface{})
			out[i] = make(Vector, cols)
			for j := range row {
				out[i][j] = row[j].(float64)
			}
		}
		return out
	}

	// Helper to cast []interface{} -> Vector ([]float64)
	toVector := func(v interface{}) Vector {
		rawList := v.([]interface{})
		out := make(Vector, len(rawList))
		for i, val := range rawList {
			out[i] = val.(float64)
		}
		return out
	}

	// Process the map
	for k, v := range raw.Weights {
		// Detect shape based on the key name or data depth
		// A simple heuristic: check the first element
		list, ok := v.([]interface{})
		if !ok {
			continue
		}

		if len(list) > 0 {
			if _, isNested := list[0].([]interface{}); isNested {
				w.Data[k] = toMatrix(v)
			} else {
				w.Data[k] = toVector(v)
			}
		}
	}

	return w, nil
}

// ============================================================================
// Types & Math Helpers (Replacing Numpy)
// ============================================================================

type Matrix [][]float64
type Vector []float64

// MatMul: Basic Matrix Multiplication (A @ B)
func MatMul(a, b Matrix) Matrix {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])

	if colsA != rowsB {
		panic("Shape mismatch in MatMul")
	}

	result := make(Matrix, rowsA)
	for i := 0; i < rowsA; i++ {
		result[i] = make(Vector, colsB)
		for j := 0; j < colsB; j++ {
			sum := 0.0
			for k := 0; k < colsA; k++ {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

// Add: Element-wise addition
func Add(a, b Matrix) Matrix {
	rows, cols := len(a), len(a[0])
	res := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		res[i] = make(Vector, cols)
		for j := 0; j < cols; j++ {
			res[i][j] = a[i][j] + b[i][j]
		}
	}
	return res
}

// Transpose: Swaps dimensions
func Transpose(m Matrix) Matrix {
	r, c := len(m), len(m[0])
	res := make(Matrix, c)
	for i := 0; i < c; i++ {
		res[i] = make(Vector, r)
		for j := 0; j < r; j++ {
			res[i][j] = m[j][i]
		}
	}
	return res
}

// ============================================================================
// Activation & Normalization Functions
// ============================================================================

func Gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// ApplyGelu: Applies GELU to a matrix element-wise
func ApplyGelu(m Matrix) Matrix {
	rows, cols := len(m), len(m[0])
	res := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		res[i] = make(Vector, cols)
		for j := 0; j < cols; j++ {
			res[i][j] = Gelu(m[i][j])
		}
	}
	return res
}

func Softmax(v Vector) Vector {
	maxVal := -math.MaxFloat64
	for _, val := range v {
		if val > maxVal {
			maxVal = val
		}
	}

	sumExp := 0.0
	res := make(Vector, len(v))
	for i, val := range v {
		res[i] = math.Exp(val - maxVal)
		sumExp += res[i]
	}
	for i := range res {
		res[i] /= sumExp
	}
	return res
}

func LayerNorm(x Matrix, w, b Vector, eps float64) Matrix {
	rows, cols := len(x), len(x[0])
	res := make(Matrix, rows)

	for i := 0; i < rows; i++ {
		// Calculate mean and variance per row
		sum := 0.0
		for _, val := range x[i] {
			sum += val
		}
		mean := sum / float64(cols)

		variance := 0.0
		for _, val := range x[i] {
			variance += (val - mean) * (val - mean)
		}
		variance /= float64(cols)

		// Normalize
		res[i] = make(Vector, cols)
		stdDev := math.Sqrt(variance + eps)
		for j := 0; j < cols; j++ {
			norm := (x[i][j] - mean) / stdDev
			res[i][j] = w[j]*norm + b[j]
		}
	}
	return res
}

// ============================================================================
// Transformer Modules
// ============================================================================

// Weights container to mimic the Python dictionary
type Weights struct {
	Data    map[string]interface{}
	Vocab   []string
	DModel  int
	NLayers int
	NHeads  int
}

func MultiHeadAttention(x Matrix, prefix string, w *Weights, mask Matrix) Matrix {
	seqLen := len(x)
	dModel := w.DModel
	nHeads := w.NHeads
	dK := dModel / nHeads

	// Get weights (Assuming already transposed for Go MatMul: Input * Weight)
	Wq := w.Data[prefix+".attn.W_q.weight"].(Matrix)
	Wk := w.Data[prefix+".attn.W_k.weight"].(Matrix)
	Wv := w.Data[prefix+".attn.W_v.weight"].(Matrix)
	Wo := w.Data[prefix+".attn.W_o.weight"].(Matrix)

	// Projections
	Q := MatMul(x, Wq)
	K := MatMul(x, Wk)
	V := MatMul(x, Wv)

	// We calculate attention per head to avoid complex 4D tensor reshaping in Go
	headOuts := make([]Matrix, nHeads)

	for h := 0; h < nHeads; h++ {
		// Slice Q, K, V for this head
		qHead := make(Matrix, seqLen)
		kHead := make(Matrix, seqLen)
		vHead := make(Matrix, seqLen)

		start := h * dK
		end := (h + 1) * dK

		for i := 0; i < seqLen; i++ {
			qHead[i] = Q[i][start:end]
			kHead[i] = K[i][start:end]
			vHead[i] = V[i][start:end]
		}

		// Attention Scores: Q @ K.T
		scores := MatMul(qHead, Transpose(kHead))

		// Scale and Mask
		scale := math.Sqrt(float64(dK))
		for r := 0; r < seqLen; r++ {
			for c := 0; c < seqLen; c++ {
				scores[r][c] /= scale
				if mask != nil && mask[r][c] == 0 {
					scores[r][c] = -1e9
				}
			}
			// Softmax per row
			scores[r] = Softmax(scores[r])
		}

		// Attention * V
		headOuts[h] = MatMul(scores, vHead)
	}

	// Concatenate heads
	concat := make(Matrix, seqLen)
	for i := 0; i < seqLen; i++ {
		concat[i] = make(Vector, dModel)
		for h := 0; h < nHeads; h++ {
			copy(concat[i][h*dK:(h+1)*dK], headOuts[h][i])
		}
	}

	// Final linear projection
	return MatMul(concat, Wo)
}

func FeedForward(x Matrix, prefix string, w *Weights) Matrix {
	W1 := w.Data[prefix+".ff.linear1.weight"].(Matrix)
	b1 := w.Data[prefix+".ff.linear1.bias"].(Vector)
	W2 := w.Data[prefix+".ff.linear2.weight"].(Matrix)
	b2 := w.Data[prefix+".ff.linear2.bias"].(Vector)

	// x @ W1 + b1
	curr := MatMul(x, W1)
	for i := range curr {
		for j := range curr[i] {
			curr[i][j] += b1[j]
		}
	}

	// GELU
	curr = ApplyGelu(curr)

	// curr @ W2 + b2
	curr = MatMul(curr, W2)
	for i := range curr {
		for j := range curr[i] {
			curr[i][j] += b2[j]
		}
	}

	return curr
}

func TransformerBlock(x Matrix, idx int, w *Weights, mask Matrix) Matrix {
	prefix := fmt.Sprintf("blocks.%d", idx)

	// 1. Layer Norm 1
	ln1W := w.Data[prefix+".norm1.weight"].(Vector)
	ln1B := w.Data[prefix+".norm1.bias"].(Vector)
	norm1 := LayerNorm(x, ln1W, ln1B, 1e-5)

	// 2. Attention + Residual
	attn := MultiHeadAttention(norm1, prefix, w, mask)
	x = Add(x, attn)

	// 3. Layer Norm 2
	ln2W := w.Data[prefix+".norm2.weight"].(Vector)
	ln2B := w.Data[prefix+".norm2.bias"].(Vector)
	norm2 := LayerNorm(x, ln2W, ln2B, 1e-5)

	// 4. Feed Forward + Residual
	ff := FeedForward(norm2, prefix, w)
	x = Add(x, ff)

	return x
}

func FullTransformer(tokens []int, w *Weights) Matrix {
	seqLen := len(tokens)

	// Embeddings
	tokenEmbed := w.Data["token_embed.weight"].(Matrix)
	posEmbed := w.Data["pos_embed.weight"].(Matrix)

	x := make(Matrix, seqLen)
	for i, t := range tokens {
		x[i] = make(Vector, w.DModel)
		// Token + Pos embedding
		for j := 0; j < w.DModel; j++ {
			x[i][j] = tokenEmbed[t][j] + posEmbed[i][j]
		}
	}

	// Causal Mask (Lower Triangular)
	mask := make(Matrix, seqLen)
	for i := 0; i < seqLen; i++ {
		mask[i] = make(Vector, seqLen)
		for j := 0; j <= i; j++ {
			mask[i][j] = 1.0
		}
	}

	// Blocks
	for i := 0; i < w.NLayers; i++ {
		x = TransformerBlock(x, i, w, mask)
	}

	// Final Norm
	lnFW := w.Data["ln_f.weight"].(Vector)
	lnFB := w.Data["ln_f.bias"].(Vector)
	x = LayerNorm(x, lnFW, lnFB, 1e-5)

	// Logits (x @ lm_head.T)
	lmHead := w.Data["lm_head.weight"].(Matrix)
	logits := MatMul(x, Transpose(lmHead))

	return logits
}

// ============================================================================
// Main Execution
// ============================================================================

func Generate(prompt string, w *Weights) {
	words := strings.Fields(prompt)
	var tokens []int

	// Simple tokenizer mapping
	for _, word := range words {
		found := false
		for idx, v := range w.Vocab {
			if v == word {
				tokens = append(tokens, idx)
				found = true
				break
			}
		}
		if !found {
			fmt.Printf("Word not in vocab: %s\n", word)
			return
		}
	}

	fmt.Printf("Input: '%s' -> Generating...\n", prompt)

	for i := 0; i < 3; i++ { // Generate 3 words
		logits := FullTransformer(tokens, w)
		lastLogits := logits[len(logits)-1]
		probs := Softmax(lastLogits)

		// Greedy selection (argmax)
		maxIdx := 0
		maxP := -1.0
		for idx, p := range probs {
			if p > maxP {
				maxP = p
				maxIdx = idx
			}
		}

		nextWord := w.Vocab[maxIdx]
		fmt.Printf(" Step %d: %s (%.2f%%)\n", i+1, nextWord, maxP*100)

		tokens = append(tokens, maxIdx)
		if nextWord == "END" {
			break
		}
	}
	fmt.Println()
}

func main() {
	// Initialize Mock Weights (In real life, load from JSON/NPZ)
	// We set d_model=4 just to make it runnable without crashing
	w, err := LoadJSONWeights("tiny_gpt_weights.json")
	if err != nil {
		panic(err)
	}
	Generate("the cat", w) // Run Inference
	Generate("the cat", w)
}

// Helper to fill weights with random data so the code runs
func initMockWeights(w *Weights) {
	// "rng" is safer naming to avoid collisions
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Helpers to create random tensors
	// We rename arguments to 'rows' and 'cols' to avoid shadowing
	randMat := func(rows, cols int) Matrix {
		m := make(Matrix, rows)
		for i := range m {
			m[i] = make(Vector, cols)
			for j := range m[i] {
				m[i][j] = rng.Float64() - 0.5
			}
		}
		return m
	}

	randVec := func(n int) Vector {
		v := make(Vector, n)
		for i := range v {
			v[i] = rng.Float64() - 0.5
		}
		return v
	}

	w.Data["token_embed.weight"] = randMat(len(w.Vocab), w.DModel)
	w.Data["pos_embed.weight"] = randMat(16, w.DModel) // Max seq len 16
	w.Data["lm_head.weight"] = randMat(len(w.Vocab), w.DModel)
	w.Data["ln_f.weight"] = randVec(w.DModel)
	w.Data["ln_f.bias"] = randVec(w.DModel)

	for i := 0; i < w.NLayers; i++ {
		p := fmt.Sprintf("blocks.%d", i)
		// Attn
		w.Data[p+".attn.W_q.weight"] = randMat(w.DModel, w.DModel)
		w.Data[p+".attn.W_k.weight"] = randMat(w.DModel, w.DModel)
		w.Data[p+".attn.W_v.weight"] = randMat(w.DModel, w.DModel)
		w.Data[p+".attn.W_o.weight"] = randMat(w.DModel, w.DModel)
		// Norms
		w.Data[p+".norm1.weight"] = randVec(w.DModel)
		w.Data[p+".norm1.bias"] = randVec(w.DModel)
		w.Data[p+".norm2.weight"] = randVec(w.DModel)
		w.Data[p+".norm2.bias"] = randVec(w.DModel)
		// FF
		w.Data[p+".ff.linear1.weight"] = randMat(w.DModel, w.DModel)
		w.Data[p+".ff.linear1.bias"] = randVec(w.DModel)
		w.Data[p+".ff.linear2.weight"] = randMat(w.DModel, w.DModel)
		w.Data[p+".ff.linear2.bias"] = randVec(w.DModel)
	}
}
