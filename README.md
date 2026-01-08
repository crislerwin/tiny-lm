# TinyLM - A Tiny Language Model from Scratch in Go

A clean, educational implementation of a transformer-based language model written in pure Go. This project demonstrates how to build a small GPT-style model without any external ML frameworks.

## Features

- **Training Support**: Backpropagation and training loop implementation
- **Pure Go Implementation**: No dependencies on TensorFlow, PyTorch, or other ML frameworks
- **Clean Architecture**: Well-organized package structure with separation of concerns
- **Educational**: Clear code with proper error handling and documentation
- **Modular Design**: Easily extensible components (tokenizer, model layers, math operations)
- **Unit Tests**: Comprehensive test coverage for core functionality

## Project Structure

```
tiny-lm/
├── main.go                    # Entry point
├── pkg/
│   ├── math/                  # Mathematical operations
│   │   ├── matrix.go         # Matrix/vector operations, activations
│   │   └── matrix_test.go    # Math tests
│   ├── tokenizer/            # Text tokenization
│   │   ├── tokenizer.go      # Word-level tokenizer
│   │   └── tokenizer_test.go # Tokenizer tests
│   ├── config/               # Model configuration
│   │   └── config.go         # Config loading and validation
│   └── model/                # Transformer model
│       ├── weights.go        # Weight loading from JSON
│       ├── layers.go         # Attention and feed-forward layers
│       └── transformer.go    # Full transformer implementation
└── tiny_gpt_weights.json     # Pre-trained weights
```

## Model Architecture

- **Embedding Layer**: Token + positional embeddings
- **Transformer Blocks**: Multi-head self-attention + feed-forward networks
- **Layer Normalization**: Applied before each sub-layer (pre-norm)
- **Residual Connections**: Skip connections around attention and FFN
- **Language Model Head**: Projects to vocabulary logits

### Configuration

The model supports the following hyperparameters:

- `d_model`: Model dimension (embedding size)
- `n_layers`: Number of transformer blocks
- `n_heads`: Number of attention heads
- `vocab`: List of vocabulary words
- `max_seq_len`: Maximum sequence length
- `eps`: Layer normalization epsilon (default: 1e-5)

## Installation

```bash
# Clone the repository
git clone https://github.com/crislerwin/tiny-lm.git
cd tiny-lm

# Initialize Go module (if needed)
go mod tidy
```

## Usage

### Training and Generation

```bash
# Run the model (Training + Generation)
go run main.go
```

### Programmatic Usage

```go
package main

import (
    "github.com/crislerwin/tiny-lm/pkg/model"
    "github.com/crislerwin/tiny-lm/pkg/tokenizer"
)

func main() {
    // Load pre-trained weights
    weights, err := model.LoadFromJSON("tiny_gpt_weights.json")
    if err != nil {
        panic(err)
    }

    // Create model and tokenizer
    transformer := model.NewTransformer(weights)
    tok := tokenizer.NewTokenizer(weights.Config.Vocab)

    // Encode input text
    tokens, _ := tok.Encode("the cat")

    // Generate continuation
    generated, _ := transformer.Generate(tokens, 3)

    // Decode output
    result, _ := tok.Decode(generated)
    println(result)
}
```

## Running Tests

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific package tests
go test ./pkg/math
go test ./pkg/tokenizer
```

## Building

```bash
# Build binary
go build -o tiny-lm

# Run binary
./tiny-lm
```

## How It Works

### 1. Tokenization
The tokenizer converts text to token IDs using a simple word-based vocabulary:
```go
tok.Encode("the cat") // Returns: [0, 1]
```

### 2. Embeddings
Each token gets an embedding vector plus a positional encoding:
```go
embedding = token_embedding[token_id] + positional_embedding[position]
```

### 3. Transformer Blocks
Each block applies:
- Layer normalization
- Multi-head self-attention with causal masking
- Residual connection
- Layer normalization
- Feed-forward network (Linear → GELU → Linear)
- Residual connection

### 4. Output Projection
Final layer norm followed by projection to vocabulary space:
```go
logits = LayerNorm(x) @ lm_head^T
probs = Softmax(logits)
```

## Mathematical Operations

All operations are implemented from scratch:

- **Matrix Multiplication**: Standard O(n³) algorithm
- **Gradients (Backpropagation)**: Manual backward pass for all operations
- **Softmax**: Numerically stable with max subtraction
- **GELU Activation**: Gaussian Error Linear Unit
- **Layer Normalization**: Per-row mean/variance normalization
- **Cross Entropy Loss**: For training evaluation

## Limitations

- **Small Vocabulary**: Limited to words in the training set
- **Greedy Decoding**: Uses argmax instead of sampling
- **CPU Only**: No GPU acceleration
- **Simple Tokenizer**: Word-level, no subword tokenization

## Future Improvements

- [ ] Add BPE/WordPiece tokenization
- [x] Implement training from scratch
- [ ] Add temperature-based sampling
- [ ] Support beam search
- [ ] Add more activation functions
- [ ] Optimize matrix operations
- [ ] Add model export/import utilities

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## Author

Created as an educational project to understand transformer architectures from first principles.
