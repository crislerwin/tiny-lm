package model

import (
	tmath "github.com/crislerwin/tiny-lm/pkg/math"
)

// AttentionCache holds intermediate values for attention backward pass
type AttentionCache struct {
	Input  tmath.Matrix
	Q      tmath.Matrix
	K      tmath.Matrix
	V      tmath.Matrix
	Scores tmath.Matrix // After softmax
}

// FeedForwardCache holds intermediate values for FFN backward pass
type FeedForwardCache struct {
	Input      tmath.Matrix
	Linear1Out tmath.Matrix // Before GELU
	GeluOut    tmath.Matrix // After GELU
}

// LayerNormCache holds intermediate values for LN backward pass
type LayerNormCache struct {
	Input tmath.Matrix
	Mean  tmath.Vector
	Std   tmath.Vector // actually we might just need Input if we recompute, strictly speaking we just need Input for the backward formula I implemented?
	// The backward implementation I wrote takes (gradOutput, x, gamma, eps).
	// It internally recomputes mean/std. So just Input is sufficient.
}

// TransformerBlockCache holds caches for one block
type TransformerBlockCache struct {
	Input     tmath.Matrix
	Norm1In   tmath.Matrix // Input to Norm1 = block input
	AttnCache *AttentionCache
	Norm2In   tmath.Matrix // Input to Norm2 = Norm1Out + AttnOut
	FFCache   *FeedForwardCache
}

// TransformerCache holds caches for the entire model
type TransformerCache struct {
	TokenEmbeds tmath.Matrix
	PosEmbeds   tmath.Matrix
	Inputs      tmath.Matrix // Sum of embeds
	BlockCaches []*TransformerBlockCache
	FinalNormIn tmath.Matrix
	Logits      tmath.Matrix
}
