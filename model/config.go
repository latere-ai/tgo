// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
)

// Config is config.json parsed: specs/004-model-graph.md §5's table, and the
// two fields §7 refuses on.
//
// Every shape the graph builds comes from here. The spec's shapes are symbolic
// and this struct is where the numbers enter (004-D8), so a checkpoint whose
// dimensions no spec anticipated builds correctly or is refused by name, and
// never builds a graph sized from a constant somebody wrote down.
type Config struct {
	// Architecture is architectures[0], the registry key.
	Architecture string

	// HiddenSize is d, the residual stream width.
	HiddenSize int

	// NumLayers is L.
	NumLayers int

	// NumHeads is H, the query heads.
	NumHeads int

	// NumKVHeads is H_kv, the key/value heads. It defaults to NumHeads — the
	// non-grouped case — and must divide it: H/H_kv queries share each cache
	// entry, and a non-integer ratio is not a grouping.
	NumKVHeads int

	// HeadDim is d_h.
	//
	// It defaults to HiddenSize/NumHeads and Qwen3 sets it explicitly to
	// something else: the 0.6B checkpoint is d=1024 over H=16 heads with
	// d_h=128, so the default would be 64 and every attention shape would be
	// half its true width. When the field is present it is used; the default is
	// never allowed to override the file (specs/004-model-graph.md §5).
	HeadDim int

	// IntermediateSize is f, the MLP width.
	IntermediateSize int

	// VocabSize is V. It must equal the rows of the embedding table, which is
	// checked against the checkpoint by [Check] rather than here.
	VocabSize int

	// RMSNormEps is ε, added under the square root of every RMSNorm.
	//
	// Held as float32 because that is the width it binds at: the graph passes
	// it as an f32 constant, and a float64 here would only be rounded at the
	// port. Rounding once, at parse, is what makes the value in this struct the
	// value the kernel sees.
	RMSNormEps float32

	// RoPETheta is the rotary base: 10^6 for Qwen3, against Llama's 10^4.
	RoPETheta float32

	// LayerTypes is what each layer of the stack holds between steps, in model
	// order. Nil is a dense stack: every layer is [LayerFullAttention].
	//
	// It is the one place [Declare] learns that a model is hybrid
	// (specs/023-cache-kinds.md §2.1). Reading it from a checkpoint is
	// specs/024-qwen3-5-architecture.md's.
	LayerTypes LayerSchedule

	// Recurrent is the geometry of the gated-delta layers, and is nil unless
	// LayerTypes has some.
	Recurrent *Recurrent

	// TieWordEmbeddings reports whether the LM head is the embedding table
	// transposed. When set, the checkpoint has no lm_head.weight and the loader
	// uploads two planes from one file tensor (004-D7).
	TieWordEmbeddings bool

	// MaxPositionEmbeddings is advisory only. Cache capacity is a session
	// parameter, not a model constant (005-D2), so nothing in the graph reads
	// this; it is parsed because a caller choosing a capacity wants to know what
	// the model was trained for.
	MaxPositionEmbeddings int
}

// KVGroup is how many query heads share one key/value head, H/H_kv. ParseConfig
// has already refused a config where the division is not exact.
func (c *Config) KVGroup() int { return c.NumHeads / c.NumKVHeads }

// QWidth is H·d_h, the width of the q projection's output — which is not d
// whenever head_dim is set explicitly to something other than d/H.
func (c *Config) QWidth() int { return c.NumHeads * c.HeadDim }

// KVWidth is H_kv·d_h, the width of the k and v projections' output.
func (c *Config) KVWidth() int { return c.NumKVHeads * c.HeadDim }

// rawConfig is config.json as it is written.
//
// Pointers rather than values throughout, because "absent" and "zero" are
// different answers for every field here: head_dim absent takes a default and
// head_dim: 0 is a config to refuse, and reporting the second as the first
// would hand a caller a graph built from d/H when their file said something
// else.
type rawConfig struct {
	Architectures         []string        `json:"architectures"`
	HiddenSize            *int            `json:"hidden_size"`
	NumHiddenLayers       *int            `json:"num_hidden_layers"`
	NumAttentionHeads     *int            `json:"num_attention_heads"`
	NumKeyValueHeads      *int            `json:"num_key_value_heads"`
	HeadDim               *int            `json:"head_dim"`
	IntermediateSize      *int            `json:"intermediate_size"`
	VocabSize             *int            `json:"vocab_size"`
	RMSNormEps            *float64        `json:"rms_norm_eps"`
	RoPETheta             *float64        `json:"rope_theta"`
	TieWordEmbeddings     *bool           `json:"tie_word_embeddings"`
	MaxPositionEmbeddings *int            `json:"max_position_embeddings"`
	RoPEScaling           json.RawMessage `json:"rope_scaling"`
	SlidingWindow         *int            `json:"sliding_window"`
	UseSlidingWindow      *bool           `json:"use_sliding_window"`
}

// ParseConfig parses and validates config.json against
// specs/004-model-graph.md §5 and §7.
//
// Every refusal names the config field that caused it. A config the builder
// cannot honour is refused here, at parse, rather than approximated: §7's rows
// are each a case where the wrong answer runs, produces shapes that check, and
// degrades output somewhere a test does not look.
func ParseConfig(raw json.RawMessage) (*Config, error) {
	var r rawConfig
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, fmt.Errorf("model: parse config: %w", err)
	}
	if len(r.Architectures) == 0 || r.Architectures[0] == "" {
		return nil, fmt.Errorf("model: config: architectures[0] is required; it is the registry key")
	}
	c := &Config{Architecture: r.Architectures[0]}

	var err error
	if c.HiddenSize, err = requirePositive("hidden_size", r.HiddenSize); err != nil {
		return nil, err
	}
	if c.NumLayers, err = requirePositive("num_hidden_layers", r.NumHiddenLayers); err != nil {
		return nil, err
	}
	if c.NumHeads, err = requirePositive("num_attention_heads", r.NumAttentionHeads); err != nil {
		return nil, err
	}
	if c.IntermediateSize, err = requirePositive("intermediate_size", r.IntermediateSize); err != nil {
		return nil, err
	}
	if c.VocabSize, err = requirePositive("vocab_size", r.VocabSize); err != nil {
		return nil, err
	}
	if c.MaxPositionEmbeddings, err = requirePositive("max_position_embeddings", r.MaxPositionEmbeddings); err != nil {
		return nil, err
	}
	if c.RMSNormEps, err = requireFloat("rms_norm_eps", r.RMSNormEps); err != nil {
		return nil, err
	}
	if c.RoPETheta, err = requireFloat("rope_theta", r.RoPETheta); err != nil {
		return nil, err
	}

	// head_dim: the file wins wherever it speaks. Only a config that omits it
	// gets d/H, and only when that division is exact -- an inexact one means
	// the default this code would invent is not the width the weights have.
	switch {
	case r.HeadDim != nil:
		c.HeadDim = *r.HeadDim
	case c.HiddenSize%c.NumHeads != 0:
		return nil, fmt.Errorf("model: config: head_dim is absent and its default "+
			"hidden_size/num_attention_heads is not an integer (%d/%d)", c.HiddenSize, c.NumHeads)
	default:
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	if c.HeadDim <= 0 {
		return nil, fmt.Errorf("model: config: head_dim is %d; it must be positive", c.HeadDim)
	}
	if c.HeadDim%2 != 0 {
		// RoPE rotates channel pairs, so an odd rotary dimension has a channel
		// with no partner. accel refuses it at graph time; refusing it here
		// names the config field that produced it.
		return nil, fmt.Errorf("model: config: head_dim is %d; RoPE rotates pairs of "+
			"channels and refuses an odd rotary dimension", c.HeadDim)
	}

	// num_key_value_heads: absent means the model is not grouped, so H_kv = H.
	c.NumKVHeads = c.NumHeads
	if r.NumKeyValueHeads != nil {
		if c.NumKVHeads, err = requirePositive("num_key_value_heads", r.NumKeyValueHeads); err != nil {
			return nil, err
		}
	}
	if c.NumHeads%c.NumKVHeads != 0 {
		return nil, fmt.Errorf("model: config: num_attention_heads (%d) is not a multiple of "+
			"num_key_value_heads (%d); H/H_kv queries share each cache entry and a "+
			"non-integer ratio is not a grouping", c.NumHeads, c.NumKVHeads)
	}

	if r.TieWordEmbeddings != nil {
		c.TieWordEmbeddings = *r.TieWordEmbeddings
	}

	if err := checkRoPEScaling(r.RoPEScaling); err != nil {
		return nil, err
	}
	if err := checkSlidingWindow(r.SlidingWindow, r.UseSlidingWindow); err != nil {
		return nil, err
	}
	return c, nil
}

// checkRoPEScaling refuses any rope_scaling this graph does not implement,
// which today is every one of them.
//
// A wrong scaling is not a wrong answer at first. It is correct for the
// positions the model was trained on and diverges past them, so the failure
// arrives after four thousand tokens of good output, which is the worst place
// for a failure to arrive.
func checkRoPEScaling(raw json.RawMessage) error {
	if isJSONNull(raw) {
		return nil
	}
	// A present key with the value null is how every real Hugging Face config
	// spells "no scaling", and json.RawMessage captures those four bytes rather
	// than leaving the field nil. Refusing on key presence would refuse Qwen3.
	var kind struct {
		Type     string `json:"type"`
		RoPEType string `json:"rope_type"`
	}
	name := "unnamed"
	if err := json.Unmarshal(raw, &kind); err == nil {
		if kind.RoPEType != "" {
			name = kind.RoPEType
		} else if kind.Type != "" {
			name = kind.Type
		}
	}
	return fmt.Errorf("model: config: rope_scaling is set (%s) and no scaling is "+
		"supported; a wrong scaling is fine for four thousand tokens and then is not", name)
}

// checkSlidingWindow refuses a config that asks for windowed attention.
//
// The graph has no window: every query attends to every cached key. A model
// trained with a window and run without one is not slightly wrong, it is
// correct up to the window length and wrong after it, with nothing in the
// output to say where the boundary was.
//
// The reading here is that the window is configured when use_sliding_window is
// true, or when sliding_window carries a positive length and use_sliding_window
// does not explicitly turn it off. Qwen3-0.6B ships sliding_window: null with
// use_sliding_window: false, which is the shape both clauses must accept.
func checkSlidingWindow(window *int, use *bool) error {
	if use != nil && *use {
		return fmt.Errorf("model: config: use_sliding_window is true and this graph " +
			"attends to the whole cache; output past the window would be silently wrong")
	}
	// A length with no flag beside it: some configs carry sliding_window alone
	// and mean it. use_sliding_window: false is the explicit off switch, and
	// honouring it is what lets Qwen3-0.6B through.
	if window != nil && *window > 0 && use == nil {
		return fmt.Errorf("model: config: sliding_window is %d and this graph attends "+
			"to the whole cache; output past the window would be silently wrong", *window)
	}
	return nil
}

// isJSONNull reports whether a raw field is absent or literally null.
func isJSONNull(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

// requirePositive reads a required dimension, separating "the file does not say"
// from "the file says something impossible" so the error tells the reader which
// of the two they are looking at.
func requirePositive(field string, v *int) (int, error) {
	if v == nil {
		return 0, fmt.Errorf("model: config: %s is required", field)
	}
	if *v <= 0 {
		return 0, fmt.Errorf("model: config: %s is %d; it must be positive", field, *v)
	}
	return *v, nil
}

// requireFloat reads a required f32 constant. It refuses a value that does not
// survive the narrowing: the graph binds these as f32 scalars, so a magnitude
// float64 holds and float32 does not would reach the kernel as an infinity.
func requireFloat(field string, v *float64) (float32, error) {
	if v == nil {
		return 0, fmt.Errorf("model: config: %s is required", field)
	}
	if *v <= 0 {
		return 0, fmt.Errorf("model: config: %s is %g; it must be positive", field, *v)
	}
	f := float32(*v)
	if math.IsInf(float64(f), 0) {
		return 0, fmt.Errorf("model: config: %s is %g, which is not finite as an f32 "+
			"scalar; it would reach the kernel as an infinity", field, *v)
	}
	if f == 0 {
		return 0, fmt.Errorf("model: config: %s is %g, which rounds to zero as an f32 "+
			"scalar", field, *v)
	}
	return f, nil
}
