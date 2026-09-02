// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"encoding/json"
	"fmt"

	"github.com/latere-ai/tgo/chat"
)

// Qwen3Architecture is the architectures[0] value Qwen3 checkpoints carry, and
// the key this package registers under.
const Qwen3Architecture = "Qwen3ForCausalLM"

// The checkpoint tensor names outside the layer stack.
const (
	qwen3Embed     = "model.embed_tokens.weight"
	qwen3FinalNorm = "model.norm.weight"
	qwen3LMHead    = "lm_head.weight"
)

// layerPrefix is the templating specs/004-model-graph.md §4 writes as
// "model.layers.ℓ.".
const layerPrefix = "model.layers.%d."

func init() { Register(Qwen3Architecture, newQwen3) }

// qwen3 is the Qwen3 builder. It holds the parsed config and nothing else: the
// weight map is a function of the config, so there is no state to keep in step.
type qwen3 struct{ cfg *Config }

// newQwen3 is the registry constructor.
func newQwen3(raw json.RawMessage) (Builder, error) {
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	return &qwen3{cfg: c}, nil
}

// Config is the parsed config.json.
func (m *qwen3) Config() *Config { return m.cfg }

// Template is the Qwen3 chat renderer (specs/003-chat-template.md).
func (m *qwen3) Template() chat.Renderer { return chat.Qwen3() }

// Weights is specs/004-model-graph.md §4's table with the layer templating
// expanded and the shapes filled in from the config.
func (m *qwen3) Weights() []WeightSpec { return qwen3Weights(m.cfg) }

// layerTensor is one row of §4's table for a tensor inside the layer stack. The
// shapes are functions of the config because §4's table is written in symbols
// and 004-D8 keeps it that way: a spec cannot go stale against a checkpoint it
// does not contain.
type layerTensor struct {
	suffix    string // the part after "model.layers.ℓ."
	port      string // the part after "ℓ."
	shape     func(c *Config) []int
	kind      Kind
	transpose bool
	permute   bool
	heads     func(c *Config) int // only read when permute is set
}

// qwen3Layer is §4's table, layer rows only, in the order a layer uses them.
//
// The permute column is the one that is not obvious from the shapes. q_proj and
// k_proj carry rotated channels, so their output channels are reordered into
// accel's interleaved pairing. v_proj does not: RoPE is applied to q and k only,
// so V never enters the rotated basis, and o_proj contracts against attention
// output, which is a mixture of unrotated V rows and is therefore unrotated too. The q_norm and k_norm gains permute with 1 head because their d_h
// values are one head's channels, shared across heads: the gain must follow the
// channels it scales or QK-norm scales the wrong ones (004-D9).
var qwen3Layer = []layerTensor{
	{
		suffix: "input_layernorm.weight", port: "attn_norm",
		shape: func(c *Config) []int { return []int{c.HiddenSize} },
		kind:  KindGain,
	},
	{
		suffix: "self_attn.q_proj.weight", port: "wq",
		shape:     func(c *Config) []int { return []int{c.QWidth(), c.HiddenSize} },
		kind:      KindProjection,
		transpose: true,
		permute:   true,
		heads:     func(c *Config) int { return c.NumHeads },
	},
	{
		suffix: "self_attn.k_proj.weight", port: "wk",
		shape:     func(c *Config) []int { return []int{c.KVWidth(), c.HiddenSize} },
		kind:      KindProjection,
		transpose: true,
		permute:   true,
		heads:     func(c *Config) int { return c.NumKVHeads },
	},
	{
		suffix: "self_attn.v_proj.weight", port: "wv",
		shape:     func(c *Config) []int { return []int{c.KVWidth(), c.HiddenSize} },
		kind:      KindProjection,
		transpose: true,
	},
	{
		suffix: "self_attn.o_proj.weight", port: "wo",
		shape:     func(c *Config) []int { return []int{c.HiddenSize, c.QWidth()} },
		kind:      KindProjection,
		transpose: true,
	},
	{
		suffix: "self_attn.q_norm.weight", port: "qnorm",
		shape:   func(c *Config) []int { return []int{c.HeadDim} },
		kind:    KindGain,
		permute: true,
		heads:   func(*Config) int { return 1 },
	},
	{
		suffix: "self_attn.k_norm.weight", port: "knorm",
		shape:   func(c *Config) []int { return []int{c.HeadDim} },
		kind:    KindGain,
		permute: true,
		heads:   func(*Config) int { return 1 },
	},
	{
		suffix: "post_attention_layernorm.weight", port: "ffn_norm",
		shape: func(c *Config) []int { return []int{c.HiddenSize} },
		kind:  KindGain,
	},
	{
		suffix: "mlp.gate_proj.weight", port: "wgate",
		shape:     func(c *Config) []int { return []int{c.IntermediateSize, c.HiddenSize} },
		kind:      KindProjection,
		transpose: true,
	},
	{
		suffix: "mlp.up_proj.weight", port: "wup",
		shape:     func(c *Config) []int { return []int{c.IntermediateSize, c.HiddenSize} },
		kind:      KindProjection,
		transpose: true,
	},
	{
		suffix: "mlp.down_proj.weight", port: "wdown",
		shape:     func(c *Config) []int { return []int{c.HiddenSize, c.IntermediateSize} },
		kind:      KindProjection,
		transpose: true,
	},
}

// qwen3Weights expands §4's table for one config.
//
// The result is ordered: the embedding, then the layers in order, then the
// final norm and the head. A loader walking it in order touches the checkpoint
// roughly in the order the shards were written.
func qwen3Weights(c *Config) []WeightSpec {
	out := make([]WeightSpec, 0, 2+len(qwen3Layer)*c.NumLayers+2)

	// The embedding table does not transpose: GatherRows reads it by row and
	// [V, d] is already row-per-token.
	out = append(out, WeightSpec{
		Tensor: qwen3Embed,
		Port:   "embed",
		Shape:  []int{c.VocabSize, c.HiddenSize},
		Layer:  -1,
		Kind:   KindEmbedding,
	})

	for l := 0; l < c.NumLayers; l++ {
		prefix := fmt.Sprintf(layerPrefix, l)
		for _, t := range qwen3Layer {
			s := WeightSpec{
				Tensor:    prefix + t.suffix,
				Port:      fmt.Sprintf("%d.%s", l, t.port),
				Shape:     t.shape(c),
				Layer:     l,
				Kind:      t.kind,
				Transpose: t.transpose,
				Permute:   t.permute,
			}
			if t.permute {
				s.Heads = t.heads(c)
			}
			out = append(out, s)
		}
	}

	out = append(out, WeightSpec{
		Tensor: qwen3FinalNorm,
		Port:   "final_norm",
		Shape:  []int{c.HiddenSize},
		Layer:  -1,
		Kind:   KindGain,
	})

	// A tied head is two planes from one file tensor: [V,d] untransposed for
	// GatherRows, and [d,V] transposed for the MatMul. They are different
	// layouts, so they cannot share a device buffer and "tied" means one
	// source tensor rather than one buffer (004-D7). Alias carries the name a
	// tied checkpoint must not also ship.
	head := WeightSpec{
		Tensor:    qwen3LMHead,
		Port:      "lm_head",
		Shape:     []int{c.VocabSize, c.HiddenSize},
		Layer:     -1,
		Kind:      KindLMHead,
		Transpose: true,
	}
	if c.TieWordEmbeddings {
		head.Tensor = qwen3Embed
		head.Alias = qwen3LMHead
	}
	return append(out, head)
}
