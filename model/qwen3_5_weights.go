// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"fmt"
	"strings"
)

// specs/024-qwen3-5-architecture.md §4.5's weight map, read from
// `Qwen/Qwen3.5-27B`'s index and safetensors headers on 2026-08-29.
//
// The map is `[]WeightSpec` built by walking the schedule and emitting one of
// two row sets per layer, which is [qwen3Weights] with a branch.
// `model/weights.go` does not change: `Check` already refuses a tensor the map
// does not name and a tensor the map names and the file lacks, and a hybrid map
// is still a map.

// qwen35Full is a full-attention layer's rows.
//
// 004 §4's table with one correction and no addition: `q_proj` is 2·H·d_h wide,
// because `attn_output_gate` is true and **there is no gate tensor** -- the gate
// is the second half of q_proj's output. The header settles it, which is what
// §4.5's draft said it would.
var qwen35Full = []layerTensor{
	{
		suffix: "input_layernorm.weight", port: "attn_norm",
		shape: func(c *Config) []int { return []int{c.HiddenSize} },
		kind:  KindGain,
	},
	{
		suffix: "self_attn.q_proj.weight", port: "wq",
		// 2·H·d_h: q and the output gate, concatenated.
		shape:     func(c *Config) []int { return []int{2 * c.QWidth(), c.HiddenSize} },
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
}

// qwen35Linear is a gated-delta layer's rows.
//
// Four projections and not two: the draft followed Qwen3-Next's fused
// `in_proj_qkvz` and `in_proj_ba`, and `qwen3_5` splits them. Nothing here
// permutes -- `Permute` reorders a projection's output channels into accel's
// interleaved rotary pairing and a linear layer has no rotation (§4.5).
var qwen35Linear = []layerTensor{
	{
		suffix: "input_layernorm.weight", port: "attn_norm",
		shape: func(c *Config) []int { return []int{c.HiddenSize} },
		kind:  KindGain,
	},
	{
		suffix: "linear_attn.in_proj_qkv.weight", port: "lin_qkv",
		// 2·W_k + W_v: q, k and v concatenated.
		shape: func(c *Config) []int {
			r := c.Recurrent
			return []int{2*r.Heads*r.KeyDim + r.ValueWidth(), c.HiddenSize}
		},
		kind:      KindProjection,
		transpose: true,
	},
	{
		suffix: "linear_attn.in_proj_z.weight", port: "lin_z",
		shape: func(c *Config) []int {
			return []int{c.Recurrent.ValueWidth(), c.HiddenSize}
		},
		kind:      KindProjection,
		transpose: true,
	},
	{
		// One scalar per value head, which is what makes the gate per head and
		// [C27](../specs/010-conformance.md) blocking.
		suffix: "linear_attn.in_proj_b.weight", port: "lin_b",
		shape: func(c *Config) []int {
			return []int{c.Recurrent.ValueHeads, c.HiddenSize}
		},
		kind:      KindProjection,
		transpose: true,
	},
	{
		suffix: "linear_attn.in_proj_a.weight", port: "lin_a",
		shape: func(c *Config) []int {
			return []int{c.Recurrent.ValueHeads, c.HiddenSize}
		},
		kind:      KindProjection,
		transpose: true,
	},
	{
		// [C_conv, 1, K] in the file and [K, C_conv] in the block: a squeeze of
		// the middle axis and then a transpose of a rank-2 plane. WeightSpec
		// reverses a rank-2 shape and weights.targetShape refuses any other
		// rank, so this row is the one the map cannot express today (§4.5).
		suffix: "linear_attn.conv1d.weight", port: "lin_taps",
		shape: func(c *Config) []int {
			return []int{c.Recurrent.ConvWidth, 1, c.Recurrent.Taps}
		},
		kind: KindGain,
	},
	{
		suffix: "linear_attn.dt_bias", port: "lin_dt",
		shape: func(c *Config) []int { return []int{c.Recurrent.ValueHeads} },
		kind:  KindGain,
	},
	{
		suffix: "linear_attn.A_log", port: "lin_alog",
		shape: func(c *Config) []int { return []int{c.Recurrent.ValueHeads} },
		kind:  KindGain,
	},
	{
		// [d_v] and not the folded [3·d_v]: one gain per **value** head,
		// applied across all of them. A graph holding only §2.2's folded
		// geometry would scale three value heads by one head's gain (§4.5).
		suffix: "linear_attn.norm.weight", port: "lin_norm",
		shape: func(c *Config) []int {
			r := c.Recurrent
			return []int{r.ValueWidth() / r.ValueHeads}
		},
		kind: KindGain,
	},
	{
		suffix: "linear_attn.out_proj.weight", port: "lin_out",
		shape: func(c *Config) []int {
			return []int{c.HiddenSize, c.Recurrent.ValueWidth()}
		},
		kind:      KindProjection,
		transpose: true,
	},
}

// qwen35Shared is what both kinds carry: the MLP and its norm.
var qwen35Shared = []layerTensor{
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

// ValueWidth is W_v = H_v · d_v, the width of v, z and the state.
//
// It is the same number under either reading of the head count -- 48 heads of
// 128 or 16 bands of 384 -- which is what makes §2.2's folding an identity for
// the state and not for the output norm.
func (r Recurrent) ValueWidth() int { return r.Heads * r.ValueDim }

// qwen35Weights expands §4.5's map for one config.
func qwen35Weights(q *qwen35Config) []WeightSpec {
	c := q.Config
	out := make([]WeightSpec, 0, 2+len(qwen35Linear)*c.NumLayers+2)

	out = append(out, WeightSpec{
		Tensor: qwen35Embed, Port: "embed",
		Shape: []int{c.VocabSize, c.HiddenSize}, Layer: -1, Kind: KindEmbedding,
	})
	for l := range c.NumLayers {
		prefix := fmt.Sprintf(qwen35LayerPrefix, l)
		rows := qwen35Linear
		if c.LayerTypes.Kind(l) == LayerFullAttention {
			rows = qwen35Full
		}
		for _, t := range append(append([]layerTensor(nil), rows...), qwen35Shared...) {
			s := WeightSpec{
				Tensor: prefix + t.suffix, Port: fmt.Sprintf("%d.%s", l, t.port),
				Shape: t.shape(c), Layer: l, Kind: t.kind, Transpose: t.transpose,
			}
			if t.permute {
				s.Permute, s.Heads = true, t.heads(c)
			}
			out = append(out, s)
		}
	}
	out = append(out, WeightSpec{
		Tensor: qwen35FinalNorm, Port: "final_norm",
		Shape: []int{c.HiddenSize}, Layer: -1, Kind: KindGain,
	})
	head := WeightSpec{
		Tensor: qwen35LMHead, Port: "lm_head",
		Shape: []int{c.VocabSize, c.HiddenSize}, Layer: -1, Kind: KindProjection,
		Transpose: true,
	}
	if c.TieWordEmbeddings {
		head.Tensor, head.Alias = qwen35Embed, qwen35LMHead
	}
	out = append(out, head)
	return out
}

// Qwen35Ignored reports whether a checkpoint tensor belongs to a tower this
// graph does not read.
//
// 024-D13. `Qwen/Qwen3.5-27B` holds 1199 tensors: 850 of the text tower, 333
// under `model.visual.` and 15 under `mtp.`. `model.Check` refuses a tensor the
// map does not name, so without this the checkpoint is refused wholesale --
// and dropping them silently is the failure §7's general row is about, one
// level up. The set is explicit and the two prefixes say which tower they are.
func Qwen35Ignored(tensor string) bool {
	return strings.HasPrefix(tensor, qwen35VisionPrefix) ||
		strings.HasPrefix(tensor, qwen35MTPPrefix)
}
