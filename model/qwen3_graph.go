// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"fmt"

	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// Forward records Qwen3's forward pass, specs/004-model-graph.md §3, one node
// per row of that table:
//
//	h = Embed(ids)
//	for l in 0..L-1:
//	    h = h + Attention(RMSNorm(h))
//	    h = h + MLP(RMSNorm(h))
//	logits = LMHead(RMSNorm(h[-1]))
//
// Pre-norm, with a residual around each sub-block, and the last row sliced off
// before the head (§3.2).
//
// The returned tensor is [1, V] f32 and is not recorded as an output: [Record]
// does that, and a caller assembling a larger graph may want the logits as an
// operand rather than as a port.
//
// An Inputs that does not describe this config returns nil and leaves the
// reason on g, which is what [nn.Graph.Err] reports.
func (m *qwen3) Forward(g *nn.Graph, in Inputs) *tensor.Tensor {
	c := m.cfg
	if g == nil || g.B == nil {
		return nil
	}
	if err := in.Validate(c); err != nil {
		// Recorded through a port declaration accel refuses, because nn.Graph
		// collects diagnostics and exports no way to add one. The message is
		// the model's; the mechanism is a zero-extent value, which accel
		// rejects by name.
		poison(g, err)
		return nil
	}
	b := g.B
	t := in.IDs.Shape()[0]

	// Row 3. The embedding table is [V, d] and is not transposed: GatherRows
	// reads it by row, and a row is already a token's vector. Its two forms
	// are separate operators rather than one taking an Operand, so the branch
	// is here -- nn has no embedding block, and the projections' Operand
	// exists because a *MatMul* takes either form.
	g.Prefix = ""
	table := g.Weight(portEmbed, tensor.Shape{c.VocabSize, c.HiddenSize})
	var h *tensor.Tensor
	switch {
	case table.IsQuant():
		h = tensor.QuantGatherRows(b, table.Quant, in.IDs)
	default:
		h = tensor.GatherRows(b, table.Dense, in.IDs)
	}

	cfg := nn.AttentionConfig{
		QHeads: c.NumHeads, KVHeads: c.NumKVHeads, HeadDim: c.HeadDim,
		RoPEBase: ScalarRoPEBase, ScaleName: ScalarScale, BaseName: in.Base,
		// Qwen3's one departure from Llama: Q and K are normalized per head,
		// over head_dim, before RoPE (§2.4).
		QKNorm: true,
	}

	for l := range c.NumLayers {
		g.Prefix = fmt.Sprintf("%d.", l)

		// Rows 4 to 18. One layer's window of the two caches is a view of the
		// whole state, taken from the parent each time: LayerState carries the
		// parent's version chain and binding identity, so L layers are two
		// bound buffers rather than 2L (specs/005-kv-cache.md §2.1).
		a := nn.Attention(g, nn.RMSNorm(g, h, g.Gain("attn_norm", c.HiddenSize)),
			nn.AttentionWeights{
				Q:     g.Weight("wq", tensor.Shape{c.HiddenSize, c.QWidth()}),
				K:     g.Weight("wk", tensor.Shape{c.HiddenSize, c.KVWidth()}),
				V:     g.Weight("wv", tensor.Shape{c.HiddenSize, c.KVWidth()}),
				O:     g.Weight("wo", tensor.Shape{c.QWidth(), c.HiddenSize}),
				QNorm: g.Gain("qnorm", c.HeadDim),
				KNorm: g.Gain("knorm", c.HeadDim),
			},
			tensor.LayerState(b, in.Keys, l), tensor.LayerState(b, in.Values, l),
			in.PosQ, in.PosK, in.Slots, in.Lengths, cfg)

		// Row 19, then rows 20 to 24, then row 25.
		h = tensor.Add(b, h, a)
		mlp := nn.SwiGLUMLP(g, nn.RMSNorm(g, h, g.Gain("ffn_norm", c.HiddenSize)),
			g.Weight("wgate", tensor.Shape{c.HiddenSize, c.IntermediateSize}),
			g.Weight("wup", tensor.Shape{c.HiddenSize, c.IntermediateSize}),
			g.Weight("wdown", tensor.Shape{c.IntermediateSize, c.HiddenSize}))
		h = tensor.Add(b, h, mlp)
	}
	g.Prefix = ""

	// Row 26, and the single largest avoidable cost in the graph (§3.2). Only
	// the last position's logits are wanted: running the head over all T costs
	// T*V f32 values, which for a 2000-token prompt at Qwen3's V is 1.2 GB of
	// numbers nobody reads.
	//
	// Contiguous after Slice because a slice is a view at a non-zero offset
	// and accel refuses a strided operand into MatMul rather than copying
	// behind the caller's back. At [1, d] the copy is kilobytes.
	last := tensor.Contiguous(b, tensor.Slice(b, h, 0, t-1, t))

	// Rows 27 and 28. lm_head is [d, V]: the checkpoint's [V, d] transposed at
	// load, or the embedding table transposed when the two are tied -- a
	// second plane from one file tensor, because the two layouts cannot share
	// a buffer (004-D7).
	return nn.Linear(g, nn.RMSNorm(g, last, g.Gain(portFinalNorm, c.HiddenSize)),
		g.Weight(portLMHead, tensor.Shape{c.HiddenSize, c.VocabSize}))
}

// poison records a model-level refusal on a graph.
//
// [nn.Graph] collects diagnostics from nn's blocks and from accel's operators
// and exports no way for a third package to add one, so the message is carried
// down as a port name. A zero-extent value is refused by accel with the name
// quoted, which puts the model's sentence in the compile error where a caller
// already looks for it.
func poison(g *nn.Graph, err error) {
	g.Weight(err.Error(), tensor.Shape{0})
}
