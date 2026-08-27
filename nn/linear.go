// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package nn

import (
	"golang.design/x/accel/tensor"
)

// LinearConfig carries the model constants a gated delta layer reads.
type LinearConfig struct {
	// Heads is how many recurrent heads the layer carries, KeyDim is the width
	// q and k contract against, and ValueDim is v's.
	//
	// The two widths differ in Qwen3.8 — 16 key heads of 128 against 48 value
	// heads of 128 — which is why the state is [slots, heads, ValueDim, KeyDim]
	// and not a square. Transposing the last two runs when they are equal and
	// is wrong when they are not, which is why accel checks the shape rather
	// than inferring it.
	Heads, KeyDim, ValueDim int
}

// LinearAttention is one gated delta layer's recurrence.
//
// specs/018-hybrid-models.md. Three of every four layers of Qwen3.8-27B are
// this, so the operator tgo spent its design on covers a quarter of that model.
//
//	u = S k,   S ← αS + β·k·(v−u)ᵀ,   o = S q
//
// # The state does not grow with the context
//
// That is the whole appeal, and it is why nothing here takes lengths or a page
// table: a matrix per sequence per head, and no positions to address. A
// 262K-context model has no key/value cache for these layers at all.
//
// # It returns the next state as well as the output
//
// A recurrence reads the state and writes it in one kernel, so unlike a
// key/value cache — where ScatterRows writes and Attention reads, as two nodes
// — there is no way to separate them. The returned version is what lets a later
// operator say which contents it meant.
//
// # What it returns, and the shape it does not
//
// [T, heads, ValueDim], as accel shapes it. Flattening to [T, heads*ValueDim]
// — which is what every projection around it is — would put a Reshape in front
// of the result, and a Reshape in front of a *graph output* silently produces
// zeros (accel#26). A caller that feeds this into another operator reshapes at
// their own call site, where the result is an operand and a view is ordinary.
//
// # What this block does not do
//
// The depthwise causal convolution that precedes the recurrence in Qwen3.8, and
// [§4.1](../specs/018-hybrid-models.md) says why it is not here: it needs the
// K−1 inputs before this step, and accel has no operator that joins them to
// this step's. The projections are computed *inside* the graph, so there is
// nothing to pad them with.
//
// # Where α sits is not settled here
//
// accel's kernel decays the state and not the correction, which is the
// expansion of the equation above. Qwen3.5's published form places α outside
// the whole bracket, and the two differ in whether the correction term is
// decayed. Both are "the gated delta rule" in the literature, and which one
// this model was trained under is a question only the checkpoint answers. This
// composes what accel documents.
func LinearAttention(g *Graph, q, k, v, alpha, beta, extents *tensor.Tensor,
	s *tensor.State, cfg LinearConfig) (*tensor.Tensor, *tensor.State) {

	fail := func(format string, args ...any) (*tensor.Tensor, *tensor.State) {
		g.fail("LinearAttention", format, args...)
		return nil, nil
	}
	if cfg.Heads <= 0 || cfg.KeyDim <= 0 || cfg.ValueDim <= 0 {
		return fail("the layer is %d heads of %d against %d; each is at least one",
			cfg.Heads, cfg.KeyDim, cfg.ValueDim)
	}
	t, w, ok := rows(q)
	if !ok {
		return fail("q is %v; the projections are [T, heads*dim]", shapeOf(q))
	}
	if want := cfg.Heads * cfg.KeyDim; w != want {
		return fail("q is %d wide and %d heads of %d is %d", w, cfg.Heads, cfg.KeyDim, want)
	}
	if kt, kw, ok := rows(k); !ok || kt != t || kw != w {
		return fail("q is %v and k is %v; they contract against the same axis of the "+
			"state, so they are the same shape", shapeOf(q), shapeOf(k))
	}
	if vt, vw, ok := rows(v); !ok || vt != t || vw != cfg.Heads*cfg.ValueDim {
		return fail("v is %v; it is the same tokens and the same heads as q and differs "+
			"only in width, which is %d heads of %d", shapeOf(v), cfg.Heads, cfg.ValueDim)
	}

	// Returned as the operator shapes it, [T, heads, ValueDim], and **not**
	// flattened to [T, heads*ValueDim] the way every projection around it is.
	//
	// That is not a style choice. A caller who declares this result as a graph
	// output gets zeros if a Reshape sits in front of it: accel's Output
	// accepts a view, records the right shape, and never writes the buffer
	// (accel#26). Flattening here would put that Reshape in every caller's
	// path whether they output the result or not, and the failure is silent —
	// correct shape, no refusal, all zeros.
	//
	// A caller feeding this into another operator reshapes at their own call
	// site, where the result is an operand and views are ordinary.
	out, next := tensor.LinearAttention(g.B,
		tensor.Reshape(g.B, q, tensor.Shape{t, cfg.Heads, cfg.KeyDim}),
		tensor.Reshape(g.B, k, tensor.Shape{t, cfg.Heads, cfg.KeyDim}),
		tensor.Reshape(g.B, v, tensor.Shape{t, cfg.Heads, cfg.ValueDim}),
		s, tensor.LinearOptions{Alpha: alpha, Beta: beta, QueryExtents: extents})
	return out, next
}

// shapeOf is a tensor's shape, or nil for a nil tensor, so a diagnostic about a
// missing operand does not panic before it is printed.
func shapeOf(x *tensor.Tensor) tensor.Shape {
	if x == nil {
		return nil
	}
	return x.Shape()
}
