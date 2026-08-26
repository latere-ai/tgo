// Copyright 2026 The tgo Authors. All rights reserved.

package nn

import (
	"golang.design/x/accel/tensor"
)

// AttentionWeights are the ports one attention block reads.
//
// The norm gains are Qwen3-specific and are [AttentionConfig.HeadDim] values
// each, shared across heads: the norm is per head, over the head dimension, so
// its gain is a head wide and not a row wide.
type AttentionWeights struct {
	Q, K, V, O   Operand
	QNorm, KNorm *tensor.Tensor
}

// AttentionConfig carries the model constants attention reads.
type AttentionConfig struct {
	QHeads, KVHeads, HeadDim int

	// RoPEBase is a declared f32 scalar, the rotary frequency base.
	RoPEBase string

	// ScaleName is a declared f32 scalar, 1/sqrt(HeadDim).
	ScaleName string

	// QKNorm normalizes Q and K per head before RoPE, which is what Qwen3
	// does and Llama does not.
	QKNorm bool

	// BaseName is a declared u32 scalar, the prefill's first position within
	// the cache. It decides what the causal mask hides. Required when more
	// than one token is scored; unread at T=1, which is a decode.
	BaseName string

	// Block is how many positions one physical block holds, and zero means the
	// cache is contiguous.
	//
	// It travels with the page table and never without it: accel refuses a
	// table with no block size, because a table addresses blocks and how big
	// one is is not derivable from the table's shape.
	Block int
}

// Attention is grouped-query attention with Qwen3's QK-norm and RoPE.
//
// The order is the whole point of the block, and specs/004-model-graph.md
// section 2.4 is emphatic about it: Q and K are normalized per head, over the
// head dimension, before RoPE. Normalizing after the rotation, or over the
// whole H*HeadDim row instead of one head, gives a model that produces
// plausible tokens and loses coherence after a few sentences.
//
// posQ and posK are separate because under GQA the q and k row counts differ,
// T*QHeads against T*KVHeads, so one positions tensor cannot serve both. Each
// holds one position per *row*, which repeats a token's position once per head.
//
// slots are the cache destinations, one per token, and lengths is how much of
// the cache holds real tokens. Both are runtime data rather than constants:
// specs/043-per-row-values.md draws that line and accel enforces it.
//
// # What the contiguous cache requires of slots and positions
//
// This block records no page table, so the cache is contiguous and the kernels
// address it by position: a token's slot is its position. A prefill asks for
// more than that -- its causal limit is BaseName+s for the s-th query, so its
// positions are consecutive from BaseName. Neither is checkable here, because
// both are runtime data and the graph sees only their extents; measured
// against accel's prefill kernel, non-consecutive positions silently mask the
// wrong keys. A cache addressed otherwise is the paged path, which
// tensor.AttentionOptions.Pages carries and this signature does not.
//
// The returned tensor is the block's output, [T, hidden] f32, already projected
// through O. The residual around it is the model's, not this block's.
//
// The key and value caches take the projections as they are, so they are f32
// states. A narrow cache halves the largest allocation a serving process has
// after the weights, and reaching it needs a Cast on the rows that accel names
// and this block does not record: a [tensor.State] does not report its dtype,
// so nn cannot tell which case it is in.
func Attention(g *Graph, x *tensor.Tensor, w AttentionWeights,
	k, v *tensor.State, posQ, posK, slots, lengths, pages *tensor.Tensor,
	cfg AttentionConfig) *tensor.Tensor {

	// The page table and the block size are one binding in two values, so a
	// caller that supplied one of them supplied half a cache addressing and
	// the other half is whatever the zero value happens to be. accel refuses
	// a table with no block; this refuses the other direction too, because a
	// block size with no table is silently a contiguous read.
	if (pages == nil) != (cfg.Block == 0) {
		return g.fail("Attention", "the page table is %v and Block is %d; a table "+
			"addresses blocks and a block size without one addresses nothing, so "+
			"they are set together or neither is (specs/005-kv-cache.md §2.2)",
			pages != nil, cfg.Block)
	}

	t, _, ok := rows(x)
	if !ok {
		if x == nil || len(x.Shape()) == 0 {
			// A poisoned or absent operand is somebody else's diagnostic
			// already; repeating it here would be the second one.
			return nil
		}
		return g.fail("Attention", "x is %v; the hidden state is [T, hidden]", x.Shape())
	}
	if !f32(x) {
		return g.fail("Attention", "x is %v and a hidden state is f32", x.DType())
	}
	if cfg.QHeads <= 0 || cfg.KVHeads <= 0 || cfg.HeadDim <= 0 {
		return g.fail("Attention", "QHeads is %d, KVHeads is %d and HeadDim is %d; all "+
			"three are positive", cfg.QHeads, cfg.KVHeads, cfg.HeadDim)
	}
	if cfg.QHeads%cfg.KVHeads != 0 {
		return g.fail("Attention", "num_attention_heads is %d and num_key_value_heads is "+
			"%d; several query heads share one cache entry, so the first is a multiple "+
			"of the second", cfg.QHeads, cfg.KVHeads)
	}
	if cfg.HeadDim%2 != 0 {
		return g.fail("Attention", "head_dim is %d; RoPE rotates pairs, so it is even",
			cfg.HeadDim)
	}
	if cfg.QKNorm && (w.QNorm == nil || w.KNorm == nil) {
		return g.fail("Attention", "QKNorm is set and the q or k norm gain is missing; "+
			"Qwen3 normalizes both before RoPE (specs/004-model-graph.md section 2.4)")
	}
	// The projections. No cast in front of any of them: C8 closed.
	q := Linear(g, x, w.Q)
	keys := Linear(g, x, w.K)
	vals := Linear(g, x, w.V)

	// The projection widths against the config, checked here because accel
	// sees them one reshape later and reports two shapes rather than the field
	// that disagreed. head_dim is the one that bites: it is not always
	// hidden_size/num_attention_heads (specs/004-model-graph.md section 5).
	if !g.wide("q_proj", q, cfg.QHeads*cfg.HeadDim) ||
		!g.wide("k_proj", keys, cfg.KVHeads*cfg.HeadDim) ||
		!g.wide("v_proj", vals, cfg.KVHeads*cfg.HeadDim) {
		return nil
	}

	// Per head, over the head dimension. The reshape is what makes the axis
	// right: [T, H*HeadDim] holds H heads side by side in one row, and
	// [T*H, HeadDim] gives each head its own row, which is the axis
	// [tensor.RMSNorm] reduces over.
	qh := tensor.Reshape(g.B, q, tensor.Shape{t * cfg.QHeads, cfg.HeadDim})
	kh := tensor.Reshape(g.B, keys, tensor.Shape{t * cfg.KVHeads, cfg.HeadDim})
	if cfg.QKNorm {
		qh = RMSNorm(g, qh, w.QNorm)
		kh = RMSNorm(g, kh, w.KNorm)
	}

	// Then RoPE, over the whole head: Qwen3's rotary dimension is HeadDim.
	//
	// accel rotates *interleaved* pairs and Qwen3 is half-split, which is not
	// reconciled here: specs/004-model-graph.md section 2.5.2 permutes the
	// projection's output channels -- and the QK-norm gains with them -- at
	// load time, so the interleaved kernel computes the half-split rotation
	// (004-D9). That is a byte layout tgo owns; nothing in this graph shows it.
	qh = tensor.RoPE(g.B, qh, cfg.HeadDim, cfg.RoPEBase, posQ)
	kh = tensor.RoPE(g.B, kh, cfg.HeadDim, cfg.RoPEBase, posK)

	// The cache is written with one row per token, not one per head: a slot
	// holds a token's whole key.
	kNext := tensor.ScatterRows(g.B, k,
		tensor.Reshape(g.B, kh, tensor.Shape{t, cfg.KVHeads * cfg.HeadDim}), slots)
	vNext := tensor.ScatterRows(g.B, v, vals, slots)

	// q's rank says which computation this is: [QHeads, HeadDim] is a decode
	// and [T, QHeads, HeadDim] is a prefill. A rank is not a hint, it is the
	// shape of the computation (specs/004-model-graph.md section 3.1).
	shaped := tensor.Reshape(g.B, qh, tensor.Shape{t, cfg.QHeads, cfg.HeadDim})
	opts := tensor.AttentionOptions{
		Lengths:   lengths,
		Pages:     pages,
		Block:     cfg.Block,
		ScaleName: cfg.ScaleName,
		BaseName:  cfg.BaseName,
	}
	if t == 1 {
		shaped = tensor.Reshape(g.B, qh, tensor.Shape{cfg.QHeads, cfg.HeadDim})
		// A decode has one query token and therefore no causal mask to place,
		// and accel refuses a BaseName that would reach nothing. One config
		// serves both plans (004-D3), so dropping it here is what lets a model
		// declare the scalar once and record either graph from it.
		opts.BaseName = ""
	}
	a := tensor.Attention(g.B, shaped, kNext, vNext, opts)
	return Linear(g, tensor.Reshape(g.B, a, tensor.Shape{t, cfg.QHeads * cfg.HeadDim}), w.O)
}

// wide reports whether a projection produced the width the config implies, and
// records which port disagreed when it did not.
//
// A poisoned projection is silent: its own diagnostic was recorded where it
// happened, and a width complaint about a value that does not exist would be
// the second one.
func (g *Graph) wide(port string, x *tensor.Tensor, want int) bool {
	_, got, ok := rows(x)
	if !ok {
		return false
	}
	if got != want {
		g.fail("Attention", "%s is %d wide and the config asks for %d; check head_dim "+
			"against num_attention_heads (specs/004-model-graph.md section 5)",
			port, got, want)
		return false
	}
	return true
}
