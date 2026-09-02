// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

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

// ConvState is the rolling window a depthwise causal convolution reads.
//
// **Flat**: [R, C] with R = B(K-1) + T, holding every slot's carry rows and
// then this step's tokens. [C26](../specs/010-conformance.md) is why it is a
// state and not a padded operand — `tensor` joins no two tensors along an axis,
// and the rows a convolution needs in front of a decode step are the *previous
// step's* anyway, not zeros.
//
// Axis 0 is time and there is no slot axis, because a state has one row axis:
// a slot axis in front of it would make one row a whole sequence and a token
// write inexpressible (specs/023-cache-kinds.md §3). The slot is arithmetic in
// the index ports instead, which [ConvIndex] computes.
type ConvState struct {
	// State is the window, [R, C].
	State *tensor.State

	// Write is where this step's rows go, a [T] u32 port.
	Write *tensor.Tensor

	// Taps is one [T] u32 port per tap: Taps[i] holds the row tap i reads for
	// each output row. There are K of them and they are what makes the slot a
	// number rather than an axis.
	Taps []*tensor.Tensor

	// Carry is which rows become the next step's leading K-1 per slot, and
	// CarryWrite is where they go: both [B(K-1)] u32 ports.
	Carry, CarryWrite *tensor.Tensor
}

// DepthwiseCausalConv convolves each channel of x with its own K taps, over the
// K−1 rows before it.
//
// specs/018-hybrid-models.md §4.1. taps is [K, C]: one row of weights per tap,
// one column per channel, so a channel never mixes with another.
// `linear_conv_kernel_dim: 4` in Qwen3.8's config is K.
//
// # Causality is structural, not checked
//
// Tap i reads the window starting at K−1−i of a buffer whose first K−1 rows are
// what came *before* this step. Position t therefore sees t−K+1 through t and
// nothing after, because there is nothing after to slice — no operator has to
// know the convolution is causal, and none can get it wrong.
//
// # The carry is written after the window is read, and the versions say so
//
// The step scatters its rows into the window, reads it, and then scatters the
// window's last K−1 rows to the front for the next step. Those are two writes
// to one state with an order that matters, which is exactly what
// [tensor.State]'s versions express: the read names the version between them.
//
// It returns the window's next version, which the caller binds for the step
// after this one.
//
// # What it costs
//
// K gathers, each multiplied by a broadcast tap row, summed — plus two scatters
// and a gather. Roughly 3K+5 dispatches for what one kernel would do, and K
// copies of a [T, C] tensor per layer. Over 48 linear layers that is real, and
// it is one less kernel to be *blocked on* rather than one less kernel to
// *want*.
//
// A gather where a slice and a contiguous used to be: the same bytes through an
// index, and the index is what carries the slot (specs/023-cache-kinds.md §3).
func DepthwiseCausalConv(g *Graph, x, taps *tensor.Tensor, w ConvState,
	k int) (*tensor.Tensor, *tensor.State) {

	fail := func(format string, args ...any) (*tensor.Tensor, *tensor.State) {
		g.fail("DepthwiseCausalConv", format, args...)
		return nil, nil
	}
	t, c, ok := rows(x)
	if !ok {
		return fail("x is %v; the input is [T, C]", shapeOf(x))
	}
	if !f32(x) {
		return fail("x is %v and this composition is f32", x.DType())
	}
	if k <= 1 {
		return fail("the kernel is %d taps; a causal convolution this composition can "+
			"express has at least two, and one tap is an elementwise scale", k)
	}
	if kt, kc, ok := rows(taps); !ok || kt != k || kc != c {
		return fail("the taps are %v; they are [K, C] = [%d, %d], one row of weights "+
			"per tap and one column per channel", shapeOf(taps), k, c)
	}
	if w.State == nil || w.Write == nil || w.Carry == nil || w.CarryWrite == nil {
		return fail("the window needs a state and three index ports; the rows before " +
			"this step are the previous step's and there is nothing else to pad with " +
			"(specs/018-hybrid-models.md §4.1.1)")
	}
	if len(w.Taps) != k {
		return fail("the window has %d tap index ports and the kernel is %d taps; "+
			"each tap reads its own rows and the slot is arithmetic in the index "+
			"(specs/023-cache-kinds.md §3)", len(w.Taps), k)
	}
	for i, idx := range w.Taps {
		sh := shapeOf(idx)
		if len(sh) != 1 || sh[0] != t {
			return fail("tap %d's index port is %v; it is [T] = [%d], one row per "+
				"output row", i, sh, t)
		}
	}

	// This step's rows into the window, after the K-1 that were already there.
	filled := tensor.ScatterRows(g.B, w.State, x, w.Write)
	window := tensor.ReadState(g.B, filled)

	var acc *tensor.Tensor
	for i := range k {
		// Tap i reads the rows its own index port names, which for output row
		// r is r's position minus K-1-i — so tap K-1 lands on the current row
		// and tap 0 on the one K-1 back:
		//
		//	y[t] = sum_i taps[i] * x[t-K+1+i]
		//
		// which is the convention the weights are trained under. Reversing it
		// runs, produces plausible numbers, and convolves the window backwards.
		// [ConvIndex] is where that convention lives now.
		rowsOf := tensor.GatherRows(g.B, window, w.Taps[i])
		// Contiguous twice, and each one is for a different reason.
		//
		// The tap row first: a slice at row i is a view at an offset, and a
		// broadcast starts from a contiguous run. Then the broadcast itself,
		// because Mul takes operands and not views — accel's refusal says
		// "make it contiguous in the shape you want", and this is that shape.
		//
		// It costs a [T, C] copy per tap. One kernel would cost none, which is
		// [C26](../specs/010-conformance.md)'s point restated in dispatches.
		tap := tensor.Contiguous(g.B, tensor.Slice(g.B, taps, 0, i, i+1))
		wide := tensor.Contiguous(g.B, tensor.Broadcast(g.B, tap, tensor.Shape{t, c}))
		term := tensor.Mul(g.B, rowsOf, wide)
		if acc == nil {
			acc = term
			continue
		}
		acc = tensor.Add(g.B, acc, term)
	}

	// The last K-1 rows of the window become the next step's leading ones.
	// After the read, because a write before it would convolve rows this step
	// has not produced.
	carried := tensor.GatherRows(g.B, window, w.Carry)
	return acc, tensor.ScatterRows(g.B, filled, carried, w.CarryWrite)
}
