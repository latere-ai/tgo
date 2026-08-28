// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package nn_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

const (
	linHeads = 2
	linKey   = 3
	linValue = 4
	linSlots = 2
	linT     = 3
)

// TestLinearAttentionMatchesAReferenceComputedBeside is specs/010-conformance.md
// §5 for the gated delta layer: a float64 reference written from
// specs/018-hybrid-models.md §2's equation rather than from this block's code.
//
// The widths differ on purpose. At KeyDim == ValueDim a state with its last two
// axes transposed still runs, so a fixture that made them equal could not tell
// a correct block from one that swapped them — which is CONTRIBUTING's rule
// about two dimensions of a fixture being equal.
func TestLinearAttentionMatchesAReferenceComputedBeside(t *testing.T) {
	r := newRig(t, 1e-6)

	q := r.input("q", accel.F32, tensor.Shape{linT, linHeads * linKey})
	k := r.input("k", accel.F32, tensor.Shape{linT, linHeads * linKey})
	v := r.input("v", accel.F32, tensor.Shape{linT, linHeads * linValue})
	alpha := r.input("alpha", accel.F32, tensor.Shape{linT})
	beta := r.input("beta", accel.F32, tensor.Shape{linT})
	ext := r.input("extents", accel.U32, tensor.Shape{linSlots})

	state := tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "s", DType: accel.F32,
		Shape: tensor.Shape{linSlots, linHeads, linValue, linKey},
	})

	qv := ramp(linT*linHeads*linKey, 7)
	kv := ramp(linT*linHeads*linKey, 13)
	vv := ramp(linT*linHeads*linValue, 29)
	av := gates(linT, 0.85, 0.1)
	bv := gates(linT, 0.30, 0.2)
	sv := ramp(linSlots*linHeads*linValue*linKey, 41)
	// Two sequences: one contributing two tokens and one contributing one, so
	// the extent is exercised rather than assumed.
	ev := []uint32{2, 1}

	r.f32("q", qv)
	r.f32("k", kv)
	r.f32("v", vv)
	r.f32("alpha", av)
	r.f32("beta", bv)
	r.f32("s", sv)
	r.u32("extents", ev)

	out, next := nn.LinearAttention(r.g, q, k, v, alpha, beta, ext, state,
		nn.LinearConfig{Heads: linHeads, KeyDim: linKey, ValueDim: linValue})
	if next == nil {
		t.Fatal("the block returned no next state")
	}
	// Output the operator's own shape. A Reshape in front of a graph output
	// silently produces zeros (accel#26), which is what this test found and
	// what the block's return shape is chosen to avoid.
	got, _ := r.run(out)

	want := linearReference(qv, kv, vv, av, bv, sv, []int{2, 1})
	// The scan carries each step's error into the next, so the budget is the
	// contractions plus the longest chain of steps any sequence takes: two.
	closeTo(t, got, want, 1e-5, "gated delta")
}

// TestLinearAttentionRefusals: each names the field.
func TestLinearAttentionRefusals(t *testing.T) {
	for _, c := range []struct {
		name string
		cfg  nn.LinearConfig
		qw   int
		want string
	}{
		{"no heads", nn.LinearConfig{KeyDim: linKey, ValueDim: linValue},
			linHeads * linKey, "each is at least one"},
		{"a q that is not the heads times the key width",
			nn.LinearConfig{Heads: linHeads, KeyDim: linKey, ValueDim: linValue},
			linHeads*linKey + 1, "heads of"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := newRig(t, 1e-6)
			q := r.input("q", accel.F32, tensor.Shape{linT, c.qw})
			k := r.input("k", accel.F32, tensor.Shape{linT, linHeads * linKey})
			v := r.input("v", accel.F32, tensor.Shape{linT, linHeads * linValue})
			a := r.input("alpha", accel.F32, tensor.Shape{linT})
			b := r.input("beta", accel.F32, tensor.Shape{linT})
			e := r.input("extents", accel.U32, tensor.Shape{linSlots})
			s := tensor.NewState(r.g.B, tensor.StateDesc{
				Name: "s", DType: accel.F32,
				Shape: tensor.Shape{linSlots, linHeads, linValue, linKey},
			})
			out, _ := nn.LinearAttention(r.g, q, k, v, a, b, e, s, c.cfg)
			r.refuses(out, c.want)
		})
	}
	// v differs from q only in width, and a v that differs in *tokens* is the
	// mistake the check exists for.
	t.Run("a v with the wrong token count", func(t *testing.T) {
		r := newRig(t, 1e-6)
		q := r.input("q", accel.F32, tensor.Shape{linT, linHeads * linKey})
		k := r.input("k", accel.F32, tensor.Shape{linT, linHeads * linKey})
		v := r.input("v", accel.F32, tensor.Shape{linT + 1, linHeads * linValue})
		a := r.input("alpha", accel.F32, tensor.Shape{linT})
		b := r.input("beta", accel.F32, tensor.Shape{linT})
		e := r.input("extents", accel.U32, tensor.Shape{linSlots})
		s := tensor.NewState(r.g.B, tensor.StateDesc{
			Name: "s", DType: accel.F32,
			Shape: tensor.Shape{linSlots, linHeads, linValue, linKey},
		})
		out, _ := nn.LinearAttention(r.g, q, k, v, a, b, e, s,
			nn.LinearConfig{Heads: linHeads, KeyDim: linKey, ValueDim: linValue})
		r.refuses(out, "the same tokens and the same heads")
	})
}

// ramp is deterministic values that vary in sign and magnitude, so an index
// error shows up as a value error rather than as a near-mean.
func ramp(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		x := float64((i*seed)%97) / 97.0
		out[i] = float32(math.Sin(6*x) * (0.5 + x))
	}
	return out
}

// gates is a per-token gate that is neither constant nor equal to the other
// one: a block that swapped alpha for beta would still run.
func gates(n int, base, swing float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(base + swing*math.Sin(float64(i)))
	}
	return out
}

// linearReference walks specs/018-hybrid-models.md §2's recurrence in float64,
// from the equation and not from the block.
func linearReference(q, k, v, alpha, beta, s0 []float32, extents []int) []float64 {
	tokens := 0
	for _, n := range extents {
		tokens += n
	}
	out := make([]float64, tokens*linHeads*linValue)
	at := 0
	for seq, n := range extents {
		for h := range linHeads {
			S := make([][]float64, linValue)
			for i := range S {
				S[i] = make([]float64, linKey)
				for j := range S[i] {
					S[i][j] = float64(s0[((seq*linHeads+h)*linValue+i)*linKey+j])
				}
			}
			for step := range n {
				tok := at + step
				a, b := float64(alpha[tok]), float64(beta[tok])
				kk := make([]float64, linKey)
				for j := range kk {
					kk[j] = float64(k[tok*linHeads*linKey+h*linKey+j])
				}
				// u = S k, the state's reading of this key, before the decay.
				u := make([]float64, linValue)
				for i := range u {
					for j := range kk {
						u[i] += S[i][j] * kk[j]
					}
				}
				for i := range linValue {
					g := b * (float64(v[tok*linHeads*linValue+h*linValue+i]) - u[i])
					for j := range linKey {
						S[i][j] = a*S[i][j] + g*kk[j]
					}
				}
				for i := range linValue {
					acc := 0.0
					for j := range linKey {
						acc += S[i][j] * float64(q[tok*linHeads*linKey+h*linKey+j])
					}
					out[tok*linHeads*linValue+h*linValue+i] = acc
				}
			}
		}
		at += n
	}
	return out
}

const (
	convK = 3
	convC = 4
	convT = 5
)

// convPorts binds the window state and the index ports a step needs, for one
// slot contributing every row.
func convPorts(r *rig, t, k, c int) (nn.ConvState, []float32) {
	return convSlots(r, []int{t}, t, k, c, 0)
}

// convSlots is convPorts over several slots: counts[j] is what slot j
// contributes and rows is the step's padded row count.
//
// Every index is [nn.ConvIndex]'s, so a test cannot pass against arithmetic
// written twice.
func convSlots(r *rig, counts []int, rows, k, c, capacity int) (nn.ConvState, []float32) {
	idx, err := nn.ConvIndex(counts, rows, k, capacity)
	if err != nil {
		r.t.Fatalf("ConvIndex(%v, %d, %d, %d): %v", counts, rows, k, capacity, err)
	}
	st := tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "w", DType: accel.F32, Shape: tensor.Shape{idx.Rows, c},
	})
	zero := make([]float32, idx.Rows*c)
	// Only if this rig has not carried the buffer over from a previous step:
	// binding it again would zero the carry, which is the state the second
	// step exists to read.
	if _, held := r.buffers["w"]; !held {
		r.f32("w", zero)
	}
	r.u32("write", idx.Write)
	r.u32("carry", idx.Carry)
	r.u32("carryTo", idx.CarryWrite)
	w := nn.ConvState{
		State:      st,
		Write:      r.input("write", accel.U32, tensor.Shape{rows}),
		Carry:      r.input("carry", accel.U32, tensor.Shape{len(idx.Carry)}),
		CarryWrite: r.input("carryTo", accel.U32, tensor.Shape{len(idx.CarryWrite)}),
	}
	for i, tap := range idx.Taps {
		name := fmt.Sprintf("tap%d", i)
		r.u32(name, tap)
		w.Taps = append(w.Taps, r.input(name, accel.U32, tensor.Shape{rows}))
	}
	return w, zero
}

// TestADepthwiseCausalConvMatchesAReferenceComputedBeside is
// specs/018-hybrid-models.md §4.1 over a rolling window: the composition, with
// the K-1 rows before this step supplied by the state rather than by a pad
// nothing can build.
//
// The reference is written from the definition — position t reads t-K+1 through
// t, each tap scaling its own channel — and not from the composition.
func TestADepthwiseCausalConvMatchesAReferenceComputedBeside(t *testing.T) {
	r := newRig(t, 1e-6)
	x := r.input("x", accel.F32, tensor.Shape{convT, convC})
	taps := r.input("taps", accel.F32, tensor.Shape{convK, convC})
	xv := ramp(convT*convC, 7)
	tv := ramp(convK*convC, 23)
	r.f32("x", xv)
	r.f32("taps", tv)
	w, _ := convPorts(r, convT, convK, convC)

	out, next := nn.DepthwiseCausalConv(r.g, x, taps, w, convK)
	if next == nil {
		t.Fatal("the block returned no next window")
	}
	got, _ := r.run(out)

	// A first step sees zeros before it, which is what an empty window holds.
	want := convReference(xv, tv, make([]float32, (convK-1)*convC))
	closeTo(t, got, want, 1e-6, "depthwise causal conv")
}

// TestADepthwiseCausalConvCarriesAcrossSteps is the half a padded operand could
// never have: a decode step has no earlier rows in its own tensor, so the
// window is where they live.
//
// One five-token step, then a one-token step, against a reference that convolves
// all six at once. If the carry were dropped the second step would see zeros
// before it and the two would part.
func TestADepthwiseCausalConvCarriesAcrossSteps(t *testing.T) {
	r := newRig(t, 1e-6)
	tv := ramp(convK*convC, 23)
	both := ramp((convT+1)*convC, 7)

	// Step one: the first convT rows.
	x1 := r.input("x", accel.F32, tensor.Shape{convT, convC})
	taps := r.input("taps", accel.F32, tensor.Shape{convK, convC})
	r.f32("x", both[:convT*convC])
	r.f32("taps", tv)
	w, _ := convPorts(r, convT, convK, convC)
	out1, _ := nn.DepthwiseCausalConv(r.g, x1, taps, w, convK)
	r.run(out1)

	// Step two reuses the same device and the same window buffer, which is
	// what a State is for: the carry step one wrote is still in it.
	r2 := r.reuse(t, "w", "taps")
	x2 := r2.input("x2", accel.F32, tensor.Shape{1, convC})
	taps2 := r2.input("taps", accel.F32, tensor.Shape{convK, convC})
	r2.f32("x2", both[convT*convC:])
	// The same window buffer, so the same declared row count: a decode step is
	// a smaller bucket over the shared allocation, not a smaller window.
	w2, _ := convSlots(r2, []int{1}, 1, convK, convC, convK-1+convT)
	out2, _ := nn.DepthwiseCausalConv(r2.g, x2, taps2, w2, convK)
	got, _ := r2.run(out2)

	// The last row of a reference over all six tokens.
	whole := convReference(both, tv, make([]float32, (convK-1)*convC))
	want := whole[convT*convC:]
	closeTo(t, got, want, 1e-6, "a one-token step after a five-token one")
}

// TestDepthwiseCausalConvRefusals: each names the field.
func TestDepthwiseCausalConvRefusals(t *testing.T) {
	for _, c := range []struct {
		name  string
		k     int
		tapsR int
		state bool
		want  string
	}{
		{"one tap", 1, convK, true, "at least two"},
		{"taps that are not [K, C]", convK, convK + 1, true, "one row of weights per tap"},
		{"no window", convK, convK, false, "nothing else to pad with"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := newRig(t, 1e-6)
			x := r.input("x", accel.F32, tensor.Shape{convT, convC})
			taps := r.input("taps", accel.F32, tensor.Shape{c.tapsR, convC})
			var w nn.ConvState
			if c.state {
				w, _ = convPorts(r, convT, convK, convC)
			}
			out, _ := nn.DepthwiseCausalConv(r.g, x, taps, w, c.k)
			r.refuses(out, c.want)
		})
	}
}

// convReference is the definition: y[t][c] = sum_i taps[i][c] * x[t-K+1+i][c],
// with the rows before the step taken from the carry.
func convReference(x, taps, carry []float32) []float64 {
	t := len(x) / convC
	out := make([]float64, t*convC)
	at := func(row, ch int) float64 {
		if row >= 0 {
			return float64(x[row*convC+ch])
		}
		// -1 is the newest carried row, -2 the one before it.
		i := len(carry)/convC + row
		if i < 0 {
			return 0
		}
		return float64(carry[i*convC+ch])
	}
	for row := range t {
		for ch := range convC {
			acc := 0.0
			for i := range convK {
				acc += float64(taps[i*convC+ch]) * at(row-convK+1+i, ch)
			}
			out[row*convC+ch] = acc
		}
	}
	return out
}

// TestADepthwiseCausalConvCarriesPerSlot is specs/023-cache-kinds.md §10's
// row, and it is the one the single-slot carry test cannot fail on: a carry
// indexed without the slot term passes that test and fails this one.
//
// Two slots step together, then step again. Each slot's second step must
// convolve over its **own** earlier tokens, so the reference is two independent
// convolutions and not one over the flat step.
func TestADepthwiseCausalConvCarriesPerSlot(t *testing.T) {
	const c = convC
	// Five and three, not four and four: equal counts make "the carry is per
	// slot" and "the carry is the last K-1 rows of the step" the same
	// prediction, so the assertion could not tell them apart.
	first := []int{5, 3}
	// Different ramps, so a row read from the wrong slot is a wrong number and
	// not a coincidence.
	a := ramp(6*c, 7)
	b := ramp(4*c, 31)
	tv := ramp(convK*c, 23)

	// Step one: slot 0's first five, then slot 1's first three, in slot order.
	step1 := append(append([]float32(nil), a[:5*c]...), b[:3*c]...)
	r := newRig(t, 1e-6)
	x1 := r.input("x", accel.F32, tensor.Shape{len(step1) / c, c})
	taps := r.input("taps", accel.F32, tensor.Shape{convK, c})
	r.f32("x", step1)
	r.f32("taps", tv)
	w1, _ := convSlots(r, first, len(step1)/c, convK, c, 0)
	out1, _ := nn.DepthwiseCausalConv(r.g, x1, taps, w1, convK)
	r.run(out1)

	// Step two: one token each, into the same window buffer.
	capacity := len(first)*(convK-1) + len(step1)/c
	step2 := append(append([]float32(nil), a[5*c:6*c]...), b[3*c:4*c]...)
	r2 := r.reuse(t, "w", "taps")
	x2 := r2.input("x2", accel.F32, tensor.Shape{2, c})
	taps2 := r2.input("taps", accel.F32, tensor.Shape{convK, c})
	r2.f32("x2", step2)
	w2, _ := convSlots(r2, []int{1, 1}, 2, convK, c, capacity)
	out2, _ := nn.DepthwiseCausalConv(r2.g, x2, taps2, w2, convK)
	got, _ := r2.run(out2)

	// Two independent convolutions, each over its own slot's whole history.
	wantA := convReference(a, tv, make([]float32, (convK-1)*c))[5*c : 6*c]
	wantB := convReference(b, tv, make([]float32, (convK-1)*c))[3*c : 4*c]
	closeTo(t, got[:c], wantA, 1e-6, "slot 0's sixth token")
	closeTo(t, got[c:], wantB, 1e-6, "slot 1's fourth token")
}

// TestADepthwiseCausalConvPadRowReadsZeros is §3's "two properties this layout
// inherits for free", asserted rather than assumed.
//
// A bucketed step's rows past the last slot's tokens carry an index of R for
// both the read and the write. GatherRows writes zeros for an index at or above
// the table's rows and ScatterRows drops a write at or above capacity, so a pad
// row needs no mask.
//
// Two claims, and only one of them is observable. The **read** is: a pad row
// gathers zeros, so its output is zero and the real rows beside it are what an
// exact step produces. The **write** is not observable in this layout and the
// test says so rather than pretending: a pad row's write index, were the
// sentinel dropped, would land past every row any tap or carry reads, so there
// is no value to catch it with. What the sentinel buys there is that the window
// holds no row nothing wrote — which matters to a reader of a dump and to
// nothing the graph computes.
func TestADepthwiseCausalConvPadRowReadsZeros(t *testing.T) {
	const c = convC
	tv := ramp(convK*c, 23)
	all := ramp(4*c, 7)

	// Exact: three tokens in a step of three rows.
	exact := runConvStep(t, [][]float32{all[:3*c]}, []int{3}, 3, tv, c)
	// Padded: the same three tokens in a step of five rows, the last two of
	// which are whatever the bucket's buffer held.
	padded := runConvStep(t, [][]float32{all[:3*c], ramp(2*c, 99)}, []int{3}, 5, tv, c)

	want := make([]float64, len(exact))
	for i, v := range exact {
		want[i] = float64(v)
	}
	closeTo(t, padded[:3*c], want, 1e-6,
		"a padded step's real rows against an exact step's")
	// And the pad rows themselves are zero, which is the read half: a gather at
	// or above the window's rows writes zeros, and zero times any tap is zero.
	for i, v := range padded[3*c:] {
		if v != 0 {
			t.Errorf("pad element %d is %v, want 0: a pad row gathers no window row",
				i, v)
		}
	}
}

// runConvStep runs one convolution step and returns its output. parts are
// concatenated into x, counts is what each slot contributes, and rows is the
// step's padded row count.
func runConvStep(t *testing.T, parts [][]float32, counts []int, rows int,
	taps []float32, c int) []float32 {

	t.Helper()
	var xv []float32
	for _, p := range parts {
		xv = append(xv, p...)
	}
	r := newRig(t, 1e-6)
	x := r.input("x", accel.F32, tensor.Shape{rows, c})
	tp := r.input("taps", accel.F32, tensor.Shape{len(taps) / c, c})
	r.f32("x", xv)
	r.f32("taps", taps)
	w, _ := convSlots(r, counts, rows, len(taps)/c, c, 0)
	out, _ := nn.DepthwiseCausalConv(r.g, x, tp, w, len(taps)/c)
	got, _ := r.run(out)
	return got
}

// TestConvIndexRefusesWhatItCannotAddress: each names the number.
func TestConvIndexRefusesWhatItCannotAddress(t *testing.T) {
	for _, c := range []struct {
		name         string
		counts       []int
		rows, k, cap int
		want         string
	}{
		{"one tap", []int{2}, 2, 1, 0, "at least two"},
		{"no slots", nil, 2, 3, 0, "convolves nothing"},
		{"a negative count", []int{2, -1}, 2, 3, 0, "contributes -1 rows"},
		{"more tokens than rows", []int{3, 3}, 4, 3, 0, "cannot carry 6 tokens"},
		{"a window too small", []int{2}, 2, 3, 3, "holds 3 rows and this step needs 4"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := nn.ConvIndex(c.counts, c.rows, c.k, c.cap)
			if err == nil {
				t.Fatalf("ConvIndex(%v, %d, %d, %d) was accepted", c.counts, c.rows,
					c.k, c.cap)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("the refusal %q does not say %q", err, c.want)
			}
		})
	}
}
