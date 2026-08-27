// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package nn_test

import (
	"math"
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

// convRig binds the window state and the three index ports a step needs.
func convPorts(r *rig, t, k, c int) (nn.ConvState, []float32) {
	w := k - 1 + t
	st := tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "w", DType: accel.F32, Shape: tensor.Shape{w, c},
	})
	write := make([]uint32, t)
	for i := range write {
		write[i] = uint32(k - 1 + i)
	}
	carry := make([]uint32, k-1)
	carryTo := make([]uint32, k-1)
	for i := range carry {
		carry[i] = uint32(t + i)
		carryTo[i] = uint32(i)
	}
	zero := make([]float32, w*c)
	r.f32("w", zero)
	r.u32("write", write)
	r.u32("carry", carry)
	r.u32("carryTo", carryTo)
	return nn.ConvState{
		State:      st,
		Write:      r.input("write", accel.U32, tensor.Shape{t}),
		Carry:      r.input("carry", accel.U32, tensor.Shape{k - 1}),
		CarryWrite: r.input("carryTo", accel.U32, tensor.Shape{k - 1}),
	}, zero
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
	w2 := nn.ConvState{
		State: tensor.NewState(r2.g.B, tensor.StateDesc{
			Name: "w", DType: accel.F32, Shape: tensor.Shape{convK - 1 + convT, convC},
		}),
		Write:      r2.input("write2", accel.U32, tensor.Shape{1}),
		Carry:      r2.input("carry2", accel.U32, tensor.Shape{convK - 1}),
		CarryWrite: r2.input("carryTo2", accel.U32, tensor.Shape{convK - 1}),
	}
	r2.u32("write2", []uint32{convK - 1})
	r2.u32("carry2", []uint32{1, 2})
	r2.u32("carryTo2", []uint32{0, 1})
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
