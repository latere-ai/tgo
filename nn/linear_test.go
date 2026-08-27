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
