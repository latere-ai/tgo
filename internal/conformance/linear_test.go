// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// The 018 probe: the gated delta layer that 48 of Qwen3.8-27B's 64 layers are.
//
// specs/018-hybrid-models.md carried status: blocked on accel#17 from the day
// it was written. The operator now exists, and specs/010-conformance.md §2.2.1
// is the reason that is not the same statement as "the row is closed": four
// rows once closed upstream and stayed open here. So this asserts a value.
//
// The recurrence, from accel specs/047-linear-attention.md:
//
//	S_t = S_{t-1}(alpha_t I - beta_t k_t k_t^T) + beta_t v_t k_t^T
//	o_t = S_t q_t
//
// with S of shape [valueDim, keyDim] per sequence per head.

// linearShape is the probe's geometry. valueDim differs from keyDim on purpose:
// at equal widths a transposed state runs and is wrong, which is the mistake
// accel's own refusal names, and a probe that cannot see it is not a probe.
type linearShape struct {
	heads, keyDim, valueDim int
}

var linShape = linearShape{heads: 2, keyDim: 3, valueDim: 4}

// linCase is a step: one sequence contributing a chunk, one contributing a
// single token, one contributing nothing.
var linExtents = []int{3, 1, 0}

type linearInputs struct {
	q, k, v, alpha, beta, state []float32
	extents                     []uint32
	sh                          linearShape
}

func newLinearInputs(sh linearShape, extents []int) linearInputs {
	tokens := 0
	for _, n := range extents {
		tokens += n
	}
	in := linearInputs{sh: sh}
	in.q = spread(tokens*sh.heads*sh.keyDim, 11)
	in.k = spread(tokens*sh.heads*sh.keyDim, 23)
	in.v = spread(tokens*sh.heads*sh.valueDim, 37)
	in.state = spread(len(extents)*sh.heads*sh.valueDim*sh.keyDim, 41)
	// The gates are what make this a *gated* delta rule, so they are neither
	// constant nor equal to each other: alpha near one decays slowly and beta
	// away from zero writes hard, and a kernel that swapped them would still
	// run.
	in.alpha = make([]float32, tokens)
	in.beta = make([]float32, tokens)
	for i := range in.alpha {
		in.alpha[i] = float32(0.85 + 0.1*math.Sin(float64(i)))
		in.beta[i] = float32(0.3 + 0.2*math.Cos(float64(2*i)))
	}
	for _, n := range extents {
		in.extents = append(in.extents, uint32(n))
	}
	return in
}

func (in linearInputs) tokens() int { return len(in.alpha) }

// oracle walks the recurrence in float64, one sequence and head at a time,
// straight from the equation above.
func (in linearInputs) oracle() ([]float64, budget) {
	sh := in.sh
	out := make([]float64, in.tokens()*sh.heads*sh.valueDim)
	var b budget
	b.positions = sh.keyDim
	tok := 0
	for seq, n := range linExtents {
		for h := 0; h < sh.heads; h++ {
			// S[i][j], i over valueDim and j over keyDim.
			S := make([][]float64, sh.valueDim)
			for i := range S {
				S[i] = make([]float64, sh.keyDim)
				for j := range S[i] {
					S[i][j] = float64(in.state[((seq*sh.heads+h)*sh.valueDim+i)*sh.keyDim+j])
				}
			}
			for t := 0; t < n; t++ {
				at := tok + t
				alpha := float64(in.alpha[at])
				beta := float64(in.beta[at])
				kv := make([]float64, sh.keyDim)
				for j := range kv {
					kv[j] = float64(in.k[(at*sh.heads+h)*sh.keyDim+j])
				}
				vv := make([]float64, sh.valueDim)
				for i := range vv {
					vv[i] = float64(in.v[(at*sh.heads+h)*sh.valueDim+i])
				}
				// S k, the current state's reading of this key.
				Sk := make([]float64, sh.valueDim)
				for i := 0; i < sh.valueDim; i++ {
					for j := 0; j < sh.keyDim; j++ {
						Sk[i] += S[i][j] * kv[j]
					}
				}
				// S <- alpha*S + beta*(v - S k) k^T, which is what
				// S(alpha I - beta k k^T) + beta v k^T distributes to: the
				// alpha multiplies the state and not the state's reading of
				// this key. Getting that wrong is a plausible-looking scan
				// that decays twice, and it is what the first draft of this
				// reference did.
				for i := 0; i < sh.valueDim; i++ {
					w := beta * (vv[i] - Sk[i])
					for j := 0; j < sh.keyDim; j++ {
						S[i][j] = alpha*S[i][j] + w*kv[j]
					}
					if a := math.Abs(S[i][0]); a > b.value {
						b.value = a
					}
				}
				for i := 0; i < sh.valueDim; i++ {
					acc := 0.0
					for j := 0; j < sh.keyDim; j++ {
						acc += S[i][j] * float64(in.q[(at*sh.heads+h)*sh.keyDim+j])
					}
					out[(at*sh.heads+h)*sh.valueDim+i] = acc
					if a := math.Abs(acc); a > b.value {
						b.value = a
					}
				}
			}
		}
		tok += n
	}
	return out, b
}

func runLinear(t *testing.T, in linearInputs) ([]float32, *tensor.Plan) {
	t.Helper()
	sh := in.sh
	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c23-linear"})

	s := tensor.NewState(r.G.B, tensor.StateDesc{Name: "s", DType: accel.F32,
		Shape: tensor.Shape{len(linExtents), sh.heads, sh.valueDim, sh.keyDim}})
	r.F32("s", in.state)

	q := r.Input("q", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.keyDim})
	k := r.Input("k", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.keyDim})
	v := r.Input("v", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.valueDim})
	r.F32("q", in.q)
	r.F32("k", in.k)
	r.F32("v", in.v)
	alpha := r.Input("alpha", accel.F32, tensor.Shape{in.tokens()})
	beta := r.Input("beta", accel.F32, tensor.Shape{in.tokens()})
	r.F32("alpha", in.alpha)
	r.F32("beta", in.beta)
	extents := r.Input("extents", accel.U32, tensor.Shape{len(linExtents)})
	r.U32("extents", in.extents)

	out, _ := tensor.LinearAttention(r.G.B, q, k, v, s, tensor.LinearOptions{
		Alpha: alpha, Beta: beta, QueryExtents: extents,
	})
	return r.Run(out)
}

// TestTheGatedDeltaScanMatchesItsOracle is the value probe.
func TestTheGatedDeltaScanMatchesItsOracle(t *testing.T) {
	in := newLinearInputs(linShape, linExtents)
	got, plan := runLinear(t, in)

	want, b := in.oracle()
	// The state is walked token by token, so the error of each step feeds the
	// next: keyDim contractions, valueDim outer-product updates, and the
	// longest chain of steps any sequence takes.
	longest := 0
	for _, n := range linExtents {
		if n > longest {
			longest = n
		}
	}
	terms := AccumF32(linShape.keyDim).
		And(AccumF32(linShape.valueDim)).
		And(RoundF32(3 * longest)).
		And(Magnitude(b.value))
	Compare(t, got, want, terms, "the gated delta scan")

	var kernel string
	for _, s := range plan.Selections() {
		if s.Op == "LinearAttention" {
			kernel = s.Kernel
		}
	}
	if !strings.Contains(kernel, "LinearAttention") {
		t.Fatalf("the step selected %q", kernel)
	}
	t.Logf("gated delta: %d tokens over %d sequences, kernel %s, state [%d, %d] per head",
		in.tokens(), len(linExtents), kernel, linShape.valueDim, linShape.keyDim)
}

// TestTheGatedDeltaGatesAreRead varies alpha and requires the output to move.
//
// A recurrence whose decay is ignored still produces plausible numbers, and
// that is the failure mode a value probe against one input cannot see: the
// oracle and the kernel would have to disagree, and a kernel that dropped
// alpha would disagree only on inputs where alpha is not one.
func TestTheGatedDeltaGatesAreRead(t *testing.T) {
	in := newLinearInputs(linShape, linExtents)
	base, _ := runLinear(t, in)

	moved := in
	moved.alpha = append([]float32(nil), in.alpha...)
	for i := range moved.alpha {
		moved.alpha[i] *= 0.5
	}
	got, _ := runLinear(t, moved)

	same := true
	for i := range got {
		if got[i] != base[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("halving every alpha produced the same output; alpha decays the " +
			"state and a step blind to it is a delta rule rather than a gated one")
	}
	want, b := moved.oracle()
	longest := 0
	for _, n := range linExtents {
		if n > longest {
			longest = n
		}
	}
	Compare(t, got, want, AccumF32(linShape.keyDim).
		And(AccumF32(linShape.valueDim)).
		And(RoundF32(3*longest)).
		And(Magnitude(b.value)), "the halved-alpha scan")
}

// TestTheGatedDeltaStateAxesAreNotInterchangeable is the refusal accel's own comment
// says is the mistake worth preventing, checked rather than trusted.
func TestTheGatedDeltaStateAxesAreNotInterchangeable(t *testing.T) {
	in := newLinearInputs(linShape, linExtents)
	sh := in.sh
	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c23-transposed"})

	// [slots, heads, keyDim, valueDim] -- the last two the wrong way round.
	s := tensor.NewState(r.G.B, tensor.StateDesc{Name: "s", DType: accel.F32,
		Shape: tensor.Shape{len(linExtents), sh.heads, sh.keyDim, sh.valueDim}})
	q := r.Input("q", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.keyDim})
	k := r.Input("k", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.keyDim})
	v := r.Input("v", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.valueDim})
	alpha := r.Input("alpha", accel.F32, tensor.Shape{in.tokens()})
	beta := r.Input("beta", accel.F32, tensor.Shape{in.tokens()})
	extents := r.Input("extents", accel.U32, tensor.Shape{len(linExtents)})

	tensor.LinearAttention(r.G.B, q, k, v, s, tensor.LinearOptions{
		Alpha: alpha, Beta: beta, QueryExtents: extents,
	})
	if err := r.G.Err(); err == nil {
		t.Fatal("a state with valueDim and keyDim transposed recorded; at unequal " +
			"widths that is a shape error and at equal widths it would be a silently " +
			"wrong answer")
	}
}

// TestTheGatedDeltaGateHasNoHeadAxis is C27's probe.
//
// The gate accel takes is one f32 per token: LinearOptions documents Alpha and
// Beta as "one entry per token, in the flat order q is in", and the kernel
// reads alpha[tok] with no head term. A model that gives each value head its
// own decay is therefore inexpressible — every head of a token shares one
// alpha and one beta.
//
// This asks the question by value rather than by reading the comment, because
// the answer that matters is not "does it refuse" but "what does it do". A
// refusal is a gap tgo can report and wait on. An accept-and-broadcast is the
// class [010 §5](../../specs/010-conformance.md) exists for: 48 layers of a
// 27B model computing a plausible wrong answer, with nothing red.
func TestTheGatedDeltaGateHasNoHeadAxis(t *testing.T) {
	in := newLinearInputs(linShape, linExtents)
	sh := in.sh
	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c27-per-head-gate"})

	s := tensor.NewState(r.G.B, tensor.StateDesc{Name: "s", DType: accel.F32,
		Shape: tensor.Shape{len(linExtents), sh.heads, sh.valueDim, sh.keyDim}})
	q := r.Input("q", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.keyDim})
	k := r.Input("k", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.keyDim})
	v := r.Input("v", accel.F32, tensor.Shape{in.tokens(), sh.heads, sh.valueDim})
	extents := r.Input("extents", accel.U32, tensor.Shape{len(linExtents)})

	// [tokens, heads] rather than [tokens]: a decay per head per token, which
	// is the shape a Qwen3.5 `in_proj_ba` of width 2*valueHeads would produce.
	alpha := r.Input("alpha", accel.F32, tensor.Shape{in.tokens(), sh.heads})
	beta := r.Input("beta", accel.F32, tensor.Shape{in.tokens(), sh.heads})

	tensor.LinearAttention(r.G.B, q, k, v, s, tensor.LinearOptions{
		Alpha: alpha, Beta: beta, QueryExtents: extents,
	})
	if err := r.G.Err(); err == nil {
		t.Fatal("a per-head gate recorded; accel would then be reading the first " +
			"heads-worth of a [tokens, heads] gate as though it were [tokens], " +
			"which is a wrong decay on every head but the first and produces no " +
			"error anywhere")
	} else {
		t.Logf("refused, which is the good answer: %v", err)
	}
}
