// Copyright 2026 The tgo Authors. All rights reserved.

package nn_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// The shapes one attention block is tested at. Small enough to compute by hand
// beside it, and grouped -- four query heads over two key/value heads -- because
// GQA is where the q and k row counts stop agreeing, which is why posQ and posK
// are separate arguments.
//
// Two key/value heads rather than one, deliberately. At attKVHeads = 1 three
// separate properties of the k side collapse into identities: the reshape to
// [T*Hkv, headDim] is a no-op, the reshape back to [T, Hkv*headDim] is a no-op,
// and posK per row equals posK per token. Deleting k's reshapes entirely then
// passes every test in this file. Two heads is the smallest count where each of
// the three is a real statement, and it keeps qHeads/kvHeads an integer.
const (
	// Four tokens rather than two. The first token of a prefill attends only
	// to itself, so its output does not depend on q at all; two tokens left
	// one row of the output carrying the whole ordering signal, and the
	// deviation between the two QK-norm orders fell to 4e-4 once the head
	// counts below became non-degenerate.
	attT       = 4 // tokens
	attHidden  = 4
	attQHeads  = 4
	attKVHeads = 2
	attHeadDim = 4
	attCap     = 4       // cache capacity
	attEps     = 1e-6    // rms_norm_eps
	attBase    = 10000.0 // rope_theta
)

// Weights that f16 holds exactly: every value is a multiple of 1/8 below 8, and
// f16's mantissa is 11 bits. A reference computed in f64 then differs from the
// graph only by the f32 accumulation, which is what the tolerances name.
//
// The heads after the first are scaled by 8, in both the query and the key
// projection, so the heads have deliberately different magnitudes. Without
// that, a norm over one head and a norm over the whole H*headDim row divide by
// the same number and the axis test cannot tell them apart.
func attWeights() (wq, wk, wv, wo, qnorm, knorm, x []float32) {
	wq = make([]float32, attHidden*attQHeads*attHeadDim)
	for i := range wq {
		wq[i] = float32((i%7)-3) / 4
		if i%(attQHeads*attHeadDim) >= attHeadDim {
			wq[i] *= 8
		}
	}
	wk = make([]float32, attHidden*attKVHeads*attHeadDim)
	for i := range wk {
		wk[i] = float32((i%5)-2) / 4
		// As for wq, and for the same reason: the key heads need different
		// magnitudes or a norm over one head and a norm over the whole row
		// divide by the same number.
		if i%(attKVHeads*attHeadDim) >= attHeadDim {
			wk[i] *= 8
		}
	}
	wv = make([]float32, attHidden*attKVHeads*attHeadDim)
	for i := range wv {
		wv[i] = float32((i%9)-4) / 8
	}
	wo = make([]float32, attQHeads*attHeadDim*attHidden)
	for i := range wo {
		wo[i] = float32((i%11)-5) / 8
	}
	// Non-uniform gains, which the ordering test needs: RMSNorm scales a row by
	// a scalar and RoPE is a rotation, so with a gain of all ones the two
	// commute exactly and normalizing before or after the rotation is the same
	// arithmetic. The gain is what breaks the commutation.
	qnorm = []float32{0.25, 1, 2, 4}
	knorm = []float32{1, 0.25, 4, 2}
	x = []float32{1, -2, 0.5, 0.25, 0.75, 1.5, -1, 2, -0.5, 0.25, 2, -1.5, 1.25, -0.75, -2, 0.5}
	return
}

// qkPlacement selects which arrangement of QK-norm the reference computes.
type qkPlacement int

const (
	normPerHeadBeforeRoPE qkPlacement = iota // what Qwen3 does
	normPerHeadAfterRoPE                     // the ordering mistake
	normPooledBeforeRoPE                     // the axis mistake
)

// rope rotates one row's interleaved pairs, which is accel's convention.
//
// specs/004-model-graph.md section 2.5.2 permutes q and k at load so that this
// kernel computes Qwen3's half-split rotation. The permutation is the loader's;
// what this reference has to match is the kernel.
func rope(row []float64, pos int, base float64) {
	d := len(row)
	for k := 0; k < d/2; k++ {
		theta := float64(pos) * math.Pow(base, -2*float64(k)/float64(d))
		c, s := math.Cos(theta), math.Sin(theta)
		lo, hi := 2*k, 2*k+1
		row[lo], row[hi] = row[lo]*c-row[hi]*s, row[lo]*s+row[hi]*c
	}
}

// rmsnorm scales a row by its own root mean square and a gain.
func rmsnorm(row []float64, gain []float64, eps float64) {
	var sq float64
	for _, v := range row {
		sq += v * v
	}
	scale := 1 / math.Sqrt(sq/float64(len(row))+eps)
	for i := range row {
		row[i] *= scale * gain[i%len(gain)]
	}
}

// referenceAttention is the block from its definition, in f64.
//
// It shares no structure with nn: the projections are three nested loops, the
// rotation is a rotation, and the softmax is a softmax. positions are given per
// token; this reference repeats them per head where the graph binds them per
// row, which is the same statement seen from the two sides.
func referenceAttention(place qkPlacement, tokens int, positions []int, cacheK, cacheV [][]float64) []float64 {
	wq, wk, wv, wo, qnormF, knormF, xF := attWeights()
	qnorm, knorm := f64s(qnormF), f64s(knormF)
	x := f64s(xF)[:tokens*attHidden]

	q := matmul(x, f64s(wq), tokens, attHidden, attQHeads*attHeadDim)
	k := matmul(x, f64s(wk), tokens, attHidden, attKVHeads*attHeadDim)
	v := matmul(x, f64s(wv), tokens, attHidden, attKVHeads*attHeadDim)

	// The per-head rows, which is the reshape of rows 9 to 11 of section 3.
	for t := range tokens {
		if place == normPooledBeforeRoPE {
			// The axis mistake: one statistic for the whole row, so both heads
			// are divided by the same number.
			rmsnorm(q[t*attQHeads*attHeadDim:(t+1)*attQHeads*attHeadDim], qnorm, attEps)
		}
		for h := range attQHeads {
			row := q[t*attQHeads*attHeadDim+h*attHeadDim:][:attHeadDim]
			switch place {
			case normPerHeadBeforeRoPE:
				rmsnorm(row, qnorm, attEps)
				rope(row, positions[t], attBase)
			case normPerHeadAfterRoPE:
				rope(row, positions[t], attBase)
				rmsnorm(row, qnorm, attEps)
			case normPooledBeforeRoPE:
				rope(row, positions[t], attBase)
			}
		}
		if place == normPooledBeforeRoPE {
			rmsnorm(k[t*attKVHeads*attHeadDim:(t+1)*attKVHeads*attHeadDim], knorm, attEps)
		}
		for h := range attKVHeads {
			row := k[t*attKVHeads*attHeadDim+h*attHeadDim:][:attHeadDim]
			switch place {
			case normPerHeadBeforeRoPE:
				rmsnorm(row, knorm, attEps)
				rope(row, positions[t], attBase)
			case normPerHeadAfterRoPE:
				rope(row, positions[t], attBase)
				rmsnorm(row, knorm, attEps)
			case normPooledBeforeRoPE:
				rope(row, positions[t], attBase)
			}
		}
	}

	// The cache, written at the slots this step owns. Whatever the caller put
	// there before stays: that is what a prefill extending a cache means.
	keys := append([][]float64{}, cacheK...)
	vals := append([][]float64{}, cacheV...)
	for t := range tokens {
		slot := positions[t]
		for len(keys) <= slot {
			keys = append(keys, make([]float64, attKVHeads*attHeadDim))
			vals = append(vals, make([]float64, attKVHeads*attHeadDim))
		}
		keys[slot] = append([]float64{}, k[t*attKVHeads*attHeadDim:][:attKVHeads*attHeadDim]...)
		vals[slot] = append([]float64{}, v[t*attKVHeads*attHeadDim:][:attKVHeads*attHeadDim]...)
	}

	scale := 1 / math.Sqrt(attHeadDim)
	attended := make([]float64, tokens*attQHeads*attHeadDim)
	for t := range tokens {
		for h := range attQHeads {
			kvHead := h / (attQHeads / attKVHeads)
			limit := positions[t]
			scores := make([]float64, 0, limit+1)
			for p := 0; p <= limit && p < len(keys); p++ {
				var dot float64
				for i := range attHeadDim {
					dot += q[t*attQHeads*attHeadDim+h*attHeadDim+i] *
						keys[p][kvHead*attHeadDim+i]
				}
				scores = append(scores, dot*scale)
			}
			max := math.Inf(-1)
			for _, s := range scores {
				max = math.Max(max, s)
			}
			var sum float64
			for i, s := range scores {
				scores[i] = math.Exp(s - max)
				sum += scores[i]
			}
			for p, w := range scores {
				for i := range attHeadDim {
					attended[t*attQHeads*attHeadDim+h*attHeadDim+i] +=
						w / sum * vals[p][kvHead*attHeadDim+i]
				}
			}
		}
	}
	return matmul(attended, f64s(wo), tokens, attQHeads*attHeadDim, attHidden)
}

// attentionRig records one attention block over the tokens x[first:first+n] at
// the given absolute positions, with the cache pre-filled from cacheK, cacheV.
func attentionRig(t *testing.T, first, tokens int, positions []int, cacheK, cacheV []float32) (*rig, *tensor.Tensor) {
	t.Helper()
	wq, wk, wv, wo, qnorm, knorm, x := attWeights()
	r := newRig(t, attEps)
	r.scalarF32("rope_base", attBase)
	r.scalarF32("scale", float32(1/math.Sqrt(attHeadDim)))
	r.scalarU32("base", uint32(positions[0]))

	in := r.input("x", accel.F32, tensor.Shape{tokens, attHidden})
	r.f32("x", x[first*attHidden:(first+tokens)*attHidden])

	posQ := make([]uint32, 0, tokens*attQHeads)
	posK := make([]uint32, 0, tokens*attKVHeads)
	slots := make([]uint32, tokens)
	for i, p := range positions {
		slots[i] = uint32(p)
		// One position per *row*, which repeats a token's position once per
		// head: section 2.5's positions formula.
		for range attQHeads {
			posQ = append(posQ, uint32(p))
		}
		for range attKVHeads {
			posK = append(posK, uint32(p))
		}
	}
	pq := r.input("posq", accel.U32, tensor.Shape{len(posQ)})
	pk := r.input("posk", accel.U32, tensor.Shape{len(posK)})
	sl := r.input("slots", accel.U32, tensor.Shape{tokens})
	ln := r.input("lengths", accel.U32, tensor.Shape{1})
	r.u32("posq", posQ)
	r.u32("posk", posK)
	r.u32("slots", slots)
	r.u32("lengths", []uint32{uint32(positions[len(positions)-1] + 1)})

	w := nn.AttentionWeights{
		Q:     r.g.Weight("wq", tensor.Shape{attHidden, attQHeads * attHeadDim}),
		K:     r.g.Weight("wk", tensor.Shape{attHidden, attKVHeads * attHeadDim}),
		V:     r.g.Weight("wv", tensor.Shape{attHidden, attKVHeads * attHeadDim}),
		O:     r.g.Weight("wo", tensor.Shape{attQHeads * attHeadDim, attHidden}),
		QNorm: r.g.Gain("qnorm", attHeadDim),
		KNorm: r.g.Gain("knorm", attHeadDim),
	}
	r.f16("block.wq", wq)
	r.f16("block.wk", wk)
	r.f16("block.wv", wv)
	r.f16("block.wo", wo)
	r.f32("block.qnorm", qnorm)
	r.f32("block.knorm", knorm)

	kc := tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "k", DType: accel.F32, Shape: tensor.Shape{attCap, attKVHeads, attHeadDim},
	})
	vc := tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{attCap, attKVHeads, attHeadDim},
	})
	k0 := make([]float32, attCap*attKVHeads*attHeadDim)
	v0 := make([]float32, attCap*attKVHeads*attHeadDim)
	copy(k0, cacheK)
	copy(v0, cacheV)
	r.f32("k", k0)
	r.f32("v", v0)

	out := nn.Attention(r.g, in, w, kc, vc, pq, pk, sl, ln, nn.AttentionConfig{
		QHeads: attQHeads, KVHeads: attKVHeads, HeadDim: attHeadDim,
		RoPEBase: "rope_base", ScaleName: "scale", BaseName: "base", QKNorm: true,
	})
	return r, out
}

func TestAttentionMatchesAReferenceComputedBeside(t *testing.T) {
	r, out := attentionRig(t, 0, attT, []int{0, 1, 2, 3}, nil, nil)
	got, plan := r.run(out)

	want := referenceAttention(normPerHeadBeforeRoPE, attT, []int{0, 1, 2, 3}, nil, nil)
	// 1e-4 relative: the f32 softmax, whose exponential dominates every other
	// term here -- the projections accumulate over K=4 and the weights are
	// exact in f16.
	closeTo(t, got, want, 1e-4, "attention")

	sel := selected(t, plan, "Attention")
	if !strings.Contains(sel.Reason, "causal prefill kernel") {
		t.Fatalf("attention over %d tokens became %q because %q; want the prefill kernel",
			attT, sel.Kernel, sel.Reason)
	}
	// No cast anywhere: the projections read f32 activations against f16
	// weights directly (C8), and the cache is f32.
	for _, s := range plan.Selections() {
		if s.Op == "Cast" {
			t.Fatalf("the graph casts: %v", s)
		}
	}
}

// The Qwen3 ordering: Q and K are normalized before RoPE, not after.
//
// The two orders differ only because a gain does not commute with a rotation,
// and only at a nonzero position -- at position 0 RoPE is the identity. This
// test therefore asserts both directions: the graph matches the order Qwen3
// uses, and *disagrees* with the other one. Without the second assertion a
// test like this can pass while measuring nothing.
func TestQKNormRunsBeforeRoPE(t *testing.T) {
	r, out := attentionRig(t, 0, attT, []int{0, 1, 2, 3}, nil, nil)
	got, _ := r.run(out)

	closeTo(t, got, referenceAttention(normPerHeadBeforeRoPE, attT, []int{0, 1, 2, 3}, nil, nil),
		1e-4, "qk-norm before rope")
	// 1e-2, against a measured maximum relative deviation of 0.107 between the
	// two orders: the margin is what says the test discriminates, and it is
	// two orders of magnitude above the 1e-4 the matching half uses.
	differs(t, got, referenceAttention(normPerHeadAfterRoPE, attT, []int{0, 1, 2, 3}, nil, nil),
		1e-2, "qk-norm after rope")
}

// The Qwen3 axis: the norm reduces over headDim, one head at a time, not over
// the whole H*headDim row.
//
// The two agree when every head has the same root mean square, which is why
// attWeights scales the trailing heads by 8 -- in wk as well as wq, or the k
// side of the property has no test.
func TestQKNormNormalisesOverTheHeadDimension(t *testing.T) {
	r, out := attentionRig(t, 0, attT, []int{0, 1, 2, 3}, nil, nil)
	got, _ := r.run(out)

	closeTo(t, got, referenceAttention(normPerHeadBeforeRoPE, attT, []int{0, 1, 2, 3}, nil, nil),
		1e-4, "per-head norm")
	// 1e-2, against a measured maximum relative deviation of 0.704 between the
	// two axes, which is what the 8x scaling of the trailing heads buys.
	differs(t, got, referenceAttention(normPooledBeforeRoPE, attT, []int{0, 1, 2, 3}, nil, nil),
		1e-2, "pooled norm")
}

// Decode is the same graph at T=1, and accel tells the two apart by q's rank
// (section 3.1). The cache is filled with what the prefill would have written,
// so the answer must be the prefill's last row.
func TestDecodeAtOneTokenEqualsThePrefillsLastRow(t *testing.T) {
	prefill, out := attentionRig(t, 0, attT, []int{0, 1, 2, 3}, nil, nil)
	whole, _ := prefill.run(out)

	// The keys and values of every token before the last, as the prefill wrote
	// them, computed here and bound as the decode's starting cache.
	k0, v0 := cachedRow(0, 0)
	k1, v1 := cachedRow(1, 1)
	k2, v2 := cachedRow(2, 2)
	decode, dOut := attentionRig(t, attT-1, 1, []int{attT - 1},
		concat(k0, k1, k2), concat(v0, v1, v2))
	one, plan := decode.run(dOut)

	sel := selected(t, plan, "Attention")
	if strings.Contains(sel.Reason, "prefill") {
		t.Fatalf("attention over one token became %q because %q; want a decode kernel",
			sel.Kernel, sel.Reason)
	}
	// 1e-3 relative: the same f32 softmax, plus the cached key and value having
	// made a round trip through the f64 reference and back into f32.
	closeTo(t, one, f64s(whole[(attT-1)*attHidden:]), 1e-3, "decode against prefill")
}

// concat joins the cached rows of the tokens a decode step follows.
func concat(rows ...[]float32) []float32 {
	var out []float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// cachedRow returns one token's normalized, rotated key and its value, which is
// what a prefill leaves in that token's slot.
//
// The position is a separate argument from the token index because they are
// separate things: the token index selects a row of x, and the position is what
// RoPE rotates by.
func cachedRow(token, pos int) (k, v []float32) {
	wk, wv := func() ([]float32, []float32) {
		_, wk, wv, _, _, _, _ := attWeights()
		return wk, wv
	}()
	_, _, _, _, _, knormF, xF := attWeights()
	x := f64s(xF)[token*attHidden : (token+1)*attHidden]
	krow := matmul(x, f64s(wk), 1, attHidden, attKVHeads*attHeadDim)
	vrow := matmul(x, f64s(wv), 1, attHidden, attKVHeads*attHeadDim)
	// Per head, which is what the block does. At one key/value head this was
	// the same call over the whole row; at two it is not.
	for h := range attKVHeads {
		row := krow[h*attHeadDim:][:attHeadDim]
		rmsnorm(row, f64s(knormF), attEps)
		rope(row, pos, attBase)
	}
	k = make([]float32, len(krow))
	v = make([]float32, len(vrow))
	for i := range krow {
		k[i] = float32(krow[i])
	}
	for i := range vrow {
		v[i] = float32(vrow[i])
	}
	return k, v
}
