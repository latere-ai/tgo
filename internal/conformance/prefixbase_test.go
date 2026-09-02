// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/internal/conformance"
	"github.com/latere-ai/tgo/internal/oracle"
)

// The shape of a partial prefix hit: a prompt of prefixLen+suffixLen tokens
// whose first prefixLen rows were already in the cache, so only the suffix is
// prefilled and it starts at a nonzero causal base.
//
// prefixLen is not a multiple of suffixLen and neither divides the other, so a
// kernel that confused one for the other would be caught rather than agree by
// arithmetic accident.
const (
	prefixLen = 9
	suffixLen = 5
	promptLen = prefixLen + suffixLen

	baseQHeads  = 4
	baseKVHeads = 2
	baseHeadDim = 8
)

// TestPartialHitAttentionAtANonzeroBase is specs/016-prefix-cache.md §8's one
// uncovered row.
//
// 016 §9 records that [C13](../../specs/010-conformance.md) — a paged prefill —
// was marked verified by an assertion about the `base` *value* rather than
// about the attention *output*. This asserts the output, in the shape a partial
// hit produces: the cache holds every row of the prompt, the queries are the
// suffix only, and `BaseName` says where the suffix starts.
//
// Two things are checked, and the second is the one a value probe adds:
//
//  1. the suffix's output matches a float64 reference at causalBase =
//     prefixLen. A kernel that masked against the query index rather than
//     against causalBase+t would let row 0 of the suffix see one cache row
//     where it should see prefixLen+1, which is a finite, plausible, wrong
//     answer;
//  2. it equals the same rows of a **cold** prefill of the whole prompt at base
//  0. That is what "a warm request produces the same tokens as a cold one"
//     means one layer down, and it is the property the prefix cache sells.
func TestPartialHitAttentionAtANonzeroBase(t *testing.T) {
	q := seeded(promptLen*baseQHeads*baseHeadDim, 31)
	k := seeded(promptLen*baseKVHeads*baseHeadDim, 32)
	v := seeded(promptLen*baseKVHeads*baseHeadDim, 33)
	scale := 1 / math.Sqrt(baseHeadDim)

	// The queries the suffix actually submits. A partial hit re-embeds nothing
	// it hit on, so these are rows prefixLen.. of the same prompt.
	qSuffix := q[prefixLen*baseQHeads*baseHeadDim:]

	want := oracle.Attention(f64(qSuffix), f64(k), f64(v),
		suffixLen, baseQHeads, baseKVHeads, baseHeadDim, promptLen, scale, prefixLen)

	warm, _ := runAttention(t, "warm-suffix", qSuffix, k, v, suffixLen, prefixLen, want)

	// The cold run: the whole prompt prefilled from an empty cache. Its last
	// suffixLen rows are the same tokens at the same absolute positions
	// attending over the same keys, so they must be the same values.
	cold, _ := runAttention(t, "cold-whole", q, k, v, promptLen, 0,
		oracle.Attention(f64(q), f64(k), f64(v),
			promptLen, baseQHeads, baseKVHeads, baseHeadDim, promptLen, scale, 0))

	tail := cold[prefixLen*baseQHeads*baseHeadDim:]
	terms := attentionTerms(want, scale)
	for i := range warm {
		if diff := math.Abs(float64(warm[i] - tail[i])); diff > terms.Bound(math.Abs(float64(tail[i]))) {
			t.Fatalf("row %d of a suffix prefilled at base %d differs from the same "+
				"row of a cold whole-prompt prefill by %.3g, over a budget of %.3g. "+
				"A prefix hit that changes the answer is not a cache.\n%s",
				i/(baseQHeads*baseHeadDim), prefixLen, diff,
				terms.Bound(math.Abs(float64(tail[i]))), terms.Explain())
		}
	}
}

// runAttention prefills qSeq query rows against a cache already holding the
// whole prompt, starting at causal position base, and checks the result against
// want.
func runAttention(t *testing.T, label string, q, k, v []float32, qSeq, base int,
	want []float64) ([]float32, *tensor.Plan) {
	t.Helper()

	scale := 1 / math.Sqrt(baseHeadDim)
	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: 1e-6, Label: label})

	qt := r.Input("q", accel.F32, tensor.Shape{qSeq, baseQHeads, baseHeadDim})
	r.F32("q", q[:qSeq*baseQHeads*baseHeadDim])

	// Every row of the prompt is live, including the prefixLen the hit
	// supplied: a partial hit reuses cache rows rather than recomputing them,
	// so the queries are short and the cache is not.
	lengths := r.Input("lengths", accel.U32, tensor.Shape{1})
	r.U32("lengths", []uint32{promptLen})

	ks := tensor.NewState(r.G.B, tensor.StateDesc{Name: "k", DType: accel.F32,
		Shape: tensor.Shape{promptLen, baseKVHeads, baseHeadDim}})
	vs := tensor.NewState(r.G.B, tensor.StateDesc{Name: "v", DType: accel.F32,
		Shape: tensor.Shape{promptLen, baseKVHeads, baseHeadDim}})
	r.F32("k", k)
	r.F32("v", v)

	r.ScalarF32("scale", float32(scale))
	r.ScalarU32("base", uint32(base))

	out := tensor.Attention(r.G.B, qt, ks, vs, tensor.AttentionOptions{
		Lengths: lengths, ScaleName: "scale", BaseName: "base"})

	return r.Parity(out, func() []float64 { return want }, attentionTerms(want, scale))
}

// attentionTerms is the budget for one attention output element.
//
// Three stages, and the middle one is why [conformance.SoftmaxWeight] exists:
// the score is a dot product over headDim accumulated in f32, the exponential
// turns that absolute error into a relative one on the weight, and the weighted
// sum over the visible rows accumulates again. The magnitude is the reference's
// own, so a near-zero output is charged an absolute floor rather than a
// relative bound it cannot meet.
func attentionTerms(want []float64, scale float64) conformance.Terms {
	// The largest score any row could have reached. The reference's inputs are
	// bounded by 2 (seeded quantizes to [-2, 2)), so a dot over headDim is
	// bounded by 4*headDim before scaling.
	maxScore := 4 * float64(baseHeadDim) * scale
	mag := 0.0
	for _, x := range want {
		if a := math.Abs(x); a > mag {
			mag = a
		}
	}
	return conformance.SoftmaxWeight(maxScore, baseHeadDim).
		And(conformance.AccumF32(promptLen)).
		And(conformance.Magnitude(mag))
}
