// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package oracle

// Attention computes grouped-query causal attention:
//
//	Attn(q, K, V) = softmax(q·K^T · scale) · V
//
// as specs/004-model-graph.md §2.4 defines it, with scale passed in rather
// than derived. accel binds 1/sqrt(d_h) as a scalar because it is a model
// constant, and a caller that wants a different one — a scaled variant, or a
// deliberately wrong one in a probe — says so here.
//
// Shapes, all row-major:
//
//	q:   [qSeq, qHeads, headDim]
//	k,v: [kvLen, kvHeads, headDim] or longer
//	out: [qSeq, qHeads, headDim]
//
// k and v may be longer than kvLen*kvHeads*headDim and the tail is ignored.
// The cache they come from is a State of capacity C with a separate length
// (004 §3 rows 14-16), so a caller binds the whole cache and says how much of
// it is live.
//
// Grouping: qHeads/kvHeads query heads share each cache entry, and query head
// h reads kv head h/(qHeads/kvHeads). That is floor division of the query head
// by the group size, not h%kvHeads: the queries that share a cache entry are
// adjacent, not strided. accel derives the same grouping from the shapes and
// refuses a non-integer ratio.
//
// Causal masking: causalBase is the absolute position of the first query row,
// so query t sits at causalBase+t and attends to cache rows 0..causalBase+t.
// A prefill from empty passes 0; a decode at position p passes p with qSeq 1,
// and then every one of the p+1 cached rows is visible. The current step's own
// keys are scattered into the cache before attention runs, so causalBase+qSeq
// can never exceed kvLen in a well-formed graph and a caller that says
// otherwise has an off-by-one rather than a shorter mask.
func Attention(q, k, v []float64, qSeq, qHeads, kvHeads, headDim, kvLen int, scale float64, causalBase int) []float64 {
	if qSeq < 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || kvLen <= 0 {
		panic("oracle: Attention has a non-positive dimension")
	}
	if qHeads%kvHeads != 0 {
		panic("oracle: Attention query heads are not a multiple of key/value heads")
	}
	if len(q) != qSeq*qHeads*headDim {
		panic("oracle: Attention q length does not match qSeq*qHeads*headDim")
	}
	if len(k) < kvLen*kvHeads*headDim {
		panic("oracle: Attention k is shorter than kvLen*kvHeads*headDim")
	}
	if len(v) < kvLen*kvHeads*headDim {
		panic("oracle: Attention v is shorter than kvLen*kvHeads*headDim")
	}
	if causalBase < 0 {
		panic("oracle: Attention causalBase is negative")
	}
	if causalBase+qSeq > kvLen {
		panic("oracle: Attention causalBase+qSeq exceeds kvLen; the step's own keys are not in the cache")
	}

	group := qHeads / kvHeads
	out := make([]float64, qSeq*qHeads*headDim)
	for t := range qSeq {
		// Visible cache rows for this query: everything up to its own
		// position. The mask is applied by shortening the score vector rather
		// than by adding -Inf, so the softmax never sees a term it has to
		// cancel.
		visible := causalBase + t + 1
		for h := range qHeads {
			kvh := h / group
			qrow := q[(t*qHeads+h)*headDim : (t*qHeads+h)*headDim+headDim]

			scores := make([]float64, visible)
			for j := range visible {
				krow := k[(j*kvHeads+kvh)*headDim : (j*kvHeads+kvh)*headDim+headDim]
				dot := 0.0
				for c := range headDim {
					dot += qrow[c] * krow[c]
				}
				scores[j] = dot * scale
			}
			w := Softmax(scores)

			orow := out[(t*qHeads+h)*headDim : (t*qHeads+h)*headDim+headDim]
			for j := range visible {
				vrow := v[(j*kvHeads+kvh)*headDim : (j*kvHeads+kvh)*headDim+headDim]
				for c := range headDim {
					orow[c] += w[j] * vrow[c]
				}
			}
		}
	}
	return out
}
