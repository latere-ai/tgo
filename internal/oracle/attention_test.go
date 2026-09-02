// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package oracle

import (
	"math"
	"math/rand/v2"
	"testing"
)

func randf(n int, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.NormFloat64()
	}
	return out
}

func TestAttentionWithOneKeyReturnsTheValue(t *testing.T) {
	// One visible key means softmax over one score, which is exactly 1
	// whatever the score is, so the output is the value row bit for bit. This
	// is the case that catches a transposed cache index or a dropped scale
	// path, because nothing else can hide behind the weights.
	q := []float64{0.5, -2, 100}
	k := []float64{1, 1, 1}
	v := []float64{7, 8, 9}
	got := Attention(q, k, v, 1, 1, 1, 3, 1, 0.125, 0)
	for i := range v {
		if got[i] != v[i] {
			t.Fatalf("out[%d] = %.17g, want exactly %.17g", i, got[i], v[i])
		}
	}
}

func TestAttentionGroupsQueryHeadsByFloorDivision(t *testing.T) {
	// specs/004-model-graph.md §2.4: H_q/H_kv adjacent query heads share one
	// cache entry. Four query heads over two kv heads means heads 0 and 1 read
	// kv head 0 while heads 2 and 3 read kv head 1. The modulo grouping that
	// is easy to write by mistake would give 0,1,0,1 instead, and this shape
	// is the smallest one that tells the two apart.
	const qSeq, qHeads, kvHeads, headDim, kvLen = 1, 4, 2, 2, 2
	q := randf(qSeq*qHeads*headDim, 21)
	k := randf(kvLen*kvHeads*headDim, 22)

	// Every value row of kv head 0 is 10 and of kv head 1 is 20, so the output
	// names the kv head that was read regardless of the attention weights.
	v := make([]float64, kvLen*kvHeads*headDim)
	for j := range kvLen {
		for h := range kvHeads {
			for c := range headDim {
				v[(j*kvHeads+h)*headDim+c] = float64(10 * (h + 1))
			}
		}
	}

	got := Attention(q, k, v, qSeq, qHeads, kvHeads, headDim, kvLen, 0.7, 1)
	want := []float64{10, 10, 10, 10, 20, 20, 20, 20}
	checkSlice(t, "GQA grouping", got, want, 4*tolULP)
}

func TestAttentionGroupedHeadsAttendToTheSameKeys(t *testing.T) {
	// The grouping claim is about K as well as V: the queries in one group
	// read the same keys, so two query heads holding the same vector produce
	// the same output, and a head in the other group does not. The
	// constant-value test above cannot see the K index, because its weights
	// never reach the output.
	const qSeq, qHeads, kvHeads, headDim, kvLen = 1, 4, 2, 3, 3
	k := randf(kvLen*kvHeads*headDim, 91)
	v := randf(kvLen*kvHeads*headDim, 92)

	// Every query head holds the same vector, so any difference in the output
	// comes from which cache entry was read.
	one := randf(headDim, 93)
	q := make([]float64, qSeq*qHeads*headDim)
	for h := range qHeads {
		copy(q[h*headDim:], one)
	}

	got := Attention(q, k, v, qSeq, qHeads, kvHeads, headDim, kvLen, 0.6, kvLen-1)
	head := func(h int) []float64 { return got[h*headDim : (h+1)*headDim] }
	for c := range headDim {
		// Same query, same keys, same values, same order: exactly equal.
		if head(0)[c] != head(1)[c] {
			t.Errorf("heads 0 and 1 share kv head 0 but differ at %d: %.17g vs %.17g", c, head(0)[c], head(1)[c])
		}
		if head(2)[c] != head(3)[c] {
			t.Errorf("heads 2 and 3 share kv head 1 but differ at %d: %.17g vs %.17g", c, head(2)[c], head(3)[c])
		}
	}
	same := true
	for c := range headDim {
		if head(0)[c] != head(2)[c] {
			same = false
		}
	}
	if same {
		t.Error("the two groups produced the same output; the kv head index is not being used")
	}
}

func TestAttentionMasksTheFuture(t *testing.T) {
	// Query t may read cache rows 0..causalBase+t. Rewriting a later row must
	// leave every earlier query's output bit-identical; if it moves, the mask
	// is not applied.
	const qSeq, kvLen, headDim = 3, 3, 2
	q := randf(qSeq*headDim, 31)
	k := randf(kvLen*headDim, 32)
	v := randf(kvLen*headDim, 33)
	before := Attention(q, k, v, qSeq, 1, 1, headDim, kvLen, 0.5, 0)

	k2 := append([]float64(nil), k...)
	v2 := append([]float64(nil), v...)
	for c := range headDim {
		k2[2*headDim+c] += 5
		v2[2*headDim+c] += 5
	}
	after := Attention(q, k2, v2, qSeq, 1, 1, headDim, kvLen, 0.5, 0)

	for i := range 2 * headDim {
		if before[i] != after[i] {
			t.Errorf("query %d saw the future: %.17g became %.17g", i/headDim, before[i], after[i])
		}
	}
	changed := false
	for i := 2 * headDim; i < 3*headDim; i++ {
		if before[i] != after[i] {
			changed = true
		}
	}
	if !changed {
		t.Error("the last query did not see the row it is allowed to see")
	}
}

func TestAttentionFirstQueryReadsOnlyItsOwnKey(t *testing.T) {
	// The strongest form of the mask: at causalBase 0 the first query has one
	// visible key, so its output is the first value row exactly, whatever else
	// is in the cache.
	const kvLen, headDim = 4, 3
	q := randf(kvLen*headDim, 41)
	k := randf(kvLen*headDim, 42)
	v := randf(kvLen*headDim, 43)
	got := Attention(q, k, v, kvLen, 1, 1, headDim, kvLen, 0.3, 0)
	for c := range headDim {
		if got[c] != v[c] {
			t.Errorf("out[%d] = %.17g, want exactly %.17g", c, got[c], v[c])
		}
	}
}

func TestDecodeEqualsPrefillLastRow(t *testing.T) {
	// specs/004-model-graph.md §3.1: decode is the same computation at T=1
	// with causalBase at the current position. It is the same arithmetic in
	// the same order, so the agreement is exact.
	const qSeq, qHeads, kvHeads, headDim = 3, 4, 2, 5
	q := randf(qSeq*qHeads*headDim, 51)
	k := randf(qSeq*kvHeads*headDim, 52)
	v := randf(qSeq*kvHeads*headDim, 53)
	prefill := Attention(q, k, v, qSeq, qHeads, kvHeads, headDim, qSeq, 0.44, 0)

	last := q[(qSeq-1)*qHeads*headDim:]
	decode := Attention(last, k, v, 1, qHeads, kvHeads, headDim, qSeq, 0.44, qSeq-1)
	for i := range decode {
		if decode[i] != prefill[(qSeq-1)*qHeads*headDim+i] {
			t.Fatalf("channel %d: decode %.17g, prefill %.17g", i, decode[i], prefill[(qSeq-1)*qHeads*headDim+i])
		}
	}
}

func TestAttentionIgnoresTheCacheTail(t *testing.T) {
	// The cache is a State of capacity C bound whole, with a separate length
	// (specs/004-model-graph.md §3 rows 14-16), so rows past kvLen hold
	// whatever was there before and must not be read.
	const headDim, live, capacity = 2, 2, 6
	q := randf(headDim, 61)
	k := randf(capacity*headDim, 62)
	v := randf(capacity*headDim, 63)
	short := Attention(q, k[:live*headDim], v[:live*headDim], 1, 1, 1, headDim, live, 0.5, live-1)

	dirty := append([]float64(nil), k...)
	for i := live * headDim; i < len(dirty); i++ {
		dirty[i] = 1e9
	}
	full := Attention(q, dirty, v, 1, 1, 1, headDim, live, 0.5, live-1)
	for i := range short {
		if short[i] != full[i] {
			t.Fatalf("channel %d: %.17g with a trimmed cache, %.17g with a dirty tail", i, short[i], full[i])
		}
	}
}

func TestAttentionScaleZeroAveragesTheValues(t *testing.T) {
	// scale 0 flattens every score, so the output is the mean of the visible
	// value rows. It checks that the scale multiplies the logits rather than
	// the values.
	const kvLen, headDim = 4, 2
	q := randf(headDim, 71)
	v := []float64{1, 10, 2, 20, 3, 30, 4, 40}
	k := randf(kvLen*headDim, 72)
	got := Attention(q, k, v, 1, 1, 1, headDim, kvLen, 0, kvLen-1)
	want := []float64{2.5, 25}
	// Four normalized weights and four accumulated products: about
	// 8*eps_64 from the divisions and the sum.
	checkSlice(t, "mean of values", got, want, 8*tolULP)
}

func TestAttentionMatchesAHandComputedSoftmax(t *testing.T) {
	// One head, two keys, a scale that makes the arithmetic checkable: the
	// scores are q.k0*s and q.k1*s and the output is the softmax-weighted mix
	// of the two value rows, computed here from the definition rather than
	// from the implementation.
	q := []float64{1, 0}
	k := []float64{1, 0, 0, 1}
	v := []float64{1, 2, 3, 4}
	const scale = 2.0
	got := Attention(q, k, v, 1, 1, 1, 2, 2, scale, 1)

	e0, e1 := math.Exp(2.0), math.Exp(0.0)
	w0, w1 := e0/(e0+e1), e1/(e0+e1)
	want := []float64{w0*1 + w1*3, w0*2 + w1*4}
	checkSlice(t, "hand-computed attention", got, want, 4*tolULP)
}

func TestAttentionRefusals(t *testing.T) {
	q := randf(4, 81)
	k := randf(4, 82)
	v := randf(4, 83)
	mustPanic(t, "Attention has a non-positive dimension", func() { Attention(q, k, v, 1, 0, 1, 4, 1, 1, 0) })
	mustPanic(t, "Attention has a non-positive dimension", func() { Attention(q, k, v, 1, 1, 0, 4, 1, 1, 0) })
	mustPanic(t, "Attention has a non-positive dimension", func() { Attention(q, k, v, 1, 1, 1, 0, 1, 1, 0) })
	mustPanic(t, "Attention has a non-positive dimension", func() { Attention(q, k, v, 1, 1, 1, 4, 0, 1, 0) })
	mustPanic(t, "Attention has a non-positive dimension", func() { Attention(q, k, v, -1, 1, 1, 4, 1, 1, 0) })
	mustPanic(t, "Attention query heads are not a multiple", func() { Attention(q, k, v, 1, 2, 3, 2, 1, 1, 0) })
	// q must match its shape exactly, from both sides. k and v may be longer
	// than kvLen*kvHeads*headDim on purpose — they are a whole cache with a
	// separate live length — so only their short side is a refusal.
	mustPanic(t, "Attention q length", func() { Attention(q, k, v, 2, 1, 1, 4, 1, 1, 0) })
	mustPanic(t, "Attention q length", func() { Attention(q, k, v, 1, 1, 1, 3, 1, 1, 0) })
	mustPanic(t, "Attention k is shorter", func() { Attention(q, k[:2], v, 1, 1, 1, 4, 1, 1, 0) })
	mustPanic(t, "Attention v is shorter", func() { Attention(q, k, v[:2], 1, 1, 1, 4, 1, 1, 0) })
	mustPanic(t, "Attention causalBase is negative", func() { Attention(q, k, v, 1, 1, 1, 4, 1, 1, -1) })
	// The step's own keys are scattered into the cache before attention runs,
	// so a query past kvLen is an off-by-one in the caller rather than a
	// shorter mask.
	mustPanic(t, "Attention causalBase+qSeq exceeds kvLen", func() { Attention(q, k, v, 1, 1, 1, 4, 1, 1, 1) })
}
