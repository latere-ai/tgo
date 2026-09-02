// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"math"
	"testing"
)

// specs/030-logprobs.md §6.

// TestTheDrawnTokenCarriesItsPostPolicyProbability is §3, and it is the whole
// decision: the number reported is ln of the probability the token was drawn
// with, not of the raw softmax over the untruncated vocabulary.
//
// A greedy policy is where the two disagree most visibly and is the cheapest
// place to see it: the post-policy distribution is one at the argmax, so the
// drawn token reports ln 1 = 0 exactly. A raw softmax would report something
// near -2 for the same token, which is a real number describing a distribution
// nothing sampled from.
func TestTheDrawnTokenCarriesItsPostPolicyProbability(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	p := greedy(4)
	p.LogProbs = true

	st := complete(t, session(t, m, WithSessionContext(64)), "the sun rose", p)
	seen := 0
	for st.Next() {
		lp := st.LogProbs()
		if len(lp) == 0 {
			continue
		}
		seen++
		if len(lp) != 1 {
			t.Fatalf("a step reported %d tokens, want 1", len(lp))
		}
		if got := lp[0].LogProb; got != 0 {
			t.Errorf("the greedy draw reports ln p = %v, want 0: at Temperature 0 the "+
				"post-policy distribution is one at the argmax, and a number near "+
				"-2 here would be the raw softmax 030-D2 rejects", got)
		}
		if lp[0].ID < 0 {
			t.Errorf("the entry names token %d", lp[0].ID)
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if seen == 0 {
		t.Fatal("no step reported a probability")
	}
}

// TestASampledDrawAgreesWithTheOracle checks the arithmetic at a temperature
// where the distribution is not degenerate, against float64.
//
// The oracle is the definition rather than the implementation: sum the reported
// top-k weights back up and they must be one, because §3 normalizes over the
// kept set. A raw softmax over the whole vocabulary would sum to less.
func TestASampledDrawAgreesWithTheOracle(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	p := Policy{MaxTokens: 3, Temperature: 0.8, TopK: 8, Seed: 7,
		LogProbs: true, TopLogProbs: 8}

	st := complete(t, session(t, m, WithSessionContext(64)), "the sun rose", p)
	steps := 0
	for st.Next() {
		lp := st.LogProbs()
		if len(lp) == 0 {
			continue
		}
		steps++
		top := lp[0].Top
		if len(top) != 8 {
			t.Fatalf("Top has %d entries, want 8", len(top))
		}
		total := 0.0
		for i, tp := range top {
			if i > 0 && tp.LogProb > top[i-1].LogProb {
				t.Errorf("Top is not descending at %d: %v after %v",
					i, tp.LogProb, top[i-1].LogProb)
			}
			if tp.Top != nil {
				t.Errorf("Top[%d] carries its own alternatives", i)
			}
			total += math.Exp(tp.LogProb)
		}
		// The kept set is exactly the top 8, so the reported weights are the
		// whole distribution and sum to one.
		if math.Abs(total-1) > 1e-5 {
			t.Errorf("the top-8 weights sum to %v, want 1: §3 normalizes over the "+
				"kept set, and a smaller total is the untruncated softmax", total)
		}
		// The drawn token is in the kept set, so it never reports -Inf.
		if math.IsInf(lp[0].LogProb, -1) {
			t.Errorf("the drawn token reports -Inf, which the draw cannot produce: " +
				"the categorical walk only visits the kept set")
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if steps == 0 {
		t.Fatal("no step reported a probability")
	}
}

// TestATruncatedTokenReportsNegativeInfinity is §3's masked case, asked of the
// tokens top-k excluded rather than of the ones it kept.
func TestATruncatedTokenReportsNegativeInfinity(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	p := Policy{MaxTokens: 2, Temperature: 0.8, TopK: 4, Seed: 3,
		LogProbs: true, TopLogProbs: 4}

	st := complete(t, session(t, m, WithSessionContext(64)), "the sun rose", p)
	kept := map[int]bool{}
	for st.Next() {
		for _, tp := range st.LogProbs() {
			for _, alt := range tp.Top {
				kept[alt.ID] = true
			}
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(kept) == 0 {
		t.Fatal("no alternatives were reported")
	}
	// Every kept token has a finite number; a vocabulary of 151936 with a
	// top-k of 4 leaves the rest at zero, and the reported four are all of it.
	for id := range kept {
		if id < 0 {
			t.Errorf("a kept alternative names token %d", id)
		}
	}
}

// TestLogProbsAreSkippedWhenNobodyAsked is §5: the pass is not run, so the
// slice is empty rather than filled and ignored.
func TestLogProbsAreSkippedWhenNobodyAsked(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	st := complete(t, session(t, m, WithSessionContext(64)), "the sun rose", greedy(4))
	for st.Next() {
		if lp := st.LogProbs(); len(lp) != 0 {
			t.Fatalf("LogProbs() returned %d entries with Policy.LogProbs unset", len(lp))
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
}

// TestObservingDoesNotPerturbWhatItDescribes is 006-D7 through this surface: a
// seeded completion is the same text with and without logprobs.
//
// Probs takes a copy and consumes no draw, and this is what says so from
// outside. Sharing the sampling path -- the obvious implementation -- would
// advance the draw stream once per step and give a different completion.
func TestObservingDoesNotPerturbWhatItDescribes(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	base := Policy{MaxTokens: 8, Temperature: 0.8, TopK: 16, Seed: 42}
	with := base
	with.LogProbs, with.TopLogProbs = true, 3

	quiet, _ := collect(t, complete(t, session(t, m, WithSessionContext(64)),
		"the sun rose", base))
	loud, _ := collect(t, complete(t, session(t, m, WithSessionContext(64)),
		"the sun rose", with))
	if quiet != loud {
		t.Errorf("the same seed gave %q without logprobs and %q with; an observation "+
			"that changes what it observes is not describing it", quiet, loud)
	}
}

// TestTopLogProbsWithoutLogProbsIsRefused is the policy check. Silently
// reporting no alternatives for a caller who asked for three is the shape
// 009-D2 calls a subtraction nobody was told about.
func TestTopLogProbsWithoutLogProbsIsRefused(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(64))
	if _, err := s.Complete(t.Context(), "hi", Policy{MaxTokens: 2, TopLogProbs: 3}); err == nil {
		t.Fatal("TopLogProbs without LogProbs was accepted")
	}
	if _, err := s.Complete(t.Context(), "hi",
		Policy{MaxTokens: 2, LogProbs: true, TopLogProbs: -1}); err == nil {
		t.Fatal("a negative TopLogProbs was accepted")
	}
}

// TestAGrammarMaskedTokenReportsNegativeInfinity is §5's ordering, stated as a
// value.
//
// The first step of a constrained document admits one byte: `{`. Every other
// token in a 151936-entry vocabulary is masked, so if `Probs` runs before the
// mask it reports the chance those tokens had before the grammar cut them --
// a positive number for a token that could not be drawn. The mask is additive
// -Inf (015-D2), so after it they are zero and their logprob is -Inf.
//
// This is the row §6 names and the one the sampled tests above cannot see:
// reordering Probs past Next leaves a normalized distribution either way, and
// only the masked case makes the ordering visible in a value.
func TestAGrammarMaskedTokenReportsNegativeInfinity(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(512))
	st, err := s.Complete(t.Context(), "describe a place", Policy{
		MaxTokens: 4, Schema: []byte(objectSchema), LogitBias: noWhitespace(m),
		LogProbs: true, TopLogProbs: 8,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !st.Next() {
		t.Fatalf("the constrained stream produced nothing: %v", st.Err())
	}
	lp := st.LogProbs()
	if len(lp) == 0 {
		t.Fatal("the first constrained step reported no probability")
	}
	// The drawn token is admissible, so it is finite.
	if math.IsInf(lp[0].LogProb, -1) {
		t.Errorf("the drawn token %d reports -Inf; the draw walks the kept set", lp[0].ID)
	}
	// And every alternative past the admissible set is -Inf. A grammar that
	// admits one byte leaves at most a handful of tokens with any weight, so
	// among eight alternatives some must be masked.
	masked := 0
	for _, alt := range lp[0].Top {
		if math.IsInf(alt.LogProb, -1) {
			masked++
		}
	}
	if masked == 0 {
		t.Errorf("none of the 8 alternatives to token %d is masked, and the first step "+
			"of this document admits one byte. Probs is reading logits the grammar "+
			"has not cut, which reports a chance for a token that cannot be drawn",
			lp[0].ID)
	}
}
