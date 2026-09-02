// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package sample turns a row of logits into a token id, on the host.
//
// accel puts argmax, categorical sampling and top-k/top-p truncation on the
// device (accel 028) and specifies temperature, the penalties and the
// composition order in accel 039 -- which is drafted and not built. Until it
// is, the whole policy runs here: O(V) over 152k floats is microseconds beside
// a model step (specs/006-sampling.md 006-D1). When 039 lands, the stages move
// down and this package stays as what the device path is checked against, so
// it is written as a reference implementation rather than as a stopgap: where
// a choice is arbitrary, it is made the way accel's kernel makes it, and the
// comment says which kernel.
//
// The composition order is specs/006-sampling.md section 3 and every adjacency
// in it is a decision:
//
//	logits -> bias -> penalties -> /T -> top-k -> top-p -> softmax -> draw
//
// Temperature zero is a distinct greedy branch taken after the penalties, not
// a division by zero: bias and the penalties reorder candidates, so they are
// not argmax-invariant and must run before the branch; temperature and the two
// truncations cannot change which token is largest, so they run after it.
//
// A Sampler is a seeded stream, not a function. Step i consumes exactly one
// draw whether or not the policy is greedy (006-D2), so a caller that changes
// temperature mid-request still reproduces its own completion from the same
// seed. Probs observes the same distribution without moving the stream
// (006-D7).
//
// # What is not here
//
// Stop-string matching. specs/006-sampling.md section 5 matches stop strings on
// the decoded text rather than on token ids, because a stop string need not
// align to a token boundary, so it belongs with the engine's detokenizer and
// its hold-back buffer (006-D4, specs/002-tokenizer.md 002-D8). EOS ids and a
// token budget are the engine's counters for the same reason: this package
// sees one row of logits and knows nothing about a sequence.
package sample

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// TopMaxRounds is how many candidates either truncation stage can keep.
//
// accel 028's bound, restated here because tgo cannot import it: it lives in
// accel's internal kernel package, and accel 039 -- which would export it from
// the tensor layer -- is not built. Both of accel's masks walk the
// distribution one entry per round and stop at 128 rounds, so a policy that
// asks for more is a policy the device cannot reproduce.
const TopMaxRounds = 128

// Policy is one sequence's sampling configuration. It is plain numbers: copying
// it copies the policy and not the stream, which is why the draw lives in
// Sampler (specs/006-sampling.md section 4).
//
// The zero value is greedy: argmax, no bias, no penalties, no truncation.
type Policy struct {
	// Temperature divides the logits. Zero means greedy, exactly -- a separate
	// branch, not a division. Negative or NaN is refused.
	Temperature float32

	// TopK keeps the k largest candidates. Zero means the stage is absent;
	// otherwise 1..TopMaxRounds, and anything above is refused rather than
	// clamped, because accel's kernel clamps silently and a caller who
	// believes they asked for 500 would be running top-128.
	TopK int

	// TopP keeps the smallest set of candidates whose mass reaches p. Zero
	// means the stage is absent -- not "keep one token": accel's kernel
	// computes its threshold as total*P, so P = 0 keeps nothing there
	// (006-D6). Any other value must lie in (0, 1].
	TopP float32

	// RepetitionPenalty is divisive and sign-asymmetric (section 3.1). One and
	// zero both mean no penalty; negative or NaN is refused.
	RepetitionPenalty float32

	// PresencePenalty is subtracted once from any token that appears in the
	// window.
	PresencePenalty float32

	// FrequencyPenalty is subtracted once per occurrence in the window.
	FrequencyPenalty float32

	// PenaltyWindow is how many tokens back the penalties read. Zero means the
	// whole context. The window covers prompt and generated tokens together:
	// penalising only what the model produced lets it repeat the prompt
	// verbatim, which is the failure users report as "it echoes my question".
	PenaltyWindow int

	// LogitBias is the caller's absolute statement about a token, applied
	// before everything else. A negative infinity bans a token; an id outside
	// the vocabulary, a NaN or a positive infinity is refused.
	LogitBias map[int]float32
}

// Sampler holds the seeded draw stream for one sequence.
//
// Not safe for concurrent use: it is per sequence by construction, which is
// also what lets Next reuse one scratch buffer across steps instead of
// allocating a vocabulary of floats per token.
type Sampler struct {
	rng     *rand.Rand
	scratch []float32
}

// goldenGamma is the second half of the PCG state.
//
// math/rand/v2's PCG takes two words and the public surface offers one seed, so
// the second is fixed. A non-zero odd constant rather than zero, so that
// New(0) -- the seed a caller who wants "just reproducible" reaches for -- does
// not start from an all-zero state.
const goldenGamma = 0x9e3779b97f4a7c15

// New returns a Sampler whose draws are determined by seed.
//
// The same seed and the same sequence of Next calls give the same tokens on one
// device, whatever the policies are, including a policy that changes mid-stream
// (006-D2).
func New(seed uint64) *Sampler {
	return &Sampler{rng: rand.New(rand.NewPCG(seed, goldenGamma))}
}

// Next returns the sampled token id for one step.
//
// logits is modified in place: the bias, the penalties and the temperature are
// applied to the caller's slice, which is the readback buffer and is about to
// be overwritten anyway. history is the token window the penalties read, prompt
// and generated tokens together, oldest first.
//
// Exactly one draw is consumed, whatever the policy (006-D2). Consuming one
// only when actually sampling would make reproducibility hold in every test and
// fail the moment a caller changed temperature mid-request.
//
// Next panics on a policy or a history the device could not reproduce; the
// refusals are listed on Policy.
func (s *Sampler) Next(logits []float32, history []int, p Policy) int {
	check(logits, history, p)

	// The draw is taken here, before any branch, so that "one draw per step"
	// is a property of the code's shape rather than something each return has
	// to remember.
	u := s.draw()

	s.scratch = ensure(s.scratch, len(logits))
	d := policyDist(logits, history, p, s.scratch)
	return d.walk(u)
}

// Probs returns the post-policy distribution over the vocabulary, for logprobs.
//
// Truncated tokens carry zero and the rest sum to one. A greedy policy returns
// the distribution it samples from, which is one at the argmax and zero
// everywhere else.
//
// It consumes no draw and does not modify logits (006-D7): computing logprobs
// is an observation, and an observation that perturbed the completion it
// describes would not be describing it.
func (s *Sampler) Probs(logits []float32, history []int, p Policy) []float32 {
	check(logits, history, p)

	// A copy, because the stages write in place and this call must leave the
	// caller's row alone.
	work := make([]float32, len(logits))
	copy(work, logits)

	d := policyDist(work, history, p, make([]float32, len(logits)))
	out := make([]float32, len(logits))
	d.spread(out)
	return out
}

// draw returns one u in [0, 1), as the float32 the device would be given.
//
// float32 because accel binds the draw as a float32 scalar, so a host reference
// that drew in float64 would round to a different value than the device saw.
// The clamp mirrors SampleCategorical's: rounding a float64 just below one to
// float32 can produce exactly one, and a draw of one lands on the total, which
// no partial sum exceeds.
func (s *Sampler) draw() float32 {
	u := float32(s.rng.Float64())
	if u > maxDraw {
		u = maxDraw
	}
	return u
}

// maxDraw is the largest float32 below one, accel's SampleCategorical clamp.
const maxDraw = 0.99999994

// ensure returns buf grown to at least n, reusing it when it already is.
func ensure(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

// check refuses what the device could not reproduce, or what is meaningless.
//
// A panic rather than an error because the surface in specs/006-sampling.md
// section 2 returns a token id. A policy is configuration: it is wrong once,
// before the first token, not intermittently. The engine will want an
// error-returning validator of its own, the way accel 039 exports
// SamplingOptions.Validate.
func check(logits []float32, history []int, p Policy) {
	if len(logits) == 0 {
		panic("sample: empty logits")
	}
	if p.Temperature < 0 || isNaN(p.Temperature) {
		panic(fmt.Sprintf("sample: temperature %v is negative or NaN", p.Temperature))
	}
	if p.TopK < 0 {
		panic(fmt.Sprintf("sample: TopK %d is negative", p.TopK))
	}
	if p.TopK > TopMaxRounds {
		// Refused rather than clamped: accel's TopKMask walks TopMaxRounds
		// rounds and keeps 128, so computing top-500 here would be a reference
		// for something the device cannot do.
		panic(fmt.Sprintf("sample: TopK %d is above TopMaxRounds %d", p.TopK, TopMaxRounds))
	}
	if p.TopP < 0 || p.TopP > 1 || isNaN(p.TopP) {
		panic(fmt.Sprintf("sample: TopP %v is outside [0, 1]; zero disables the stage", p.TopP))
	}
	if p.RepetitionPenalty < 0 || isNaN(p.RepetitionPenalty) {
		panic(fmt.Sprintf("sample: repetition penalty %v is negative or NaN", p.RepetitionPenalty))
	}
	if isNaN(p.PresencePenalty) || isNaN(p.FrequencyPenalty) {
		panic("sample: presence or frequency penalty is NaN")
	}
	if p.PenaltyWindow < 0 {
		panic(fmt.Sprintf("sample: penalty window %d is negative", p.PenaltyWindow))
	}
	for id, b := range p.LogitBias {
		if id < 0 || id >= len(logits) {
			panic(fmt.Sprintf("sample: logit bias id %d is outside the vocabulary of %d", id, len(logits)))
		}
		// Negative infinity is how a caller bans a token and is allowed.
		// Positive infinity is not: it makes every softmax weight NaN.
		if isNaN(b) || math.IsInf(float64(b), 1) {
			panic(fmt.Sprintf("sample: logit bias for id %d is %v", id, b))
		}
	}
	for i, id := range history {
		if id < 0 || id >= len(logits) {
			panic(fmt.Sprintf("sample: history[%d] = %d is outside the vocabulary of %d", i, id, len(logits)))
		}
	}
}

func isNaN(x float32) bool { return x != x }
