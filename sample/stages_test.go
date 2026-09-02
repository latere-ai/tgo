// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package sample

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/latere-ai/tgo/internal/oracle"
)

// softmaxTol is the tolerance on a probability checked against the float64
// oracle.
//
// The package computes exp and the normalizing division in float32, so each
// probability carries the float32 rounding of one exp plus one divide plus the
// running sum over V terms: about sqrt(V)*eps32 relative for the sum
// (specs/010-conformance.md section 5.1, with eps32 = 2^-24 = 6e-8), which for
// the vocabularies used here is under 1e-6. Nothing in this file has a
// tolerance that had to be raised.
const softmaxTol = 1e-6

func f64(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}
	return out
}

// closeTo reports whether got is within tol of want.
func closeTo(got, want float64, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func TestSoftmaxMatchesTheOracle(t *testing.T) {
	logits := []float32{2.5, -1.25, 0, 7, -3.5, 1}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1})
	want := oracle.Softmax(f64(logits))
	for i := range want {
		if !closeTo(float64(got[i]), want[i], softmaxTol) {
			t.Errorf("probs[%d] = %v, oracle says %v", i, got[i], want[i])
		}
	}
}

func TestTemperatureDividesTheLogitsBeforeTheSoftmax(t *testing.T) {
	logits := []float32{2, 1, 0, -1}
	const temp = 0.25
	got := New(1).Probs(logits, nil, Policy{Temperature: temp})

	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = float64(v / temp)
	}
	want := oracle.Softmax(scaled)
	for i := range want {
		if !closeTo(float64(got[i]), want[i], softmaxTol) {
			t.Errorf("probs[%d] = %v, oracle says %v", i, got[i], want[i])
		}
	}
}

// TestPenaltyArithmeticIsExact checks section 3.1's formula term by term. The
// stage is a handful of float32 operations, so the reference is the same
// operations written out and the comparison is exact rather than toleranced.
func TestPenaltyArithmeticIsExact(t *testing.T) {
	const r, alpha, beta = 2, 0.5, 0.25
	logits := []float32{3, -3, 0, 5}
	history := []int{0, 0, 1} // token 0 twice, token 1 once, token 2 and 3 never

	got := make([]float32, len(logits))
	copy(got, logits)
	penalize(got, history, Policy{RepetitionPenalty: r, PresencePenalty: alpha, FrequencyPenalty: beta})

	want := []float32{
		logits[0]/r - alpha - beta*2, // positive: divided, seen twice
		logits[1]*r - alpha - beta*1, // negative: multiplied, seen once
		logits[2],                    // absent from the window
		logits[3],                    // absent from the window
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("logit[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestRepetitionPenaltyIsSignAsymmetric is the classic bug: dividing a negative
// logit by r > 1 moves it toward zero and makes a penalised token more likely.
func TestRepetitionPenaltyIsSignAsymmetric(t *testing.T) {
	const r = 2
	logits := []float32{-4, -4}
	penalize(logits[:1], []int{0}, Policy{RepetitionPenalty: r})

	if logits[0] >= logits[1] {
		t.Fatalf("penalised negative logit %v is not below the unpenalised %v", logits[0], logits[1])
	}
	if want := float32(-8); logits[0] != want {
		t.Errorf("penalised logit = %v, want %v (multiplied, not divided)", logits[0], want)
	}
}

// TestRepetitionPenaltyPutsZeroInTheMultiplyBranch pins the boundary. A
// negative logit passes under either comparison; only zero tells > 0 from >= 0
// apart, and section 3.1 puts it in the multiply branch.
func TestRepetitionPenaltyPutsZeroInTheMultiplyBranch(t *testing.T) {
	logits := []float32{0}
	penalize(logits, []int{0}, Policy{RepetitionPenalty: 4})
	if logits[0] != 0 {
		t.Errorf("zero logit became %v; the multiply branch leaves it at zero", logits[0])
	}
}

// TestRepetitionPenaltyAppliesOncePerToken: r is not raised to the occurrence
// count, however often the token appears.
func TestRepetitionPenaltyAppliesOncePerToken(t *testing.T) {
	const r = 2
	once := []float32{8}
	many := []float32{8}
	penalize(once, []int{0}, Policy{RepetitionPenalty: r})
	penalize(many, []int{0, 0, 0, 0}, Policy{RepetitionPenalty: r})
	if once[0] != many[0] {
		t.Errorf("four occurrences gave %v, one gave %v; the penalty is per token", many[0], once[0])
	}
}

// TestPenaltiesReadPromptTokens is the "it echoes my question" failure: the
// window is prompt and generated tokens together.
func TestPenaltiesReadPromptTokens(t *testing.T) {
	// history is all prompt: nothing has been generated yet.
	logits := []float32{5, 4}
	prompt := []int{0}
	got := New(7).Next(logits, prompt, Policy{RepetitionPenalty: 4})
	if got != 1 {
		t.Fatalf("token %d; the prompt token 0 was not penalised", got)
	}
}

func TestPenaltyWindowLimitsHowFarBack(t *testing.T) {
	logits := []float32{5, 4}
	history := []int{0, 0, 0, 0, 1}
	p := Policy{FrequencyPenalty: 1}

	// A window of one sees only the trailing 1, so token 0 keeps its logit.
	windowed := p
	windowed.PenaltyWindow = 1
	if got := New(1).Next([]float32{logits[0], logits[1]}, history, windowed); got != 0 {
		t.Fatalf("windowed token %d, want 0: the four 0s are outside a window of one", got)
	}
	// The whole context counts them, and four occurrences outweigh the gap.
	if got := New(1).Next([]float32{logits[0], logits[1]}, history, p); got != 1 {
		t.Fatalf("unwindowed token %d, want 1: the four 0s are in the window", got)
	}
}

// The order tests. Each takes one adjacent pair of section 3 and finds a case
// where the swapped composition returns a different answer.

// TestBiasBeforePenalties: the bias is the caller's absolute statement, so the
// penalty reads the biased logit and not the other way round.
func TestBiasBeforePenalties(t *testing.T) {
	// Token 0 at -1 with a bias of +2 and a repetition penalty of 2.
	//   as specified: (-1 + 2) = 1 > 0, divided  -> 0.5
	//   swapped:      (-1 * 2) = -2, then biased -> 0.0
	// Token 1 sits between the two, so the greedy answer names the order.
	logits := []float32{-1, 0.25}
	p := Policy{RepetitionPenalty: 2, LogitBias: map[int]float32{0: 2}}
	if got := New(3).Next(logits, []int{0}, p); got != 0 {
		t.Fatalf("token %d, want 0: the penalty read an unbiased logit", got)
	}
	if logits[0] != 0.5 {
		t.Errorf("penalised logit = %v, want 0.5", logits[0])
	}
}

// TestPenaltiesBeforeTemperature: a penalty is a logit adjustment with a fixed
// meaning. Subtracting alpha after dividing by T is subtracting alpha*T before
// it, so the same policy would behave differently at every temperature.
func TestPenaltiesBeforeTemperature(t *testing.T) {
	logits := []float32{2, 0}
	p := Policy{Temperature: 0.5, PresencePenalty: 1}
	got := New(1).Probs(logits, []int{0}, p)

	// as specified: (2 - 1)/0.5 = 2 against 0, so the ratio is exp(2).
	// swapped:      2/0.5 - 1 = 3 against 0, so it would be exp(3).
	ratio := float64(got[0] / got[1])
	if !closeTo(ratio, math.Exp(2), 1e-4) {
		t.Fatalf("p0/p1 = %v, want exp(2) = %v (exp(3) = %v is the swapped order)",
			ratio, math.Exp(2), math.Exp(3))
	}
}

// TestTemperatureBeforeTruncation: top-p is a mass threshold and temperature is
// what changes the mass, so truncating first makes p mean a different set at
// every temperature.
func TestTemperatureBeforeTruncation(t *testing.T) {
	logits := []float32{2, 1, 0}
	p := Policy{Temperature: 2, TopP: 0.6}
	got := New(1).Probs(logits, nil, p)

	// as specified: /2 first gives 0.507, 0.307, 0.186; 0.507 < 0.6, so the
	// nucleus takes two tokens.
	// swapped: the nucleus over the untempered 0.665, 0.245, 0.090 crosses on
	// the first token and takes one.
	if got[1] == 0 {
		t.Fatalf("token 1 was truncated: the nucleus was taken before the division")
	}
	if got[2] != 0 {
		t.Errorf("token 2 survived a nucleus of two")
	}
}

// TestTopKBeforeTopP: k is a hard cap on the candidate count and p trims within
// it, so the nucleus mass is a fraction of the top-k set rather than of the
// whole vocabulary.
func TestTopKBeforeTopP(t *testing.T) {
	// Probabilities 0.5, 0.3, 0.2 at T = 1.
	logits := []float32{float32(math.Log(0.5)), float32(math.Log(0.3)), float32(math.Log(0.2))}
	p := Policy{Temperature: 1, TopK: 2, TopP: 0.6}
	got := New(1).Probs(logits, nil, p)

	// as specified: top-2 has mass 0.8, so the target is 0.48 and the first
	// token alone crosses it.
	// swapped: the nucleus over the full vocabulary targets 0.6, which needs
	// two tokens, and top-2 then keeps both.
	if got[0] != 1 {
		t.Fatalf("probs = %v; the nucleus was taken over the full vocabulary", got)
	}
}

// TestTopPKeepsTheCrossingToken is 006-D6: the set is the smallest one reaching
// p, so the token that crosses the threshold is kept rather than dropped.
func TestTopPKeepsTheCrossingToken(t *testing.T) {
	logits := []float32{float32(math.Log(0.5)), float32(math.Log(0.3)), float32(math.Log(0.2))}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopP: 0.6})
	if got[1] == 0 {
		t.Fatalf("probs = %v; token 1 crosses 0.6 and must be kept", got)
	}
	if got[2] != 0 {
		t.Fatalf("probs = %v; token 2 is past the threshold", got)
	}
}

// TestTopPTiesBreakByID is section 3.2: a tie at the boundary is resolved by
// token id ascending, so the answer does not depend on the sort's stability.
func TestTopPTiesBreakByID(t *testing.T) {
	logits := []float32{1, 0, 0}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopP: 0.7})
	if got[1] == 0 || got[2] != 0 {
		t.Fatalf("probs = %v; the tie between 1 and 2 must keep the lower id", got)
	}
}

// TestTopKTiesBreakByID is the same rule in the other stage: accel's TopKMask
// compares lexicographically on (value, index), so top-2 of three equal logits
// is exactly ids 0 and 1.
func TestTopKTiesBreakByID(t *testing.T) {
	logits := []float32{0, 0, 0}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopK: 2})
	if got[0] == 0 || got[1] == 0 || got[2] != 0 {
		t.Fatalf("probs = %v; top-2 of an all-tie row is ids 0 and 1", got)
	}
}

func TestTopNOrdersByValueThenID(t *testing.T) {
	logits := []float32{1, 3, 3, 2}
	got := topN(logits, 3)
	want := []int{1, 2, 3}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("topN = %v, want %v", got, want)
		}
	}
}

// TestNucleusStopsAtTopMaxRounds: accel's TopPMask walks 128 rounds and stops
// whether or not it reached its mass, so a wider nucleus is capped rather than
// computed.
func TestNucleusStopsAtTopMaxRounds(t *testing.T) {
	logits := make([]float32, 200) // uniform: every token carries 1/200
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopP: 0.99})

	kept := 0
	for _, v := range got {
		if v != 0 {
			kept++
		}
	}
	if kept != TopMaxRounds {
		t.Fatalf("kept %d candidates, want the %d the kernel can walk", kept, TopMaxRounds)
	}
}

func TestTopKAboveTheVocabularyKeepsEverything(t *testing.T) {
	logits := []float32{3, 2, 1, 0, -1}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopK: 10})
	want := oracle.Softmax(f64(logits))
	for i := range want {
		if !closeTo(float64(got[i]), want[i], softmaxTol) {
			t.Errorf("probs[%d] = %v, oracle says %v", i, got[i], want[i])
		}
	}
}

// TestTopPOfOneKeepsEverything: the running float32 sum need not reach
// total*1, so the walk must fall off the end of the candidate list rather than
// leave a token behind.
func TestTopPOfOneKeepsEverything(t *testing.T) {
	logits := []float32{3, 2, 1, 0, -1}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopP: 1})
	want := oracle.Softmax(f64(logits))
	for i := range want {
		if !closeTo(float64(got[i]), want[i], softmaxTol) {
			t.Errorf("probs[%d] = %v, oracle says %v", i, got[i], want[i])
		}
	}
}

// TestTopPOfZeroIsTheStageBeingAbsent is 006-D6 against section 6's table row:
// zero disables the stage. It cannot mean "keep one token", because accel's
// kernel would compute a threshold of total*0 and keep nothing.
func TestTopPOfZeroIsTheStageBeingAbsent(t *testing.T) {
	logits := []float32{3, 2, 1, 0, -1}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopP: 0})
	want := oracle.Softmax(f64(logits))
	for i := range want {
		if !closeTo(float64(got[i]), want[i], softmaxTol) {
			t.Errorf("probs[%d] = %v, oracle says %v", i, got[i], want[i])
		}
	}
}

func TestTopKOfZeroIsTheStageBeingAbsent(t *testing.T) {
	logits := []float32{3, 2, 1, 0, -1}
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, TopK: 0})
	if got[4] == 0 {
		t.Fatalf("probs = %v; TopK zero removes the stage", got)
	}
}

func TestProbsSumToOneOverTheKeptSet(t *testing.T) {
	logits := []float32{4, 3, 2, 1, 0}
	for _, p := range []Policy{
		{Temperature: 1},
		{Temperature: 0.7, TopK: 2},
		{Temperature: 1, TopP: 0.9},
		{Temperature: 1, TopK: 3, TopP: 0.8},
		{}, // greedy
	} {
		sum := 0.0
		for _, v := range New(1).Probs(logits, nil, p) {
			sum += float64(v)
		}
		if !closeTo(sum, 1, softmaxTol) {
			t.Errorf("policy %+v: probabilities sum to %v", p, sum)
		}
	}
}

func TestGreedyProbsAreOneHot(t *testing.T) {
	logits := []float32{1, 9, 3}
	got := New(1).Probs(logits, nil, Policy{})
	want := []float32{0, 1, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("greedy probs = %v, want %v", got, want)
		}
	}
}

func TestArgmaxTiesGoToTheLowestID(t *testing.T) {
	if got := argmax([]float32{2, 7, 7, 1}); got != 1 {
		t.Errorf("argmax = %d, want 1", got)
	}
}

// TestLogitBiasBansAToken: negative infinity is the caller's way to remove a
// token, and it must survive the softmax as a zero rather than as a NaN.
func TestLogitBiasBansAToken(t *testing.T) {
	logits := []float32{5, 4}
	inf := float32(math.Inf(-1))
	got := New(1).Probs(logits, nil, Policy{Temperature: 1, LogitBias: map[int]float32{0: inf}})
	if got[0] != 0 || got[1] != 1 {
		t.Fatalf("probs = %v, want the banned token at zero", got)
	}
}

// TestTopKSelectsWhereAccelSelects pins the one case where selecting on the
// logits and selecting on the softmax weights disagree.
//
// exp is monotonic, so the two rules agree everywhere it is injective in f32.
// It is not injective at the tail: two logits that differ as f32 can round to
// one f32 weight, and there the rules diverge. On logits the larger one wins;
// accel's TopKMask runs after the softmax, sees two equal values, and breaks
// the tie on the index, keeping the smaller id.
//
// Put the larger logit at the higher id and the boundary between them, and the
// two rules pick different tokens. A k = 128 boundary over a 152k vocabulary
// sits deep in this range, which is why the case is worth a test rather than a
// note.
func TestTopKSelectsWhereAccelSelects(t *testing.T) {
	// Find the pair by search rather than by asserting a constant: which
	// logits collapse depends on where the denormals fall, and a hard-coded
	// pair would be a test that passes for the wrong reason if that moves.
	hi, lo, ok := collapsingPair()
	if !ok {
		t.Skip("no two f32 logits on this platform share one f32 weight")
	}
	if hi <= lo {
		t.Fatalf("the pair is not ordered: %v, %v", hi, lo)
	}
	if expf(hi) != expf(lo) {
		t.Fatalf("the pair does not collapse: exp(%v)=%v, exp(%v)=%v",
			hi, expf(hi), lo, expf(lo))
	}
	if expf(hi) == 0 {
		t.Fatalf("the shared weight is zero, so the choice cannot be observed")
	}

	// ids 0 and 1 take the first two slots; the third is contested between id 2
	// (the smaller logit) and id 3 (the larger). Weights are exp(l - 0).
	logits := []float32{0, -1, lo, hi}
	d := policyDist(logits, nil, Policy{Temperature: 1, TopK: 3},
		make([]float32, len(logits)))

	got := append([]int(nil), d.ids...)
	slices.Sort(got)
	want := []int{0, 1, 2}
	if !slices.Equal(got, want) {
		t.Errorf("top-3 over %v = %v, want %v\n"+
			"id 3 has the larger logit and id 2 the smaller, and the two share "+
			"one f32 weight; accel breaks that tie on the index, so the set "+
			"keeps id 2", logits, got, want)
	}
}

// collapsingPair returns two distinct f32 logits whose f32 weights are equal
// and nonzero, searching the denormal range where exp stops being injective.
func collapsingPair() (hi, lo float32, ok bool) {
	// Below about -87 an f32 exp result is denormal, and denormals are spaced
	// far more coarsely in relative terms than the logits that produce them.
	for x := float32(-88); x > -103; x -= 0.5 {
		for d := float32(1e-4); d < 1; d *= 2 {
			if y := x - d; expf(x) == expf(y) && expf(x) != 0 && x != y {
				return x, y, true
			}
		}
	}
	return 0, 0, false
}

// TestPenaltiesAgainstTheOracle is specs/006-sampling.md §6's missing row.
//
// The table test above computes its expectation by hand, in the same file, from
// the same reading of §3.1 — so it checks that penalize does what its author
// thought the rules said, not that it does what they say. 010 §5 draws that
// line: a reference is an independent implementation, and internal/oracle is
// where tgo keeps them.
//
// The inputs are randomised over the shapes that make the rules differ from
// each other: a token seen once against one seen many times separates presence
// from frequency, and a negative logit separates dividing from multiplying. A
// fixed seed keeps a failure reproducible.
func TestPenaltiesAgainstTheOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x5eed, 0xC0FFEE))

	for trial := range 200 {
		n := 4 + rng.IntN(12)
		logits := make([]float32, n)
		for i := range logits {
			// Straddle zero: the sign is what selects the branch.
			logits[i] = (rng.Float32() - 0.5) * 8
		}
		// A history with repeats, so counts differ per token.
		history := make([]int, rng.IntN(20))
		for i := range history {
			history[i] = rng.IntN(n)
		}

		p := Policy{
			RepetitionPenalty: []float32{0, 1, 1.2, 2}[rng.IntN(4)],
			PresencePenalty:   []float32{0, 0.3, -0.5}[rng.IntN(3)],
			FrequencyPenalty:  []float32{0, 0.25, -0.1}[rng.IntN(3)],
			PenaltyWindow:     []int{0, 3, 8}[rng.IntN(3)],
		}

		got := make([]float32, n)
		copy(got, logits)
		penalize(got, history, p)

		in64 := make([]float64, n)
		for i, v := range logits {
			in64[i] = float64(v)
		}
		want := oracle.Penalize(in64, history, float64(p.RepetitionPenalty),
			float64(p.PresencePenalty), float64(p.FrequencyPenalty), p.PenaltyWindow)

		for i := range got {
			// The host works in f32 and the reference in f64, so the tolerance
			// is the rounding of the operations §3.1 names: at most one divide
			// or multiply, then one subtract.
			const ulps = 4
			if diff := math.Abs(float64(got[i]) - want[i]); diff > ulps*ulpOf(want[i]) {
				t.Fatalf("trial %d, logit %d: got %v, oracle %v (policy %+v, history %v)",
					trial, i, got[i], want[i], p, history)
			}
		}
	}
}

// ulpOf is the spacing of float32 near v, which is the unit an f32 computation
// of an f64 quantity can differ by.
func ulpOf(v float64) float64 {
	f := float32(v)
	if f == 0 {
		return math.SmallestNonzeroFloat32
	}
	return math.Abs(float64(math.Nextafter32(f, float32(math.Inf(1))) - f))
}

// TestNextRewritesTheRowItWasGiven is why specs/030-logprobs.md §5 puts Probs
// before the draw and not after it.
//
// The stages of §3 write in place: the bias is added, the penalties are applied
// and the row is divided by the temperature, all on the caller's slice. So
// after Next the slice is not the logits the token was drawn from, and a Probs
// taken there would penalise and divide a second time -- reporting a sharper
// distribution than the one that produced the token, with no error anywhere.
//
// Probs itself copies (006-D7), which is what makes the ordering the only thing
// a caller has to get right.
func TestNextRewritesTheRowItWasGiven(t *testing.T) {
	logits := []float32{1, 2, 3, 4}
	before := append([]float32(nil), logits...)
	p := Policy{Temperature: 0.5, RepetitionPenalty: 1.5}

	New(1).Next(logits, []int{0, 1}, p)
	if slices.Equal(logits, before) {
		t.Fatal("Next left the row unchanged; 030 §5's ordering rests on it not doing " +
			"that, and if this is now true the requirement can be dropped rather " +
			"than merely satisfied")
	}

	// And Probs does not, which is the half a caller relies on.
	quiet := append([]float32(nil), before...)
	New(1).Probs(quiet, []int{0, 1}, p)
	if !slices.Equal(quiet, before) {
		t.Errorf("Probs rewrote its input: %v, want %v", quiet, before)
	}
}
