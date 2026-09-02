// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package sample

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/latere-ai/tgo/internal/oracle"
)

// rows returns count rows of vocab logits from a seeded generator, so a stream
// test feeds the same inputs to every run without holding a golden file.
func rows(seed uint64, count, vocab int) [][]float32 {
	r := rand.New(rand.NewPCG(seed, 99))
	out := make([][]float32, count)
	for i := range out {
		row := make([]float32, vocab)
		for j := range row {
			row[j] = float32(r.NormFloat64() * 3)
		}
		out[i] = row
	}
	return out
}

func copyRow(x []float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	return out
}

// run feeds the rows through one sampler, taking the policy for each step from
// pick, and returns the tokens.
func run(seed uint64, in [][]float32, pick func(step int) Policy) []int {
	s := New(seed)
	out := make([]int, len(in))
	for i, row := range in {
		out[i] = s.Next(copyRow(row), nil, pick(i))
	}
	return out
}

func same(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestSameSeedGivesTheSameCompletion(t *testing.T) {
	in := rows(4, 32, 64)
	p := func(int) Policy { return Policy{Temperature: 0.9, TopK: 8, TopP: 0.95} }
	if a, b := run(11, in, p), run(11, in, p); !same(a, b) {
		t.Fatalf("two runs of one seed differ:\n%v\n%v", a, b)
	}
}

func TestDifferentSeedsDiverge(t *testing.T) {
	in := rows(4, 32, 64)
	p := func(int) Policy { return Policy{Temperature: 1} }
	if same(run(11, in, p), run(12, in, p)) {
		t.Fatal("two seeds produced the same 32 tokens; the seed is not reaching the draw")
	}
}

// TestStreamSurvivesAPolicyChangeMidStream is 006-D2, the decision that is easy
// to get wrong.
//
// Step i consumes exactly one draw whether or not the policy is greedy. Two
// runs that differ only in the policy of their first ten steps therefore reach
// step ten with the same draw, and their tails agree. Under the natural
// implementation -- draw only when sampling -- the greedy run would be ten
// draws behind and every token after the change would differ.
func TestStreamSurvivesAPolicyChangeMidStream(t *testing.T) {
	in := rows(5, 24, 64)
	const change = 10
	tail := Policy{Temperature: 1}

	greedyFirst := run(21, in, func(i int) Policy {
		if i < change {
			return Policy{} // greedy: consumes a draw all the same
		}
		return tail
	})
	sampledFirst := run(21, in, func(i int) Policy {
		if i < change {
			return Policy{Temperature: 1.4, TopK: 4}
		}
		return tail
	})

	if !same(greedyFirst[change:], sampledFirst[change:]) {
		t.Fatalf("the stream shifted at the policy change:\n%v\n%v",
			greedyFirst[change:], sampledFirst[change:])
	}
	if same(greedyFirst[:change], sampledFirst[:change]) {
		t.Fatal("the two policies produced the same first ten tokens; the test proves nothing")
	}
}

// TestProbsDoesNotMoveTheStream is 006-D7: an observation must not perturb the
// completion it describes.
func TestProbsDoesNotMoveTheStream(t *testing.T) {
	in := rows(6, 16, 32)
	p := Policy{Temperature: 1, TopP: 0.9}

	plain := run(31, in, func(int) Policy { return p })

	s := New(31)
	observed := make([]int, len(in))
	for i, row := range in {
		s.Probs(row, nil, p)
		s.Probs(row, nil, Policy{}) // a greedy observation must not draw either
		observed[i] = s.Next(copyRow(row), nil, p)
	}
	if !same(plain, observed) {
		t.Fatalf("Probs moved the stream:\n%v\n%v", plain, observed)
	}
}

// TestGreedyIsDeterministic is section 4.1: greedy is bit-exact across runs on
// one device, and independent of the seed.
func TestGreedyIsDeterministic(t *testing.T) {
	logits := rows(7, 1, 512)[0]
	want := New(0).Next(copyRow(logits), nil, Policy{})
	for i := range 100 {
		got := New(uint64(i)).Next(copyRow(logits), nil, Policy{})
		if got != want {
			t.Fatalf("run %d gave token %d, want %d", i, got, want)
		}
	}
	if want != argmax(logits) {
		t.Fatalf("greedy token %d is not the argmax %d", want, argmax(logits))
	}
}

// TestTopKOfOneEqualsGreedy: one candidate leaves the walk no choice, whatever
// the draw and whatever the temperature.
func TestTopKOfOneEqualsGreedy(t *testing.T) {
	in := rows(8, 40, 128)
	greedy := run(41, in, func(int) Policy { return Policy{} })
	k1 := run(41, in, func(int) Policy { return Policy{Temperature: 1.3, TopK: 1} })
	if !same(greedy, k1) {
		t.Fatalf("k=1 is not greedy:\n%v\n%v", greedy, k1)
	}
}

// TestCategoricalMatchesAnIndependentWalk checks the untruncated draw against a
// float64 reference built from the oracle's softmax.
//
// A step is skipped when the scaled draw lands within boundaryBand of a
// cumulative boundary, where the float32 walk and the float64 reference are
// allowed to disagree: that band is the accumulated float32 rounding of the
// running sum, sqrt(V)*eps32 relative (specs/010-conformance.md section 5.1),
// which is 5e-7 here and is rounded up to 1e-5 to leave room for the exp.
func TestCategoricalMatchesAnIndependentWalk(t *testing.T) {
	const boundaryBand = 1e-5

	in := rows(9, 200, 48)
	s := New(51)
	draws := New(51) // the same stream, read alongside
	checked := 0
	for step, row := range in {
		u := float64(draws.draw())
		got := s.Next(copyRow(row), nil, Policy{Temperature: 1})

		probs := oracle.Softmax(f64(row))
		acc, want, near := 0.0, len(probs)-1, false
		for i, v := range probs {
			acc += v
			if math.Abs(acc-u) < boundaryBand {
				near = true
			}
			if acc > u && want == len(probs)-1 && i < len(probs)-1 {
				want = i
			}
		}
		if near {
			continue
		}
		checked++
		if got != want {
			t.Fatalf("step %d: token %d, the float64 walk says %d for u = %v", step, got, want, u)
		}
	}
	if checked < len(in)/2 {
		t.Fatalf("only %d of %d steps were checked; the band is swallowing the test", checked, len(in))
	}
}

// TestNextIsTheQuantileOfProbs ties the two entry points together: the token
// Next returns is the one the distribution Probs reports puts the draw in.
func TestNextIsTheQuantileOfProbs(t *testing.T) {
	in := rows(10, 50, 64)
	s := New(61)
	draws := New(61)
	for step, row := range in {
		p := Policy{Temperature: 0.8, TopK: 16, TopP: 0.9}
		u := float64(draws.draw())
		probs := s.Probs(row, nil, p)
		got := s.Next(copyRow(row), nil, p)

		acc, want := 0.0, -1
		for i, v := range probs {
			acc += float64(v)
			if acc > u && v > 0 {
				want = i
				break
			}
		}
		if want >= 0 && got != want {
			t.Fatalf("step %d: Next chose %d, Probs puts u = %v at %d", step, got, u, want)
		}
	}
}

// TestSampledFrequenciesFollowTheDistribution: the walk is unbiased. Two
// tokens three logits apart appear in the ratio their probabilities state.
func TestSampledFrequenciesFollowTheDistribution(t *testing.T) {
	logits := []float32{0, 1}
	const trials = 20000

	s := New(71)
	ones := 0
	for range trials {
		if s.Next(copyRow(logits), nil, Policy{Temperature: 1}) == 1 {
			ones++
		}
	}
	want := math.Exp(1) / (1 + math.Exp(1))
	got := float64(ones) / trials
	// Three standard errors of a binomial with p = 0.73 over 20000 trials is
	// about 0.009; the bound is that, not a number tuned to the outcome.
	if math.Abs(got-want) > 0.01 {
		t.Fatalf("token 1 appeared %v of the time, want %v", got, want)
	}
}

func TestNextModifiesTheLogitsInPlace(t *testing.T) {
	logits := []float32{2, 1}
	New(1).Next(logits, nil, Policy{Temperature: 0.5, LogitBias: map[int]float32{1: 1}})
	want := []float32{4, 4} // biased, then divided by the temperature
	for i := range want {
		if logits[i] != want[i] {
			t.Fatalf("logits = %v, want %v", logits, want)
		}
	}
}

func TestProbsLeavesTheLogitsAlone(t *testing.T) {
	logits := []float32{2, 1}
	New(1).Probs(logits, []int{0}, Policy{Temperature: 0.5, RepetitionPenalty: 2})
	want := []float32{2, 1}
	for i := range want {
		if logits[i] != want[i] {
			t.Fatalf("Probs modified the logits: %v, want %v", logits, want)
		}
	}
}

// TestTopKAtTheBoundIsAcceptedAndAboveIsRefused: TopMaxRounds is a hard edge,
// not a clamp. accel's kernel walks 128 rounds and would silently keep 128.
func TestTopKAtTheBoundIsAcceptedAndAboveIsRefused(t *testing.T) {
	logits := rows(12, 1, 256)[0]
	New(1).Next(copyRow(logits), nil, Policy{Temperature: 1, TopK: TopMaxRounds})

	defer func() {
		if recover() == nil {
			t.Fatal("TopK = 129 was accepted")
		}
	}()
	New(1).Next(copyRow(logits), nil, Policy{Temperature: 1, TopK: TopMaxRounds + 1})
}

// TestRefusals: every refusal on Policy, through both entry points.
func TestRefusals(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	cases := []struct {
		name    string
		logits  []float32
		history []int
		p       Policy
	}{
		{"empty logits", nil, nil, Policy{}},
		{"negative temperature", []float32{1, 2}, nil, Policy{Temperature: -1}},
		{"NaN temperature", []float32{1, 2}, nil, Policy{Temperature: nan}},
		{"negative k", []float32{1, 2}, nil, Policy{TopK: -1}},
		{"k above the bound", []float32{1, 2}, nil, Policy{TopK: TopMaxRounds + 1}},
		{"negative p", []float32{1, 2}, nil, Policy{TopP: -0.5}},
		{"p above one", []float32{1, 2}, nil, Policy{TopP: 1.5}},
		{"NaN p", []float32{1, 2}, nil, Policy{TopP: nan}},
		{"negative repetition penalty", []float32{1, 2}, nil, Policy{RepetitionPenalty: -1}},
		{"NaN repetition penalty", []float32{1, 2}, nil, Policy{RepetitionPenalty: nan}},
		{"NaN presence penalty", []float32{1, 2}, nil, Policy{PresencePenalty: nan}},
		{"NaN frequency penalty", []float32{1, 2}, nil, Policy{FrequencyPenalty: nan}},
		{"negative window", []float32{1, 2}, nil, Policy{PenaltyWindow: -1}},
		{"bias id below the vocabulary", []float32{1, 2}, nil, Policy{LogitBias: map[int]float32{-1: 1}}},
		{"bias id above the vocabulary", []float32{1, 2}, nil, Policy{LogitBias: map[int]float32{2: 1}}},
		{"NaN bias", []float32{1, 2}, nil, Policy{LogitBias: map[int]float32{0: nan}}},
		{"infinite bias", []float32{1, 2}, nil, Policy{LogitBias: map[int]float32{0: inf}}},
		{"history token below the vocabulary", []float32{1, 2}, []int{-1}, Policy{}},
		{"history token above the vocabulary", []float32{1, 2}, []int{2}, Policy{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			mustPanic(t, "Next", func() { New(1).Next(copyRow(c.logits), c.history, c.p) })
			mustPanic(t, "Probs", func() { New(1).Probs(copyRow(c.logits), c.history, c.p) })
		})
	}
}

func mustPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s accepted what the spec refuses", what)
		}
	}()
	f()
}

// TestRefusalCostsNoDraw: a refused policy is a caller error before the step,
// so a sampler that recovers from one is still where it was.
func TestRefusalCostsNoDraw(t *testing.T) {
	logits := []float32{3, 1, 2}
	s := New(81)
	func() {
		defer func() { _ = recover() }()
		s.Next(copyRow(logits), nil, Policy{Temperature: -1})
	}()
	if got, want := s.Next(copyRow(logits), nil, Policy{Temperature: 1}), New(81).Next(copyRow(logits), nil, Policy{Temperature: 1}); got != want {
		t.Fatalf("token %d after a refusal, want %d: the refusal consumed a draw", got, want)
	}
}

// TestDrawStaysBelowOne: the float32 clamp mirrors accel's SampleCategorical.
// A draw of exactly one lands on the total, which no partial sum exceeds, and
// the walk would fall through to its last candidate.
func TestDrawStaysBelowOne(t *testing.T) {
	s := New(91)
	for range 10000 {
		u := s.draw()
		if u < 0 || u >= 1 {
			t.Fatalf("draw %v is outside [0, 1)", u)
		}
		if u > maxDraw {
			t.Fatalf("draw %v is above the clamp %v", u, maxDraw)
		}
	}
}

func TestScratchGrowsWithTheVocabulary(t *testing.T) {
	s := New(1)
	small := s.Next(copyRow([]float32{1, 2}), nil, Policy{Temperature: 1})
	big := s.Next(rows(13, 1, 4096)[0], nil, Policy{Temperature: 1})
	if small < 0 || big < 0 {
		t.Fatal("a token id is negative")
	}
	if len(s.scratch) != 4096 {
		t.Fatalf("scratch is %d long after a 4096 vocabulary", len(s.scratch))
	}
}
