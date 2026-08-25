// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	"github.com/latere-ai/tgo/bench"
)

func TestFirstDifferenceFindsWhereTwoRunsPart(t *testing.T) {
	for _, c := range []struct {
		name string
		a, b []int32
		want int
	}{
		{"identical runs", []int32{1, 2, 3}, []int32{1, 2, 3}, -1},
		{"a divergence in the middle", []int32{1, 2, 3}, []int32{1, 9, 3}, 1},
		{"a divergence at the first token", []int32{4}, []int32{5}, 0},
		{"one run stopped early", []int32{1, 2, 3}, []int32{1, 2}, 2},
		{"the other stopped early", []int32{1, 2}, []int32{1, 2, 3}, 2},
		{"two empty runs", nil, nil, -1},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := FirstDifference(c.a, c.b); got != c.want {
				t.Fatalf("FirstDifference = %d, want %d", got, c.want)
			}
		})
	}
}

func TestTopTwoMarginIsTheDecisionsCloseness(t *testing.T) {
	for _, c := range []struct {
		name   string
		logits []float32
		want   float64
	}{
		{"the top is first", []float32{5, 4, 1}, 1},
		{"the top is last", []float32{1, 4, 5}, 1},
		{"a near tie", []float32{2, 2}, 0},
		{"the runner-up displaces the second", []float32{1, 2, 3, 2.5}, 0.5},
		{"negatives", []float32{-8, -1, -3}, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := TopTwoMargin(c.logits); math.Abs(got-c.want) > 1e-6 {
				t.Fatalf("TopTwoMargin = %v, want %v", got, c.want)
			}
		})
	}
}

func TestTopTwoMarginRefusesAOneTokenVocabulary(t *testing.T) {
	defer func() {
		// The message and not just the panic. Dropping the guard would still
		// panic -- on an index out of range -- and a refusal that reads as a
		// bounds bug sends the reader into this package rather than to the
		// binding that was wrong.
		r := recover()
		if r == nil {
			t.Fatal("a margin over one logit was computed rather than refused")
		}
		if msg, _ := r.(string); !strings.Contains(msg, "fewer than two logits") {
			t.Fatalf("the refusal reads %v, want it to name the one-token vocabulary", r)
		}
	}()
	TopTwoMargin([]float32{1})
}

func TestDivergenceReportsBothOutcomes(t *testing.T) {
	agreed := Divergence{Prompt: "hi", Tokens: 64, Index: -1}
	if got := agreed.Value(); !strings.Contains(got, "no divergence over 64 tokens") {
		t.Errorf("an agreeing run reads %q", got)
	}
	parted := Divergence{Prompt: "hi", Tokens: 64, Index: 17, TopTwoMargin: 0.002}
	got := parted.Value()
	for _, want := range []string{"token 17 of 64", "0.002"} {
		if !strings.Contains(got, want) {
			t.Errorf("a diverging run reads %q, want it to mention %q", got, want)
		}
	}
}

// decodeReport is a bench report with a readback term big enough to be the
// share C6 costs.
func decodeReport(t *testing.T, steps int) bench.Report {
	t.Helper()
	rec := bench.NewRecorder(steps)
	for i := 0; i < steps; i++ {
		rec.Step(bench.Step{
			Phase: bench.Decode, Tokens: 1, Batch: 1,
			Host:     1 * time.Millisecond,
			Submit:   1 * time.Millisecond,
			Device:   6 * time.Millisecond,
			Readback: 2 * time.Millisecond,
		})
	}
	return rec.Report()
}

func TestReadbackShareComesOutOfABenchReport(t *testing.T) {
	const vocab, bytes = 151936, 151936 * 4
	r := ReadbackFrom(decodeReport(t, 8), vocab, bytes)
	if math.Abs(r.Share-0.2) > 1e-9 {
		t.Fatalf("readback share = %v, want 2 ms of a 10 ms step", r.Share)
	}
	if r.Median != 2*time.Millisecond {
		t.Fatalf("readback p50 = %v, want 2ms", r.Median)
	}
	if r.Steps != 8 || r.Vocab != vocab || r.Bytes != bytes {
		t.Fatalf("readback carries %d steps over V=%d at %d B", r.Steps, r.Vocab, r.Bytes)
	}
	got := r.Value()
	for _, want := range []string{"20.0%", "2ms", "607744 B", "V=151936", "8 steps"} {
		if !strings.Contains(got, want) {
			t.Errorf("the readback line reads %q, want it to mention %q", got, want)
		}
	}
}

// outlierWeights are what §3 asks the quantization number to be measured on:
// a block with one channel far outside the others, which is what trained
// transformer weights have and what synthetic ones do not.
func outlierWeights(n int) []float32 {
	rng := rand.New(rand.NewPCG(7, 11))
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.02
	}
	for i := 0; i < n; i += 4 * quant.Int8Block {
		w[i] = 2.5
	}
	return w
}

func TestQuantizationErrorStaysUnderAccelsBound(t *testing.T) {
	const n = 8 * quant.Int8Block
	w := outlierWeights(n)
	x := make([]float32, n)
	rng := rand.New(rand.NewPCG(3, 5))
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := MeasureQuantization("layers.0.mlp.down_proj", x, w)
	if q.Blocks != n/quant.Int8Block {
		t.Fatalf("%d blocks over %d weights", q.Blocks, n)
	}
	if q.Worst > q.Bound {
		t.Fatalf("int8 error %v exceeds quant.Int8ErrorBound %v. "+
			"specs/010-conformance.md §1: where the oracle and accel disagree "+
			"one of them is wrong, and this is a finding against accel",
			q.Worst, q.Bound)
	}
	if q.Used() <= 0 || q.Used() > 1 {
		t.Fatalf("the bound was %v%% spent, which is outside (0, 1]", 100*q.Used())
	}
	// The bound has to be built one scale per term. quant.Int8ErrorBound's own
	// documentation records that indexing scales[i/Int8Block] *inside* the
	// bound was the bug the signature change fixed, so a measurement that
	// cannot tell a per-term array from a constant is not testing the thing
	// the API was changed for.
	_, scales := quant.Int8Quantize(w)
	uniform := func(s accel.Float16) float64 {
		a := make([]accel.Float16, n)
		for i := range a {
			a[i] = s
		}
		return quant.Int8ErrorBound(x, a)
	}
	perTerm := make([]accel.Float16, n)
	for i := range perTerm {
		perTerm[i] = scales[i/quant.Int8Block]
	}
	if want := quant.Int8ErrorBound(x, perTerm); q.Bound != want {
		t.Fatalf("bound = %v, want %v: the bound was not taken over one scale "+
			"per term of the dot product", q.Bound, want)
	}
	// And the fixture discriminates: if every block carried the same scale,
	// the check above would hold for a constant too. The outlier channels are
	// what make the blocks differ, which is why §3 insists on real weights.
	for i, s := range scales {
		if uniform(s) == q.Bound {
			t.Fatalf("the bound over block %d's scale alone equals the "+
				"measured bound, so this fixture cannot tell a per-term array "+
				"from a constant one", i)
		}
	}
	if got := q.Value(); !strings.Contains(got, "down_proj") ||
		!strings.Contains(got, "8 blocks") {
		t.Errorf("the quantization line reads %q", got)
	}
}

// TestAnAllZeroWeightSpendsNoBound covers the degenerate block: a zero scale
// makes the bound zero, and Used has to answer without dividing by it.
func TestAnAllZeroWeightSpendsNoBound(t *testing.T) {
	n := quant.Int8Block
	q := MeasureQuantization("zeros", make([]float32, n), make([]float32, n))
	if q.Bound != 0 || q.Worst != 0 || q.Used() != 0 {
		t.Fatalf("a zero weight measured %+v", q)
	}
}

func TestMeasureQuantizationRefusesMismatchedInputs(t *testing.T) {
	for _, c := range []struct {
		name string
		x, w []float32
		says string
	}{
		{"different lengths", make([]float32, 32), make([]float32, 64), "different lengths"},
		{"nothing to measure", nil, nil, "empty weight"},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				// The message and not just the panic: without the guard a
				// mismatched pair panics anyway, on an index out of range,
				// and that names this package instead of the caller's error.
				r := recover()
				if r == nil {
					t.Fatal("the measurement was taken rather than refused")
				}
				if msg, _ := r.(string); !strings.Contains(msg, c.says) {
					t.Fatalf("the refusal reads %v, want it to mention %q", r, c.says)
				}
			}()
			MeasureQuantization("bad", c.x, c.w)
		})
	}
}

func TestCompileTimeSummarizesTheBucketSet(t *testing.T) {
	c := Compile{
		Buckets: []Bucket{{Tokens: 1, Elapsed: 2 * time.Millisecond},
			{Tokens: 256, Elapsed: 8 * time.Millisecond}},
		Hits: 3, Misses: 1,
	}
	if c.Total() != 10*time.Millisecond {
		t.Fatalf("total compile time = %v, want 10ms", c.Total())
	}
	if c.HitRate() != 0.75 {
		t.Fatalf("hit rate = %v, want 0.75", c.HitRate())
	}
	got := c.Value()
	for _, want := range []string{"1 tok 2ms", "256 tok 8ms", "75%", "4 lookups"} {
		if !strings.Contains(got, want) {
			t.Errorf("the compile line reads %q, want it to mention %q", got, want)
		}
	}
}

// TestNoBucketWasCompiled covers a session that compiled nothing: the hit rate
// of no lookups is not a division, and a table row saying "0%" over no lookups
// would read as a cache that never hits.
func TestNoBucketWasCompiled(t *testing.T) {
	var c Compile
	if c.HitRate() != 0 || c.Total() != 0 {
		t.Fatalf("an empty session reported %v over %v", c.HitRate(), c.Total())
	}
	if got := c.Value(); !strings.Contains(got, "no bucket compiled") {
		t.Fatalf("an empty session reads %q", got)
	}
}

func TestTransientIsMeasuredAgainstTheWorkingSet(t *testing.T) {
	tr := Transient{
		Label:      "qwen3-0.6b decode",
		Memory:     accel.GraphMemory{TransientBytes: 300, UnaliasedBytes: 1000, PeakBytes: 1200},
		WorkingSet: 250,
	}
	if tr.Saved() != 700 {
		t.Fatalf("aliasing saved %d B, want 700", tr.Saved())
	}
	if math.Abs(tr.Overhead()-1.2) > 1e-9 {
		t.Fatalf("overhead = %v, want 1.2", tr.Overhead())
	}
	if got := tr.Value(); !strings.Contains(got, "700 B saved") ||
		!strings.Contains(got, "1.20× the floor") {
		t.Errorf("the transient line reads %q", got)
	}
	var none Transient
	if none.Overhead() != 0 {
		t.Fatalf("an unmeasured working set produced an overhead of %v", none.Overhead())
	}
}

// TestAnUnmeasuredNumberSaysSo is the point of the pointer fields: an engine
// being built in parallel has not taken three of these numbers yet, and a
// report that printed 0.00 for them would be a false measurement rather than a
// missing one.
func TestAnUnmeasuredNumberSaysSo(t *testing.T) {
	doc := Measurements{}.Document()
	if n := strings.Count(doc, notMeasured); n != 5 {
		t.Fatalf("an empty report names %d unmeasured numbers, want 5:\n%s", n, doc)
	}
	for _, want := range []string{
		"CPU/Metal divergence", "readback share of a decode step",
		"quantization error against `Int8ErrorBound`", "plan compile time per bucket",
		"transient bytes against the working set",
	} {
		if !strings.Contains(doc, want) {
			t.Errorf("the report has no row for %q:\n%s", want, doc)
		}
	}
}

func TestAFullReportPrintsEveryNumber(t *testing.T) {
	m := Measurements{
		Divergence:   &Divergence{Prompt: "why is the sky blue?", Tokens: 128, Index: 41, TopTwoMargin: 1e-4},
		Readback:     ptr(ReadbackFrom(decodeReport(t, 4), 151936, 607744)),
		Quantization: ptr(MeasureQuantization("q_proj", []float32{1, -2, 3}, []float32{0.5, 0.25, -0.75})),
		Compile:      &Compile{Buckets: []Bucket{{Tokens: 1, Elapsed: time.Millisecond}}, Hits: 9, Misses: 1},
		Transient: &Transient{Label: "decode", WorkingSet: 100,
			Memory: accel.GraphMemory{TransientBytes: 100, UnaliasedBytes: 400}},
	}
	doc := m.Document()
	if strings.Contains(doc, notMeasured) {
		t.Fatalf("a full report says a number is missing:\n%s", doc)
	}
	if n := strings.Count(doc, "\n|"); n != 6 {
		t.Fatalf("the report has %d table lines after the header, want 6:\n%s", n, doc)
	}
	// The report is a record, so it round-trips through JSON.
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Measurements
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Divergence == nil || back.Divergence.Index != 41 || back.Transient == nil ||
		back.Transient.Memory.UnaliasedBytes != 400 {
		t.Fatalf("the report did not survive a round trip: %s", b)
	}
	if empty, err := json.Marshal(Measurements{}); err != nil || string(empty) != "{}" {
		t.Fatalf("an unmeasured report marshals as %s (%v), want {}", empty, err)
	}
}

func ptr[T any](v T) *T { return &v }

// TestPublishIsTheGeneratedDocument is 010 §6: the suite emits the register and
// the numbers as one Markdown document, from the tests rather than beside them.
func TestPublishIsTheGeneratedDocument(t *testing.T) {
	doc := Publish(Register(), Measurements{})
	if !strings.Contains(doc, Document(Register())) {
		t.Error("the published document does not carry the register table")
	}
	if !strings.Contains(doc, Measurements{}.Document()) {
		t.Error("the published document does not carry the measurements")
	}
	for _, want := range []string{"# Conformance", "## The register",
		"## Numbers tgo reports back", "register.go"} {
		if !strings.Contains(doc, want) {
			t.Errorf("the published document has no %q", want)
		}
	}
}
