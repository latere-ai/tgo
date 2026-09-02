// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package bench_test

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/latere-ai/tgo/bench"
)

// shareTolerance bounds how far the four fractions of ShareOfStep may sum from
// 1, and it is not slack: it is the largest error IEEE-754 double rounding can
// produce for this expression.
//
// The four sums are int64 nanosecond counts, exact in a float64 below 2^53 ns
// (104 days of accumulated time), so the conversions contribute nothing. What
// remains is four divisions and, in this test, three additions: seven
// operations, each rounded to the nearest double with relative error at most
// 2^-53 = 1.11e-16. Seven of those bound the deviation at 7*2^-53 = 7.8e-16,
// which rounds up to 1e-15.
//
// The four terms are exhaustive by construction: specs/017-benchmarks.md §1
// defines the step as their sum, and no wall-clock total is recorded, so there
// is no unmeasured gap for the shares to lose. Rounding is the only error term.
const shareTolerance = 1e-15

func sumShares(t *testing.T, m map[string]float64) float64 {
	t.Helper()
	if len(m) != 4 {
		t.Fatalf("ShareOfStep has %d keys, want 4: %v", len(m), m)
	}
	return m[bench.ShareHost] + m[bench.ShareSubmit] + m[bench.ShareDevice] + m[bench.ShareReadback]
}

// TestDisabledRecorderAllocatesZero pins 017-D3: off by default, and off means
// no memory touched. A nil recorder, the zero value, and an explicitly sized
// zero recorder are all the disabled case.
func TestDisabledRecorderAllocatesZero(t *testing.T) {
	cases := map[string]*bench.Recorder{
		"nil":            nil,
		"zero value":     new(bench.Recorder),
		"zero capacity":  bench.NewRecorder(0),
		"negative capac": bench.NewRecorder(-1),
	}
	for name, r := range cases {
		t.Run(name, func(t *testing.T) {
			if r.Enabled() {
				t.Fatalf("Enabled() = true, want false")
			}
			s := bench.Step{Phase: bench.Decode, Tokens: 1, Host: time.Microsecond}
			if got := testing.AllocsPerRun(1000, func() { r.Step(s) }); got != 0 {
				t.Errorf("Step allocated %v times per run, want 0", got)
			}
			if got := testing.AllocsPerRun(1000, func() { r.TTFT(time.Millisecond) }); got != 0 {
				t.Errorf("TTFT allocated %v times per run, want 0", got)
			}

			rep := r.Report()
			if rep.Steps != 0 {
				t.Errorf("Steps = %d, want 0", rep.Steps)
			}
			// Nothing was discarded by an instrument that never recorded.
			if rep.Dropped != 0 {
				t.Errorf("Dropped = %d, want 0: a disabled recorder drops nothing", rep.Dropped)
			}
			if got := sumShares(t, rep.Decode.ShareOfStep); got != 0 {
				t.Errorf("empty ShareOfStep sums to %v, want 0", got)
			}
		})
	}
}

// TestEnabledRecorderAllocatesZero pins the other half of 017-D3: an
// instrument that allocates while measuring is measuring itself.
func TestEnabledRecorderAllocatesZero(t *testing.T) {
	const runs = 1000
	// Two spare slots per stream: AllocsPerRun calls the closure runs+1 times,
	// and the Dropped assertion below is what stops this test passing because
	// the recorder filled up and took the cheap reject path instead.
	r := bench.NewRecorder(runs + 2)

	s := bench.Step{Phase: bench.Decode, Tokens: 1, Batch: 4, Host: time.Microsecond, Device: time.Millisecond}
	if got := testing.AllocsPerRun(runs, func() { r.Step(s) }); got != 0 {
		t.Errorf("Step allocated %v times per run, want 0", got)
	}
	if rep := r.Report(); rep.Dropped != 0 || rep.Steps != runs+1 {
		t.Fatalf("Steps = %d, Dropped = %d; want %d and 0 (the run must fit, or zero allocations proves nothing)",
			rep.Steps, rep.Dropped, runs+1)
	}

	r.Reset()
	if got := testing.AllocsPerRun(runs, func() { r.TTFT(time.Millisecond) }); got != 0 {
		t.Errorf("TTFT allocated %v times per run, want 0", got)
	}
	if rep := r.Report(); rep.Dropped != 0 || rep.TTFT.N != runs+1 {
		t.Fatalf("TTFT.N = %d, Dropped = %d; want %d and 0", rep.TTFT.N, rep.Dropped, runs+1)
	}
}

// TestFullRecorderKeepsTheMostRecent is the negative test for the bound above:
// a recorder past capacity must say so rather than truncate in silence, and
// what it keeps must be the end of the run rather than the beginning.
//
// It kept the beginning until 2026-08-27, which made
// server/generate.go's "reports quantiles over its most recent steps" false for
// exactly the completions long enough to need a window. A request past capacity
// published percentiles for its own warm-up, and nothing said so: Dropped was
// non-zero, and no reader treats a count as "these numbers are the wrong ones".
func TestFullRecorderKeepsTheMostRecent(t *testing.T) {
	r := bench.NewRecorder(2)
	for i := range 5 {
		r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Host: time.Duration(i+1) * time.Microsecond})
		r.TTFT(time.Duration(i+1) * time.Millisecond)
	}
	rep := r.Report()
	if rep.Steps != 2 || rep.TTFT.N != 2 {
		t.Errorf("Steps = %d, TTFT.N = %d; want 2 and 2", rep.Steps, rep.TTFT.N)
	}
	if want := 6; rep.Dropped != want {
		t.Errorf("Dropped = %d, want %d (3 steps and 3 TTFTs overwritten)", rep.Dropped, want)
	}
	// The kept observations are the last two, 4µs and 5µs. The old behaviour
	// gave 2µs here, so this assertion is the one that changes with it.
	if got, want := rep.Decode.Host.P99, 5*time.Microsecond; got != want {
		t.Errorf("Host.P99 = %v, want %v: the recorder keeps the suffix", got, want)
	}
	if got, want := rep.Decode.Host.P50, 4*time.Microsecond; got != want {
		t.Errorf("Host.P50 = %v, want %v", got, want)
	}
	if got, want := rep.TTFT.P50, 4*time.Millisecond; got != want {
		t.Errorf("TTFT.P50 = %v, want %v: TTFT is a ring too", got, want)
	}

	// Reset returns it to empty, ring state included: a recorder that had
	// wrapped must not report its stale head as the oldest entry.
	r.Reset()
	r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Host: 9 * time.Microsecond})
	rep = r.Report()
	if rep.Steps != 1 || rep.Dropped != 0 || rep.Decode.Host.P50 != 9*time.Microsecond {
		t.Errorf("after Reset: Steps = %d, Dropped = %d, P50 = %v; want 1, 0, 9µs",
			rep.Steps, rep.Dropped, rep.Decode.Host.P50)
	}
}

// TestARingReportsOldestFirst pins the order Report hands back, which no
// quantile depends on and every reader lining a dump against a log does.
func TestARingReportsOldestFirst(t *testing.T) {
	r := bench.NewRecorder(3)
	for i := range 5 {
		r.Step(bench.Step{Phase: bench.Prefill, Tokens: i + 1})
	}
	// Five recorded into three: 3, 4, 5 survive, in that order.
	rep := r.Report()
	if rep.Steps != 3 {
		t.Fatalf("Steps = %d, want 3", rep.Steps)
	}
	if got, want := rep.Prefill.Tokens, 3+4+5; got != want {
		t.Errorf("Tokens = %d, want %d: the surviving steps are the last three", got, want)
	}
}

// TestPercentilesKnownDistribution checks the aggregation against a
// distribution whose answers are computable by hand, at two sizes. The second
// size is chosen so that every percentile differs from the linearly
// interpolated answer: at n=10 an interpolating definition returns 5.5, 9.1 and
// 9.91, none of which is a sample. A test at n=100 alone passes under either
// definition and would not catch the swap.
func TestPercentilesKnownDistribution(t *testing.T) {
	tests := []struct {
		name          string
		n             int
		p50, p90, p99 time.Duration
	}{
		{"n=100, values 1..100ns", 100, 50, 90, 99},
		{"n=10, values 1..10ns", 10, 5, 9, 10},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := bench.NewRecorder(tc.n)
			// Fed back to front, so a report that forgot to sort fails.
			for i := tc.n; i >= 1; i-- {
				r.Step(bench.Step{Phase: bench.Decode, Host: time.Duration(i)})
			}
			got := r.Report().Decode.Host
			want := bench.Quantiles{P50: tc.p50, P90: tc.p90, P99: tc.p99, N: tc.n}
			if got != want {
				t.Errorf("Host quantiles = %+v, want %+v", got, want)
			}
		})
	}
}

// TestPercentilesReturnRealSamples states the property that decided the
// definition: every reported percentile is a step that happened. A bimodal
// sample with no mass in the middle catches an interpolating implementation,
// which would report a value no step ever took.
func TestPercentilesReturnRealSamples(t *testing.T) {
	r := bench.NewRecorder(10)
	for range 9 {
		r.Step(bench.Step{Phase: bench.Decode, Host: time.Millisecond})
	}
	r.Step(bench.Step{Phase: bench.Decode, Host: time.Second}) // the stall

	q := r.Report().Decode.Host
	for _, v := range []time.Duration{q.P50, q.P90, q.P99} {
		if v != time.Millisecond && v != time.Second {
			t.Errorf("percentile %v is not an observed sample", v)
		}
	}
	if q.P50 != time.Millisecond {
		t.Errorf("P50 = %v, want 1ms", q.P50)
	}
	// Rank ceil(0.99*10) = 10, the tenth sample: the tail is reported, not
	// averaged away.
	if q.P99 != time.Second {
		t.Errorf("P99 = %v, want 1s", q.P99)
	}
}

func TestQuantilesEmptyAndSingle(t *testing.T) {
	empty := bench.NewRecorder(4).Report()
	if got := (empty.Decode.Host); got != (bench.Quantiles{}) {
		t.Errorf("no samples gives %+v, want the zero Quantiles", got)
	}
	if empty.TTFT != (bench.Quantiles{}) {
		t.Errorf("TTFT = %+v, want the zero Quantiles", empty.TTFT)
	}

	r := bench.NewRecorder(4)
	r.Step(bench.Step{Phase: bench.Prefill, Device: 7 * time.Millisecond})
	q := r.Report().Prefill.Device
	if q != (bench.Quantiles{P50: 7 * time.Millisecond, P90: 7 * time.Millisecond, P99: 7 * time.Millisecond, N: 1}) {
		t.Errorf("one sample gives %+v, want that sample at every percentile", q)
	}
}

// TestShareOfStepSumsToOne is 017-D1's arithmetic, on sums whose ratios are not
// representable in binary so that the tolerance is exercised rather than
// stepped around.
func TestShareOfStepSumsToOne(t *testing.T) {
	r := bench.NewRecorder(8)
	for i := 1; i <= 7; i++ {
		r.Step(bench.Step{
			Phase:    bench.Decode,
			Tokens:   1,
			Host:     time.Duration(i) * time.Nanosecond,
			Submit:   time.Duration(3*i) * time.Nanosecond,
			Device:   time.Duration(11*i) * time.Nanosecond,
			Readback: time.Duration(2*i) * time.Nanosecond,
		})
	}
	m := r.Report().Decode.ShareOfStep
	if got := sumShares(t, m); math.Abs(got-1) > shareTolerance {
		t.Errorf("shares sum to %v, off from 1 by more than %g", got, shareTolerance)
	}
	// Ratios 1:3:11:2 of a 17-part step, whichever i the step carried.
	for key, want := range map[string]float64{
		bench.ShareHost:     1.0 / 17.0,
		bench.ShareSubmit:   3.0 / 17.0,
		bench.ShareDevice:   11.0 / 17.0,
		bench.ShareReadback: 2.0 / 17.0,
	} {
		if math.Abs(m[key]-want) > shareTolerance {
			t.Errorf("share %q = %v, want %v", key, m[key], want)
		}
	}
}

// TestShareOfStepExactBreakdown checks the shares against a breakdown chosen to
// be exact, so a failure is the arithmetic and not the tolerance.
func TestShareOfStepExactBreakdown(t *testing.T) {
	r := bench.NewRecorder(1)
	r.Step(bench.Step{
		Phase:    bench.Decode,
		Tokens:   1,
		Host:     100 * time.Microsecond,
		Submit:   200 * time.Microsecond,
		Device:   600 * time.Microsecond,
		Readback: 100 * time.Microsecond,
	})
	m := r.Report().Decode.ShareOfStep
	want := map[string]float64{
		bench.ShareHost: 0.1, bench.ShareSubmit: 0.2,
		bench.ShareDevice: 0.6, bench.ShareReadback: 0.1,
	}
	if !reflect.DeepEqual(m, want) {
		t.Errorf("ShareOfStep = %v, want %v", m, want)
	}
	if got := sumShares(t, m); math.Abs(got-1) > shareTolerance {
		t.Errorf("shares sum to %v, want 1", got)
	}
}

func TestTokensPerSecond(t *testing.T) {
	r := bench.NewRecorder(4)
	// Four steps of 250ms each, eight tokens: one second, eight tokens.
	for range 4 {
		r.Step(bench.Step{Phase: bench.Decode, Tokens: 2, Batch: 2, Device: 250 * time.Millisecond})
	}
	if got := r.Report().Decode.TokensPerSecond; got != 8 {
		t.Errorf("TokensPerSecond = %v, want 8", got)
	}
	// A phase that recorded no time reports no rate rather than an infinity.
	z := bench.NewRecorder(1)
	z.Step(bench.Step{Phase: bench.Prefill, Tokens: 32})
	if got := z.Report().Prefill.TokensPerSecond; got != 0 {
		t.Errorf("TokensPerSecond = %v with zero elapsed, want 0", got)
	}
}

func TestTTFTQuantiles(t *testing.T) {
	r := bench.NewRecorder(10)
	for i := 10; i >= 1; i-- {
		r.TTFT(time.Duration(i) * time.Millisecond)
	}
	got := r.Report().TTFT
	want := bench.Quantiles{P50: 5 * time.Millisecond, P90: 9 * time.Millisecond, P99: 10 * time.Millisecond, N: 10}
	if got != want {
		t.Errorf("TTFT = %+v, want %+v", got, want)
	}
}

func TestReset(t *testing.T) {
	r := bench.NewRecorder(1)
	r.Step(bench.Step{Phase: bench.Decode, Host: time.Second})
	r.Step(bench.Step{Phase: bench.Decode, Host: time.Second}) // dropped
	r.TTFT(time.Second)

	r.Reset()
	if rep := r.Report(); rep.Steps != 0 || rep.Dropped != 0 || rep.TTFT.N != 0 {
		t.Errorf("after Reset: %+v, want everything zero", rep)
	}
	// The storage survives, so the recorder still records without allocating.
	r.Step(bench.Step{Phase: bench.Prefill, Host: 3 * time.Second})
	if rep := r.Report(); rep.Steps != 1 || rep.Prefill.Host.P50 != 3*time.Second {
		t.Errorf("after Reset the recorder did not record again: %+v", rep)
	}

	var nilRec *bench.Recorder
	nilRec.Reset() // must not panic
}

func TestPhaseString(t *testing.T) {
	for p, want := range map[bench.Phase]string{
		bench.Prefill:  "prefill",
		bench.Decode:   "decode",
		bench.Phase(9): "unknown",
	} {
		if got := p.String(); got != want {
			t.Errorf("Phase(%d).String() = %q, want %q", uint8(p), got, want)
		}
	}
}

func TestStepTotal(t *testing.T) {
	s := bench.Step{Host: 1, Submit: 2, Device: 4, Readback: 8}
	if got := s.Total(); got != 15 {
		t.Errorf("Total() = %v, want 15ns", got)
	}
}

// TestRealisticSequence feeds one prefill step and a hundred decode steps, the
// shape of a single short request, and checks the report reads the way an
// operator would need it to.
func TestRealisticSequence(t *testing.T) {
	const decodes = 100
	r := bench.NewRecorder(256)

	prefill := bench.Step{
		Phase: bench.Prefill, Tokens: 64, Batch: 1,
		Host: time.Millisecond, Submit: 2 * time.Millisecond,
		Device: 20 * time.Millisecond, Readback: 2 * time.Millisecond,
	}
	r.Step(prefill)
	r.TTFT(prefill.Total())

	var decodeTotal time.Duration
	for i := range decodes {
		s := bench.Step{
			Phase: bench.Decode, Tokens: 1, Batch: 1,
			Host:     100 * time.Microsecond,
			Submit:   50 * time.Microsecond,
			Device:   time.Duration(1000+10*i) * time.Microsecond,
			Readback: 25 * time.Microsecond,
		}
		r.Step(s)
		decodeTotal += s.Total()
	}

	rep := r.Report()
	if rep.Steps != decodes+1 || rep.Dropped != 0 {
		t.Fatalf("Steps = %d, Dropped = %d; want %d and 0", rep.Steps, rep.Dropped, decodes+1)
	}
	if rep.Prefill.Steps != 1 || rep.Decode.Steps != decodes {
		t.Fatalf("phase split = %d prefill, %d decode; want 1 and %d", rep.Prefill.Steps, rep.Decode.Steps, decodes)
	}

	// Prefill: 64 tokens in 25ms.
	if got := rep.Prefill.TokensPerSecond; got != 2560 {
		t.Errorf("prefill tokens/s = %v, want 2560", got)
	}
	if got, want := rep.TTFT.P50, 25*time.Millisecond; got != want {
		t.Errorf("TTFT p50 = %v, want %v", got, want)
	}

	// Decode device times are 1000..1990µs in 10µs steps, so the ranks are
	// exact: p50 is the 50th sample, p90 the 90th, p99 the 99th.
	wantDevice := bench.Quantiles{P50: 1490 * time.Microsecond, P90: 1890 * time.Microsecond, P99: 1980 * time.Microsecond, N: decodes}
	if rep.Decode.Device != wantDevice {
		t.Errorf("decode device = %+v, want %+v", rep.Decode.Device, wantDevice)
	}
	// Host is constant, so every percentile is that constant. This is the
	// number 000 §11's bet is stated in.
	if got := rep.Decode.Host.P99; got != 100*time.Microsecond {
		t.Errorf("decode host p99 = %v, want 100µs", got)
	}
	if got := float64(decodes) / decodeTotal.Seconds(); math.Abs(rep.Decode.TokensPerSecond-got) > 1e-9 {
		t.Errorf("decode tokens/s = %v, want %v", rep.Decode.TokensPerSecond, got)
	}

	// The breakdown attributes this workload to the device, which is the
	// attribution a regression has to survive to be tgo's fault.
	m := rep.Decode.ShareOfStep
	if math.Abs(sumShares(t, m)-1) > shareTolerance {
		t.Errorf("decode shares sum to %v, want 1", sumShares(t, m))
	}
	// hostSum 10ms of a 167ms phase.
	if want := 10.0 / 167.0; math.Abs(m[bench.ShareHost]-want) > shareTolerance {
		t.Errorf("decode host share = %v, want %v", m[bench.ShareHost], want)
	}
	if m[bench.ShareDevice] < m[bench.ShareHost] {
		t.Errorf("device share %v below host share %v: the attribution is inverted",
			m[bench.ShareDevice], m[bench.ShareHost])
	}
}

// TestReportJSONStable is 017-D6: the record a regression check diffs must be
// the same bytes for the same measurement, and must survive a round trip.
func TestReportJSONStable(t *testing.T) {
	build := func() bench.Report {
		r := bench.NewRecorder(64)
		r.Step(bench.Step{Phase: bench.Prefill, Tokens: 8, Batch: 1, Host: 1, Submit: 2, Device: 30, Readback: 3})
		for i := range 5 {
			r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Batch: 1,
				Host: time.Duration(i + 1), Submit: 2, Device: time.Duration(20 + i), Readback: 1})
		}
		r.TTFT(36)
		return r.Report()
	}

	first, err := json.Marshal(build())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	second, err := json.Marshal(build())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(first) != string(second) {
		t.Errorf("two marshals of the same measurement differ:\n%s\n%s", first, second)
	}

	var back bench.Report
	if err := json.Unmarshal(first, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(back, build()) {
		t.Errorf("round trip changed the report:\ngot  %+v\nwant %+v", back, build())
	}

	// The bytes are the schema. A golden record rather than a scan for tag
	// names, because "host" and the rest appear twice for unrelated reasons --
	// once as a PhaseStats tag and once as a ShareOfStep key -- so a scan is
	// satisfied by the map key alone and would pass through a renamed field.
	// This is what a regression check diffs, so this is what is pinned.
	if string(first) != goldenReport {
		t.Errorf("JSON record changed shape.\ngot  %s\nwant %s", first, goldenReport)
	}
}

// goldenReport is TestReportJSONStable's measurement, encoded. Every stored
// baseline is read against this schema, so a change here is a break of all of
// them and has to be a deliberate one.
const goldenReport = `{"prefill":{"tokens_per_second":222222222.22222224,"host":{"p50_ns":1,"p90_ns":1,"p99_ns":1,"n":1},"submit":{"p50_ns":2,"p90_ns":2,"p99_ns":2,"n":1},"device":{"p50_ns":30,"p90_ns":30,"p99_ns":30,"n":1},"readback":{"p50_ns":3,"p90_ns":3,"p99_ns":3,"n":1},"share_of_step":{"device":0.8333333333333334,"host":0.027777777777777776,"readback":0.08333333333333333,"submit":0.05555555555555555},"steps":1,"tokens":8},"decode":{"tokens_per_second":35714285.71428571,"host":{"p50_ns":3,"p90_ns":5,"p99_ns":5,"n":5},"submit":{"p50_ns":2,"p90_ns":2,"p99_ns":2,"n":5},"device":{"p50_ns":22,"p90_ns":24,"p99_ns":24,"n":5},"readback":{"p50_ns":1,"p90_ns":1,"p99_ns":1,"n":5},"share_of_step":{"device":0.7857142857142857,"host":0.10714285714285714,"readback":0.03571428571428571,"submit":0.07142857142857142},"steps":5,"tokens":5},"ttft":{"p50_ns":36,"p90_ns":36,"p99_ns":36,"n":1},"steps":6,"dropped":0}`

// TestEmptyReportMarshals keeps the first run of a regression check readable: a
// phase with no steps must still produce a four-key breakdown, not null.
func TestEmptyReportMarshals(t *testing.T) {
	b, err := json.Marshal(bench.NewRecorder(4).Report())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// A nil map marshals as null, which the first run of a regression check
	// cannot diff against a later populated one. The breakdown is always four
	// keys, zero when the phase recorded nothing.
	const zeroShares = `"share_of_step":{"device":0,"host":0,"readback":0,"submit":0}`
	if !strings.Contains(string(b), zeroShares) {
		t.Errorf("empty report is missing %s:\n%s", zeroShares, b)
	}
	var back bench.Report
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(back.Decode.ShareOfStep) != 4 {
		t.Errorf("ShareOfStep round tripped to %v, want four zero keys", back.Decode.ShareOfStep)
	}
}

// TestARecorderIsWrittenByOneGoroutineAndReadByAnother is what
// specs/022-batched-serving.md made structural: one driver goroutine steps the
// whole batch and writes every request's recorder, and the request's own
// goroutine reads it.
//
// It fails under -race without the lock, which is how it was found: 017-D3's
// "one recorder per stepping goroutine" assumed the stepping goroutine and the
// reader are the same one, and under a batch they cannot be.
func TestARecorderIsWrittenByOneGoroutineAndReadByAnother(t *testing.T) {
	r := bench.NewRecorder(16)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 500 {
			r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Batch: 2, Device: time.Microsecond})
			r.TTFT(time.Millisecond)
		}
	}()
	for range 500 {
		_ = r.Report()
		_ = r.Steps()
	}
	<-done
	if got := r.Report().Decode.Steps; got == 0 {
		t.Error("nothing was recorded")
	}
}
