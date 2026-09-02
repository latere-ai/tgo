// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/sample"
	"github.com/latere-ai/tgo/weights"
)

// syntheticReport builds a bench.Report from steps whose four terms are known,
// so that every number the renderers print can be checked against arithmetic
// rather than against whatever a run happened to produce.
func syntheticReport(t *testing.T) bench.Report {
	t.Helper()
	r := bench.NewRecorder(256)
	r.TTFT(40 * time.Millisecond)
	r.Step(bench.Step{
		Phase: bench.Prefill, Tokens: 128, Batch: 1,
		Host: 1 * time.Millisecond, Submit: 2 * time.Millisecond,
		Device: 30 * time.Millisecond, Readback: 7 * time.Millisecond,
	})
	for i := 1; i <= 100; i++ {
		n := i
		r.Step(bench.Step{
			Phase: bench.Decode, Tokens: 1, Batch: 1,
			Host: time.Duration(n) * time.Microsecond, Submit: time.Duration(2*n) * time.Microsecond,
			Device: time.Duration(10*n) * time.Microsecond, Readback: time.Duration(3*n) * time.Microsecond,
		})
	}
	return r.Report()
}

// syntheticRecord is a whole benchRecord with every condition filled in, which
// is what 017-D4 requires of a record: hardware, model, precision, policy, the
// build and the prompt.
func syntheticRecord(t *testing.T) benchRecord {
	t.Helper()
	b := syntheticBuilder(t)
	rep, err := describe("models/synthetic", b, describeOptions{Context: 2048},
		hardware{Backend: "cpu", Device: "go", Vendor: "golang.design", CPUs: 8, MaxPoolBytes: 1 << 30},
		environment{Go: "go1.27.0", GOOS: "darwin", GOARCH: "arm64", Accel: "v0.0.0-20260824185610"})
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	return newRecord(
		conditionsOf(rep,
			samplingOf(sample.Policy{Temperature: 0.7, TopP: 0.9, TopK: 40}, 42, 128),
			promptFacts{Kind: "synthetic", Recipe: syntheticRecipe(128), RequestedTokens: 128,
				MeasuredTokens: 129, Text: syntheticPrompt(4)},
			8),
		batchAxis{Points: []int{1}, Note: singleBatchNote},
		[]batchPoint{{
			Batch: 1, Cold: coldFacts{Open: 900 * time.Millisecond, FirstToken: 940 * time.Millisecond},
			Resident: rep.Memory.ResidentBytes, Tokens: 100, Wall: 2 * time.Second,
			Report: syntheticReport(t),
		}},
	)
}

// TestMarkdownCarriesEveryCondition is 017-D4 as a test. A tokens-per-second
// figure without the hardware, the model, the precision and the policy is
// decoration, and the Go version, GOOS/GOARCH and the accel backend are what
// make two records comparable at all.
func TestMarkdownCarriesEveryCondition(t *testing.T) {
	r := syntheticRecord(t)
	var sb strings.Builder
	renderMarkdown(&sb, r)
	out := sb.String()

	for what, want := range map[string]string{
		"the model directory": r.Conditions.Model.Dir,
		"the architecture":    r.Conditions.Model.Architecture,
		"the precision":       r.Conditions.Precision.Chosen,
		"why that precision":  r.Conditions.Precision.Why,
		"the accel backend":   r.Conditions.Hardware.Backend,
		"the device name":     r.Conditions.Hardware.Device,
		"the Go version":      r.Conditions.Environment.Go,
		"GOOS":                r.Conditions.Environment.GOOS,
		"GOARCH":              r.Conditions.Environment.GOARCH,
		"the accel version":   r.Conditions.Environment.Accel,
		"the sampling policy": "temperature 0.7",
		"the seed":            "seed 42",
		"the prompt recipe":   r.Conditions.Prompt.Recipe,
		"the cache dtype":     r.Conditions.Memory.CacheDType,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the table does not carry %s (%q):\n%s", what, want, out)
		}
	}
}

// TestMarkdownReportsTheBreakdownAndNotOneNumber is 017-D1 and 017-D2: four
// terms and three percentiles, per phase.
func TestMarkdownReportsTheBreakdownAndNotOneNumber(t *testing.T) {
	r := syntheticRecord(t)
	var sb strings.Builder
	renderMarkdown(&sb, r)
	out := sb.String()

	for _, want := range []string{
		"| batch | tokens/s | steps | p50 | p90 | p99 | host | submit | device | readback |",
		"## Decode", "## Prefill", "## Time to first token",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the table is missing %q:\n%s", want, out)
		}
	}
	// The decode shares are printed, and they sum to a whole step.
	d := r.Batches[0].Report.Decode
	var sum float64
	for _, k := range []string{bench.ShareHost, bench.ShareSubmit, bench.ShareDevice, bench.ShareReadback} {
		sum += d.ShareOfStep[k]
		if !strings.Contains(out, percent(d.ShareOfStep[k])) {
			t.Errorf("the %s share (%s) is not in the table", k, percent(d.ShareOfStep[k]))
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("the shares sum to %v, want 1", sum)
	}
	// The three percentiles of a decode step differ, so a renderer that
	// collapsed them into a mean would be visible here.
	p50, p90, p99 := stepAt(d, quantileP50), stepAt(d, quantileP90), stepAt(d, quantileP99)
	if p50 >= p90 || p90 >= p99 {
		t.Errorf("p50=%v p90=%v p99=%v, want a tail", p50, p90, p99)
	}
	if want := d.Host.P90 + d.Submit.P90 + d.Device.P90 + d.Readback.P90; p90 != want {
		t.Errorf("the modeled step at p90 = %v, want the sum of the four terms %v", p90, want)
	}
}

// TestBatchAxisIsAnAxis is 017-D5. tgo does not batch, and the record says so
// in the axis rather than by omitting it: a scalar batch size would be
// indistinguishable from a framework that measured one point of a curve it has.
func TestBatchAxisIsAnAxis(t *testing.T) {
	r := syntheticRecord(t)
	b, err := encodeRecord(r)
	if err != nil {
		t.Fatalf("encodeRecord: %v", err)
	}
	var decoded struct {
		BatchAxis struct {
			Points []int  `json:"points"`
			Note   string `json:"note"`
		} `json:"batch_axis"`
		Batches []json.RawMessage `json:"batches"`
	}
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("the record does not round-trip: %v", err)
	}
	if len(decoded.BatchAxis.Points) != 1 || decoded.BatchAxis.Points[0] != 1 {
		t.Errorf("batch_axis.points = %v, want the array [1]", decoded.BatchAxis.Points)
	}
	if len(decoded.Batches) != len(decoded.BatchAxis.Points) {
		t.Errorf("%d batches for %d axis points", len(decoded.Batches), len(decoded.BatchAxis.Points))
	}
	for _, want := range []string{"008-scheduler.md", "one point"} {
		if !strings.Contains(decoded.BatchAxis.Note, want) {
			t.Errorf("the axis note does not mention %q: %q", want, decoded.BatchAxis.Note)
		}
	}
	var sb strings.Builder
	renderMarkdown(&sb, r)
	if !strings.Contains(sb.String(), "## Batch axis") || !strings.Contains(sb.String(), "008") {
		t.Errorf("the Markdown drops the batch axis:\n%s", sb.String())
	}
	// And the comparison tgo has not run is named rather than left out
	// (017 §4 rule 1).
	if !strings.Contains(sb.String(), "vLLM") {
		t.Error("the report does not say that no vLLM comparison is in it")
	}
}

// TestRecordEncodesStably is 017-D6: the JSON is what a regression check reads,
// so two encodings of equal records must be equal bytes or every check is a
// spurious diff.
func TestRecordEncodesStably(t *testing.T) {
	a, err := encodeRecord(syntheticRecord(t))
	if err != nil {
		t.Fatalf("encodeRecord: %v", err)
	}
	b, err := encodeRecord(syntheticRecord(t))
	if err != nil {
		t.Fatalf("encodeRecord: %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Error("two encodings of the same record differ")
	}
	if !bytes.HasSuffix(a, []byte("\n")) {
		t.Error("the record does not end in a newline, so a diff of it has no last line")
	}
	var m map[string]any
	if err := json.Unmarshal(a, &m); err != nil {
		t.Fatalf("the record is not valid JSON: %v", err)
	}
	if m["schema"] != recordSchema {
		t.Errorf("schema = %v, want %q: a checker that meets a record it cannot read has to know",
			m["schema"], recordSchema)
	}
	// The four terms survive the encoding as nanosecond integers, which is what
	// a check compares.
	got := m["batches"].([]any)[0].(map[string]any)["report"].(map[string]any)
	decode := got["decode"].(map[string]any)
	if decode["host"].(map[string]any)["p50_ns"].(float64) == 0 {
		t.Error("the decode host p50 encoded as zero")
	}
}

func TestDescribePolicy(t *testing.T) {
	greedy := describePolicy(samplingOf(sample.Policy{}, 7, 32))
	if !strings.Contains(greedy, "greedy") || !strings.Contains(greedy, "seed 7") {
		t.Errorf("greedy policy = %q", greedy)
	}
	full := describePolicy(samplingOf(sample.Policy{
		Temperature: 0.8, TopK: 40, TopP: 0.95, RepetitionPenalty: 1.1}, 1, 16))
	for _, want := range []string{"temperature 0.8", "top-k 40", "top-p 0.95", "repetition penalty 1.1"} {
		if !strings.Contains(full, want) {
			t.Errorf("policy line %q is missing %q", full, want)
		}
	}
}

func TestWrapNoteAndAxisPoints(t *testing.T) {
	for line := range strings.SplitSeq(wrapNote(singleBatchNote), "\n") {
		if len(line) > 92 {
			t.Errorf("a wrapped line is %d columns: %q", len(line), line)
		}
	}
	if strings.Fields(wrapNote(singleBatchNote))[0] != strings.Fields(singleBatchNote)[0] {
		t.Error("wrapping changed the note's first word")
	}
	if got := axisPoints(nil); got != "none" {
		t.Errorf("axisPoints(nil) = %q, want none", got)
	}
	if got := axisPoints([]int{1, 2, 4}); got != "1, 2, 4" {
		t.Errorf("axisPoints = %q", got)
	}
}

// TestDroppedObservationsAreReported pins the one case where the percentiles do
// not describe the run: a recorder that filled reports a prefix, and the table
// has to say so rather than print numbers that look complete.
func TestDroppedObservationsAreReported(t *testing.T) {
	r := bench.NewRecorder(2)
	for range 5 {
		r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Host: time.Millisecond})
	}
	rec := syntheticRecord(t)
	rec.Batches[0].Report = r.Report()
	var sb strings.Builder
	renderMarkdown(&sb, rec)
	if !strings.Contains(sb.String(), "dropped 3 observations") {
		t.Errorf("a truncated report is not flagged:\n%s", sb.String())
	}
}

func TestHumanFormats(t *testing.T) {
	for _, tc := range []struct {
		in   int64
		want string
	}{{0, "0 B"}, {512, "512 B"}, {1024, "1.00 KiB"}, {1 << 30, "1.00 GiB"}, {-2048, "-2.00 KiB"}} {
		if got := weights.HumanBytes(tc.in); got != tc.want {
			t.Errorf("weights.HumanBytes(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, tc := range []struct {
		in   int64
		want string
	}{{7, "7"}, {1500, "1.50K"}, {596_049_920, "596.05M"}, {4_000_000_000, "4.00B"}, {-1500, "-1.50K"}} {
		if got := humanCount(tc.in); got != tc.want {
			t.Errorf("humanCount(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, tc := range []struct {
		in   time.Duration
		want string
	}{
		{0, "0"}, {500 * time.Nanosecond, "500ns"}, {1500 * time.Nanosecond, "1.50µs"},
		{2500 * time.Microsecond, "2.50ms"}, {90 * time.Second, "90.00s"},
	} {
		if got := humanDuration(tc.in); got != tc.want {
			t.Errorf("humanDuration(%v) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if got := percent(0.1234); got != "12.34%" {
		t.Errorf("percent = %q", got)
	}
}

// TestBreakdownIsReportedAsMissingRatherThanAsZeros is the arm every real run
// takes today, and the one no other test here reaches.
//
// A session opened without a tgo.WithRecorder has a time to first token and no
// four-way breakdown. A Report from a recorder nothing wrote to marshals as a
// full set of fields holding zero times and zero shares, which reads as a
// measurement of an impossibly fast model. The record says the terms are
// missing, and names the option that supplies them.
func TestBreakdownIsReportedAsMissingRatherThanAsZeros(t *testing.T) {
	r := bench.NewRecorder(8)
	r.TTFT(30 * time.Millisecond)

	rec := newRecord(syntheticRecord(t).Conditions, batchAxis{Points: []int{1}, Note: singleBatchNote},
		[]batchPoint{{Batch: 1, Tokens: 12, Wall: 2 * time.Second, Report: r.Report()}})
	if rec.Breakdown.Available {
		t.Fatal("a report holding only a time to first token claims to carry the breakdown")
	}
	for _, want := range []string{"017-D1", "WithRecorder", "table of zeros"} {
		if !strings.Contains(rec.Breakdown.Note, want) {
			t.Errorf("the note does not mention %q: %q", want, rec.Breakdown.Note)
		}
	}
	var sb strings.Builder
	renderMarkdown(&sb, rec)
	out := sb.String()
	if !strings.Contains(out, "## Where the time went") || !strings.Contains(out, "007-engine.md") {
		t.Errorf("the table does not say the breakdown is missing:\n%s", out)
	}
	// And it prints no share table, rather than one of zeros.
	if strings.Contains(out, "| host | submit | device | readback |") {
		t.Errorf("a report with no breakdown printed the breakdown table anyway:\n%s", out)
	}
	// The time to first token survives: it is the one thing that was measured.
	if !strings.Contains(out, "## Time to first token") || !strings.Contains(out, "30.00ms") {
		t.Errorf("the one measurement that exists is missing:\n%s", out)
	}
}

// TestBreakdownRefusesAStepOfFourZeros is why availability is a question about
// time and not about a step count. A step recorded with four zero durations
// passes any count test and renders as a share table summing to nothing.
func TestBreakdownRefusesAStepOfFourZeros(t *testing.T) {
	r := bench.NewRecorder(8)
	r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Batch: 1})
	rep := r.Report()
	if rep.Decode.Steps != 1 {
		t.Fatalf("the recorder holds %d steps, want the one that was written", rep.Decode.Steps)
	}
	if hasBreakdown(rep) {
		t.Error("a step whose four terms are all zero was reported as a breakdown")
	}
	// One non-zero term is a measurement, however small.
	r.Reset()
	r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Batch: 1, Readback: time.Nanosecond})
	if !hasBreakdown(r.Report()) {
		t.Error("a step with a measured readback was reported as no breakdown")
	}
}

// TestNewRecordDividesTheThroughput pins the field a struct literal forgets.
// The wall-clock rate is the only throughput a process outside the engine can
// measure, and a record that carried a zero would report a model that produced
// nothing.
func TestNewRecordDividesTheThroughput(t *testing.T) {
	rec := newRecord(conditions{}, batchAxis{Points: []int{1}},
		[]batchPoint{{Batch: 1, Tokens: 50, Wall: 2 * time.Second}})
	if got := rec.Batches[0].TokensPerSecond; got != 25 {
		t.Errorf("tokens/second = %v, want 50 over two seconds", got)
	}
	if rec.Schema != recordSchema {
		t.Errorf("schema = %q, want %q", rec.Schema, recordSchema)
	}
	// A run that produced nothing reports no rate rather than a division.
	empty := newRecord(conditions{}, batchAxis{}, []batchPoint{{Batch: 1, Wall: time.Second}})
	if got := empty.Batches[0].TokensPerSecond; got != 0 {
		t.Errorf("a run with no tokens reported %v tokens/second", got)
	}
	// The constructor does not write through to the caller's points.
	points := []batchPoint{{Batch: 1, Tokens: 4, Wall: time.Second}}
	newRecord(conditions{}, batchAxis{}, points)
	if points[0].TokensPerSecond != 0 {
		t.Error("newRecord mutated the slice it was given")
	}
}

// TestEveryUnmeasuredAxisIsNamed is 017-D6 applied to the axes this build
// cannot measure.
//
// specs/017-benchmarks.md §3 names six measurements. Three of them are not in
// this record -- the four-term breakdown, the batch curve, and the plan compile
// time with its cache hit rate -- and each is carried as a field holding the
// reason. A check that cannot tell "not measured here" from "no such axis"
// silently stops gating the axis the day it disappears, which is the drift
// 017-D6 exists to catch.
// TestAnUnmeasuredAxisNamesAGapThatIsStillOpen is the other half. A note has to
// cite a spec, and it also has to be true: a note naming a gap somebody closed
// sends a reader to fix something that works, and nothing here would have said
// so -- the note for the breakdown claimed specs/007-engine.md §1 exports no
// way to set a bench.Recorder for three waves after WithRecorder shipped, past
// a test that only checked it cited a spec.
func TestAnUnmeasuredAxisNamesAGapThatIsStillOpen(t *testing.T) {
	// The API each note says is missing, against what the engine exports. A
	// claim about a name that exists is a claim that has expired.
	for what, note := range map[string]string{
		"the breakdown":  noBreakdownNote,
		"the plan cache": noPlanStatsNote,
	} {
		if strings.Contains(note, "no way to set") || strings.Contains(note, "no way to read") {
			t.Errorf("the note for %s says the engine exports no way to set or read a "+
				"recorder, and tgo.WithRecorder does exactly that: %q", what, note)
		}
	}
	// WithRecorder is the name the corrected note points a reader at, so a
	// rename that leaves the note behind is caught here rather than by a reader.
	if !strings.Contains(noBreakdownNote, "WithRecorder") {
		t.Error("the breakdown note does not name the option a caller has to pass, " +
			"which is the whole of what a reader who sees no breakdown needs")
	}
}

func TestEveryUnmeasuredAxisIsNamed(t *testing.T) {
	rec := newRecord(conditions{}, batchAxis{Points: []int{1}, Note: singleBatchNote}, nil)
	for what, note := range map[string]string{
		"the breakdown":  rec.Breakdown.Note,
		"the plan cache": rec.PlanStats.Note,
		"the comparison": rec.Comparison.Note,
		"the batch axis": rec.BatchAxis.Note,
	} {
		if note == "" {
			t.Errorf("%s is absent from the record with no reason given", what)
		}
		if !strings.Contains(note, "specs/") && !strings.Contains(note, "017-D") {
			t.Errorf("the note for %s cites no spec: %q", what, note)
		}
	}
	// The reasons survive the encoding a regression check reads.
	b, err := encodeRecord(rec)
	if err != nil {
		t.Fatalf("encodeRecord: %v", err)
	}
	for _, key := range []string{`"breakdown"`, `"plan_stats"`, `"comparison"`, `"batch_axis"`} {
		if !strings.Contains(string(b), key) {
			t.Errorf("the record has no %s field", key)
		}
	}
	// And a reader of the table meets all four.
	var sb strings.Builder
	renderMarkdown(&sb, rec)
	for _, heading := range []string{"## Where the time went", "## Batch axis",
		"## Plan compilation", "## Comparisons"} {
		if !strings.Contains(sb.String(), heading) {
			t.Errorf("the table has no %q section:\n%s", heading, sb.String())
		}
	}
	// 007-D2 is the decision the plan numbers would settle, so the note names it.
	if !strings.Contains(rec.PlanStats.Note, "007-D2") {
		t.Errorf("the plan note does not say what the missing numbers would decide: %q", rec.PlanStats.Note)
	}
}

// TestBreakdownCountsEveryPhaseAndEveryTerm is the other half of what
// availability means.
//
// 017-D1's breakdown is four terms over two phases, and a check that read only
// some of them would report a measured run as carrying nothing: an engine that
// instruments its prefill and not its decode has measured the breakdown, and so
// has one whose only non-zero term is the readback -- which is the term
// specs/007-engine.md 007-D4 says tgo exists to measure. Each of the eight
// cells is asserted on its own, so a reader dropped from [hasBreakdown] is a
// failure rather than a record that quietly says "not measured".
func TestBreakdownCountsEveryPhaseAndEveryTerm(t *testing.T) {
	for _, phase := range []bench.Phase{bench.Prefill, bench.Decode} {
		for name, term := range map[string]func(*bench.Step){
			"host":     func(s *bench.Step) { s.Host = time.Millisecond },
			"submit":   func(s *bench.Step) { s.Submit = time.Millisecond },
			"device":   func(s *bench.Step) { s.Device = time.Millisecond },
			"readback": func(s *bench.Step) { s.Readback = time.Millisecond },
		} {
			t.Run(phase.String()+"/"+name, func(t *testing.T) {
				step := bench.Step{Phase: phase, Tokens: 1, Batch: 1}
				term(&step)
				r := bench.NewRecorder(8)
				r.Step(step)
				if !hasBreakdown(r.Report()) {
					t.Errorf("a %v step whose %s was measured was reported as no breakdown", phase, name)
				}
			})
		}
	}
}
