// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/sample"
)

// recordSchema names the shape of the JSON record.
//
// 017-D6 makes the JSON what a regression check reads, which makes it an
// interface with a version: a checker that meets a record it does not
// understand has to be able to say so rather than compare fields that moved.
const recordSchema = "tgo.bench/1"

// samplingFacts is the policy every number was produced under.
//
// 017-D4: different frameworks default to different policies, and comparing
// greedy against top-p is comparing nothing. Greedy is carried as its own
// field, and not left to be inferred from a temperature of zero, because that
// inference is exactly the one a reader of a comparison table gets wrong.
type samplingFacts struct {
	Greedy            bool    `json:"greedy"`
	Temperature       float32 `json:"temperature"`
	TopK              int     `json:"top_k"`
	TopP              float32 `json:"top_p"`
	RepetitionPenalty float32 `json:"repetition_penalty"`
	PresencePenalty   float32 `json:"presence_penalty"`
	FrequencyPenalty  float32 `json:"frequency_penalty"`
	PenaltyWindow     int     `json:"penalty_window"`
	Seed              uint64  `json:"seed"`
	MaxTokens         int     `json:"max_tokens"`
}

// samplingOf records a policy as measured conditions.
func samplingOf(p sample.Policy, seed uint64, maxTokens int) samplingFacts {
	return samplingFacts{
		Greedy: p.Temperature == 0, Temperature: p.Temperature, TopK: p.TopK, TopP: p.TopP,
		RepetitionPenalty: p.RepetitionPenalty, PresencePenalty: p.PresencePenalty,
		FrequencyPenalty: p.FrequencyPenalty, PenaltyWindow: p.PenaltyWindow,
		Seed: seed, MaxTokens: maxTokens,
	}
}

// promptFacts is what was sent, exactly.
//
// specs/017-benchmarks.md §4 rule 4: prompt length decides the prefill/decode
// ratio, which decides everything, so a record that does not carry the prompt
// cannot be reproduced. Requested and measured lengths are both here because
// they differ: a synthetic prompt is built from a word repeated, and only the
// tokenizer knows how many tokens that came to.
type promptFacts struct {
	Kind            string `json:"kind"`
	Recipe          string `json:"recipe"`
	RequestedTokens int    `json:"requested_tokens"`
	MeasuredTokens  int    `json:"measured_tokens"`
	Text            string `json:"text"`
}

// conditions is 017-D4 in one struct: the model, the precision, the memory, the
// machine, the build, the policy and the prompt. Every number in the record
// below is qualified by it, and a number printed without it is decoration.
type conditions struct {
	Model       modelFacts     `json:"model"`
	Precision   precisionFacts `json:"precision"`
	Memory      memoryFacts    `json:"memory"`
	Hardware    hardware       `json:"hardware"`
	Environment environment    `json:"environment"`
	Sampling    samplingFacts  `json:"sampling_policy"`
	Prompt      promptFacts    `json:"prompt"`
	WarmupSteps int            `json:"warmup_steps"`
}

// conditionsOf assembles the conditions from a model description and a request.
func conditionsOf(r modelReport, s samplingFacts, p promptFacts, warmup int) conditions {
	return conditions{
		Model: r.Model, Precision: r.Precision, Memory: r.Memory,
		Hardware: r.Hardware, Environment: r.Environment,
		Sampling: s, Prompt: p, WarmupSteps: warmup,
	}
}

// coldFacts is the cold measurement: what a user waits through before the first
// token of the first request.
//
// specs/017-benchmarks.md §3 keeps it apart from the warm one because they are
// different products. Open is the model load and the plan compile; FirstToken
// is the whole wait, Open included.
type coldFacts struct {
	Open       time.Duration `json:"open_ns"`
	FirstToken time.Duration `json:"first_token_ns"`
}

// batchPoint is one point on the batch axis.
//
// TokensPerSecond is the wall clock's, not bench.PhaseStats's: it is the tokens
// the engine reported over the time the command line waited, which is the one
// throughput a caller outside the engine can measure. It is reported beside the
// breakdown and never instead of it -- 017-D1 exists because this number alone
// cannot say whether a regression belongs to tgo or to accel.
type batchPoint struct {
	Batch           int           `json:"batch"`
	Cold            coldFacts     `json:"cold"`
	Resident        int64         `json:"resident_bytes"`
	Tokens          int           `json:"generated_tokens"`
	Wall            time.Duration `json:"wall_ns"`
	TokensPerSecond float64       `json:"tokens_per_second"`
	Report          bench.Report  `json:"report"`
}

// batchAxis is 017-D5: the axis, and why it has the points it has.
//
// Points is an array even when it holds one number. A record that reported
// `"batch": 1` would have dropped the axis, which is the silent omission D5
// forbids: a reader could not tell a framework that measured one batch size
// from one that has no batching at all.
type batchAxis struct {
	Points []int  `json:"points"`
	Note   string `json:"note"`
}

// singleBatchNote is why the curve is one point.
const singleBatchNote = "tgo does not batch yet. specs/008-scheduler.md is drafted and unbuilt, " +
	"so exactly one sequence is ever in flight and this axis has one point. " +
	"017-D5 asks for a curve because 008 §1 shows the throughput ceiling falls as context grows, " +
	"and a single batch size hides that shape. The axis is reported with one point, rather than " +
	"dropped, so that the missing shape is visible in the record rather than absent from it."

// comparisonNote is 017 §4 rule 1 applied to a record that has no rival in it.
const comparisonNote = "No vLLM or sglang row is in this record. specs/017-benchmarks.md §3 compares " +
	"against both on the same model, hardware, prompts and policy, and specs/011-sequencing.md schedules " +
	"that for Wave 5. Publishing tgo's own numbers under a heading that implies a comparison would be the " +
	"decoration 017-D4 refuses, so the comparison is named as missing."

// breakdownFacts says whether the record carries 017-D1's four terms, and if
// not, why.
//
// A missing breakdown is stated rather than rendered as zeros. A Report from a
// recorder nothing wrote to marshals as a full set of fields holding zero
// times and zero shares, which reads as a measurement of a very fast model
// instead of as no measurement at all.
type breakdownFacts struct {
	Available bool   `json:"available"`
	Note      string `json:"note"`
}

// noBreakdownNote is the reason the four terms can be missing.
const noBreakdownNote = "The host/submit/device/readback breakdown is not in this record. " +
	"017-D1 makes that breakdown the deliverable, and the engine of specs/007-engine.md " +
	"instruments its own decode loop with a bench.Recorder that its public surface (§1) exports " +
	"no way to set and no way to read, so a process outside that package cannot obtain the four " +
	"terms. What is left is the wall-clock throughput and the time to first token, and a " +
	"throughput on its own is precisely the number 017-D1 says cannot attribute a regression to " +
	"tgo or to accel. It is reported as missing rather than as a table of zeros."

// noPlanStatsNote is the other row of specs/017-benchmarks.md §3 this record
// cannot carry.
//
// §3 asks for the plan compile time per bucket and the plan cache hit rate,
// because they are what says whether 007-D2's bucket set is right. A Model owns
// the plan cache and its public surface (specs/007-engine.md §1) reports
// neither, so the cost of a cache miss reaches this process only folded into
// the cold time to first token, where it cannot be separated from the model
// load. Named rather than dropped, for the same reason the batch axis is.
const noPlanStatsNote = "Plan compile time per bucket and plan cache hit rate are not in this record. " +
	"specs/017-benchmarks.md §3 asks for both because they are the evidence for 007-D2's bucket set, " +
	"and specs/007-engine.md §1 exports neither: the plan cache is unexported and a Model reports no " +
	"statistics about it. What survives is the cold time to first token, which includes the first " +
	"compile of every bucket the request touched along with the model load, and does not separate them."

// benchRecord is the whole of one `tgo bench` run: the conditions, the axis,
// and a measurement at each point of it.
//
// The three note fields are the axes specs/017-benchmarks.md §3 asks for that
// this build cannot measure. They are fields rather than omissions because
// 017-D6 makes this record what a regression check reads, and a check that
// cannot tell "this axis was not measured" from "this axis does not exist"
// silently stops gating the axis the day it disappears.
type benchRecord struct {
	Schema     string         `json:"schema"`
	Conditions conditions     `json:"conditions"`
	BatchAxis  batchAxis      `json:"batch_axis"`
	Breakdown  breakdownFacts `json:"breakdown"`
	PlanStats  breakdownFacts `json:"plan_stats"`
	Comparison breakdownFacts `json:"comparison"`
	Batches    []batchPoint   `json:"batches"`
}

// newRecord assembles a record and derives the fields that are properties of
// the measurement rather than of the request.
//
// A constructor rather than a struct literal, because the derived fields are
// exactly the ones a literal forgets: a throughput nobody divided reads as
// zero tokens per second, and a breakdown nobody classified reads as absent.
// Both renderers read this one struct, so a field the caller left unset is a
// wrong number in the table and in the record at once -- and in whatever reads
// the record after that. 017-D6 asks for a gate over it that this tree does not
// have: internal/covercheck is the model named, and nothing is its benchmark
// counterpart. See this package's reported discrepancies.
func newRecord(c conditions, axis batchAxis, points []batchPoint) benchRecord {
	measured := make([]batchPoint, len(points))
	copy(measured, points)

	available := false
	for i := range measured {
		p := &measured[i]
		if p.Tokens > 0 && p.Wall > 0 {
			p.TokensPerSecond = float64(p.Tokens) / p.Wall.Seconds()
		}
		available = available || hasBreakdown(p.Report)
	}

	breakdown := breakdownFacts{Available: available}
	if !available {
		breakdown.Note = noBreakdownNote
	}
	return benchRecord{
		Schema: recordSchema, Conditions: c, BatchAxis: axis,
		Breakdown:  breakdown,
		PlanStats:  breakdownFacts{Note: noPlanStatsNote},
		Comparison: breakdownFacts{Note: comparisonNote},
		Batches:    measured,
	}
}

// hasBreakdown reports whether a report carries 017-D1's four terms rather than
// only a time to first token.
//
// Non-zero time rather than a non-zero step count: a step recorded with four
// zero durations passes a count test and renders as a share table of zeros,
// which reads as a measurement of an impossibly fast model rather than as no
// measurement at all. What the table asks is whether the four terms hold
// anything, so that is what is tested.
func hasBreakdown(r bench.Report) bool {
	for _, s := range []bench.PhaseStats{r.Prefill, r.Decode} {
		for _, q := range []bench.Quantiles{s.Host, s.Submit, s.Device, s.Readback} {
			if q.P50 > 0 || q.P90 > 0 || q.P99 > 0 {
				return true
			}
		}
	}
	return false
}

// encodeRecord marshals the record a regression check reads.
//
// Indented, because a record that lands in a repository is read in a diff, and
// a one-line JSON document diffs as one changed line whatever moved. The
// encoding is stable for equal records: struct fields marshal in declaration
// order and encoding/json sorts the keys of the one map in the tree,
// bench.PhaseStats.ShareOfStep.
func encodeRecord(r benchRecord) ([]byte, error) {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encoding the benchmark record: %w", err)
	}
	return append(b, '\n'), nil
}

// renderMarkdown writes the table a person reads, from the same struct the JSON
// is encoded from.
//
// One struct and two renderers, rather than two structs: 017-D4 requires the
// conditions to travel with every number, and a Markdown table assembled from
// its own fields would drift from the record a regression check reads, which is
// the drift nobody notices until the two disagree.
func renderMarkdown(w io.Writer, r benchRecord) {
	c := r.Conditions
	fmt.Fprintf(w, "# tgo bench: %s\n\n", c.Model.Architecture)

	fmt.Fprint(w, "## Conditions\n\n")
	fmt.Fprint(w, "| what | value |\n| --- | --- |\n")
	row := func(k, format string, args ...any) {
		fmt.Fprintf(w, "| %s | %s |\n", k, fmt.Sprintf(format, args...))
	}
	row("model", "`%s`", c.Model.Dir)
	row("architecture", "%s, %s parameters, %d layers, %d heads over %d kv heads, head_dim %d",
		c.Model.Architecture, humanCount(c.Model.Parameters), c.Model.Layers,
		c.Model.Heads, c.Model.KVHeads, c.Model.HeadDim)
	row("precision", "**%s** — %s", c.Precision.Chosen, c.Precision.Why)
	row("context", "%d positions, kv cache %s (%s)", c.Memory.Context,
		humanBytes(c.Memory.KVBytes), c.Memory.CacheDType)
	row("resident", "%s (weights %s plus cache; activations and host heap excluded)",
		humanBytes(c.Memory.ResidentBytes), humanBytes(c.Memory.WeightBytes))
	row("hardware", "%s — %s (%s), software=%t, unified memory=%t, %d cpus",
		c.Hardware.Backend, c.Hardware.Device, c.Hardware.Vendor,
		c.Hardware.Software, c.Hardware.UnifiedMemory, c.Hardware.CPUs)
	row("build", "%s %s/%s, accel %s", c.Environment.Go, c.Environment.GOOS,
		c.Environment.GOARCH, c.Environment.Accel)
	row("sampling policy", "%s", describePolicy(c.Sampling))
	row("prompt", "%s, %d tokens requested, %d measured — %s", c.Prompt.Kind,
		c.Prompt.RequestedTokens, c.Prompt.MeasuredTokens, c.Prompt.Recipe)
	row("warm-up", "%d steps run and discarded before measuring (§4 rule 3)", c.WarmupSteps)

	fmt.Fprint(w, "\n## Throughput\n\n")
	fmt.Fprint(w, "| batch | generated tokens | wall | tokens/s |\n| ---: | ---: | ---: | ---: |\n")
	for _, p := range r.Batches {
		fmt.Fprintf(w, "| %d | %d | %s | %.2f |\n", p.Batch, p.Tokens,
			humanDuration(p.Wall), p.TokensPerSecond)
	}

	if !r.Breakdown.Available {
		fmt.Fprintf(w, "\n## Where the time went\n\n%s\n", wrapNote(r.Breakdown.Note))
	} else {
		fmt.Fprint(w, "\n## Decode\n\n")
		fmt.Fprint(w, "Every measurement is the host/submit/device/readback breakdown (017-D1), "+
			"reported at percentiles rather than means (017-D2).\n\n")
		phaseTable(w, r.Batches, func(p batchPoint) bench.PhaseStats { return p.Report.Decode })

		fmt.Fprint(w, "\n## Prefill\n\n")
		phaseTable(w, r.Batches, func(p batchPoint) bench.PhaseStats { return p.Report.Prefill })
	}

	fmt.Fprint(w, "\n## Time to first token\n\n")
	fmt.Fprint(w, "| batch | cold: open | cold: first token | warm p50 | warm p90 | warm p99 | n |\n")
	fmt.Fprint(w, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, p := range r.Batches {
		t := p.Report.TTFT
		fmt.Fprintf(w, "| %d | %s | %s | %s | %s | %s | %d |\n", p.Batch,
			humanDuration(p.Cold.Open), humanDuration(p.Cold.FirstToken),
			humanDuration(t.P50), humanDuration(t.P90), humanDuration(t.P99), t.N)
	}
	fmt.Fprint(w, "\nCold includes the model load and the plan compile; warm is prefill only. "+
		"They are different products (017 §3).\n")

	fmt.Fprintf(w, "\n## Batch axis\n\n%s\n", wrapNote(r.BatchAxis.Note))
	fmt.Fprintf(w, "\nPoints measured: %s.\n", axisPoints(r.BatchAxis.Points))

	fmt.Fprintf(w, "\n## Plan compilation\n\n%s\n", wrapNote(r.PlanStats.Note))

	fmt.Fprintf(w, "\n## Comparisons\n\n%s\n", wrapNote(r.Comparison.Note))

	for _, p := range r.Batches {
		if p.Report.Dropped > 0 {
			fmt.Fprintf(w, "\n> Batch %d dropped %d observations: the recorder filled and the "+
				"percentiles above describe the start of the run rather than the run.\n",
				p.Batch, p.Report.Dropped)
		}
	}
}

// phaseTable writes one phase's row per batch point.
func phaseTable(w io.Writer, points []batchPoint, pick func(batchPoint) bench.PhaseStats) {
	fmt.Fprint(w, "| batch | tokens/s | steps | p50 | p90 | p99 | host | submit | device | readback |\n")
	fmt.Fprint(w, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, p := range points {
		s := pick(p)
		fmt.Fprintf(w, "| %d | %.2f | %d | %s | %s | %s | %s | %s | %s | %s |\n",
			p.Batch, s.TokensPerSecond, s.Steps,
			humanDuration(stepAt(s, quantileP50)), humanDuration(stepAt(s, quantileP90)),
			humanDuration(stepAt(s, quantileP99)),
			percent(s.ShareOfStep[bench.ShareHost]), percent(s.ShareOfStep[bench.ShareSubmit]),
			percent(s.ShareOfStep[bench.ShareDevice]), percent(s.ShareOfStep[bench.ShareReadback]))
	}
}

// The three percentiles a step is reported at.
const (
	quantileP50 = iota
	quantileP90
	quantileP99
)

// stepAt sums the four terms at one percentile.
//
// It is a sum of four independent distributions and not a percentile of the
// step, because bench.PhaseStats holds the four terms separately: a step's host
// time and its device time are separate samples, and pairing them would report
// a step that never happened. The sum is the modeled step at that percentile,
// which is what 017-D1's breakdown is for, and it is labeled as the step in the
// table because every column beside it is one of its four terms.
func stepAt(s bench.PhaseStats, q int) time.Duration {
	pick := func(v bench.Quantiles) time.Duration {
		switch q {
		case quantileP90:
			return v.P90
		case quantileP99:
			return v.P99
		default:
			return v.P50
		}
	}
	return pick(s.Host) + pick(s.Submit) + pick(s.Device) + pick(s.Readback)
}

// describePolicy renders the sampling policy as one line, greedy first.
func describePolicy(s samplingFacts) string {
	if s.Greedy {
		return fmt.Sprintf("greedy (temperature 0), seed %d, max %d tokens", s.Seed, s.MaxTokens)
	}
	parts := []string{fmt.Sprintf("temperature %g", s.Temperature)}
	if s.TopK > 0 {
		parts = append(parts, fmt.Sprintf("top-k %d", s.TopK))
	}
	if s.TopP > 0 {
		parts = append(parts, fmt.Sprintf("top-p %g", s.TopP))
	}
	if s.RepetitionPenalty != 0 && s.RepetitionPenalty != 1 {
		parts = append(parts, fmt.Sprintf("repetition penalty %g", s.RepetitionPenalty))
	}
	parts = append(parts, fmt.Sprintf("seed %d", s.Seed), fmt.Sprintf("max %d tokens", s.MaxTokens))
	return strings.Join(parts, ", ")
}

// axisPoints renders the batch sizes measured.
func axisPoints(points []int) string {
	if len(points) == 0 {
		return "none"
	}
	s := make([]string, len(points))
	for i, p := range points {
		s[i] = fmt.Sprint(p)
	}
	return strings.Join(s, ", ")
}

// wrapNote folds a long note at 92 columns, so that the Markdown a milestone
// copies into specs/011-sequencing.md §5 reads as prose rather than as one long
// line.
func wrapNote(s string) string {
	const width = 92
	var b strings.Builder
	col := 0
	for i, word := range strings.Fields(s) {
		switch {
		case i == 0:
		case col+1+len(word) > width:
			b.WriteByte('\n')
			col = 0
		default:
			b.WriteByte(' ')
			col++
		}
		b.WriteString(word)
		col += len(word)
	}
	return b.String()
}
