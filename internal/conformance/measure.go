// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"fmt"
	"math"
	"strings"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	"github.com/latere-ai/tgo/bench"
)

// Measurements are the five numbers of specs/010-conformance.md §3: what tgo
// reports back to accel, measured rather than asserted, and re-measured each
// release.
//
// Each is a question accel cannot answer about itself, because each needs a
// real model -- a 151936-wide vocabulary, trained weights with outlier
// channels, a graph of some five hundred nodes -- and accel's own tests have
// none of those.
//
// Every field is a pointer, and a nil one means *not measured*. That is not the
// same as zero and the difference matters here: two of the five come from an
// engine being built in parallel, and a report that printed 0.00 for a number
// nobody has taken yet would be a false measurement rather than a missing one.
type Measurements struct {
	Divergence   *Divergence   `json:"divergence,omitempty"`
	Readback     *Readback     `json:"readback,omitempty"`
	Quantization *Quantization `json:"quantization,omitempty"`
	Compile      *Compile      `json:"compile,omitempty"`
	Transient    *Transient    `json:"transient,omitempty"`
}

// Divergence is where a greedy CPU run and a greedy Metal run part, over one
// prompt.
//
// §3: it decides whether "the same result on both backends" is a claim tgo can
// make. Reduction order differs between the backends, so the two runs are not
// required to be identical -- they are required to differ, when they differ, on
// a decision that was close. The margin is what says which happened.
type Divergence struct {
	// Prompt is what was run, so the number can be reproduced.
	Prompt string `json:"prompt"`

	// Tokens is how many were generated on each backend.
	Tokens int `json:"tokens"`

	// Index is the first token position the two runs disagree at, or -1 when
	// they agree for the whole run.
	Index int `json:"index"`

	// TopTwoMargin is the logit gap at Index between the token the CPU run
	// took and the one it ranked second, on the CPU run, which 010-D5 makes
	// the reference.
	//
	// It is the top-two margin and not the difference between the backends'
	// logits. §3 asks for "the logit gap there" and the question it decides
	// is how close the decision was: a margin near zero is reduction-order
	// noise flipping a near-tie, and a wide one is a backend computing
	// something else.
	TopTwoMargin float64 `json:"top_two_margin"`
}

// FirstDifference is the index of the first token two runs disagree at, or -1.
//
// Runs of different length whose common prefix agrees differ at the shorter
// one's end: one of them stopped and the other did not, which is a divergence
// in the decode and not a shorter answer.
func FirstDifference(a, b []int32) int {
	n := min(len(b), len(a))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// TopTwoMargin is the gap between the largest and second-largest logit.
//
// It panics on fewer than two logits: a vocabulary with one token in it is a
// binding mistake, not a run whose margin is infinite.
func TopTwoMargin(logits []float32) float64 {
	if len(logits) < 2 {
		panic("conformance: TopTwoMargin over fewer than two logits")
	}
	first, second := float64(logits[0]), float64(logits[1])
	if second > first {
		first, second = second, first
	}
	for _, l := range logits[2:] {
		v := float64(l)
		if v > first {
			first, second = v, first
		} else if v > second {
			second = v
		}
	}
	return first - second
}

// Value renders the measurement for the report.
func (d Divergence) Value() string {
	if d.Index < 0 {
		return fmt.Sprintf("no divergence over %d tokens", d.Tokens)
	}
	return fmt.Sprintf("token %d of %d, at a top-two margin of %.3g",
		d.Index, d.Tokens, d.TopTwoMargin)
}

// Readback is the readback share of a decode step: what C6 costs, in one
// number.
//
// §3 names it as the size of C6, because sampling on the host means every token
// carries the whole logits vector back across the bus, and no amount of kernel
// work removes a term that is bus time.
type Readback struct {
	// Vocab is the vocabulary the logits are over. The share is meaningless
	// without it: the readback is one float per token of the vocabulary.
	Vocab int `json:"vocab"`

	// Bytes is one token's logits readback.
	Bytes int `json:"bytes"`

	// Share is the readback's fraction of a decode step, from
	// [bench.PhaseStats.ShareOfStep].
	Share float64 `json:"share"`

	// Median is the readback term's p50 over the run.
	Median time.Duration `json:"median_ns"`

	// Steps is how many decode steps the share is over, because a share over
	// four steps is not a share.
	Steps int `json:"steps"`
}

// ReadbackFrom takes the decode phase's readback share out of a bench report.
//
// bench already partitions a step into host, submit, device and readback and
// reports the four as fractions (017-D1), so the measurement §3 asks for is a
// projection of a report tgo takes anyway rather than a second instrument.
func ReadbackFrom(r bench.Report, vocab, bytesPerToken int) Readback {
	return Readback{
		Vocab:  vocab,
		Bytes:  bytesPerToken,
		Share:  r.Decode.ShareOfStep[bench.ShareReadback],
		Median: r.Decode.Readback.P50,
		Steps:  r.Decode.Steps,
	}
}

// Value renders the measurement for the report.
func (r Readback) Value() string {
	return fmt.Sprintf("%.1f%% of a decode step, %v at p50, %d B per token over "+
		"V=%d, from %d steps", 100*r.Share, r.Median, r.Bytes, r.Vocab, r.Steps)
}

// Quantization is int8's error on real blocks, against the bound accel
// promises.
//
// §3 insists on real blocks and the reason is in the scheme: quant.Int8Quantize
// scales a block of 32 by its largest magnitude, so a weight's error is
// proportional to the largest weight *in its block* and not to its own.
// Synthetic weights from one distribution have no outlier channels and
// therefore flatter the scheme. Trained transformer weights have them, which is
// the whole reason mixed-precision schemes exist, so the two numbers are
// different numbers and only one is worth reporting.
type Quantization struct {
	// Tensor is which weight was measured.
	Tensor string `json:"tensor"`

	// Blocks is how many blocks of quant.Int8Block it covered.
	Blocks int `json:"blocks"`

	// Worst is the absolute error of the dot product measured. One product
	// and not a matrix: a caller that wants a whole projection's worst case
	// loops its rows and keeps the largest, and doing it here would hide
	// which row produced the number.
	Worst float64 `json:"worst"`

	// Bound is quant.Int8ErrorBound over the same inputs. accel promises the
	// error stays under it; Worst above Bound is a finding against accel.
	Bound float64 `json:"bound"`
}

// Used is the fraction of the bound the real weights actually spent.
//
// A number near 1 says the bound is tight and int8 is as bad as advertised; a
// number near 0 says the bound is loose over this tensor and a test asserting
// against it proves less than it looks. Zero when the bound is zero, which
// happens only for a weight that is all zeros.
func (q Quantization) Used() float64 {
	if q.Bound == 0 {
		return 0
	}
	return q.Worst / q.Bound
}

// MeasureQuantization quantizes w, multiplies it by x, and compares the error
// against quant.Int8ErrorBound over the same inputs.
//
// A dot product and not a per-weight round trip, because the error a model
// suffers is the error of the product: the bound accel states is a bound on a
// dot product, and comparing a per-weight difference against it would compare
// two different quantities and call the difference a margin.
func MeasureQuantization(tensor string, x, w []float32) Quantization {
	if len(x) != len(w) {
		panic("conformance: MeasureQuantization over an activation and a weight " +
			"of different lengths")
	}
	if len(w) == 0 {
		panic("conformance: MeasureQuantization over an empty weight")
	}
	quants, scales := quant.Int8Quantize(w)
	back := quant.Int8Dequantize(quants, scales)
	exact, got := 0.0, 0.0
	terms := make([]accel.Float16, len(w))
	for i := range w {
		exact += float64(x[i]) * float64(w[i])
		got += float64(x[i]) * float64(back[i])
		terms[i] = scales[i/quant.Int8Block]
	}
	return Quantization{
		Tensor: tensor,
		Blocks: len(scales),
		Worst:  math.Abs(got - exact),
		Bound:  quant.Int8ErrorBound(x, terms),
	}
}

// Value renders the measurement for the report.
func (q Quantization) Value() string {
	return fmt.Sprintf("%s: %.3g against a bound of %.3g (%.0f%% of it) over %d "+
		"blocks", q.Tensor, q.Worst, q.Bound, 100*q.Used(), q.Blocks)
}

// Bucket is one bucket's compile time: the shape a plan was compiled for and
// what compiling it cost.
type Bucket struct {
	// Tokens is the bucket, in tokens per submission.
	Tokens int `json:"tokens"`

	// Elapsed is what compiling that bucket's plan took.
	Elapsed time.Duration `json:"elapsed_ns"`
}

// Compile is plan compile time per bucket and the plan cache's hit rate over a
// session.
//
// §3: it decides whether 007-D2's bucket set is right. A bucket nobody hits is
// compile time spent for nothing, and a session that misses often is a bucket
// set that does not match the shapes real requests have.
type Compile struct {
	// Buckets are the compile times, in bucket order.
	Buckets []Bucket `json:"buckets"`

	// Hits and Misses are the plan cache over one session.
	Hits   int `json:"hits"`
	Misses int `json:"misses"`
}

// HitRate is the fraction of plan lookups that found a compiled plan, or 0 when
// there were no lookups.
func (c Compile) HitRate() float64 {
	n := c.Hits + c.Misses
	if n == 0 {
		return 0
	}
	return float64(c.Hits) / float64(n)
}

// Total is the compile time of every bucket together: the cold start 010 §3.1
// claims tgo wins.
func (c Compile) Total() time.Duration {
	var t time.Duration
	for _, b := range c.Buckets {
		t += b.Elapsed
	}
	return t
}

// Value renders the measurement for the report.
func (c Compile) Value() string {
	parts := make([]string, 0, len(c.Buckets))
	for _, b := range c.Buckets {
		parts = append(parts, fmt.Sprintf("%d tok %v", b.Tokens, b.Elapsed))
	}
	if len(parts) == 0 {
		parts = append(parts, "no bucket compiled")
	}
	return fmt.Sprintf("%s; %v in total, cache hit rate %.0f%% over %d lookups",
		strings.Join(parts, ", "), c.Total(), 100*c.HitRate(), c.Hits+c.Misses)
}

// Transient is what accel's planner says a graph's transients cost, against
// what the working set actually is.
//
// §3: it decides whether accel's aliasing helps by the amount it claims. The
// planner reports both what it needs after aliasing and what it would have
// needed without, and the gap is what planning bought -- but a gap is only
// meaningful next to a working set computed by hand from the graph's lifetimes,
// because a planner that over-allocated and then aliased half of it away has
// bought nothing.
type Transient struct {
	// Memory is what accel's Plan.Memory() reported.
	Memory accel.GraphMemory `json:"memory"`

	// WorkingSet is the hand-computed live set at the graph's widest point,
	// in bytes: the number aliasing cannot go below.
	WorkingSet int `json:"working_set"`

	// Label names the graph, because the number is per-graph and a report
	// with one unlabelled memory figure in it invites the reader to think it
	// is the model's.
	Label string `json:"label"`
}

// Saved is what aliasing removed: the unaliased requirement less the aliased
// one, as accel reports them.
func (t Transient) Saved() int { return t.Memory.UnaliasedBytes - t.Memory.TransientBytes }

// Overhead is the aliased requirement as a multiple of the hand-computed
// working set, or 0 when no working set was computed.
//
// One means the planner reached the floor. Above one is what it holds beyond
// the live set, and that -- not the saving it reports against its own
// unaliased figure -- is the number §3 asks for.
func (t Transient) Overhead() float64 {
	if t.WorkingSet == 0 {
		return 0
	}
	return float64(t.Memory.TransientBytes) / float64(t.WorkingSet)
}

// Value renders the measurement for the report.
func (t Transient) Value() string {
	return fmt.Sprintf("%s: %d B transient against %d B unaliased (%d B saved) "+
		"and a %d B working set, %.2f× the floor",
		t.Label, t.Memory.TransientBytes, t.Memory.UnaliasedBytes, t.Saved(),
		t.WorkingSet, t.Overhead())
}

// notMeasured is what a nil measurement prints.
//
// It says who takes the number, because the reader's next question about a
// missing measurement is whose job it is, and a row that says only "-" invites
// the reader to read it as zero.
const notMeasured = "not measured"

// Document renders the five numbers as the Markdown table
// specs/010-conformance.md §3 publishes.
//
// A row per measurement whether or not it was taken. Emitting only the ones
// that exist would produce a report that shrinks as the suite loses coverage,
// which is the shape of every measurement that quietly stopped running.
func (m Measurements) Document() string {
	rows := [][2]string{
		{"CPU/Metal divergence", value(m.Divergence)},
		{"readback share of a decode step", value(m.Readback)},
		{"quantization error against `Int8ErrorBound`", value(m.Quantization)},
		{"plan compile time per bucket", value(m.Compile)},
		{"transient bytes against the working set", value(m.Transient)},
	}
	var b strings.Builder
	b.WriteString("| measurement | value |\n| --- | --- |\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| **%s** | %s |\n", r[0], r[1])
	}
	return b.String()
}

// valuer is what a measurement renders itself with.
type valuer interface{ Value() string }

// value renders a measurement, or says it was not taken.
//
// It takes a pointer to a type with a Value method so that a nil measurement
// and a zero one are different: the generic parameter keeps the nil check on
// the concrete pointer rather than on an interface that a typed nil would have
// made non-nil.
func value[T valuer](p *T) string {
	if p == nil {
		return notMeasured
	}
	return (*p).Value()
}

// Publish is the generated document specs/010-conformance.md §6 asks the suite
// to emit: the §2 register and the §3 numbers, from the tests rather than
// maintained beside them.
func Publish(rows []Row, m Measurements) string {
	var b strings.Builder
	b.WriteString("# Conformance\n\n" +
		"Generated by internal/conformance from the register and the suite's " +
		"measurements (specs/010-conformance.md §6). Edit the register in " +
		"internal/conformance/register.go; this file is output.\n\n" +
		"## The register\n\n")
	b.WriteString(Document(rows))
	b.WriteString("\n## Numbers tgo reports back\n\n")
	b.WriteString(m.Document())
	return b.String()
}
