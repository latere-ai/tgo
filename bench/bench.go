// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package bench records where a step's time goes and reports it as a
// breakdown.
//
// A decode step is
//
//	t_step = t_host + t_submit + t_device + t_readback
//
// and tgo's claim is that a compiled language wins t_host while accel decides
// t_device. One throughput number cannot tell those apart, so it cannot say
// whether a regression belongs to tgo or to accel. Every measurement this
// package produces is therefore a four-way breakdown, reported as percentiles
// (specs/017-benchmarks.md 017-D1, 017-D2).
//
// The Recorder is off in its zero value and allocates nothing while recording,
// because an instrument that changes what it measures is not an instrument
// (017-D3). The Report carries explicit JSON tags and marshals stably, because
// a regression check diffs its bytes (017-D6).
//
// The package takes durations and returns statistics. It depends on nothing
// else in tgo.
package bench

import "time"

// Phase says whether a step consumed prompt tokens or produced one.
type Phase uint8

// The phases a step can belong to. Prefill is the zero value so that a Step
// built by a caller that forgot the field is attributed to the phase whose
// statistics matter least.
const (
	Prefill Phase = iota
	Decode
)

// String returns the phase name used in reports and error text.
func (p Phase) String() string {
	switch p {
	case Prefill:
		return "prefill"
	case Decode:
		return "decode"
	default:
		return "unknown"
	}
}

// Step is one decode or prefill step.
//
// The four durations partition the step: specs/017-benchmarks.md §1 treats
// them as exhaustive, and no separate wall-clock total is recorded. A caller
// that measures a gap outside the four has found a term the model does not
// have, which is a finding rather than a rounding error.
type Step struct {
	Phase  Phase // Prefill or Decode
	Tokens int   // tokens this step consumed
	Batch  int   // sequences in flight for this step

	Host     time.Duration // sampling, detokenizing, bookkeeping
	Submit   time.Duration // building bindings and handing the plan to the queue
	Device   time.Duration // fence wait
	Readback time.Duration // logits to host
}

// Total is the modeled step time: the sum of the four measured parts.
func (s Step) Total() time.Duration {
	return s.Host + s.Submit + s.Device + s.Readback
}

// Recorder collects steps with no allocation on the hot path.
//
// The zero Recorder is disabled, and so is a nil one: Step and TTFT return
// without touching memory. On a nil receiver that costs a single branch; on a
// disabled non-nil receiver it costs two, since the capacity test doubles as
// the enabled test.
//
// Storage is fixed at construction and never grows, which is what makes the
// allocation-free claim structural rather than lucky. Observations past
// capacity are discarded and counted in Report.Dropped; a report with a
// non-zero Dropped is truncated, and its percentiles describe the start of the
// run rather than the run.
//
// A Recorder is not safe for concurrent use. One recorder per stepping
// goroutine, merged by the harness, keeps the hot path free of a lock.
type Recorder struct {
	steps []Step // pre-sized; len is the capacity, never appended to
	n     int

	ttfts []time.Duration
	nttft int

	dropped int
}

// NewRecorder returns an enabled Recorder sized for capacity observations of
// each kind. A capacity of zero or less returns a disabled Recorder, so that
// switching the instrument off is a number rather than a second code path.
//
// Size capacity above the measurement window. specs/017-benchmarks.md §4 rule
// 3 says to warm up and then measure: the intended shape is to record through
// warm-up, call Reset, and record the window. A recorder that fills during
// warm-up reports drops for the part that mattered.
func NewRecorder(capacity int) *Recorder {
	if capacity <= 0 {
		return &Recorder{}
	}
	return &Recorder{
		steps: make([]Step, capacity),
		ttfts: make([]time.Duration, capacity),
	}
}

// Enabled reports whether the Recorder records. Callers use it to skip reading
// the clock at all when the instrument is off, since time.Now on both sides of
// four regions is itself a cost the measurement would otherwise carry.
func (r *Recorder) Enabled() bool { return r != nil && len(r.steps) > 0 }

// Step records one step. It is a no-op on a nil or disabled Recorder, and it
// allocates nothing: s is copied into storage reserved by NewRecorder.
func (r *Recorder) Step(s Step) {
	if r == nil || r.n == len(r.steps) {
		// Disabled means zero capacity, which the same test catches. Only a
		// recorder that is on can drop: nothing was discarded by an instrument
		// that was never recording.
		if r != nil && len(r.steps) > 0 {
			r.dropped++
		}
		return
	}
	r.steps[r.n] = s
	r.n++
}

// TTFT records one time to first token.
//
// It is a separate stream because time to first token is a property of a
// request and a Step is a property of a step: no aggregation over steps
// recovers it once prefill is chunked, and a cold measurement includes model
// load and plan compile, which no Step ever sees.
func (r *Recorder) TTFT(d time.Duration) {
	if r == nil || r.nttft == len(r.ttfts) {
		if r != nil && len(r.ttfts) > 0 {
			r.dropped++
		}
		return
	}
	r.ttfts[r.nttft] = d
	r.nttft++
}

// Reset discards everything recorded so far and keeps the storage. It is how a
// caller drops the warm-up steps, whose plan compilation and page faults belong
// to a cold measurement rather than a warm one.
func (r *Recorder) Reset() {
	if r == nil {
		return
	}
	r.n, r.nttft, r.dropped = 0, 0, 0
}
