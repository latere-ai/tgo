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

import (
	"sync"
	"time"
)

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
// allocation-free claim structural rather than lucky. It is a **ring**:
// observations past capacity overwrite the oldest, and Report.Dropped counts
// how many were overwritten. So a report with a non-zero Dropped describes the
// most recent capacity observations rather than the whole run.
//
// It kept the *first* capacity observations until 2026-08-27, which made
// server/generate.go's "reports quantiles over its most recent steps" false for
// exactly the completions long enough to need a window: a request past capacity
// published percentiles for its own warm-up and never noticed. Keeping the
// newest is the answer 027-D5 chose over refusing to publish a truncated
// report, because a long request is the one whose current behaviour a reader
// wants, and refusing would report nothing at all in that case.
//
// # Why there is a lock
//
// A Recorder is safe for concurrent use. It was not until 2026-08-28, on
// 017-D3's argument that one recorder per stepping goroutine keeps the hot path
// free of a lock -- which assumed the stepping goroutine and the reader are the
// same one. Under specs/022-batched-serving.md they are structurally different:
// one driver goroutine steps the whole batch and writes every request's
// recorder, and the request's own goroutine reads it. No discipline fixes that,
// because the two goroutines are what batching is.
//
// The cost is one uncontended mutex against a forward pass, and the allocation
// claim is unchanged: storage is still fixed at construction and Step still
// copies into it.
type Recorder struct {
	mu sync.Mutex

	steps []Step // pre-sized; len is the capacity, never appended to
	first int    // index of the oldest step once the ring has wrapped
	n     int

	ttfts     []time.Duration
	ttftFirst int
	nttft     int

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
//
// Past capacity it overwrites the oldest step. That keeps the cost of a step
// constant whether or not the ring has wrapped, which is what the hot path
// needs; the bookkeeping is one modulo and one counter.
func (r *Recorder) Step(s Step) {
	if r == nil || len(r.steps) == 0 {
		// Disabled means zero capacity. Only a recorder that is on can drop:
		// nothing was discarded by an instrument that was never recording.
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.steps[(r.first+r.n)%len(r.steps)] = s
	if r.n == len(r.steps) {
		r.first = (r.first + 1) % len(r.steps)
		r.dropped++
		return
	}
	r.n++
}

// TTFT records one time to first token.
//
// It is a separate stream because time to first token is a property of a
// request and a Step is a property of a step: no aggregation over steps
// recovers it once prefill is chunked, and a cold measurement includes model
// load and plan compile, which no Step ever sees.
func (r *Recorder) TTFT(d time.Duration) {
	if r == nil || len(r.ttfts) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ttfts[(r.ttftFirst+r.nttft)%len(r.ttfts)] = d
	if r.nttft == len(r.ttfts) {
		r.ttftFirst = (r.ttftFirst + 1) % len(r.ttfts)
		r.dropped++
		return
	}
	r.nttft++
}

// Reset discards everything recorded so far and keeps the storage. It is how a
// caller drops the warm-up steps, whose plan compilation and page faults belong
// to a cold measurement rather than a warm one.
func (r *Recorder) Reset() {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.first, r.n, r.ttftFirst, r.nttft, r.dropped = 0, 0, 0, 0, 0
}
