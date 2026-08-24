// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package bench

import (
	"sort"
	"time"
)

// The keys of PhaseStats.ShareOfStep. They name the four terms of
// specs/017-benchmarks.md §1 and are always all present.
const (
	ShareHost     = "host"
	ShareSubmit   = "submit"
	ShareDevice   = "device"
	ShareReadback = "readback"
)

// Quantiles is a distribution reported at its median and its two tails.
//
// Percentiles rather than a mean, because a mean hides the tail and the tail is
// what a user feels as a stall (017-D2). N is carried with them: a p99 over
// four samples is not a p99, and a reader cannot see that from the number.
type Quantiles struct {
	P50 time.Duration `json:"p50_ns"`
	P90 time.Duration `json:"p90_ns"`
	P99 time.Duration `json:"p99_ns"`
	N   int           `json:"n"`
}

// PhaseStats is one phase's measurement: throughput, the four terms as
// distributions, and the breakdown as fractions of a step.
type PhaseStats struct {
	TokensPerSecond float64   `json:"tokens_per_second"`
	Host            Quantiles `json:"host"`
	Submit          Quantiles `json:"submit"`
	Device          Quantiles `json:"device"`
	Readback        Quantiles `json:"readback"`

	// ShareOfStep is the host/submit/device/readback breakdown as fractions of
	// the phase's total time, keyed by the Share constants (017-D1). The four
	// values sum to 1 to within floating-point rounding whenever the phase
	// recorded any time at all; when it recorded none, all four are zero.
	ShareOfStep map[string]float64 `json:"share_of_step"`

	Steps  int `json:"steps"`
	Tokens int `json:"tokens"`
}

// Report is the aggregate over everything a Recorder holds.
//
// Its JSON is the regression record (017-D6), so every field is tagged and the
// encoding is stable: struct fields marshal in declaration order and
// encoding/json sorts map keys, so two marshals of equal reports are equal
// bytes.
type Report struct {
	Prefill PhaseStats `json:"prefill"`
	Decode  PhaseStats `json:"decode"`
	TTFT    Quantiles  `json:"ttft"`
	Steps   int        `json:"steps"`

	// Dropped counts observations discarded because the Recorder was full.
	// Non-zero means the report describes a prefix of the run and its
	// percentiles are not the run's.
	Dropped int `json:"dropped"`
}

// Report aggregates what has been recorded. It allocates and sorts, so it
// belongs off the hot path; a nil or disabled Recorder returns the zero Report
// with its share maps populated.
func (r *Recorder) Report() Report {
	var steps []Step
	var ttfts []time.Duration
	var dropped int
	if r != nil {
		steps = r.steps[:r.n]
		ttfts = r.ttfts[:r.nttft]
		dropped = r.dropped
	}

	var prefill, decode []Step
	for _, s := range steps {
		if s.Phase == Decode {
			decode = append(decode, s)
			continue
		}
		prefill = append(prefill, s)
	}

	sorted := append([]time.Duration(nil), ttfts...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	return Report{
		Prefill: phaseStats(prefill),
		Decode:  phaseStats(decode),
		TTFT:    quantiles(sorted),
		Steps:   len(steps),
		Dropped: dropped,
	}
}

// phaseStats reduces one phase's steps. The four term slices are built and
// sorted independently: a step's host time and its device time are separate
// distributions, and pairing them would report a step that never happened.
func phaseStats(steps []Step) PhaseStats {
	host := make([]time.Duration, len(steps))
	submit := make([]time.Duration, len(steps))
	device := make([]time.Duration, len(steps))
	readback := make([]time.Duration, len(steps))

	var hostSum, submitSum, deviceSum, readbackSum time.Duration
	var tokens int
	for i, s := range steps {
		host[i], submit[i], device[i], readback[i] = s.Host, s.Submit, s.Device, s.Readback
		hostSum += s.Host
		submitSum += s.Submit
		deviceSum += s.Device
		readbackSum += s.Readback
		tokens += s.Tokens
	}
	total := hostSum + submitSum + deviceSum + readbackSum

	for _, v := range [][]time.Duration{host, submit, device, readback} {
		sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
	}

	stats := PhaseStats{
		Host:     quantiles(host),
		Submit:   quantiles(submit),
		Device:   quantiles(device),
		Readback: quantiles(readback),
		ShareOfStep: map[string]float64{
			ShareHost:     share(hostSum, total),
			ShareSubmit:   share(submitSum, total),
			ShareDevice:   share(deviceSum, total),
			ShareReadback: share(readbackSum, total),
		},
		Steps:  len(steps),
		Tokens: tokens,
	}
	if total > 0 {
		stats.TokensPerSecond = float64(tokens) / total.Seconds()
	}
	return stats
}

// share is one term as a fraction of the step. A phase that recorded no time
// has no breakdown to report, and zero is the honest answer rather than a
// quarter each.
func share(part, total time.Duration) float64 {
	if total <= 0 {
		return 0
	}
	return float64(part) / float64(total)
}

// quantiles reports p50, p90 and p99 of an already sorted sample.
func quantiles(sorted []time.Duration) Quantiles {
	return Quantiles{
		P50: nearestRank(sorted, 50),
		P90: nearestRank(sorted, 90),
		P99: nearestRank(sorted, 99),
		N:   len(sorted),
	}
}

// nearestRank returns the p-th percentile of a sorted sample by the
// nearest-rank definition: the smallest sample at or above rank ceil(p*n/100).
// p is a percentile in [1, 100].
//
// Nearest rank rather than an interpolating definition because a reported
// percentile must be a step that actually happened. An interpolated p99 is an
// average of two steps, which is the mean that 017-D2 rejects, reintroduced at
// the tail.
//
// The rank is computed in integer arithmetic. p*n/100 in floating point can
// land either side of an integer depending on the order of the multiply, and
// there the error is not a fraction of a nanosecond: it moves the answer by a
// whole sample. Integer arithmetic also makes the index provably in range, so
// there is no clamp to write and no unreachable branch to leave untested:
// ceil(p*n/100) is at least 1 for p >= 1 and n >= 1, and at most n for
// p <= 100, so i lies in [0, n-1].
func nearestRank(sorted []time.Duration, p int) time.Duration {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	return sorted[(p*n+99)/100-1]
}
