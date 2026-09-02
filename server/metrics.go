// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// The numbers §6 asks for, in Prometheus text exposition, hand-written because
// this package's dependency list is stdlib plus llmdialect.
//
// The pair that carries the point is tgo_logits_readback_seconds against
// tgo_decode_step_seconds: specs/010-conformance.md §3's readback share,
// measured in production rather than in a benchmark. tgo_queue_wait_seconds is
// what specs/008-scheduler.md's absence costs, in the units a caller feels.

// buckets are the histogram boundaries, in seconds.
//
// They span a microsecond to a minute because the two ends are both real: a
// readback on a device that works is under a millisecond, and a decode step on
// accel's CPU backend has been measured at three minutes.
var buckets = []float64{
	0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60,
}

// maxLossLabels bounds the loss counter's cardinality.
//
// A loss field name can be any top-level member a client sent, so an
// unbounded map is a client-controlled memory series. Past the bound, further
// names count under "other": the counter's job is to say which field turns up
// constantly, and that survives folding the long tail.
const maxLossLabels = 256

// histogram is a fixed-bucket cumulative histogram.
type histogram struct {
	counts []uint64
	sum    float64
	total  uint64
}

func newHistogram() *histogram { return &histogram{counts: make([]uint64, len(buckets))} }

// observe records one value, in seconds.
func (h *histogram) observe(v float64) {
	h.total++
	h.sum += v
	for i, b := range buckets {
		if v <= b {
			h.counts[i]++
		}
	}
}

// write renders the histogram's series.
func (h *histogram) write(w io.Writer, name string) {
	var cum uint64
	for i, b := range buckets {
		cum = h.counts[i]
		_, _ = fmt.Fprintf(w, "%s_bucket{le=\"%s\"} %d\n", name, formatFloat(b), cum)
	}
	_, _ = fmt.Fprintf(w, "%s_bucket{le=\"+Inf\"} %d\n", name, h.total)
	_, _ = fmt.Fprintf(w, "%s_sum %s\n", name, formatFloat(h.sum))
	_, _ = fmt.Fprintf(w, "%s_count %d\n", name, h.total)
}

// metrics holds every series this server exports.
type metrics struct {
	mu sync.Mutex

	inFlight map[string]int64
	queued   int64

	queueWait  *histogram
	decodeStep *histogram
	readback   *histogram

	loss     map[string]int64
	rejected map[string]int64
}

func newMetrics() *metrics {
	return &metrics{
		inFlight:   map[string]int64{},
		queueWait:  newHistogram(),
		decodeStep: newHistogram(),
		readback:   newHistogram(),
		loss:       map[string]int64{},
		rejected:   map[string]int64{},
	}
}

func (m *metrics) enter(dialect string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.inFlight[dialect]++
}

func (m *metrics) leave(dialect string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.inFlight[dialect]--
}

func (m *metrics) queue(delta int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queued += delta
}

func (m *metrics) waited(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queueWait.observe(d.Seconds())
}

// step records one request's median decode step and its median readback.
//
// One observation per request rather than per step: [bench.Recorder] reports
// quantiles and keeps no per-step list, so a per-step histogram would need the
// engine to hand out what it does not have. Both series are drawn the same way,
// which is what keeps their ratio meaningful.
func (m *metrics) step(decode, readback time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.decodeStep.observe(decode.Seconds())
	m.readback.observe(readback.Seconds())
}

func (m *metrics) lost(fields []string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, f := range fields {
		if _, ok := m.loss[f]; !ok && len(m.loss) >= maxLossLabels {
			f = "other"
		}
		m.loss[f]++
	}
}

func (m *metrics) reject(reason string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rejected[reason]++
}

// write renders the exposition body.
func (m *metrics) write(w io.Writer) {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, _ = fmt.Fprint(w, "# HELP tgo_requests_in_flight Requests generating right now.\n")
	_, _ = fmt.Fprint(w, "# TYPE tgo_requests_in_flight gauge\n")
	for _, d := range slices.Sorted(maps.Keys(m.inFlight)) {
		_, _ = fmt.Fprintf(w, "tgo_requests_in_flight{dialect=\"%s\"} %d\n", escape(d), m.inFlight[d])
	}

	_, _ = fmt.Fprint(w, "# HELP tgo_queue_depth Admitted requests waiting for a session slot.\n")
	_, _ = fmt.Fprint(w, "# TYPE tgo_queue_depth gauge\n")
	_, _ = fmt.Fprintf(w, "tgo_queue_depth %d\n", m.queued)

	_, _ = fmt.Fprint(w, "# HELP tgo_queue_wait_seconds Time spent waiting for a session slot.\n")
	_, _ = fmt.Fprint(w, "# TYPE tgo_queue_wait_seconds histogram\n")
	m.queueWait.write(w, "tgo_queue_wait_seconds")

	_, _ = fmt.Fprint(w, "# HELP tgo_decode_step_seconds One request's median decode step.\n")
	_, _ = fmt.Fprint(w, "# TYPE tgo_decode_step_seconds histogram\n")
	m.decodeStep.write(w, "tgo_decode_step_seconds")

	_, _ = fmt.Fprint(w, "# HELP tgo_logits_readback_seconds One request's median logits readback, "+
		"the share of a decode step spent moving a row of logits to the host.\n")
	_, _ = fmt.Fprint(w, "# TYPE tgo_logits_readback_seconds histogram\n")
	m.readback.write(w, "tgo_logits_readback_seconds")

	_, _ = fmt.Fprint(w, "# HELP tgo_request_loss_total Advisory fields accepted and not acted on.\n")
	_, _ = fmt.Fprint(w, "# TYPE tgo_request_loss_total counter\n")
	for _, f := range slices.Sorted(maps.Keys(m.loss)) {
		_, _ = fmt.Fprintf(w, "tgo_request_loss_total{field=\"%s\"} %d\n", escape(f), m.loss[f])
	}

	_, _ = fmt.Fprint(w, "# HELP tgo_sessions_rejected_total Requests refused rather than run.\n")
	_, _ = fmt.Fprint(w, "# TYPE tgo_sessions_rejected_total counter\n")
	for _, r := range slices.Sorted(maps.Keys(m.rejected)) {
		_, _ = fmt.Fprintf(w, "tgo_sessions_rejected_total{reason=\"%s\"} %d\n", escape(r), m.rejected[r])
	}
}

// labelEscaper covers the three characters the exposition format reserves in a
// label value. The values include field names a client chose, so all three are
// the client's to send.
var labelEscaper = strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)

// escape makes a label value safe.
func escape(s string) string { return labelEscaper.Replace(s) }

// formatFloat renders a bucket boundary or a sum without an exponent where one
// would be surprising, which is what a Prometheus scrape expects.
func formatFloat(v float64) string {
	if math.IsInf(v, 1) {
		return "+Inf"
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}
