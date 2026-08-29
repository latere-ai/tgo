// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"math"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"
)

// §6's numbers. 009-D7: the series that name tgo's upstream costs are the ones
// worth exporting, and the pair that carries the point is the readback against
// the step.

func TestMetricsExposeEverySeriesTheSpecNames(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("a", "b"), prompt: 5}
	s := newTestServer(t, eng)
	wantStatus(t, post(t, s, "/v1/chat/completions", routes[0].body(`,"user":"u-1"`)),
		http.StatusOK)

	w := get(t, s, "/metrics")
	wantStatus(t, w, http.StatusOK)
	if got := w.Header().Get("Content-Type"); !strings.HasPrefix(got, "text/plain") {
		t.Errorf("Content-Type = %q", got)
	}
	body := w.Body.String()
	for _, want := range []string{
		"tgo_requests_in_flight",
		"tgo_queue_depth",
		"tgo_queue_wait_seconds_count",
		"tgo_decode_step_seconds_count",
		"tgo_logits_readback_seconds_count",
		"tgo_request_loss_total",
		"tgo_sessions_rejected_total",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("no %s series:\n%s", want, body)
		}
	}
	// Every series is typed and documented, or a scrape reads it as untyped.
	for line := range strings.SplitSeq(body, "\n") {
		if strings.HasPrefix(line, "# HELP tgo_") {
			name := strings.Fields(line)[2]
			if !strings.Contains(body, "# TYPE "+name+" ") {
				t.Errorf("%s has HELP and no TYPE", name)
			}
		}
	}
}

// The readback against the step is specs/010-conformance.md §3's share,
// measured where it matters. Both are counted, and the count is the assertion:
// a duration is not, because Windows' timer granularity makes a real interval
// of a few hundred microseconds measure as exactly zero.
func TestTheReadbackAndTheStepAreBothObserved(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("a", "b", "c")}
	s := newTestServer(t, eng)
	for range 3 {
		wantStatus(t, post(t, s, "/v1/chat/completions", routes[0].body("")), http.StatusOK)
	}
	body := get(t, s, "/metrics").Body.String()
	for _, want := range []string{
		"tgo_decode_step_seconds_count 3",
		"tgo_logits_readback_seconds_count 3",
		"tgo_queue_wait_seconds_count 3",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("want %q:\n%s", want, body)
		}
	}
}

// A request that never decoded contributes no decode observation, rather than a
// zero that would drag the quantiles toward a step that did not happen.
func TestARefusedRequestObservesNoDecodeStep(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("ok")})
	wantStatus(t, post(t, s, "/v1/chat/completions", routes[0].body(`,"n":9`)),
		http.StatusBadRequest)
	if body := get(t, s, "/metrics").Body.String(); !strings.Contains(body,
		"tgo_decode_step_seconds_count 0") {
		t.Errorf("a refused request was counted as a decode step:\n%s", body)
	}
}

// The in-flight gauge is per dialect, and it comes back down.
func TestTheInFlightGaugeRisesAndFalls(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok"), hold: make(chan struct{}),
		entered: make(chan struct{}, 4)}
	s := newTestServer(t, eng, WithConcurrency(2), WithQueue(2))
	release := busy(t, s, eng)

	if body := get(t, s, "/metrics").Body.String(); !strings.Contains(body,
		`tgo_requests_in_flight{dialect="openai-chat"} 1`) {
		t.Errorf("the gauge did not rise:\n%s", body)
	}
	release()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(get(t, s, "/metrics").Body.String(),
			`tgo_requests_in_flight{dialect="openai-chat"} 0`) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Errorf("the gauge did not come back down:\n%s", get(t, s, "/metrics").Body.String())
}

// A loss field name is whatever member a client sent, so the exposition must
// survive one that carries the characters the format reserves.
func TestALossLabelIsEscaped(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("ok")})
	w := post(t, s, "/v1/chat/completions",
		routes[0].body(`,"a\"quoted\\field":1`))
	wantStatus(t, w, http.StatusOK)
	body := get(t, s, "/metrics").Body.String()
	if !strings.Contains(body, `tgo_request_loss_total{field="a\"quoted\\field"} 1`) {
		t.Errorf("the label was not escaped:\n%s", body)
	}
}

// The loss counter's cardinality is bounded, because its label is a name a
// client chose. Past the bound the long tail folds into one series, which is
// all the counter's question needs.
func TestTheLossCounterFoldsItsLongTail(t *testing.T) {
	t.Parallel()
	m := newMetrics()
	for i := range maxLossLabels + 50 {
		m.lost([]string{"field" + strings.Repeat("x", i%7) + string(rune('a'+i%26)) +
			strconv.Itoa(i)})
	}
	if len(m.loss) > maxLossLabels+1 {
		t.Errorf("the loss counter grew to %d labels, want at most %d plus \"other\"",
			len(m.loss), maxLossLabels)
	}
	if m.loss["other"] == 0 {
		t.Error("the long tail was not folded")
	}
}

func TestAHistogramCountsWhatItObserved(t *testing.T) {
	t.Parallel()
	h := newHistogram()
	for _, v := range []float64{0.00001, 0.02, 1.5, 1000} {
		h.observe(v)
	}
	var w strings.Builder
	h.write(&w, "x")
	body := w.String()
	if !strings.Contains(body, "x_count 4") {
		t.Errorf("count:\n%s", body)
	}
	if !strings.Contains(body, `x_bucket{le="+Inf"} 4`) {
		t.Errorf("the last bucket does not hold everything:\n%s", body)
	}
	if !strings.Contains(body, `x_bucket{le="0.0001"} 1`) {
		t.Errorf("the first bucket:\n%s", body)
	}
	if !strings.Contains(body, `x_bucket{le="60"} 3`) {
		t.Errorf("a value past the last boundary was counted below it:\n%s", body)
	}
}

func TestFormatFloatNamesInfinity(t *testing.T) {
	t.Parallel()
	if got := formatFloat(math.Inf(1)); got != "+Inf" {
		t.Errorf("formatFloat(+Inf) = %q", got)
	}
	if got := formatFloat(0.025); got != "0.025" {
		t.Errorf("formatFloat(0.025) = %q", got)
	}
}
