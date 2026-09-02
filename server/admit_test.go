// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// §6 and 009-D3. One session per in-flight request, a semaphore sized by KV
// memory, a bounded queue, and 429 with Retry-After when it is full. The bound
// is the point: an unbounded queue converts a load problem into an
// out-of-memory one.

// busy starts one request that occupies a session slot and does not finish
// until the returned release is called.
func busy(t *testing.T, s *Server, eng *fakeEngine) func() {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		post(t, s, "/v1/chat/completions", routes[0].body(""))
	}()
	select {
	case <-eng.entered:
	case <-time.After(10 * time.Second):
		t.Fatal("the first request never reached the engine")
	}
	return func() {
		close(eng.hold)
		<-done
	}
}

func TestAFullQueueIs429WithRetryAfter(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok"), hold: make(chan struct{}),
		entered: make(chan struct{}, 4)}
	s := newTestServer(t, eng, WithConcurrency(1), WithQueue(0))
	release := busy(t, s, eng)
	defer release()

	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusTooManyRequests)
	retry := w.Header().Get("Retry-After")
	if retry == "" {
		t.Fatal("a 429 with no Retry-After is an invitation to retry in a tight loop")
	}
	if n, err := strconv.Atoi(retry); err != nil || n < 1 {
		t.Errorf("Retry-After = %q, want a whole number of seconds of at least 1", retry)
	}
	wantNames(t, w, "queue is full")
	if body := get(t, s, "/metrics").Body.String(); !strings.Contains(body,
		`tgo_sessions_rejected_total{reason="queue_full"} 1`) {
		t.Errorf("the rejection was not counted:\n%s", body)
	}
}

// The 429 is in the caller's dialect too, and so is its Retry-After.
func TestTheOverloadedErrorIsShapedByTheDialect(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok"), hold: make(chan struct{}),
		entered: make(chan struct{}, 4)}
	s := newTestServer(t, eng, WithConcurrency(1), WithQueue(0))
	release := busy(t, s, eng)
	defer release()

	w := post(t, s, "/v1/messages", routes[1].body(""))
	wantStatus(t, w, http.StatusTooManyRequests)
	if !strings.Contains(w.Body.String(), "overloaded_error") {
		t.Errorf("the Anthropic 429 is not Anthropic's: %s", w.Body.String())
	}
	if w.Header().Get("Retry-After") == "" {
		t.Error("no Retry-After")
	}
}

// A queued request that waits out its budget is refused rather than left there.
func TestAQueuedRequestIsRefusedWhenItsWaitRunsOut(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok"), hold: make(chan struct{}),
		entered: make(chan struct{}, 4)}
	s := newTestServer(t, eng, WithConcurrency(1), WithQueue(4),
		WithQueueWait(20*time.Millisecond))
	release := busy(t, s, eng)
	defer release()

	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusTooManyRequests)
	wantNames(t, w, "waited")
	if body := get(t, s, "/metrics").Body.String(); !strings.Contains(body,
		`tgo_sessions_rejected_total{reason="queue_timeout"} 1`) {
		t.Errorf("the timeout was not counted:\n%s", body)
	}
}

// A queued request whose client hangs up gives its place back rather than
// waiting out the budget on behalf of nobody.
func TestAQueuedRequestEndsWhenItsClientDoes(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok"), hold: make(chan struct{}),
		entered: make(chan struct{}, 4)}
	s := newTestServer(t, eng, WithConcurrency(1), WithQueue(4), WithQueueWait(time.Minute))
	srv := httptest.NewServer(s)
	defer srv.Close()
	release := busy(t, s, eng)
	defer release()

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		srv.URL+"/v1/chat/completions", strings.NewReader(routes[0].body("")))
	if err != nil {
		t.Fatal(err)
	}
	errc := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			_ = resp.Body.Close()
		}
		errc <- err
	}()
	// Wait until the request is ACTUALLY in the queue, then hang up.
	//
	// This was a 50ms sleep, and it failed under -race on a loaded macOS runner:
	// a request cancelled before it reaches the queue is cancelled at the
	// transport instead, never enqueues, and never counts as client_gone — so
	// the test reported "did not end with its client" about a request that
	// never got there. tgo_queue_depth is the state the cancellation needs, so
	// the test waits for that state rather than guessing how long it takes.
	if !awaitQueueDepth(t, s, 1) {
		t.Fatal("the request never reached the queue, so there was nothing to cancel")
	}
	cancel()
	select {
	case <-errc:
	case <-time.After(10 * time.Second):
		t.Fatal("the cancelled request did not return")
	}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(get(t, s, "/metrics").Body.String(),
			`tgo_sessions_rejected_total{reason="client_gone"} 1`) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Errorf("the queued request did not end with its client:\n%s",
		get(t, s, "/metrics").Body.String())
}

// The semaphore lets exactly its width through at once, and the rest wait.
func TestTheSemaphoreAdmitsItsWidth(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok"), hold: make(chan struct{}),
		entered: make(chan struct{}, 8)}
	s := newTestServer(t, eng, WithConcurrency(2), WithQueue(0))

	done := make(chan struct{}, 2)
	for range 2 {
		go func() {
			post(t, s, "/v1/chat/completions", routes[0].body(""))
			done <- struct{}{}
		}()
	}
	for range 2 {
		select {
		case <-eng.entered:
		case <-time.After(10 * time.Second):
			t.Fatal("the second slot was never taken")
		}
	}
	// Both slots are held, so the third request has nowhere to wait.
	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusTooManyRequests)

	close(eng.hold)
	for range 2 {
		<-done
	}
	// And once they are back, a request runs again: a slot is released by the
	// handler rather than by the process ending.
	wantStatus(t, post(t, s, "/v1/chat/completions", routes[0].body("")), http.StatusOK)
}

// §6's arithmetic: the admission limit is a memory bound divided by one
// session's reservation.
func TestTheKVBudgetSetsTheConcurrency(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{}, WithKVBudget(4*fakeCacheBytes+fakeCacheBytes/2))
	if got := s.Concurrency(); got != 4 {
		t.Errorf("concurrency = %d, want 4", got)
	}
}

func TestNewRefusesAConfigurationThatCannotWork(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		opts []Option
		eng  Engine
	}{
		{"no engine", nil, nil},
		{"no concurrency", []Option{WithConcurrency(0)}, &fakeEngine{}},
		{"a negative queue", []Option{WithQueue(-1)}, &fakeEngine{}},
		{"a zero wait", []Option{WithQueueWait(0)}, &fakeEngine{}},
		{"a zero body bound", []Option{WithMaxBodyBytes(0)}, &fakeEngine{}},
		{"a zero budget", []Option{WithKVBudget(0)}, &fakeEngine{}},
		{"a budget under one session", []Option{WithKVBudget(fakeCacheBytes - 1)}, &fakeEngine{}},
		{"a budget with no cache size to divide", []Option{WithKVBudget(1 << 30)}, &sizelessEngine{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			if _, err := New(c.eng, c.opts...); err == nil {
				t.Fatal("New accepted a configuration that cannot work")
			}
		})
	}
}

// sizelessEngine reports no cache size, which is what an engine that does not
// reserve one looks like. A budget cannot be divided by it.
type sizelessEngine struct{ fakeEngine }

func (*sizelessEngine) CacheBytesPerSession() int64 { return 0 }

func TestRetryAfterRoundsUpAndNeverReachesZero(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in   time.Duration
		want string
	}{
		{time.Nanosecond, "1"},
		{time.Second, "1"},
		{1500 * time.Millisecond, "2"},
		{30 * time.Second, "30"},
	}
	for _, c := range cases {
		if got := retryAfter(c.in); got != c.want {
			t.Errorf("retryAfter(%s) = %q, want %q", c.in, got, c.want)
		}
	}
}

// awaitQueueDepth waits for the queue to hold n requests, and reports whether it
// got there. It polls a gauge rather than sleeping a guess: see the call site.
func awaitQueueDepth(t *testing.T, s *Server, n int) bool {
	t.Helper()
	want := fmt.Sprintf("tgo_queue_depth %d\n", n)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(get(t, s, "/metrics").Body.String(), want) {
			return true
		}
		runtime.Gosched()
		time.Sleep(time.Millisecond)
	}
	return false
}
