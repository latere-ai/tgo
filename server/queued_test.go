// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/latere-ai/tgo"
)

// An engine that does its own admission, and what its refusals have to look
// like on the wire.
//
// specs/022-batched-serving.md §10. The wait moved into the engine, so the 429
// moved with it: a deployment's answer must not change because the queue is one
// layer down. Scripted rather than batched, because what is under test is the
// mapping and not the queue -- the queue's own bounds are asserted in package
// tgo, and driving them through a real batch would make this a race against a
// forward pass.
type queueingEngine struct {
	fakeEngine
	wait time.Duration
}

func (e *queueingEngine) AdmissionWait() time.Duration { return e.wait }
func (e *queueingEngine) AdmissionDepth() int          { return 32 }

// newServerOver builds a server over any engine, which newTestServer cannot:
// its parameter is the concrete fake, and what is under test here is an engine
// that implements one more interface than the fake does.
func newServerOver(t *testing.T, eng Engine, opts ...Option) *Server {
	t.Helper()
	opts = append([]Option{WithNotice(&strings.Builder{})}, opts...)
	s, err := New(eng, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

// TestAnEngineThatQueuesAnswers429WithItsOwnBudget is the mapping.
func TestAnEngineThatQueuesAnswers429WithItsOwnBudget(t *testing.T) {
	t.Parallel()
	for _, c := range []struct {
		name       string
		err        error
		status     int
		retryAfter string
		reason     string
		says       string
	}{
		{
			name:   "the budget elapsed",
			err:    fmt.Errorf("tgo: waited 1.5s for a slot: %w", tgo.ErrQueueTimeout),
			status: http.StatusTooManyRequests, retryAfter: "2",
			reason: "queue_timeout", says: "waited 1.5s for a slot",
		},
		{
			name:   "the queue is full",
			err:    fmt.Errorf("tgo: 32 requests are waiting: %w", tgo.ErrQueueFull),
			status: http.StatusTooManyRequests, retryAfter: "2",
			reason: "queue_full", says: "32 requests are waiting",
		},
		{
			// A client that hung up while queued is a departure and not a
			// failure, and nothing is written to it.
			name:   "the client hung up while queued",
			err:    fmt.Errorf("tgo: %w", context.Canceled),
			status: 499, reason: "client_gone",
		},
		{
			// The door's refusal: this prompt does not fit the pool and never
			// will, which is the same answer a session gives a prompt larger
			// than its cache.
			name:   "the prompt does not fit the pool",
			err:    fmt.Errorf("tgo: a prompt of 9000: %w", tgo.ErrContextExhausted),
			status: http.StatusBadRequest, reason: "bad_request",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			eng := &queueingEngine{wait: 1500 * time.Millisecond}
			eng.chatErr = c.err
			s := newServerOver(t, eng, WithConcurrency(4))

			w := post(t, s, "/v1/completions",
				`{"model":"`+fakeName+`","max_tokens":2,"prompt":"hi"}`)
			if w.Code != c.status {
				t.Fatalf("status = %d, want %d: %s", w.Code, c.status, w.Body.String())
			}
			if got := w.Header().Get("Retry-After"); got != c.retryAfter {
				t.Errorf("Retry-After = %q, want %q", got, c.retryAfter)
			}
			if c.says != "" && !strings.Contains(w.Body.String(), c.says) {
				t.Errorf("the answer does not carry the engine's own words %q: %s",
					c.says, w.Body.String())
			}
			if strings.Contains(w.Body.String(), "tgo: tgo:") {
				t.Errorf("the message is prefixed twice: %s", w.Body.String())
			}
			m := get(t, s, "/metrics").Body.String()
			if !strings.Contains(m,
				fmt.Sprintf(`tgo_sessions_rejected_total{reason=%q} 1`, c.reason)) &&
				c.reason != "bad_request" {
				t.Errorf("the refusal was not counted as %q:\n%s", c.reason, m)
			}
		})
	}
}

// TestARetryAfterComesFromTheEngineThatWaited is 021-D6 one layer up: the
// budget is the only interval either layer can promise, and the one that
// promised it is the one that waited.
func TestARetryAfterComesFromTheEngineThatWaited(t *testing.T) {
	t.Parallel()
	eng := &queueingEngine{wait: 45 * time.Second}
	eng.chatErr = fmt.Errorf("tgo: %w", tgo.ErrQueueTimeout)
	// The server's own budget is a different number, so a Retry-After that came
	// from it would be visible.
	s := newServerOver(t, eng, WithConcurrency(4), WithQueueWait(3*time.Second))

	w := post(t, s, "/v1/completions",
		`{"model":"`+fakeName+`","max_tokens":2,"prompt":"hi"}`)
	if got, want := w.Header().Get("Retry-After"), "45"; got != want {
		t.Errorf("Retry-After = %q, want %q: the engine waited, so the engine's "+
			"budget is what a caller can be promised", got, want)
	}

	// And an engine that does not queue falls back to the admitter's budget,
	// which is the number that describes what that request waited.
	plain := &fakeEngine{}
	plain.chatErr = fmt.Errorf("tgo: %w", tgo.ErrQueueTimeout)
	s = newServerOver(t, plain, WithQueueWait(3*time.Second))
	w = post(t, s, "/v1/completions",
		`{"model":"`+fakeName+`","max_tokens":2,"prompt":"hi"}`)
	if got, want := w.Header().Get("Retry-After"), "3"; got != want {
		t.Errorf("Retry-After = %q, want %q", got, want)
	}
}
