// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"fmt"
	"strconv"
	"time"
)

// Admission, which is a memory bound wearing a concurrency bound's clothes.
//
// specs/005-kv-cache.md makes each session's cache a fixed reservation, so the
// number of sessions that can exist at once is
//
//	N_max = floor((M_available - M_weights) / M_kv(C))
//
// and [WithKVBudget] is that arithmetic. Over the limit requests wait; the wait
// is bounded and so is the number of waiters, because an unbounded queue
// converts a load problem into an out-of-memory one (009-D3). A full queue
// answers 429 with Retry-After rather than growing.
//
// A pooled engine ([WrapPool]) arrives at the same number from the other end.
// Its N sessions are reserved at startup rather than admitted against, so N is
// the concurrency, and [WithConcurrency] states it instead of dividing memory
// twice (specs/019-session-affinity.md §4). It must not be larger than the
// pool. A larger one still bounds the waiters and still answers 429 behind
// them, but the surplus waits inside the engine for a free session, where this
// queue neither counts it nor times it out -- so the Retry-After stops
// describing what that request waits. [New] refuses it.
//
// Without batching this buys no throughput and does not pretend to: concurrent
// requests interleave at submission granularity and the total is what one
// sequence gets. What it buys is that the twenty-first caller is told to come
// back instead of taking the device down.

// admitter is the semaphore and the queue in front of it.
type admitter struct {
	// slots holds one token per session that may exist at once.
	slots chan struct{}

	// queue holds one token per request waiting for a slot. Its capacity is
	// the bound; a request that cannot take a token is refused immediately.
	queue chan struct{}

	// wait is how long a queued request waits before it is refused.
	wait time.Duration

	m *metrics
}

func newAdmitter(concurrency, queue int, wait time.Duration, m *metrics) *admitter {
	return &admitter{
		slots: make(chan struct{}, concurrency),
		queue: make(chan struct{}, queue),
		wait:  wait,
		m:     m,
	}
}

// acquire takes a session slot, or says why it could not.
//
// The returned release must be called exactly once, and releasing it is what
// gives the KV reservation back: a handler that returns without releasing has
// leaked a session's worth of device memory to the process's lifetime.
func (a *admitter) acquire(ctx context.Context) (func(), *apiError) {
	select {
	case a.slots <- struct{}{}:
		// Admitted without waiting. The zero observation is recorded rather
		// than skipped, so the histogram's count is admissions and its shape
		// says what share of them waited at all.
		a.m.waited(0)
		return a.release, nil
	default:
	}

	select {
	case a.queue <- struct{}{}:
	default:
		return nil, a.overloaded("queue_full", "tgo: the queue is full: %d requests are generating and %d "+
			"are waiting", cap(a.slots), cap(a.queue))
	}
	a.m.queue(1)
	defer func() {
		a.m.queue(-1)
		<-a.queue
	}()

	start := time.Now()
	timer := time.NewTimer(a.wait)
	defer timer.Stop()

	select {
	case a.slots <- struct{}{}:
		a.m.waited(time.Since(start))
		return a.release, nil
	case <-timer.C:
		return nil, a.overloaded("queue_timeout", "tgo: waited %s for a session and did not get one: %d "+
			"requests are generating", a.wait, cap(a.slots))
	case <-ctx.Done():
		return nil, &apiError{kind: errClientGone, reason: "client_gone",
			msg: "tgo: the client hung up while queued"}
	}
}

// release returns a slot. It is a method rather than a closure so that two
// calls are a bug in this file rather than in a caller's defer.
func (a *admitter) release() { <-a.slots }

// overloaded builds the 429, with the Retry-After a caller should honour.
func (a *admitter) overloaded(reason, format string, args ...any) *apiError {
	e := &apiError{kind: errOverloaded, reason: reason, retryAfter: retryAfter(a.wait)}
	e.msg = fmt.Sprintf(format, args...)
	return e
}

// retryAfter is the queue's budget rounded up to a whole second, with a floor
// of one: a Retry-After of zero is an invitation to retry in a tight loop.
func retryAfter(wait time.Duration) string {
	s := int(wait / time.Second)
	if wait%time.Second != 0 {
		s++
	}
	if s < 1 {
		s = 1
	}
	return strconv.Itoa(s)
}
