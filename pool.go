// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
)

// Pool is N conversations' worth of key/value cache, held for the process's
// life, with each request routed to the session that already holds the longest
// matching prefix (specs/019-session-affinity.md).
//
// # Why a pool rather than a session per request
//
// [WithPrefixCache] reuses what a session already scored, and a session that is
// closed at the end of the request it was made for never sees a second turn. So
// a server that opens one session per request reuses nothing however good the
// matching is: the history it would match against was destroyed with the
// session that held it (016 §7.2). A pool keeps the history, and the request is
// routed to it.
//
// The unit of sharing is the session, not the block. Two conversations with the
// same system prompt still pay for it twice, which is what
// specs/016-prefix-cache.md §4's block pool would fix and what this cannot at
// any pool size (019-D1). What this needs, and 016 does not have, is nothing:
// a session's cache is contiguous and single-owner, so a row's index is its
// position and no page table is involved.
//
// # What it costs
//
// The key/value cache of every pooled session is reserved when the pool is
// built and released only by [Pool.Close], so a process that served one request
// holds N sessions' cache for its life (019-D2). That is the dominant resident
// cost of a large model, and it is paid whether or not a second request ever
// arrives. In exchange, admission is a counting semaphore rather than
// arithmetic over free device memory: N is the concurrency limit and the reuse
// depth at once.
//
// A Pool is safe for concurrent use. A [Lease] is not: it belongs to the one
// request that took it.
type Pool struct {
	m *Model

	// sem holds one token per free session. Taking a token is what guarantees
	// [Pool.route] finds an idle entry, which is why routing can pick the best
	// of them rather than waiting for a particular one: a channel of sessions
	// would hand out whichever was returned first.
	sem chan struct{}

	mu      sync.Mutex
	entries []*poolEntry

	// tick orders the entries by last use. A counter and not a clock: two
	// releases inside one timer granularity would be indistinguishable, and
	// the coldest of them is exactly the question this answers.
	tick uint64

	closed bool
}

// poolEntry is one pooled session and what routing needs to know about it.
type poolEntry struct {
	s    *Session
	busy bool

	// key is the affinity key of the last request that used this session, and
	// the empty string when that request carried none. Matching compares it
	// for equality in both directions, so an unkeyed request never reads a
	// keyed session's history and a keyed one never reads an unkeyed session's
	// (019-D3).
	key string

	// used is the tick at which the last request released this session. A
	// session that has never served one is zero, so it is colder than any
	// session that has, and a fresh pool fills up before it evicts anything.
	used uint64
}

// PoolRequest is what one request needs of a pooled session: the shape of the
// prompt it renders, and the key that bounds what it may match.
type PoolRequest struct {
	// Tools are the functions the model may call and Thinking says whether the
	// assistant may open a thinking block. Both are rendered into the prompt,
	// so both change the token ids a match is computed over.
	Tools    []chat.ToolSpec
	Thinking bool

	// Key is the affinity key. A request may match only a session whose last
	// request carried the same key, and the empty string is a key of its own:
	// a caller who supplies nothing shares with nobody rather than with
	// everybody (019-D3).
	//
	// tgo has no notion of a tenant (009 §7), so the key is whatever the layer
	// in front supplies. specs/016-prefix-cache.md §7.1's cache_salt is what
	// the server puts here.
	Key string

	// Recorder instruments the session's loop for the duration of the lease,
	// one [bench.Step] per prefill and per decode step. It is the caller's,
	// one per request, and is unset again when the lease is released.
	Recorder *bench.Recorder
}

// NewPool builds a pool of n sessions, allocating every one of their caches
// now.
//
// Now, rather than on the first request that needs them: n sessions' key/value
// cache is what this process will hold for its life, so a device that cannot
// hold it must say so at startup rather than under load (019-D2).
//
// The pool must be closed before the [Model] is.
func (m *Model) NewPool(n int) (*Pool, error) {
	if n < 1 {
		return nil, fmt.Errorf("tgo: a pool of %d sessions holds no conversation; it needs "+
			"at least one", n)
	}
	p := &Pool{m: m, sem: make(chan struct{}, n), entries: make([]*poolEntry, 0, n)}
	for i := range n {
		s, err := m.NewSession()
		if err != nil {
			_ = p.Close()
			return nil, fmt.Errorf("tgo: reserving session %d of %d for the pool: %w",
				i+1, n, err)
		}
		p.entries = append(p.entries, &poolEntry{s: s})
	}
	return p, nil
}

// Size is how many sessions the pool holds, which is also how many requests can
// generate at once.
func (p *Pool) Size() int { return len(p.entries) }

// Close releases every pooled session's device memory.
//
// It is called once, after the last lease is released and before the [Model] is
// closed, which is the order accel requires. A session still leased is closed
// anyway and reported: a pool closed under a live request is a shutdown that
// ran out of grace, and leaking the memory would be the worse answer.
func (p *Pool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true
	var errs []error
	leased := 0
	for _, e := range p.entries {
		if e.busy {
			leased++
		}
		errs = append(errs, e.s.Close())
	}
	if leased > 0 {
		errs = append(errs, fmt.Errorf("tgo: the pool was closed with %d session(s) still "+
			"leased; their requests were still generating", leased))
	}
	return errors.Join(errs...)
}

// Acquire takes one of the pool's sessions for one request, waiting until one
// is free.
//
// The token is taken here and the session is chosen at the first [Lease.Chat]
// or [Lease.Complete], because routing needs the request's token ids and those
// exist only once the prompt is rendered. The two are safe apart: N tokens over
// N sessions means a lease that holds a token always finds an idle session,
// whichever order the leases pick in.
//
// The returned lease must be released exactly once. A handler that returns
// without releasing has taken one of the N sessions out of the pool for the
// life of the process.
func (p *Pool) Acquire(ctx context.Context, req PoolRequest) (*Lease, error) {
	if ctx == nil {
		return nil, errors.New("tgo: the context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case p.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	p.mu.Lock()
	closed := p.closed
	p.mu.Unlock()
	if closed {
		<-p.sem
		return nil, errors.New("tgo: the pool is closed")
	}
	return &Lease{p: p, req: req}, nil
}

// Lease is one request's hold on one pooled session.
//
// It is not safe for concurrent use, and neither is the [Stream] it starts: a
// lease is one request's, exactly as a [Session] is one conversation's
// (007-D1).
type Lease struct {
	p   *Pool
	req PoolRequest

	// e is the session this lease routed to, and is nil until the first
	// generation. A second generation on one lease keeps the same session:
	// re-routing would leave the first one marked busy for the rest of the
	// process's life.
	e *poolEntry

	// st is the last stream this lease started. It is held for [Lease.Reused]
	// alone: [Stream.finish] clears the session's live pointer, so a lease
	// that read the reuse count through the session would report zero for
	// every request that ran to completion -- which is every request an
	// isolation test measures.
	st *Stream

	released bool
}

// Chat renders messages through the model's template and generates, on the
// pooled session that already holds the longest matching prefix.
//
// The render happens before the routing and not inside a session, because the
// route is chosen by comparing token ids and the ids exist only after the
// prompt is rendered and tokenized (016 §2).
func (l *Lease) Chat(ctx context.Context, msgs []chat.Message, p Policy) (*Stream, error) {
	if err := l.usable(); err != nil {
		return nil, err
	}
	prompt, err := l.p.m.renderer().Render(msgs, chat.Options{
		Thinking:          l.req.Thinking,
		Tools:             l.req.Tools,
		AddGenerationHint: true,
	})
	if err != nil {
		return nil, err
	}
	ids, err := l.p.m.encode(prompt)
	if err != nil {
		return nil, err
	}
	return l.generate(ctx, ids, p)
}

// Complete generates from raw text, with no template. It routes exactly as
// [Lease.Chat] does.
func (l *Lease) Complete(ctx context.Context, prompt string, p Policy) (*Stream, error) {
	if err := l.usable(); err != nil {
		return nil, err
	}
	return l.generate(ctx, l.p.m.tok.Encode(prompt, false), p)
}

// Reused is how many leading prompt positions the last generation took from the
// session's cache, and is zero before one has started.
//
// It is [Usage.CachedPromptTokens] for a caller who has the lease and not the
// stream, and it is the number an isolation test reads: a cache hit is faster
// than a miss, so the reuse count is what a timing oracle would be measuring
// (019 §5).
func (l *Lease) Reused() int {
	if l.st == nil {
		return 0
	}
	return l.st.reused
}

// usable reports whether this lease will accept work.
func (l *Lease) usable() error {
	if l.released {
		return errors.New("tgo: the lease has been released and its session is another " +
			"request's")
	}
	return nil
}

// generate routes the request and starts it.
func (l *Lease) generate(ctx context.Context, ids []int, p Policy) (*Stream, error) {
	if len(ids) == 0 {
		return nil, errors.New("tgo: the prompt is empty; there is nothing to condition on")
	}
	if l.e == nil {
		e, matched := l.p.route(ids, l.req.Key)
		l.e = e
		if !matched {
			// The cold path rewinds here rather than leaving it to
			// [Session.start], which recomputes the match without the key.
			// This entry was chosen for its coldness and may still share a
			// leading run with what it holds -- under another key. Emptying
			// the history is what makes 019-D3 structural rather than a
			// second comparison that has to agree with the first.
			e.s.rewind(0)
		}
	}
	s := l.e.s
	// The session's own render options are kept in step with the request's,
	// even though this path renders before it reaches the session: a session
	// whose fields said one thing while its history said another is a trap for
	// the next reader, and [Session.Chat] on a pooled session would use them.
	s.thinking, s.tools = l.req.Thinking, l.req.Tools
	s.rec = l.req.Recorder
	// The same key bounds both mechanisms. It decides which session this
	// request may be routed to (019-D3) and, under a process-scoped block
	// pool, which blocks it may match -- and they have to be the same string
	// or a request excluded from a session's history would reach the same
	// tokens through the pool a layer down.
	s.salt = l.req.Key
	if err := s.usable(); err != nil {
		return nil, err
	}
	st, err := s.start(ctx, ids, p)
	if err != nil {
		return nil, err
	}
	l.st = st
	return st, nil
}

// Release returns the session to the pool with its history intact, which is the
// one structural change specs/019-session-affinity.md §2 makes.
//
// # The history is truncated to what the device actually holds
//
// A request that ends early -- cancelled, disconnected, or failed on the device
// -- must leave the session advertising exactly the positions whose key/value
// state was written, or the next request to match against it attends to state
// that does not exist, and that is silent wrong output on a different request
// (019-D5).
//
// Session.length is that number already, and it is the number this rewinds to.
// Two invariants make it the right one:
//
//   - length and the history advance together. [Stream.advance] appends the
//     suffix and sets the length in the same branch, and only after the step
//     returned; a step that failed does neither. So the rewind moves nothing
//     for a request that ended between steps, which is every cancellation.
//   - a failed step wrote nothing below the length. [stepData.fill] gives real
//     row i the slot first+i, where first is the length the step started from,
//     and gives every pad row a slot at the capacity, which tensor.ScatterRows
//     drops. So the extent of a partial write is known rather than unknown,
//     and it is entirely above the valid prefix.
//
// What the rewind therefore does, and what nothing else does, is clear the
// failure a device error left behind. Without it one bad request would take a
// session out of the pool for the life of the process: 007-D5 refuses further
// work on a failed session until [Session.Reset], which is the right answer for
// a session a caller owns and the wrong one for a session the pool is about to
// hand to somebody else. The second invariant above is what makes clearing it
// sound rather than hopeful.
func (l *Lease) Release() {
	if l.released {
		return
	}
	l.released = true
	if e := l.e; e != nil {
		p := l.p
		p.mu.Lock()
		// The stream, if the caller abandoned it mid-completion, is over: the
		// next request owns this session, and a Next() on the old stream would
		// run a step on somebody else's conversation.
		if e.s.live != nil {
			e.s.live.abandon()
		}
		e.s.rewind(e.s.length)
		e.s.rec = nil
		e.key = l.req.Key
		e.busy = false
		p.tick++
		e.used = p.tick
		p.mu.Unlock()
	}
	<-l.p.sem
}

// route picks the session this request will run on, and reports whether it was
// picked for a prefix it already holds.
//
// specs/019-session-affinity.md §3.2: routing chooses what to destroy as much
// as what to reuse, because rewinding a session to the matched prefix throws
// away everything after it. So the longest match wins, and among equal matches
// the shortest history wins -- which is what stops a 40-token match from
// discarding an 8000-token history another conversation would have hit on. A
// request that matches nothing goes to the coldest session by last use and not
// to the emptiest: an empty session is usually empty because it was just
// evicted, and the coldest is the one whose owner is least likely to return.
//
// The caller holds a semaphore token, so at least one entry is idle and this
// never returns nil.
func (p *Pool) route(ids []int, key string) (*poolEntry, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	var best *poolEntry
	bestMatch := 0
	for _, e := range p.entries {
		if e.busy || e.key != key {
			continue
		}
		m := e.s.reusable(ids)
		if m == 0 {
			continue
		}
		if best == nil || m > bestMatch ||
			(m == bestMatch && len(e.s.history) < len(best.s.history)) {
			best, bestMatch = e, m
		}
	}
	if best == nil {
		for _, e := range p.entries {
			if e.busy {
				continue
			}
			if best == nil || e.used < best.used {
				best = e
			}
		}
	}
	best.busy = true
	return best, bestMatch > 0
}
