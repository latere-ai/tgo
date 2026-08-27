// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/internal/grammar"
	"github.com/latere-ai/tgo/internal/prefix"
	"github.com/latere-ai/tgo/model"
)

// ErrContextExhausted is what a request that does not fit the session's cache
// is refused with.
//
// A refusal and never a truncation (§7, 006 §4): silently dropping the start of
// a user's context produces an answer to a question they did not ask, and
// nothing downstream can tell that from an answer to the one they did.
var ErrContextExhausted = errors.New("tgo: the context is exhausted")

// ErrSessionFailed wraps the error that made a session unusable.
//
// A device failure mid-generation leaves the cache holding a partial write
// whose extent is unknown, and continuing from it would produce plausible text
// from a corrupt state. The session refuses further work with the original
// error attached until [Session.Reset] is called, which is explicit (007-D5).
var ErrSessionFailed = errors.New("tgo: the session failed and has not been reset")

// Session is one conversation: its key/value cache, its position in that
// cache, and the buffers one step binds.
//
// # A Session is not safe for concurrent use
//
// Two goroutines decoding one session would interleave writes into one cache,
// and serialising that internally would hide a caller's bug rather than report
// it (007-D1). Use one session per conversation, and as many sessions
// concurrently as you like: [Model] is safe for concurrent use and holds the
// submission lock that makes it so.
type Session struct {
	m        *Model
	capacity int
	buckets  tensor.Buckets

	keys, values *accel.Buffer
	// shared reports that keys and values belong to the model's block pool, so
	// this session neither cleared them nor closes them.
	shared  bool
	pageBuf *accel.Buffer

	ids        *accel.Buffer
	posq, posk *accel.Buffer
	slots      *accel.Buffer
	lengths    *accel.Buffer
	logits     *accel.Buffer

	step    stepData
	hLogits []float32

	// decodeBind and prefillBind are this session's bindings, built once per
	// shape and mutated nowhere: a decode step rebuilding three hundred weight
	// entries would put a map allocation in every step's host time.
	decodeBind  tensor.Bindings
	prefillBind tensor.Bindings
	prefillRows int

	length  int
	history []int

	// lease is this request's blocks in the model's shared pool, and pages is
	// the page-table row it hands out. Both are nil unless the model was
	// opened with WithPrefixCache(CacheProcess, ...).
	//
	// A lease lives for one request rather than for the conversation, and
	// [Stream.finish] gives it back. Nothing is lost: a complete block is
	// published as it is computed, so the next turn finds it by hash. What is
	// avoided is an idle conversation holding a reference no live one can
	// have, which with B blocks over N sessions is how a pool deadlocks
	// (016 §5, 008 §3).
	lease *prefix.Lease
	pages []int

	failed error
	closed bool
	live   *Stream

	thinking bool
	tools    []chat.ToolSpec

	// salt bounds what this conversation may match in a shared block pool.
	//
	// The empty string is a key of its own and shares with nobody, which is
	// 019-D3's rule applied to blocks rather than to sessions: the same string
	// bounds both, because they are the same question asked of two mechanisms.
	salt string

	// rec instruments the loop (specs/007-engine.md §5.1, 017-D1), one Step
	// per prefill and per decode step. Nil unless [WithRecorder] set it, and a
	// nil one costs the loop one branch per step (017-D3).
	rec *bench.Recorder

	// submit is the seam a test replaces to inject a device failure. There is
	// no other way to fake a fault that only a driver produces, and §8 requires
	// the test.
	submit func(p *tensor.Plan, b tensor.Bindings) error
}

// NewSession allocates a conversation's key/value cache and the buffers one
// step binds.
//
// The cache is the largest thing a request costs: specs/005-kv-cache.md §3's
// M_kv, which [Info.CacheBytesPerSession] reports before anything is
// allocated.
func (m *Model) NewSession(opts ...SessionOption) (*Session, error) {
	o := sessionOptions{context: m.context, thinking: true}
	for _, fn := range opts {
		fn(&o)
	}
	if o.context <= 0 {
		return nil, fmt.Errorf("tgo: the session context is %d; a cache holds at least one "+
			"position", o.context)
	}
	buckets, err := bucketsFor(o.context)
	if err != nil {
		return nil, fmt.Errorf("tgo: %w", err)
	}
	c := m.cfg
	s := &Session{
		m:        m,
		capacity: o.context,
		buckets:  buckets,
		thinking: o.thinking,
		tools:    o.tools,
		salt:     o.salt,
		rec:      o.recorder,
	}
	s.submit = func(p *tensor.Plan, b tensor.Bindings) error {
		return p.Submit(m.dev.Queue(), b).Wait()
	}

	// Every buffer at the largest shape it can take, once. A decode step then
	// binds a prefix of the same allocation rather than asking the device for
	// memory on the hot path.
	rows := o.context
	alloc := func(dst **accel.Buffer, dt accel.DType, n int, label string) error {
		b, err := m.dev.NewBuffer(accel.BufferDescriptor{
			DType: dt, Count: n, Label: label,
			Usage: accel.BufferStorage | accel.BufferCopyDst | accel.BufferCopySrc,
		})
		if err != nil {
			return fmt.Errorf("tgo: allocating %s: %w", label, err)
		}
		*dst = b
		return nil
	}
	cache := c.NumLayers * o.context * c.NumKVHeads * c.HeadDim
	ports := []struct {
		dst   **accel.Buffer
		dt    accel.DType
		n     int
		label string
	}{
		{&s.ids, accel.U32, rows, model.PortIDs},
		{&s.posq, accel.U32, rows * c.NumHeads, model.PortPosQ},
		{&s.posk, accel.U32, rows * c.NumKVHeads, model.PortPosK},
		{&s.slots, accel.U32, rows, model.PortSlots},
		{&s.lengths, accel.U32, 1, model.PortLengths},
		{&s.logits, accel.F32, c.VocabSize, model.PortLogits},
	}
	// The key and value states are the model's when the process shares a block
	// pool, and this session's otherwise. Which one is what decides whether a
	// server's cache footprint scales with concurrency: one pool for every
	// session, or one cache each.
	//
	// Shared states are not allocated here and not closed by Session.Close.
	// They outlive every session, which is the point of them.
	if m.blocks == nil {
		ports = append(ports,
			struct {
				dst   **accel.Buffer
				dt    accel.DType
				n     int
				label string
			}{&s.keys, accel.F32, cache, model.PortKeys},
			struct {
				dst   **accel.Buffer
				dt    accel.DType
				n     int
				label string
			}{&s.values, accel.F32, cache, model.PortValues})
	}
	for _, a := range ports {
		if err := alloc(a.dst, a.dt, a.n, a.label); err != nil {
			_ = s.Close()
			return nil, err
		}
	}
	if m.blocks != nil {
		s.keys, s.values = m.blocks.keys, m.blocks.values
		s.shared = true
		// One entry per block the pool holds: a sequence may in principle hold
		// every one of them, and the port's width is a graph parameter rather
		// than a request's.
		if err := alloc(&s.pageBuf, accel.U32, m.blocks.maxPages(), model.PortPages); err != nil {
			_ = s.Close()
			return nil, err
		}
	} else {
		// The cache starts zeroed rather than merely allocated: a length of
		// zero means no row is read, but a NaN left in device memory by a
		// previous tenant would reach attention through the padded rows of a
		// prefill. The shared pool is cleared once, where it is allocated.
		zero := make([]float32, cache)
		for _, b := range []*accel.Buffer{s.keys, s.values} {
			if err := m.dev.Queue().WriteBuffer(b, 0, zero); err != nil {
				_ = s.Close()
				return nil, fmt.Errorf("tgo: clearing the cache: %w", err)
			}
		}
		// A queue write is batched, and a buffer closed with one outstanding is
		// refused by accel rather than quietly dropped. The clear is part of
		// handing the session over, so it completes here.
		if err := m.dev.Queue().Flush().Wait(); err != nil {
			_ = s.Close()
			return nil, fmt.Errorf("tgo: clearing the cache: %w", err)
		}
	}

	s.step = stepData{
		ids:     make([]uint32, rows),
		posq:    make([]uint32, rows*c.NumHeads),
		posk:    make([]uint32, rows*c.NumKVHeads),
		slots:   make([]uint32, rows),
		lengths: make([]uint32, 1),
	}
	if s.shared {
		s.step.pages = make([]uint32, m.blocks.maxPages())
	}
	s.hLogits = make([]float32, c.VocabSize)
	return s, nil
}

// Reset empties the conversation and clears a failure.
//
// It is the explicit recovery §7 requires: a session that failed mid-generation
// refuses further work until this is called, because the cache holds a partial
// write whose extent is unknown and the only safe reading of it is none.
//
// It also drops everything [WithPrefixCache] could have reused, so the next
// request prefills its whole prompt. That is the cold baseline, and it is what
// a caller who reset asked for.
func (s *Session) Reset() {
	s.release()
	s.rewind(0)
}

// rewind keeps the first n positions of the conversation and drops the rest.
//
// The cache is not rewritten, because nothing reads past the length: a step
// binds lengths[0] and the kernel masks with `pos < lengths[0]`, so a row above
// n is unreachable until some later step scatters over it. Dropping the history
// with the length is what keeps the two agreeing — s.history[i] is the token
// whose key/value state sits at row i, and a reuse decision that trusted a
// stale entry would attend to a token the cache no longer holds.
func (s *Session) rewind(n int) {
	s.length = n
	s.history = s.history[:n]
	s.failed = nil
	s.live = nil
}

// adopt records that the first n positions of this request are already
// computed, whoever computed them.
//
// It is not [Session.rewind] with an argument, and the difference is what a
// shared pool introduced. s.history[i] is the token whose key/value state sits
// at logical position i, and without a pool that run is always a prefix of what
// this session itself scored -- so truncating the history to n was the whole
// operation. With a pool the reused run may have been computed by another
// conversation entirely, and this session's history can be empty while n is a
// hundred. The tokens are the same tokens either way, because that is what the
// pool matched on, so they are taken from the prompt.
func (s *Session) adopt(ids []int, n int) {
	s.length = n
	s.history = append(s.history[:0], ids[:n]...)
	s.failed = nil
	s.live = nil
}

// reusable is how many leading positions of ids this session already holds the
// key/value state for.
//
// # Why a token comparison and not a hash
//
// specs/016-prefix-cache.md keys a shared pool on chained block hashes because
// a hit there means one sequence attending to another sequence's blocks, and
// the blocks must be found before they can be compared. Nothing is shared here:
// the run being matched is this session's own, its tokens are in s.history, and
// comparing them is exact. So there is no hash to collide (016-D9 has nothing
// to protect), no block alignment to round down to (016-D4's up-to-B-1 loss is
// not paid), and no salt to mix (016 §7's oracle needs a second tenant).
//
// # The cap at one token short of the prompt
//
// The cache holds key/value state, not logits. Sampling needs the logits at the
// last prompt position and those come from a forward pass over it, so reusing
// the whole match would leave the request with a warm cache and nothing to
// sample from (016-D10). It is capped here rather than in the caller because
// the caller that would forget is the chat path, where a rendered prompt always
// ends with a fresh assistant opener and the cap never binds.
// # A failed session reuses nothing, structurally
//
// There is no check for it here because there is no path to it. A session whose
// step failed holds a partial write of unknown extent, and [Session.usable]
// refuses every request before one is built (007-D5). The only way back is
// [Session.Reset], which rewinds to zero, so the first request after a failure
// is cold whatever this would have said.
func (s *Session) reusable(ids []int) int {
	if s.m.cacheScope != CacheSession {
		return 0
	}
	n := min(s.length, len(s.history), len(ids)-1, s.m.cachePositions)
	m := 0
	for m < n && s.history[m] == ids[m] {
		m++
	}
	return m
}

// Close releases this session's device memory. It is safe to call more than
// once and reports every failure rather than the first.
//
// It flushes the device queue first, so a session closed while another is
// mid-step waits for that step: accel refuses to close a buffer with a batched
// write outstanding, and the queue is the model's rather than the session's.
// Closing every session before the [Model] is the order accel requires, since
// it closes in order rather than recursively.
func (s *Session) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	s.live = nil
	// The blocks go back before the buffers do. A session closed while holding
	// a lease would leak them out of a pool every other conversation draws
	// from, so the next request gets ErrExhausted rather than this session
	// leaking memory it alone owned.
	s.release()
	var errs []error
	// A step whose submission failed leaves its input writes batched, and
	// accel refuses to close a buffer with one outstanding. Flushing first
	// turns a failed generation into a clean close rather than a second error
	// on top of the first.
	if s.m != nil && s.m.dev != nil {
		errs = append(errs, s.m.dev.Queue().Flush().Wait())
	}
	own := []*accel.Buffer{s.ids, s.posq, s.posk, s.slots, s.lengths, s.logits, s.pageBuf}
	if !s.shared {
		// The shared states are the model's and outlive every session, so a
		// session that borrowed them closes neither.
		own = append(own, s.keys, s.values)
	}
	for _, b := range own {
		if b != nil {
			errs = append(errs, b.Close())
		}
	}
	return errors.Join(errs...)
}

// Chat renders messages through the model's template and generates.
//
// The conversation is what msgs holds: the session's cache is refilled from
// this render, so a caller carries the history in the slice and a partial
// render can never be appended to a cache that ends somewhere else.
//
// Refilled, but not necessarily recomputed. Turn n's render begins with turn
// n-1's, so under [WithPrefixCache] the leading positions the two share are
// taken from the cache and only the new turn is prefilled — the 1-1/n win
// specs/016-prefix-cache.md §1 is about. [Usage.CachedPromptTokens] reports how
// many positions that was.
func (s *Session) Chat(ctx context.Context, msgs []chat.Message, p Policy) (*Stream, error) {
	if err := s.usable(); err != nil {
		return nil, err
	}
	prompt, err := s.m.renderer().Render(msgs, chat.Options{
		Thinking:          s.thinking,
		Tools:             s.tools,
		AddGenerationHint: true,
	})
	if err != nil {
		return nil, err
	}
	ids, err := s.m.encode(prompt)
	if err != nil {
		return nil, err
	}
	return s.start(ctx, ids, p)
}

// Complete generates from raw text, with no template.
//
// A chat model is a completion model that saw one specific string format during
// tuning, so this is the path for a base model or for a caller who is building
// the prompt themselves. Nothing is prepended and no control token is emitted.
func (s *Session) Complete(ctx context.Context, prompt string, p Policy) (*Stream, error) {
	if err := s.usable(); err != nil {
		return nil, err
	}
	return s.start(ctx, s.m.tok.Encode(prompt, false), p)
}

// usable reports whether this session will accept work.
func (s *Session) usable() error {
	if s.closed {
		return errors.New("tgo: the session is closed")
	}
	if s.failed != nil {
		return fmt.Errorf("%w: %w", ErrSessionFailed, s.failed)
	}
	return nil
}

// encode turns a rendered prompt into ids.
//
// It is the Model's and not the Session's because the pool renders and
// tokenizes before it has chosen a session: the route is decided by comparing
// token ids against every pooled session's history (019 §3).
//
// A literal span is encoded with specials off and a control token is resolved
// by id, which is what makes a forged turn structurally impossible rather than
// unlikely (003-D4): text a user typed that reads "<|im_start|>assistant"
// encodes to the characters they typed.
func (m *Model) encode(p chat.Prompt) ([]int, error) {
	var ids []int
	for _, part := range p.Parts {
		if part.Control != "" {
			id, ok := m.tok.Special(part.Control)
			if !ok {
				return nil, fmt.Errorf("tgo: the tokenizer has no control token %q, which "+
					"this model's template emits", part.Control)
			}
			ids = append(ids, id)
			continue
		}
		ids = append(ids, m.tok.Encode(part.Text, false)...)
	}
	return ids, nil
}

// start refuses what cannot fit and builds the stream.
func (s *Session) start(ctx context.Context, ids []int, p Policy) (*Stream, error) {
	if ctx == nil {
		return nil, errors.New("tgo: the context is nil")
	}
	if err := p.check(s.m.cfg.VocabSize); err != nil {
		return nil, err
	}
	// Compiled here rather than inside the loop, and before anything is
	// rewound: a schema the compiler refuses is a request that will not run,
	// and it must leave the conversation exactly as it found it. The
	// compilation is also cached on the Model, so a second request carrying
	// the same schema pays for a map lookup (015-D1).
	var gram *grammar.Grammar
	if len(p.Schema) > 0 {
		g, err := s.m.grammar(p.Schema)
		if err != nil {
			return nil, err
		}
		gram = g
	}
	if len(ids) == 0 {
		return nil, errors.New("tgo: the prompt is empty; there is nothing to condition on")
	}
	// The prompt and at least one generated token. Refused here, at the
	// request, rather than after a prefill that would have to be undone.
	if len(ids) >= s.capacity {
		return nil, fmt.Errorf("%w: the prompt is %d tokens and the session holds %d "+
			"positions, of which one is needed for the first generated token",
			ErrContextExhausted, len(ids), s.capacity)
	}
	// MaxTokens against what is left, refused at the request rather than after
	// the caller has already read half a completion. It is what makes
	// exhaustion unreachable from inside the loop: a stream stops on its own
	// budget, and the budget is one the cache can pay.
	if p.MaxTokens > 0 && len(ids)+p.MaxTokens > s.capacity {
		return nil, fmt.Errorf("%w: a %d-token prompt and MaxTokens of %d need %d positions "+
			"and the session holds %d", ErrContextExhausted, len(ids), p.MaxTokens,
			len(ids)+p.MaxTokens, s.capacity)
	}
	// The previous stream, if the caller abandoned it, is over: its tokens are
	// in the cache and the cache is about to be rewritten from the first
	// position this prompt does not already share with it.
	if s.live != nil {
		s.live.abandon()
	}
	// Every refusal above this line, and nothing below it: a request that does
	// not fit must leave the conversation exactly as it found it, and a rewind
	// that ran before the check would have shortened a cache the caller can
	// still generate from.
	reuse, err := s.acquire(ids, s.salt)
	if err != nil {
		return nil, err
	}
	s.adopt(ids, reuse)
	st := newStream(ctx, s, ids, p, reuse, gram)
	s.live = st
	return st, nil
}

// acquire decides how many leading positions of ids are already computed, and
// where this request's positions will live.
//
// Two answers by two mechanisms, and the split is 016 §4 versus this session's
// own history. Without a shared pool the run being matched is this session's,
// its tokens are in s.history, and comparing them is exact. With one, a match
// is another sequence's blocks and has to be found by hash before it can be
// used at all.
func (s *Session) acquire(ids []int, salt string) (int, error) {
	if !s.shared {
		return s.reusable(ids), nil
	}
	// There is normally nothing to release here: a lease lives for one request
	// and [Stream.finish] gives it back. This covers the request that never
	// produced a stream -- a refusal after the lease, an abandoned start --
	// so a session cannot accumulate two.
	s.release()
	l, err := s.m.blocks.pool.Acquire(prefix.Request{
		IDs: ids, Session: s.salt, Salt: salt,
	})
	if err != nil {
		return 0, fmt.Errorf("tgo: leasing blocks for a %d-token prompt: %w", len(ids), err)
	}
	s.lease, s.pages = l, l.Blocks()
	return l.Reused(), nil
}

// reserve records generated tokens against the lease, allocating blocks as the
// positions need them, and takes the page table those blocks give.
//
// It runs *before* the step that computes their key and value state, because a
// token needs a row to be written to. It does not publish, and that separation
// is the invariant 016 §5 turns on: a published block is immutable and another
// sequence may attend to it before the call returns, so a block offered before
// its KV was written would be read as somebody's context and hold whatever was
// there.
func (s *Session) reserve(toks ...int) error {
	if !s.shared || s.lease == nil {
		return nil
	}
	// Blocks only. What the tokens are is settled by the step, and a hash
	// chained over a token nobody computed names a block holding something
	// else -- [Session.publish] records them once the step lands.
	if err := s.lease.Grow(len(toks)); err != nil {
		return fmt.Errorf("tgo: extending a sequence by %d token(s): %w", len(toks), err)
	}
	s.pages = s.lease.Blocks()
	return nil
}

// publish offers every block whose key/value state the last step computed, and
// adopts the page table that comes back.
//
// The table is returned rather than unchanged because two sequences can miss on
// one prefix concurrently, compute it twice and both publish: the pool keeps one
// block and the loser drops its own and takes the winner's, so the two end up
// sharing a block rather than leaking one.
func (s *Session) publish(toks ...int) error {
	if !s.shared || s.lease == nil {
		return nil
	}
	if err := s.lease.Commit(toks...); err != nil {
		return fmt.Errorf("tgo: recording %d computed token(s): %w", len(toks), err)
	}
	// The session's own length and not the lease's: the lease covers what this
	// conversation may write, and only what a step has written may be offered
	// to another sequence.
	s.pages = s.lease.Publish(s.length)
	return nil
}

// release gives this request's blocks back. Idempotent.
func (s *Session) release() {
	if s.lease != nil {
		s.lease.Release()
		s.lease, s.pages = nil, nil
	}
}

// timings is what one submission cost, in specs/017-benchmarks.md §1's terms
// minus the host's share, which the caller measures around this.
type timings struct {
	submit   time.Duration
	device   time.Duration
	readback time.Duration
}

// run submits one forward pass and returns the last position's logits.
//
// # The lock
//
// [Model.mu] is held from the first buffer write through the fence and the
// readback. tensor.PlanCache returns the same *tensor.Plan for an identical
// graph and a plan refuses a second submission while one is in flight, so two
// sessions decoding at once would otherwise share one decode plan and the
// second would get a failed fence — invisible to -race, because the failure is
// a refused submission rather than a race (007-D9). accel.Queue.ReadBuffer
// flushes the queue, so the readback is inside the lock too: a flush issued
// while another session's graph is in flight waits for work this session did
// not submit.
//
// Time spent waiting for that lock is counted as submit time, which is where
// specs/017-benchmarks.md §1 puts "handing the plan to the queue": under
// concurrency, being allowed to hand it over is what the wait is.
func (s *Session) run(rows int, toks []int, first int) ([]float32, timings, error) {
	var t timings
	if err := s.step.fill(s.m.cfg, rows, toks, first, s.layout()); err != nil {
		return nil, t, err
	}
	start := time.Now()
	plan, err := s.m.plan(rows, s.stateRows(), s.block(), 1, s.cacheDType())
	if err != nil {
		return nil, t, err
	}
	b, err := s.bindings(rows)
	if err != nil {
		return nil, t, err
	}

	s.m.mu.Lock()
	defer s.m.mu.Unlock()

	q := s.m.dev.Queue()
	for _, w := range []struct {
		buf  *accel.Buffer
		data any
	}{
		{s.ids, s.step.ids},
		{s.posq, s.step.posq},
		{s.posk, s.step.posk},
		{s.slots, s.step.slots},
		{s.lengths, s.step.lengths},
	} {
		if err := q.WriteBuffer(w.buf, 0, w.data); err != nil {
			return nil, t, fmt.Errorf("tgo: binding a step's inputs: %w", err)
		}
	}
	if s.shared {
		// The table has to have been sized, and this checks it rather than
		// trusting it because an unsized one is invisible: WriteBuffer over an
		// empty slice writes nothing, leaves the port holding whatever the
		// allocation held, and the step attends to blocks nobody chose --
		// fluent output from a page table that was never bound. That is the
		// bug this line exists for, found by a run that compared a pooled
		// session against a contiguous one and not by any refusal.
		if len(s.step.pages) != s.m.blocks.maxPages() {
			return nil, t, fmt.Errorf("tgo: the page table binding holds %d entries "+
				"and the port declares %d; a step that wrote a short table would "+
				"leave the rest of the port holding whatever was there",
				len(s.step.pages), s.m.blocks.maxPages())
		}
		// The page table is written every step because a decode that crossed
		// into a new block changed it, and because Publish may have swapped an
		// entry for the block that won a race to name the same prefix.
		for i := range s.step.pages {
			if i < len(s.pages) {
				s.step.pages[i] = uint32(s.pages[i])
				continue
			}
			s.step.pages[i] = 0
		}
		if err := q.WriteBuffer(s.pageBuf, 0, s.step.pages); err != nil {
			return nil, t, fmt.Errorf("tgo: binding the page table: %w", err)
		}
	}
	t.submit = time.Since(start)

	start = time.Now()
	if err := s.submit(plan, b); err != nil {
		return nil, t, fmt.Errorf("tgo: submitting a %d-token step: %w", len(toks), err)
	}
	t.device = time.Since(start)

	start = time.Now()
	if err := q.ReadBuffer(s.logits, 0, s.hLogits); err != nil {
		return nil, t, fmt.Errorf("tgo: reading the logits back: %w", err)
	}
	t.readback = time.Since(start)
	return s.hLogits, t, nil
}

// bindings returns this session's bindings for a step of rows tokens, building
// them at most once per shape.
func (s *Session) bindings(rows int) (tensor.Bindings, error) {
	if rows == 1 && s.decodeBind.Buffers != nil {
		return s.decodeBind, nil
	}
	if rows > 1 && s.prefillRows == rows {
		// The base scalar is the step's, not the shape's: a prefill that
		// extends an existing cache starts somewhere other than zero, and a
		// cached binding that kept the first call's value would place every
		// later causal mask at the wrong position.
		s.prefillBind.Scalars[model.ScalarBase] = tensor.U32(s.step.base)
		return s.prefillBind, nil
	}
	c := s.m.cfg
	bufs := make(map[string]accel.BufferView, len(s.m.weightBind)+8)
	for k, v := range s.m.weightBind {
		bufs[k] = v
	}
	for _, e := range []struct {
		name  string
		buf   *accel.Buffer
		count int
	}{
		{model.PortIDs, s.ids, rows},
		{model.PortPosQ, s.posq, rows * c.NumHeads},
		{model.PortPosK, s.posk, rows * c.NumKVHeads},
		{model.PortSlots, s.slots, rows},
		{model.PortLengths, s.lengths, 1},
		{model.PortKeys, s.keys, c.NumLayers * s.stateRows() * c.NumKVHeads * c.HeadDim},
		{model.PortValues, s.values, c.NumLayers * s.stateRows() * c.NumKVHeads * c.HeadDim},
		{model.PortLogits, s.logits, c.VocabSize},
	} {
		v, err := e.buf.View(0, e.count)
		if err != nil {
			return tensor.Bindings{}, fmt.Errorf("tgo: binding %q: %w", e.name, err)
		}
		bufs[e.name] = v
	}
	if s.shared {
		v, err := s.pageBuf.View(0, s.m.blocks.maxPages())
		if err != nil {
			return tensor.Bindings{}, fmt.Errorf("tgo: binding %q: %w", model.PortPages, err)
		}
		bufs[model.PortPages] = v
	}
	// The scalars a decode declares are a strict subset of a prefill's:
	// model.Declare records ScalarBase only above one token, and accel refuses
	// a binding for a scalar the plan does not declare.
	scalars := map[string]tensor.ScalarValue{
		model.ScalarRoPEBase: tensor.F32(c.RoPETheta),
		model.ScalarScale:    tensor.F32(rsqrt(c.HeadDim)),
	}
	if rows > 1 {
		scalars[model.ScalarBase] = tensor.U32(s.step.base)
	}
	b := tensor.Bindings{Buffers: bufs, Scalars: scalars}
	if rows == 1 {
		s.decodeBind = b
		return b, nil
	}
	s.prefillBind, s.prefillRows = b, rows
	return b, nil
}

// layout is where this session's positions live in the states it binds.
func (s *Session) layout() cacheLayout {
	if !s.shared {
		return cacheLayout{rows: s.capacity, limit: s.capacity}
	}
	return cacheLayout{
		rows:  s.m.blocks.positions,
		limit: s.capacity,
		pages: s.pages,
		block: CacheBlock,
	}
}

// stateRows is the row count of the key and value states this session binds,
// which is the graph's Capacity.
func (s *Session) stateRows() int {
	if s.shared {
		return s.m.blocks.positions
	}
	return s.capacity
}

// cacheDType is what this session's key and value states hold: the shared
// pool's when it borrows one, and f32 when it owns its own.
//
// A session's own cache is f32 because it is sized to one conversation and
// halving it buys one conversation's memory; the pool is every conversation's
// and is the allocation that scales with concurrency (specs/005-kv-cache.md §3).
func (s *Session) cacheDType() accel.DType {
	if s.shared {
		return s.m.blocks.dtype
	}
	return accel.F32
}

// block is the graph's block size, and zero when the cache is contiguous.
func (s *Session) block() int {
	if s.shared {
		return CacheBlock
	}
	return 0
}

// fail poisons the session with the error that ended a step.
func (s *Session) fail(err error) error {
	if s.failed == nil {
		s.failed = err
	}
	return err
}

// rsqrt is 1/sqrt(headDim), attention's scale.
//
// Computed in float64 and rounded once, so the value bound as an f32 scalar is
// the nearest f32 to the exact reciprocal square root rather than the result of
// a float32 divide of a float32 root.
func rsqrt(headDim int) float32 {
	return float32(1 / math.Sqrt(float64(headDim)))
}
