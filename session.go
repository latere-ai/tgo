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
	ids          *accel.Buffer
	posq, posk   *accel.Buffer
	slots        *accel.Buffer
	lengths      *accel.Buffer
	logits       *accel.Buffer

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

	failed error
	closed bool
	live   *Stream

	thinking bool
	tools    []chat.ToolSpec

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
	for _, a := range []struct {
		dst   **accel.Buffer
		dt    accel.DType
		n     int
		label string
	}{
		{&s.keys, accel.F32, cache, model.PortKeys},
		{&s.values, accel.F32, cache, model.PortValues},
		{&s.ids, accel.U32, rows, model.PortIDs},
		{&s.posq, accel.U32, rows * c.NumHeads, model.PortPosQ},
		{&s.posk, accel.U32, rows * c.NumKVHeads, model.PortPosK},
		{&s.slots, accel.U32, rows, model.PortSlots},
		{&s.lengths, accel.U32, 1, model.PortLengths},
		{&s.logits, accel.F32, c.VocabSize, model.PortLogits},
	} {
		if err := alloc(a.dst, a.dt, a.n, a.label); err != nil {
			_ = s.Close()
			return nil, err
		}
	}
	// The cache starts zeroed rather than merely allocated: a length of zero
	// means no row is read, but a NaN left in device memory by a previous
	// tenant would reach attention through the padded rows of a prefill.
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

	s.step = stepData{
		ids:     make([]uint32, rows),
		posq:    make([]uint32, rows*c.NumHeads),
		posk:    make([]uint32, rows*c.NumKVHeads),
		slots:   make([]uint32, rows),
		lengths: make([]uint32, 1),
	}
	s.hLogits = make([]float32, c.VocabSize)
	return s, nil
}

// Reset empties the conversation and clears a failure.
//
// It is the explicit recovery §7 requires: a session that failed mid-generation
// refuses further work until this is called, because the cache holds a partial
// write whose extent is unknown and the only safe reading of it is none.
func (s *Session) Reset() {
	s.length = 0
	s.history = s.history[:0]
	s.failed = nil
	s.live = nil
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
	var errs []error
	// A step whose submission failed leaves its input writes batched, and
	// accel refuses to close a buffer with one outstanding. Flushing first
	// turns a failed generation into a clean close rather than a second error
	// on top of the first.
	if s.m != nil && s.m.dev != nil {
		errs = append(errs, s.m.dev.Queue().Flush().Wait())
	}
	for _, b := range []*accel.Buffer{
		s.keys, s.values, s.ids, s.posq, s.posk, s.slots, s.lengths, s.logits,
	} {
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
// render can never be appended to a cache that ends somewhere else. Reusing the
// prefix two consecutive calls share is
// specs/016-prefix-cache.md's and is not done here.
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
	ids, err := s.encode(prompt)
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
// A literal span is encoded with specials off and a control token is resolved
// by id, which is what makes a forged turn structurally impossible rather than
// unlikely (003-D4): text a user typed that reads "<|im_start|>assistant"
// encodes to the characters they typed.
func (s *Session) encode(p chat.Prompt) ([]int, error) {
	var ids []int
	for _, part := range p.Parts {
		if part.Control != "" {
			id, ok := s.m.tok.Special(part.Control)
			if !ok {
				return nil, fmt.Errorf("tgo: the tokenizer has no control token %q, which "+
					"this model's template emits", part.Control)
			}
			ids = append(ids, id)
			continue
		}
		ids = append(ids, s.m.tok.Encode(part.Text, false)...)
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
	// in the cache and the cache is about to be rewritten from position zero.
	if s.live != nil {
		s.live.abandon()
	}
	s.Reset()
	st := newStream(ctx, s, ids, p)
	s.live = st
	return st, nil
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
	if err := s.step.fill(s.m.cfg, rows, toks, first, s.capacity); err != nil {
		return nil, t, err
	}
	start := time.Now()
	plan, err := s.m.plan(rows, s.capacity)
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
		{model.PortKeys, s.keys, c.NumLayers * s.capacity * c.NumKVHeads * c.HeadDim},
		{model.PortValues, s.values, c.NumLayers * s.capacity * c.NumKVHeads * c.HeadDim},
		{model.PortLogits, s.logits, c.VocabSize},
	} {
		v, err := e.buf.View(0, e.count)
		if err != nil {
			return tensor.Bindings{}, fmt.Errorf("tgo: binding %q: %w", e.name, err)
		}
		bufs[e.name] = v
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
