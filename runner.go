// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tgo

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/internal/grammar"
)

// Runner is many conversations sharing one forward pass.
//
// specs/022-batched-serving.md. A [Session] owns a forward pass, so B
// concurrent requests read the weights B times per token produced: one step
// over B sequences moves (W + B·A)/β bytes for weight bytes W, per-sequence
// traffic A and bandwidth β, and B separate steps move B·W where one step moves
// W. Throughput under a session per request is what one sequence gets.
//
// A Runner is a [Scheduler], a [Queue] in front of its admission, and one
// goroutine that drives them. A request renders, waits for a slot, and then
// reads events; it never runs a step and never sees a logits row.
//
// # Why one driver goroutine
//
// [Scheduler.Step] serialises on the scheduler's own lock and [Batch.Step]
// takes the model's submission lock (007-D9), so B goroutines calling Step
// would interleave rather than batch -- which is the behaviour a session per
// request already has, with more machinery (022-D3). One goroutine steps,
// samples every slot and sends events; the request's goroutine receives them.
//
// # What it needs of the model
//
// A shared block pool. [Model.NewBatch] refuses a model without one, because
// sequences that step together have different lengths and a contiguous cache
// would pad every one of them to the longest. So a Runner exists exactly under
// [CacheProcess] (022-D1), and [WrapPool]'s session pool stays the answer for
// the other two scopes.
type Runner struct {
	m       *Model
	sched   *Scheduler
	q       *Queue
	rec     *bench.Recorder
	backlog int

	// runs is what the driver is generating for, indexed by slot. Only the
	// driver writes an entry's decode state; the request's goroutine reads
	// nothing of it, which is what keeps a step free of a lock.
	mu   sync.Mutex
	runs []*slotRun

	// domain is the prefix of every salt this runner mints, sixteen random
	// bytes read once. See [Runner.salt].
	domain string
	minted atomic.Uint64

	// wake tells the driver that a slot was filled, so it steps rather than
	// spinning on an empty batch.
	wake chan struct{}
	done chan struct{}
	// stopped closes when the driver has returned.
	stopped chan struct{}

	closeOnce sync.Once
	closeErr  error
}

// DefaultReserve is how many positions beyond its prompt an admitted sequence
// holds blocks for when a deployment names no number.
//
// specs/008-scheduler.md §3's R, and the difference between admitting a request
// and promising it can finish. It is one number for the whole runner, which
// [022-D7](specs/022-batched-serving.md) makes per request in a later pass: a
// single R is either larger than most requests need, which admits fewer than
// the device holds, or smaller than some request needs, which is the admission
// the promise exists to prevent.
const DefaultReserve = 512

// RunnerOptions are what a deployment chooses about the batch.
type RunnerOptions struct {
	// Slots is the batch width: how many sequences generate at once. It is
	// fixed for the life of the runner, because batch is a leading dimension
	// on every port and a step that changed it would be a different compiled
	// graph (008-D1).
	Slots int

	// Chunk is how many prompt tokens one step prefills for one sequence.
	// Zero takes [DefaultChunk].
	Chunk int

	// Reserve is §3's R. Zero takes [DefaultReserve].
	Reserve int

	// Queue configures the wait in front of admission. Its zero value is
	// [NewQueue]'s defaults, which scale with Slots.
	Queue QueueOptions

	// Backlog is how many steps' events a request may fall behind by before it
	// is dropped as a slow consumer. Zero takes [DefaultBacklog].
	//
	// It is a deployment number rather than a constant because it is the one
	// place a slot's memory and a client's tolerance trade against each other:
	// larger holds more events per slot and forgives a slower reader, and
	// smaller frees a slot sooner from a reader that has stopped.
	Backlog int

	// Recorder instruments the batched loop. It is the runner's and not a
	// request's: a step carries several sequences, so the step count is a
	// property of the batch and the token count is a property of a request.
	// A per-request recorder is [RunRequest.Recorder] and sees the steps that
	// request was in.
	Recorder *bench.Recorder
}

// RunRequest is what one request needs its prompt rendered and keyed with.
type RunRequest struct {
	// Tools are the functions the model may call, rendered into the system
	// turn.
	Tools []chat.ToolSpec

	// Thinking says whether the assistant may open a thinking block.
	Thinking bool

	// Key is the request's cache_salt: the isolation domain the shared pool's
	// block hashes are seeded with (016 §7.1).
	//
	// It is not an affinity key here. Under a block pool the reuse is keyed on
	// chained block hashes shared across every slot, so which slot a request
	// lands in does not change what it reuses (022-D2).
	//
	// **Empty means share with nothing**, not share with everything: the runner
	// mints a salt unique to the request. See [Runner.salt] for why the
	// alternative is a membership test over another tenant's prompt.
	Key string

	// Recorder instruments this request. It sees the steps this request was
	// in, each with the batch width it ran at.
	Recorder *bench.Recorder
}

// NewRunner builds a batched runner over the model's shared block pool.
func (m *Model) NewRunner(o RunnerOptions) (*Runner, error) {
	if o.Reserve == 0 {
		o.Reserve = DefaultReserve
	}
	if o.Backlog == 0 {
		o.Backlog = DefaultBacklog
	}
	if o.Backlog < 1 {
		return nil, fmt.Errorf("tgo: the runner's backlog is %d; a request that may "+
			"fall behind by no steps is dropped by its first one", o.Backlog)
	}
	s, err := m.NewScheduler(o.Slots, SchedulerOptions{Chunk: o.Chunk, Reserve: o.Reserve})
	if err != nil {
		return nil, err
	}
	q, err := NewQueue(s, o.Queue)
	if err != nil {
		_ = s.Close()
		return nil, err
	}
	domain, err := mintDomain()
	if err != nil {
		_ = q.Close()
		_ = s.Close()
		return nil, err
	}
	r := &Runner{
		m: m, sched: s, q: q, rec: o.Recorder, backlog: o.Backlog, domain: domain,
		runs:    make([]*slotRun, s.Slots()),
		wake:    make(chan struct{}, 1),
		done:    make(chan struct{}),
		stopped: make(chan struct{}),
	}
	go r.drive()
	return r, nil
}

// Slots is how many sequences the runner generates at once.
func (r *Runner) Slots() int { return r.sched.Slots() }

// mintDomain reads the sixteen random bytes every synthesised salt is prefixed
// with. See [Runner.salt].
func mintDomain() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("tgo: reading the runner's isolation domain: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}

// salt is the isolation domain a request's block hashes are seeded with.
//
// A request that names one gets its own, and shares with the other requests
// that named the same one, which is what cache_salt is for
// (specs/016-prefix-cache.md §7.1).
//
// **A request that names none gets a salt unique to itself, so it shares with
// nothing.** [016-D7](specs/016-prefix-cache.md) and 019-D3 both say an unkeyed
// request shares with nobody rather than with everybody, and under a pooled
// session that is true by construction because routing compares the key. Under
// a shared block pool it is false: the seed's domain is empty for
// [CacheProcess] (internal/prefix/prefix.go), so every unsalted request hashes
// into one domain and two tenants with the same system prompt seed identically
// — the second one's first token arrives fast, which is a membership test over
// the first one's prompt (specs/022-batched-serving.md §7).
//
// The minted salt is **sixteen random bytes and a counter**, not a counter
// alone. A caller's salt is used verbatim, so any predictable namespace can be
// named by a request and shared into; randomness is what makes the domain
// unreachable rather than merely unlikely to be guessed.
//
// The cost is that a conversation's second turn through a runner reuses nothing
// unless its client sends a cache_salt. That is 016-D7's rule kept rather than
// broken: cross-request sharing is a deployment's decision, and this is what
// lets batching be the default without making sharing one (022-D8).
func (r *Runner) salt(key string) string {
	if key != "" {
		return key
	}
	return r.domain + "-" + strconv.FormatUint(r.minted.Add(1), 36)
}

// Queue is the admission queue in front of the batch, for a caller reporting
// what it measured (021 §7).
func (r *Runner) Queue() *Queue { return r.q }

// Chat renders messages through the model's template and generates.
func (r *Runner) Chat(ctx context.Context, req RunRequest, msgs []chat.Message,
	p Policy) (*SlotStream, error) {

	prompt, err := r.m.renderer().Render(msgs, chat.Options{
		Thinking: req.Thinking, Tools: req.Tools, AddGenerationHint: true,
	})
	if err != nil {
		return nil, err
	}
	ids, err := r.m.encode(prompt)
	if err != nil {
		return nil, err
	}
	return r.start(ctx, req, ids, p)
}

// Complete generates from raw text, with no template.
func (r *Runner) Complete(ctx context.Context, req RunRequest, prompt string,
	p Policy) (*SlotStream, error) {

	return r.start(ctx, req, r.m.tok.Encode(prompt, false), p)
}

// start refuses what cannot run, waits for a slot, and hands back the stream.
func (r *Runner) start(ctx context.Context, req RunRequest, ids []int, p Policy) (
	*SlotStream, error) {

	if ctx == nil {
		return nil, errors.New("tgo: the context is nil")
	}
	if err := p.check(r.m.cfg.VocabSize); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, errors.New("tgo: the prompt is empty; there is nothing to " +
			"condition on")
	}
	capacity := r.m.Info().Context
	if len(ids) >= capacity {
		return nil, fmt.Errorf("%w: the prompt is %d tokens and a sequence holds %d "+
			"positions, of which one is needed for the first generated token",
			ErrContextExhausted, len(ids), capacity)
	}
	if p.MaxTokens > 0 && len(ids)+p.MaxTokens > capacity {
		return nil, fmt.Errorf("%w: a %d-token prompt and MaxTokens of %d need %d "+
			"positions and a sequence holds %d", ErrContextExhausted, len(ids),
			p.MaxTokens, len(ids)+p.MaxTokens, capacity)
	}
	// Compiled before admission: a schema the compiler refuses is a request
	// that will not run, and it must not have taken a slot to find that out
	// (015-D1).
	var gram *grammar.Grammar
	if len(p.Schema) > 0 {
		g, err := r.m.grammar(p.Schema)
		if err != nil {
			return nil, err
		}
		gram = g
	}
	max := p.MaxTokens
	if max <= 0 {
		max = capacity - len(ids)
	}

	// The reserve is this request's own budget and not one number for the
	// deployment: a single R is either larger than most requests need, which
	// admits fewer than the device holds, or smaller than some request needs,
	// which is the admission the promise exists to prevent (022-D7). max is
	// Policy.MaxTokens, or what the capacity left after the prompt allowed.
	slot, err := r.q.Admit(ctx, r.q.NewTicket(), ids, r.salt(req.Key), max)
	if err != nil {
		return nil, err
	}
	run := &slotRun{
		dec:     newDecoder(r.m, p, max, gram),
		history: append(make([]int, 0, len(ids)+max), ids...),
		steps:   make(chan slotStep, r.backlog),
		rec:     req.Recorder,
		first:   time.Now(),
	}
	run.dec.usage.PromptTokens = len(ids)
	run.dec.usage.CachedPromptTokens = r.sched.Reused(slot)

	r.mu.Lock()
	r.runs[slot] = run
	r.mu.Unlock()
	r.nudge()

	return &SlotStream{ctx: ctx, run: run, steps: run.steps}, nil
}

// Close stops the driver, refuses the waiters and releases the batch.
func (r *Runner) Close() error {
	r.closeOnce.Do(func() {
		close(r.done)
		<-r.stopped
		r.closeErr = errors.Join(r.q.Close(), r.sched.Close())
	})
	return r.closeErr
}

func (r *Runner) nudge() {
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

// DefaultBacklog is how many steps' events a request may fall behind by.
//
// A client that stops reading must not let its slot accumulate an unbounded
// queue on the driver, and it must not stall the sequences beside it either
// (022 §4). So the channel is bounded, and a request that fills it is a slow
// consumer: the driver drops it the way it drops a disconnect, rather than
// waiting on it with the whole batch behind.
const DefaultBacklog = 256

// ErrSlowConsumer is what a request is ended with when it stopped reading its
// own events for long enough to fill its channel.
var ErrSlowConsumer = errors.New("tgo: the caller stopped reading its events")

// slotStep is one step's output for one slot, crossing from the driver to the
// request's goroutine.
//
// It is a step and not an event because a step can produce several events, and
// because the log probabilities belong to the step: the decoder reuses their
// backing array, so a reader on another goroutine must be handed the values
// rather than the slice the driver keeps writing (030-D1).
type slotStep struct {
	events []Event
	probs  []TokenProb
	usage  Usage

	done    bool
	err     error
	reason  StopReason
	stopSeq string
}

// slotRun is one request as the driver sees it.
type slotRun struct {
	dec *decoder

	// history is the tokens this sequence has scored. It is the runner's copy
	// and not the scheduler's: schedState.prompt is appended to by
	// Scheduler.Feed, and two owners of one slice is the aliasing
	// Scheduler.Admit's copy exists to prevent.
	history []int

	steps chan slotStep
	rec   *bench.Recorder
	first time.Time

	// gone is set by the request's goroutine when it stops reading, and is
	// read by the driver at the step boundary. It is atomic rather than
	// locked because it is one bit written once and read every step.
	gone atomic.Bool

	// ttft is whether the first token has been reported to the recorder.
	ttft bool

	// dropped is why the driver closed the channel without a final step. It is
	// written before the close and read after it, so the close orders the two.
	dropped error
}

// drive is the one goroutine that steps the batch.
func (r *Runner) drive() {
	defer close(r.stopped)
	for {
		select {
		case <-r.done:
			r.failAll(nil)
			return
		default:
		}
		if !r.live() {
			// Nothing admitted. Wait for one rather than spinning a step that
			// would carry no rows.
			select {
			case <-r.done:
				r.failAll(nil)
				return
			case <-r.wake:
			}
			continue
		}
		start := time.Now()
		res, t, err := r.sched.step()
		if err != nil {
			r.failAll(err)
			return
		}
		for _, p := range res.Produced {
			r.produced(p)
		}
		// The host term is what the whole loop took minus the three the step
		// measured, which is 017 §1's model: the four are exhaustive, so the
		// host's share is a subtraction rather than a fifth measurement with a
		// gap between them. Sampling, masking and detokenizing every slot are
		// in it, which is what a batched loop puts on the host that a session
		// per request also does -- once per slot either way.
		host := max(time.Since(start)-t.submit-t.device-t.readback, 0)
		r.record(res, t, host)
	}
}

// live reports whether any slot is generating.
func (r *Runner) live() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, run := range r.runs {
		if run != nil {
			return true
		}
	}
	return false
}

// produced turns one slot's share of a step into that request's events.
func (r *Runner) produced(p Produced) {
	r.mu.Lock()
	run := r.runs[p.Slot]
	r.mu.Unlock()
	if run == nil {
		return
	}
	if run.gone.Load() {
		// The caller walked away. The step it was inside has finished -- accel
		// has no cancel on a submitted queue (007-D9) -- so the slot is
		// released here, at the step boundary, and its blocks go back to the
		// shared pool (022-D5).
		r.release(p.Slot)
		return
	}
	// A chunk that is not the last chunk produces logits for a token in the
	// middle of the prompt: a real distribution over a token the caller
	// already has. Sampling it would be the mistake Sampleable exists to
	// prevent.
	if !p.Sampleable() {
		return
	}
	done, err := run.dec.consume(p.Logits, run.history)
	if err != nil {
		run.dec.end(err)
	} else if done {
		run.dec.end(nil)
	}
	tok := run.dec.feed
	over := done || err != nil
	if !run.send(r, p.Slot, over) {
		// The caller stopped reading and the driver dropped it, which released
		// the slot already.
		return
	}
	if over {
		r.release(p.Slot)
		return
	}
	run.history = append(run.history, tok)
	if ferr := r.sched.Feed(p.Slot, tok); ferr != nil {
		run.dec.end(ferr)
		run.send(r, p.Slot, true)
		r.release(p.Slot)
	}
}

// send hands one step's output to the request's goroutine, and reports whether
// the run is still alive.
func (run *slotRun) send(r *Runner, slot int, done bool) bool {
	d := run.dec
	st := slotStep{
		events: d.queue, probs: append([]TokenProb(nil), d.probs...),
		usage: d.usage, done: done, err: d.err, reason: d.reason,
		stopSeq: d.stopSeq,
	}
	// The decoder's queue is handed over rather than copied, and it takes a
	// fresh one. The events hold strings the detokenizer already allocated, so
	// what this costs is one slice header per step that produced anything.
	d.queue, d.head = nil, 0
	select {
	case run.steps <- st:
	default:
		// Full: the caller stopped reading. Dropping it here is what keeps a
		// slow consumer from becoming backpressure on every other slot in the
		// batch (022 §4).
		run.gone.Store(true)
		run.dropped = fmt.Errorf("%w: %d steps of events are unread", ErrSlowConsumer,
			cap(run.steps))
		close(run.steps)
		r.release(slot)
		return false
	}
	if done {
		close(run.steps)
	}
	return true
}

// release ends a slot and gives its blocks back.
func (r *Runner) release(slot int) {
	r.mu.Lock()
	r.runs[slot] = nil
	r.mu.Unlock()
	_ = r.sched.Finish(slot)
}

// record writes one step to the runner's recorder and to the recorder of every
// request that was in it.
//
// The batch width is the number of slots the step carried, which is the field
// specs/022-batched-serving.md §11's gate reads: a session per request sets it
// to 1 because that is what it is.
func (r *Runner) record(res StepResult, t timings, host time.Duration) {
	width := len(res.Produced)
	if width == 0 {
		return
	}
	// A step can be both: 008 §5's mix is a prefill chunk and the decodes
	// beside it in one dispatch. bench.Phase has one value, so a step that
	// carried any prompt tokens is a prefill and its tokens are the prompt
	// tokens -- which keeps a chunked prefill's cost out of the decode
	// distribution rather than averaged into it.
	phase := bench.Decode
	tokens := res.Decodes
	if res.PrefillTokens > 0 {
		phase, tokens = bench.Prefill, res.PrefillTokens
	}
	s := bench.Step{
		Phase: phase, Tokens: tokens, Batch: width,
		Host: host, Submit: t.submit, Device: t.device, Readback: t.readback,
	}
	if r.rec.Enabled() {
		r.rec.Step(s)
	}
	r.mu.Lock()
	runs := append([]*slotRun(nil), r.runs...)
	r.mu.Unlock()
	for _, p := range res.Produced {
		run := runs[p.Slot]
		if run == nil || !run.rec.Enabled() {
			continue
		}
		if !run.ttft && p.Sampleable() {
			run.ttft = true
			run.rec.TTFT(time.Since(run.first))
		}
		run.rec.Step(s)
	}
}

// failAll ends every live request, which is what a driver that is stopping owes
// the callers still reading.
func (r *Runner) failAll(err error) {
	r.mu.Lock()
	runs := r.runs
	r.runs = make([]*slotRun, len(r.runs))
	r.mu.Unlock()
	for slot, run := range runs {
		if run == nil {
			continue
		}
		if err == nil {
			err = errors.New("tgo: the runner was closed while this request was " +
				"generating")
		}
		run.dec.end(err)
		st := slotStep{events: run.dec.queue, usage: run.dec.usage, done: true, err: err}
		select {
		case run.steps <- st:
		default:
		}
		close(run.steps)
		_ = r.sched.Finish(slot)
	}
}

// SlotStream is one request's completion, read from the batch that produced it.
//
// It is [Stream]'s surface over a channel rather than over a forward pass. The
// events are the same values a single session yields, produced on the driver's
// goroutine and received here (022 §4).
type SlotStream struct {
	ctx   context.Context
	run   *slotRun
	steps <-chan slotStep

	step slotStep
	i    int
	cur  Event

	done    bool
	err     error
	usage   Usage
	probs   []TokenProb
	reason  StopReason
	stopSeq string
}

// Next advances the stream and reports whether there is an event to read.
func (s *SlotStream) Next() bool {
	for {
		if s.i < len(s.step.events) {
			s.cur = s.step.events[s.i]
			s.i++
			return true
		}
		if s.done {
			return false
		}
		select {
		case st, ok := <-s.steps:
			if !ok {
				s.done = true
				if s.err == nil {
					s.err = s.run.dropped
				}
				continue
			}
			s.step, s.i = st, 0
			s.usage, s.probs = st.usage, st.probs
			if st.done {
				s.done = true
				s.err, s.reason, s.stopSeq = st.err, st.reason, st.stopSeq
			}
		case <-s.ctx.Done():
			// The handler answers immediately and the slot is released at the
			// next step boundary: a dispatch in flight cannot be cancelled
			// (022-D5).
			s.done, s.err = true, s.ctx.Err()
			s.run.gone.Store(true)
			return false
		}
	}
}

// Event is the current event.
func (s *SlotStream) Event() Event { return s.cur }

// Text is the current event's text delta, and is empty for an event that is not
// one.
func (s *SlotStream) Text() string { return s.cur.Text }

// Usage is the prompt and completion token counts so far.
func (s *SlotStream) Usage() Usage { return s.usage }

// Err is what ended the stream, or nil if it ran to its stopping condition.
func (s *SlotStream) Err() error { return s.err }

// StopReason is why the stream ended, and is [StopRunning] until it has.
func (s *SlotStream) StopReason() StopReason { return s.reason }

// StopSequence is the stop string that ended the stream, and is empty unless
// [SlotStream.StopReason] is [StopSequence].
func (s *SlotStream) StopSequence() string { return s.stopSeq }

// LogProbs is the tokens the last [SlotStream.Next] produced, and is empty
// unless [Policy.LogProbs] is set.
//
// Unlike [Stream.LogProbs] these are the request's own copy: the decoder that
// produced them runs on the driver's goroutine and reuses its backing array,
// so handing the slice across would be a race rather than a lifetime.
func (s *SlotStream) LogProbs() []TokenProb { return s.probs }

// Close says the caller is finished, so the slot is released at the next step
// boundary even if the completion had not ended.
func (s *SlotStream) Close() error {
	s.run.gone.Store(true)
	s.done = true
	return nil
}
