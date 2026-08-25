// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"strings"
	"time"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/sample"
	"github.com/latere-ai/tgo/tokenizer"
)

// EventKind is what one [Event] says happened.
type EventKind uint8

// The kinds of event a stream yields.
//
// They are typed rather than a string because [Stream.Text] alone cannot tell a
// caller whether the current token is inside a thinking block, which is the one
// thing a chat UI must know in order to render it (007-D8). They map one to one
// onto the block types specs/009-server.md encodes, which is what keeps a
// server's adapter a translation rather than a state machine.
const (
	// TextDelta is a piece of the assistant's answer.
	TextDelta EventKind = iota

	// ThinkingDelta is a piece of a thinking block.
	ThinkingDelta

	// ToolArgsDelta is a piece of a tool call's arguments.
	ToolArgsDelta

	// BlockStart and BlockStop bracket a block. [Event.Block] says which.
	BlockStart
	BlockStop
)

// String names an EventKind.
func (k EventKind) String() string {
	switch k {
	case TextDelta:
		return "text_delta"
	case ThinkingDelta:
		return "thinking_delta"
	case ToolArgsDelta:
		return "tool_args_delta"
	case BlockStart:
		return "block_start"
	case BlockStop:
		return "block_stop"
	}
	return "unknown"
}

// Event is one thing the model produced.
type Event struct {
	// Kind is what happened.
	Kind EventKind

	// Block is the block this event belongs to, and is set on every event
	// including the deltas.
	Block chat.BlockType

	// Text is the delta, and is empty on [BlockStart] and [BlockStop].
	Text string
}

// Stream yields a completion as it is produced.
//
// It is an iterator rather than a channel: a channel obliges a caller to drain
// it or leak a goroutine, and an iterator makes early return the normal case
// (007-D6). Nothing runs between calls — one [Stream.Next] is one model step —
// so abandoning a stream holds nothing but the memory the caller still points
// at, and the session it came from is usable again after the next [Session.Chat]
// or [Session.Complete].
//
// A Stream belongs to one session and is no safer for concurrent use than the
// session is.
type Stream struct {
	s   *Session
	ctx context.Context

	pol     Policy
	sp      sample.Policy
	sampler *sample.Sampler
	dec     *tokenizer.Decoder

	prompt    []int
	maxTokens int

	queue []Event
	head  int
	cur   Event

	started bool
	done    bool
	err     error
	usage   Usage

	// feed is the token the next decode step scores: the one just emitted.
	feed int

	// pending is the decoded text held back because it could still begin a
	// stop string. It is empty whenever Policy.Stop is.
	pending string
	stopped bool

	openBlock chat.BlockType
	first     time.Time
}

// newStream prepares a request. Nothing is submitted until the first
// [Stream.Next].
func newStream(ctx context.Context, s *Session, ids []int, p Policy) *Stream {
	max := p.MaxTokens
	if max <= 0 {
		max = s.capacity - len(ids)
	}
	return &Stream{
		s:         s,
		ctx:       ctx,
		pol:       p,
		sp:        p.sampling(),
		sampler:   sample.New(p.Seed),
		dec:       s.m.tok.NewDecoder(),
		prompt:    ids,
		maxTokens: max,
		feed:      -1,
		first:     time.Now(),
	}
}

// Next advances the stream and reports whether there is an event to read.
//
// One call runs as many model steps as it takes to produce one event, which is
// usually one: a token that completes no code point and closes no block yields
// nothing, and the loop is here rather than in the caller.
func (st *Stream) Next() bool {
	for {
		if st.head < len(st.queue) {
			st.cur = st.queue[st.head]
			st.head++
			return true
		}
		st.queue, st.head = st.queue[:0], 0
		if st.done {
			return false
		}
		st.advance()
	}
}

// Event is the current event.
func (st *Stream) Event() Event { return st.cur }

// Text is the current event's text delta, and is empty for an event that is not
// one.
func (st *Stream) Text() string { return st.cur.Text }

// Err is what ended the stream, or nil if it ran to its stopping condition.
//
// A cancelled context reports its own error. A device failure reports the
// driver's, and has also left the session unusable until it is reset (§7).
func (st *Stream) Err() error { return st.err }

// Usage is the prompt and completion token counts so far.
func (st *Stream) Usage() Usage { return st.usage }

// advance runs one model step and turns its token into events.
func (st *Stream) advance() {
	if err := st.ctx.Err(); err != nil {
		st.finish(err)
		return
	}
	s := st.s
	step := time.Now()

	var (
		logits []float32
		t      timings
		err    error
		phase  bench.Phase
		count  int
	)
	if !st.started {
		st.started = true
		rows, ferr := s.buckets.For(len(st.prompt))
		if ferr != nil {
			st.finish(s.fail(ferr))
			return
		}
		logits, t, err = s.run(rows, st.prompt, 0)
		if err == nil {
			s.length = len(st.prompt)
			s.history = append(s.history, st.prompt...)
			st.usage.PromptTokens = len(st.prompt)
		}
		phase, count = bench.Prefill, len(st.prompt)
	} else {
		logits, t, err = s.run(1, []int{st.feed}, s.length)
		if err == nil {
			s.history = append(s.history, st.feed)
			s.length++
		}
		phase, count = bench.Decode, 1
	}
	if err != nil {
		st.finish(s.fail(err))
		return
	}

	// Sampling and detokenizing are the host's share, and so is everything
	// else this function does: specs/017-benchmarks.md §1 treats the four
	// terms as exhaustive, so the host's is the step minus the other three
	// rather than a fifth measurement with a gap between them.
	tok := st.sampler.Next(logits, s.history, st.sp)
	st.feed = tok

	if st.isStop(tok) {
		st.finish(nil)
	} else {
		st.usage.CompletionTokens++
		st.emit(tok)
		switch {
		case st.stopped:
			st.finish(nil)
		case st.usage.CompletionTokens >= st.maxTokens:
			st.finish(nil)
		}
	}

	if s.rec.Enabled() {
		if phase == bench.Prefill {
			s.rec.TTFT(time.Since(st.first))
		}
		host := time.Since(step) - t.submit - t.device - t.readback
		if host < 0 {
			host = 0
		}
		s.rec.Step(bench.Step{
			Phase: phase, Tokens: count, Batch: 1,
			Host: host, Submit: t.submit, Device: t.device, Readback: t.readback,
		})
	}
}

// isStop reports whether a token ends the completion without being part of it.
func (st *Stream) isStop(tok int) bool {
	sp := st.s.m.special
	return tok == sp.imEnd || tok == sp.endOfText
}

// emit turns one token into events.
//
// The structural markers are matched by id and never by looking at decoded
// text. A stop found in text has the boundary problem 003-D6 rejects for turn
// markers and fails the same way: a user who asks the model to explain
// "</think>" would have the explanation cut in half.
func (st *Stream) emit(tok int) {
	sp := st.s.m.special
	switch {
	case tok == sp.think[0] || tok == sp.think[1]:
		st.startBlock(chat.BlockThinking)
		return
	case tok == sp.toolCall:
		st.startBlock(chat.BlockToolUse)
		return
	case tok == sp.thinkEnd, tok == sp.toolEnd:
		st.endBlock()
		return
	}
	if text := st.dec.Push(tok); text != "" {
		st.delta(text)
	}
}

// delta queues a piece of text, opening a text block if none is open.
func (st *Stream) delta(text string) {
	if st.openBlock == "" {
		st.openBlock = chat.BlockText
		st.queue = append(st.queue, Event{Kind: BlockStart, Block: chat.BlockText})
	}
	st.pending += text
	st.drain(false)
}

// startBlock closes whatever block is open and opens this one.
func (st *Stream) startBlock(bt chat.BlockType) {
	st.endBlock()
	st.openBlock = bt
	st.queue = append(st.queue, Event{Kind: BlockStart, Block: bt})
}

// endBlock flushes what is held back and closes the open block.
func (st *Stream) endBlock() {
	if st.openBlock == "" {
		return
	}
	st.drain(true)
	st.queue = append(st.queue, Event{Kind: BlockStop, Block: st.openBlock})
	st.openBlock = ""
}

// drain releases as much held-back text as is safe.
//
// While a stop string is set, the longest suffix of the output that could still
// begin one is held: a stop string need not align to a token boundary, so
// matching it means matching decoded text, and text already handed to the
// caller cannot be taken back (006-D4).
func (st *Stream) drain(final bool) {
	if st.pending == "" {
		return
	}
	if i := firstStop(st.pending, st.pol.Stop); i >= 0 {
		st.emitText(st.pending[:i])
		st.pending = ""
		st.stopped = true
		return
	}
	keep := 0
	if !final {
		keep = holdBack(st.pending, st.pol.Stop)
	}
	out := st.pending[:len(st.pending)-keep]
	st.pending = st.pending[len(st.pending)-keep:]
	st.emitText(out)
}

// emitText queues one delta of the open block's kind.
func (st *Stream) emitText(s string) {
	if s == "" {
		return
	}
	st.queue = append(st.queue, Event{Kind: deltaKind(st.openBlock), Block: st.openBlock,
		Text: s})
}

// deltaKind is the delta event a block's text belongs to.
func deltaKind(bt chat.BlockType) EventKind {
	switch bt {
	case chat.BlockThinking:
		return ThinkingDelta
	case chat.BlockToolUse:
		return ToolArgsDelta
	}
	return TextDelta
}

// finish ends the stream, flushing the detokenizer and closing any open block.
//
// The flush happens even on a failure: text the model produced before the
// device failed is text the caller already paid for, and dropping it would make
// the error harder to read rather than easier.
func (st *Stream) finish(err error) {
	if st.done {
		return
	}
	st.done = true
	st.err = err
	// Through delta rather than straight onto pending: what the detokenizer
	// holds at the end of a stream is a truncated code point, and if no block
	// was ever opened — a completion that is one partial character — appending
	// to pending would leave it in a buffer endBlock does not drain.
	if rest := st.dec.Flush(); rest != "" && !st.stopped {
		st.delta(rest)
	}
	st.endBlock()
	if st.s.live == st {
		st.s.live = nil
	}
}

// abandon ends the stream with no further events, for a caller who moved on.
func (st *Stream) abandon() {
	st.done = true
	st.queue, st.head = nil, 0
	if st.s.live == st {
		st.s.live = nil
	}
}

// firstStop is the index of the earliest stop string in s, or -1.
func firstStop(s string, stops []string) int {
	best := -1
	for _, stop := range stops {
		if i := strings.Index(s, stop); i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	return best
}

// holdBack is how many trailing bytes of s could still begin a stop string.
//
// A proper prefix only: a suffix that is a whole stop string is a stop, which
// [firstStop] has already found, and holding it back would be waiting for a
// match that has happened.
func holdBack(s string, stops []string) int {
	keep := 0
	for _, stop := range stops {
		n := min(len(stop)-1, len(s))
		for k := n; k > 0; k-- {
			if strings.HasSuffix(s, stop[:k]) {
				keep = max(keep, k)
				break
			}
		}
	}
	return keep
}
