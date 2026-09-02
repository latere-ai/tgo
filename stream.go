// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"time"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/internal/grammar"
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

// TokenProb is one token and what the sampler's distribution gave it.
//
// specs/030-logprobs.md. The probability is the **post-policy** one: the
// distribution the token was actually drawn from, after the bias, the
// penalties, the temperature and the two truncations of
// specs/006-sampling.md section 3. Reporting the raw softmax over the
// untruncated vocabulary would describe a distribution nothing sampled from,
// and would give a token top-k had already excluded a positive chance
// (030-D2).
type TokenProb struct {
	// ID is the token.
	ID int

	// Text is what the token contributes to the output, and is empty for a
	// control token: it is Tokenizer.TextBytes, which is nil for an added
	// token because the token's literal spelling is not text the model
	// produced (002 section 2.1).
	Text string

	// LogProb is ln of the token's post-policy probability, and is
	// math.Inf(-1) for a token the policy gave no chance -- one outside the
	// top-k or the nucleus, one a grammar masked, or any non-argmax token at
	// Temperature 0. It is never that for a token the stream drew, because the
	// draw walks the kept set.
	LogProb float64

	// Top is the Policy.TopLogProbs most likely tokens of the same
	// distribution, descending, and is nil inside those entries.
	Top []TokenProb
}

// StopReason is why a stream ended.
//
// A caller can reconstruct two of these from what it already has -- the token
// count against MaxTokens, and its own stop strings against the text -- and not
// the third: a stop string need not align to a token boundary and the matched
// text is never emitted (006-D4), so "ended on a stop string" is not visible
// from outside. Without this a completion that hit a stop string cannot be told
// from one the model chose to end, and the three wire dialects render those
// differently (specs/009-server.md §3.2).
type StopReason uint8

// The reasons a stream ends.
const (
	// StopRunning is the zero value: the stream has not ended, or it ended
	// with an error, which [Stream.Err] reports instead.
	StopRunning StopReason = iota

	// StopEndTurn is an end-of-turn token. The model said it was done.
	StopEndTurn

	// StopSequence is one of [Policy.Stop]. [Stream.StopSequence] says which.
	StopSequence

	// StopMaxTokens is the caller's budget, from [Policy.MaxTokens] or from
	// what the session's remaining capacity allowed.
	StopMaxTokens
)

// String names a StopReason, in the vocabulary specs/009-server.md's IR uses.
func (r StopReason) String() string {
	switch r {
	case StopEndTurn:
		return "end_turn"
	case StopSequence:
		return "stop_sequence"
	case StopMaxTokens:
		return "max_tokens"
	}
	return "running"
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
	// decoder is the host half: the sampler, the grammar, the detokenizer, the
	// stop strings and the events. It is embedded rather than owned because a
	// batched step runs the same half over a slot's row, and one copy of it is
	// what keeps the two paths from drifting (specs/022-batched-serving.md §5).
	*decoder

	s   *Session
	ctx context.Context

	prompt []int

	// reused is how many leading prompt positions the session's cache already
	// holds, and therefore the position the prefill starts at. Zero is a cold
	// request: [WithPrefixCache] off, a fresh session, or a prompt that shares
	// no first token with the conversation so far.
	reused int

	started bool
	first   time.Time
}

// newStream prepares a request. Nothing is submitted until the first
// [Stream.Next].
func newStream(ctx context.Context, s *Session, ids []int, p Policy, reused int,
	g *grammar.Grammar) *Stream {

	max := p.MaxTokens
	if max <= 0 {
		max = s.capacity - len(ids)
	}
	return &Stream{
		decoder: newDecoder(s.m, p, max, g),
		s:       s,
		ctx:     ctx,
		prompt:  ids,
		reused:  reused,
		first:   time.Now(),
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

// Err is what ended the stream, or nil if it ran to its stopping condition.
//
// A cancelled context reports its own error. A device failure reports the
// driver's, and has also left the session unusable until it is reset (§7).
func (st *Stream) Err() error { return st.err }

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
		// Only the suffix, at the position the reused prefix ends at. The
		// positions are what make this safe: stepData.fill writes each row's
		// rotary position and its cache slot from first+i, and the causal mask
		// is pos <= base+s, so a suffix prefilled at zero instead of at
		// st.reused would rotate every query by the wrong angle and let its
		// first token see nothing behind it — the silent coherence loss
		// specs/004-model-graph.md §2.5.1 describes, reached from the other
		// direction (016 §4).
		suffix := st.prompt[st.reused:]
		rows, ferr := s.buckets.For(len(suffix))
		if ferr != nil {
			st.finish(s.fail(ferr))
			return
		}
		logits, t, err = s.run(rows, suffix, st.reused)
		if err == nil {
			// The session was rewound to st.reused before the stream was
			// built, so the history is the reused run and this appends the
			// rest of the prompt onto it.
			s.length = len(st.prompt)
			s.history = append(s.history, suffix...)
			st.usage.PromptTokens = len(st.prompt)
			st.usage.CachedPromptTokens = st.reused
			// After the step and never before: offering a block whose
			// key/value state is not yet written hands another sequence a
			// block holding whatever was there. The prompt is already in the
			// lease -- Acquire took it -- so nothing is committed here and
			// what publishes is bounded by the length this step just set.
			err = s.publish()
		}
		phase, count = bench.Prefill, len(suffix)
	} else {
		// The block first, then the step, then the publish. A generated token
		// needs a row before the step that computes its key and value, and the
		// block that row is in may not be allocated yet -- but it may only be
		// offered to other sequences once that step has run.
		if err = s.reserve(st.feed); err == nil {
			logits, t, err = s.run(1, []int{st.feed}, s.length)
		}
		if err == nil {
			s.history = append(s.history, st.feed)
			s.length++
			// The token is recorded here and not before the step, so a step
			// that failed leaves no hash chained over a token nobody computed.
			err = s.publish(st.feed)
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
	//
	// The history is the session's, and the token consume draws is appended to
	// it by the next step rather than here, which is the order the batched path
	// follows too (specs/022-batched-serving.md §4).
	done, derr := st.consume(logits, s.history)
	if derr != nil {
		st.finish(derr)
		return
	}
	if done {
		st.finish(nil)
	}

	if s.rec.Enabled() {
		if phase == bench.Prefill {
			s.rec.TTFT(time.Since(st.first))
		}
		host := max(time.Since(step)-t.submit-t.device-t.readback, 0)
		s.rec.Step(bench.Step{
			Phase: phase, Tokens: count, Batch: 1,
			Host: host, Submit: t.submit, Device: t.device, Readback: t.readback,
		})
	}
}

// finish ends the stream and gives the session's blocks back.
func (st *Stream) finish(err error) {
	if st.done {
		return
	}
	st.end(err)
	if st.s.live == st {
		st.s.live = nil
	}
	// The blocks go back here and not at the next request.
	//
	// A lease is a refcount, not the key/value state: every complete block was
	// published as it was computed, so releasing keeps them in the pool and the
	// next request that shares the prefix finds them by hash -- including this
	// conversation's own next turn. What a lease does hold is a *reference*,
	// and a reference held by an idle session is a block no live conversation
	// can have. With a pool of B blocks and N sessions, holding across requests
	// makes idle conversations compete with running ones for the one resource
	// the process shares, which is the shape specs/008-scheduler.md §3 calls a
	// deadlock.
	//
	// The cost is the tail: the partial block at the end, which no hash entry
	// names and which therefore cannot be found again. That is at most
	// CacheBlock-1 positions re-prefilled on the next turn, and it is
	// specs/016-prefix-cache.md 016-D4's rounding rather than a new loss.
	st.s.release()
}

// abandon ends the stream with no further events, for a caller who moved on.
func (st *Stream) abandon() {
	st.done = true
	st.queue, st.head = nil, 0
	if st.s.live == st {
		st.s.live = nil
	}
	// Same as [Stream.finish]: an abandoned stream's blocks are no more this
	// session's than a finished one's, and a caller who walked away from a
	// completion is the case most likely to leave them held for good.
	st.s.release()
}
