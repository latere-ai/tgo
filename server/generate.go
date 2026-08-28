// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
	"latere.ai/x/pkg/llmdialect"
	"latere.ai/x/pkg/llmdialect/ir"
)

// One canonical event sequence, three wire formats.
//
// Every frontend's EventEncoder writes its own SSE framing from the same
// ir.Event grammar:
//
//	MessageStart (BlockStart (TextDelta|ArgsDelta|ThinkingDelta)* BlockStop)*
//	MessageDelta MessageStop
//
// so this file emits that sequence once and the dialects fall out. [tgo.Event]
// maps onto it directly, which is what keeps this a translation rather than a
// state machine (specs/009-server.md §3.2).

// recorderCapacity is how many steps one request's recorder keeps. A completion
// longer than this reports quantiles over its most recent steps, which is the
// window that says what the device is doing now.
//
// [bench.Recorder] is a ring, so "most recent" is what it keeps. It kept the
// *first* 1024 until 2026-08-27, which made this comment false for exactly the
// completions long enough to need a window: a long request published
// percentiles for its own warm-up, and Report.Dropped being non-zero was the
// only sign — a count nobody reads as "these are the wrong numbers".
const recorderCapacity = 1024

// generate runs one request and writes the answer, streaming or not.
//
// The session is this request's alone for the length of the request, and is
// given back on the way out. What that means depends on the engine: [Wrap]
// closes it, which returns the KV reservation (§6); [WrapPool] returns it to
// the pool with its history intact, so the next turn of the same conversation
// can be routed to it (specs/019-session-affinity.md §2).
//
// The request's context is handed to the engine, not only to the generation: a
// pooled engine waits for a free session, and a client who hangs up while
// waiting must stop waiting rather than take a session it will not read from.
func (s *Server) generate(w http.ResponseWriter, r *http.Request, front llmdialect.Frontend, req *request) {
	rec := bench.NewRecorder(recorderCapacity)
	req.spec.Recorder = rec
	ctx := r.Context()

	sess, err := s.eng.NewSession(ctx, req.spec)
	if err != nil {
		if ctx.Err() != nil {
			// The client hung up while a pooled engine was waiting for a free
			// session. Nothing has been written, and writeError has no body
			// for a client that is gone, so the status is written here: a bare
			// return would let the runtime synthesize 200 with an empty body,
			// which a proxy reads as a completion that produced nothing.
			s.metrics.reject("client_gone")
			w.WriteHeader(errClientGone.status())
			return
		}
		s.fail(w, req.dialect, sessionError(err))
		return
	}
	defer func() {
		if err := sess.Close(); err != nil {
			s.notice("tgo: closing a session: %v", err)
		}
		s.report(rec)
	}()

	var st Stream
	if req.prompt != "" {
		st, err = sess.Complete(ctx, req.prompt, req.policy)
	} else {
		st, err = sess.Chat(ctx, req.msgs, req.policy)
	}
	if err != nil {
		s.fail(w, req.dialect, sessionError(err))
		return
	}

	if req.stream {
		s.stream(w, ctx, front, req, st)
		return
	}
	s.whole(w, ctx, front, req, st)
}

// stream writes the SSE answer.
//
// Three things a naive implementation gets wrong, and none of them is the
// encoder's to fix (§5):
//
//   - Flush per event. Without it the runtime buffers and the client gets the
//     whole answer at once, which passes every test that checks content and
//     defeats the point.
//   - A client disconnect cancels generation. The request's context is already
//     cancelled; a loop that does not read it holds a session and its KV
//     reservation until max_tokens.
//   - The terminal event carries the stop reason and the usage. A stream that
//     ends without one is indistinguishable from a dropped connection.
func (s *Server) stream(w http.ResponseWriter, ctx context.Context, front llmdialect.Frontend,
	req *request, st Stream) {

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	// NewResponseController rather than a w.(http.Flusher) assertion: it
	// unwraps middleware, so a wrapper added later cannot silently turn
	// flushing off while the content tests stay green.
	rc := http.NewResponseController(w)
	flush := func() { _ = rc.Flush() }

	enc := front.NewEventEncoder(w)
	id := requestID(req.dialect)

	// probs is the tokens the step that produced the pending event reported,
	// and it is nil on every route but /v1/completions -- [mapPolicy] does not
	// ask the engine for work no encoder can carry (specs/030-logprobs.md §4).
	// A streaming request must serve them wherever the whole-body one does, or
	// a caller gets a number that depends on a flag about delivery.
	var probs []tgo.TokenProb
	emit := func(ev ir.Event) bool {
		err := encodeEvent(enc, ev, probs)
		probs = nil
		if err != nil {
			s.notice("tgo: encoding a %s event: %v", ev.Type, err)
			return false
		}
		flush()
		return true
	}

	// The first event is pulled before message_start so that the prompt token
	// count is known: Anthropic puts input_tokens in message_start, and a zero
	// there is a number a client will believe.
	more := st.Next()
	if !emit(ir.Event{Type: ir.EventMessageStart, ID: id, Model: s.eng.Name(),
		Usage: usageOf(st.Usage())}) {
		return
	}

	var blocks blockIndex
	for more {
		if ctx.Err() != nil {
			// The client is gone. Nothing more is written and the deferred
			// close ends the session, which is what releases the reservation.
			return
		}
		ev, ok := blocks.translate(st.Event())
		probs = st.LogProbs()
		if ok && !emit(ev) {
			return
		}
		probs = nil
		more = st.Next()
	}
	if ev, ok := blocks.closeOpen(); ok {
		if !emit(ev) {
			return
		}
	}

	if err := st.Err(); err != nil {
		if ctx.Err() != nil {
			return
		}
		writeStreamError(w, req.dialect, streamError(err))
		flush()
		return
	}
	if !emit(ir.Event{Type: ir.EventMessageDelta, StopReason: stopReason(st, req.policy, st.Usage()),
		StopSequence: st.StopSequence(), Usage: usageOf(st.Usage())}) {
		return
	}
	emit(ir.Event{Type: ir.EventMessageStop})
}

// whole writes the non-streaming answer, which is the same sequence collected
// rather than framed.
func (s *Server) whole(w http.ResponseWriter, ctx context.Context, front llmdialect.Frontend,
	req *request, st Stream) {

	var blocks []ir.Block
	var probs []tgo.TokenProb
	for st.Next() {
		// Appended rather than kept: LogProbs is valid only until the next
		// Next (specs/030-logprobs.md §2), and the backing array is reused.
		probs = append(probs, st.LogProbs()...)
		if ctx.Err() != nil {
			// Nothing has been written yet on this path, so returning here
			// would let Go synthesize 200 with an empty body -- which a proxy
			// or an SDK reads as a successful empty completion rather than as
			// a cancellation. The status is written even though the client is
			// probably gone: the case where it is not gone is a server-side
			// deadline, and that one must not read as success.
			s.metrics.reject("client_gone")
			w.WriteHeader(errClientGone.status())
			return
		}
		ev := st.Event()
		switch ev.Kind {
		case tgo.BlockStart:
			blocks = append(blocks, ir.Block{Type: irBlockType(ev.Block)})
		case tgo.TextDelta, tgo.ThinkingDelta, tgo.ToolArgsDelta:
			if len(blocks) == 0 {
				blocks = append(blocks, ir.Block{Type: irBlockType(ev.Block)})
			}
			blocks[len(blocks)-1].Text += ev.Text
		}
	}
	if err := st.Err(); err != nil {
		if ctx.Err() != nil {
			s.metrics.reject("client_gone")
			w.WriteHeader(errClientGone.status())
			return
		}
		s.fail(w, req.dialect, streamError(err))
		return
	}

	resp := &ir.Response{
		ID:           requestID(req.dialect),
		Model:        s.eng.Name(),
		Blocks:       blocks,
		StopReason:   stopReason(st, req.policy, st.Usage()),
		StopSequence: st.StopSequence(),
		Usage:        *usageOf(st.Usage()),
	}
	body, err := encodeResponse(front, resp, probs)
	if err != nil {
		s.fail(w, req.dialect, &apiError{kind: errInternal, reason: "internal",
			msg: fmt.Sprintf("tgo: encoding the response: %v", err)})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// blockIndex numbers the output blocks, which is the one piece of state the
// translation needs: ir.Event carries an ordinal and [tgo.Event] does not.
type blockIndex struct {
	index int
	open  bool
}

// translate turns one engine event into one IR event.
//
// The mapping is one to one except for a tool call. tgo yields the model's
// tool-call text as a tool_use block, and 009-D6 says what comes back is what
// the model emitted rather than a parsed call: nothing has checked that the
// JSON is well formed, [tgo.Event] carries neither a call id nor a name, and
// all three encoders dereference both. So a tool block goes out as text, and a
// client sees the model's own output instead of a call this server would have
// had to invent.
func (b *blockIndex) translate(ev tgo.Event) (ir.Event, bool) {
	switch ev.Kind {
	case tgo.BlockStart:
		t := irBlockType(ev.Block)
		b.open = true
		out := ir.Event{Type: ir.EventBlockStart, Index: b.index, Block: &ir.Block{Type: t}}
		return out, true
	case tgo.BlockStop:
		if !b.open {
			return ir.Event{}, false
		}
		b.open = false
		out := ir.Event{Type: ir.EventBlockStop, Index: b.index}
		b.index++
		return out, true
	case tgo.ThinkingDelta:
		return ir.Event{Type: ir.EventThinkingDelta, Index: b.index, Delta: ev.Text}, true
	case tgo.TextDelta, tgo.ToolArgsDelta:
		return ir.Event{Type: ir.EventTextDelta, Index: b.index, Delta: ev.Text}, true
	}
	return ir.Event{}, false
}

// closeOpen closes a block the stream left open, which a stream that failed
// mid-block does. An encoder that never sees the stop leaves a client's parser
// inside a block forever.
func (b *blockIndex) closeOpen() (ir.Event, bool) {
	if !b.open {
		return ir.Event{}, false
	}
	b.open = false
	out := ir.Event{Type: ir.EventBlockStop, Index: b.index}
	b.index++
	return out, true
}

// irBlockType maps tgo's block types onto the IR's. A tool block becomes text;
// see [blockIndex.translate].
func irBlockType(t chat.BlockType) ir.BlockType {
	if t == chat.BlockThinking {
		return ir.BlockThinking
	}
	return ir.BlockText
}

// logProbEncoder is the half of a Frontend that can carry per-token
// probabilities. Only tgo's own /v1/completions codec implements it.
//
// An optional interface rather than a field on ir.Response, because the IR is
// llmdialect's and has no logprobs shape: specs/030-logprobs.md 030-D5 reports
// that gap rather than reaching past the codec to append a member to a body
// tgo did not write, which is what 009-D10 exists to prevent.
type logProbEncoder interface {
	EncodeResponseWithLogProbs(*ir.Response, []tgo.TokenProb) ([]byte, error)
}

// encodeResponse hands the probabilities to a Frontend that can carry them, and
// encodes normally otherwise.
//
// probs is empty on the three routes that cannot serve them, because
// [mapPolicy] does not ask the engine for work no encoder can use.
func encodeResponse(front llmdialect.Frontend, resp *ir.Response, probs []tgo.TokenProb) ([]byte, error) {
	if e, ok := front.(logProbEncoder); ok && len(probs) > 0 {
		return e.EncodeResponseWithLogProbs(resp, probs)
	}
	return front.EncodeResponse(resp)
}

// logProbEventEncoder is the streaming half of [logProbEncoder].
type logProbEventEncoder interface {
	EncodeWithLogProbs(ir.Event, []tgo.TokenProb) error
}

// encodeEvent writes one SSE frame, carrying the step's probabilities where the
// encoder can hold them.
func encodeEvent(enc llmdialect.EventEncoder, ev ir.Event, probs []tgo.TokenProb) error {
	if e, ok := enc.(logProbEventEncoder); ok && len(probs) > 0 {
		return e.EncodeWithLogProbs(ev, probs)
	}
	return enc.Encode(ev)
}

// usageOf converts one request's token counts.
func usageOf(u tgo.Usage) *ir.Usage {
	return &ir.Usage{InputTokens: int64(u.PromptTokens), OutputTokens: int64(u.CompletionTokens)}
}

// stopReason is why generation ended, in the IR's vocabulary.
//
// The stream is the authority and the policy is the fallback. A stop string
// need not align to a token boundary and the matched text is never emitted
// (006-D4), so "ended on a stop string" cannot be reconstructed out here --
// which is why [tgo.Stream] reports it and this translates rather than
// recomputes.
//
// StopRunning falls through to end_turn. A stream that ended in an error never
// reaches an encoder, and one that ran out of the session's remaining capacity
// rather than of Policy.MaxTokens is a budget the caller did not set: the
// policy branch below is what still answers max_tokens there.
//
// StopToolUse and StopRefusal stay unreachable, and neither is a gap. tgo emits
// a tool call as text (009-D6) and has no refusal classifier, so a stop reason
// naming either would be a claim about output nothing checked.
func stopReason(st Stream, p tgo.Policy, u tgo.Usage) ir.StopReason {
	switch st.StopReason() {
	case tgo.StopSequence:
		return ir.StopStopSequence
	case tgo.StopMaxTokens:
		return ir.StopMaxTokens
	}
	if p.MaxTokens > 0 && u.CompletionTokens >= p.MaxTokens {
		return ir.StopMaxTokens
	}
	return ir.StopEndTurn
}

// sessionError dresses a failure that happened before any output.
func sessionError(err error) *apiError {
	if errors.Is(err, tgo.ErrContextExhausted) {
		return badRequest("tgo: %v: the request does not fit this session's context, and is "+
			"refused rather than truncated", err)
	}
	return &apiError{kind: errInternal, reason: "internal", msg: fmt.Sprintf("tgo: %v", err)}
}

// streamError dresses a failure that happened during generation.
func streamError(err error) *apiError {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return &apiError{kind: errClientGone, reason: "client_gone",
			msg: fmt.Sprintf("tgo: %v", err)}
	}
	return &apiError{kind: errInternal, reason: "device", msg: fmt.Sprintf("tgo: %v", err)}
}

// report feeds one request's decode timings to the metrics.
//
// The step is the sum of the four medians rather than the median of the sums,
// because [bench.PhaseStats] reports the terms separately. The two differ, and
// the term that matters here is the ratio between readback and step, which the
// difference does not move.
func (s *Server) report(rec *bench.Recorder) {
	rep := rec.Report()
	d := rep.Decode
	if d.Steps == 0 {
		return
	}
	step := d.Host.P50 + d.Submit.P50 + d.Device.P50 + d.Readback.P50
	s.metrics.step(step, d.Readback.P50)
}

// requestID is one answer's id, in the shape its dialect's clients parse.
func requestID(d ir.Dialect) string {
	var raw [12]byte
	// crypto/rand.Read does not fail; it panics on a broken system rather than
	// returning, so the id is always the full width.
	_, _ = rand.Read(raw[:])
	suffix := hex.EncodeToString(raw[:])
	switch d {
	case ir.DialectAnthropicMessages:
		return "msg_" + suffix
	case ir.DialectOpenAIResponses:
		return suffix
	default:
		return "chatcmpl-" + suffix
	}
}
