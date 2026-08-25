// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
	"latere.ai/x/pkg/llmdialect/ir"
)

// The fake engine: a scripted event stream, so every handler test runs with no
// device and no weights (009-D4). e2e_test.go runs the same handlers over a
// real model built from a synthetic checkpoint, which is what proves this
// fake's contract is the real one's.

// fakeName is the model id the fake serves. It is not "test" or "model",
// because a handler that compared against a constant it had lying around would
// pass with either.
const fakeName = "qwen3-fixture-0.1b"

// The fake's shape. No two of these are equal, and that is the rule
// specs/011-sequencing.md's waves paid for: a fixture where the vocabulary
// equals the context is the identity for every confusion between them.
const (
	fakeVocab      = 640
	fakeContext    = 2048
	fakeCacheBytes = 1 << 20
)

// fakeEngine is an [Engine] whose sessions replay a script.
type fakeEngine struct {
	// script is what every stream yields, in order.
	script []tgo.Event

	// prompt is the prompt token count the stream reports.
	prompt int

	// streamErr ends the stream after the script, as a device failure does.
	streamErr error

	// sessionErr fails NewSession, as an exhausted device does.
	sessionErr error

	// chatErr fails Chat before any output, as a prompt that does not fit the
	// context does.
	chatErr error

	// gate, when set, holds each event after the first until a token arrives.
	// The flush test feeds it from the client's flushes, so a handler that
	// buffers stops the engine dead.
	gate chan struct{}

	// hold, when set, blocks Chat until it is closed, which is how the queue
	// tests keep a slot occupied.
	hold chan struct{}

	// deaf makes the stream ignore its context, which is what an engine that
	// checks for cancellation only between model steps looks like. The handler
	// must stop anyway: reading the context in exactly one of the two places
	// passes a fake-engine test and leaves a real one generating.
	deaf bool

	// entered, when set, receives once per request that has reached Chat and
	// therefore holds a session slot. Waiting on it is how a test knows the
	// semaphore is full without waiting on a clock.
	entered chan struct{}

	mu       sync.Mutex
	sessions []*fakeSession
}

func (e *fakeEngine) Name() string                { return fakeName }
func (e *fakeEngine) Context() int                { return fakeContext }
func (e *fakeEngine) VocabSize() int              { return fakeVocab }
func (e *fakeEngine) CacheBytesPerSession() int64 { return fakeCacheBytes }

func (e *fakeEngine) NewSession(spec SessionSpec) (Session, error) {
	if e.sessionErr != nil {
		return nil, e.sessionErr
	}
	s := &fakeSession{eng: e, spec: spec}
	e.mu.Lock()
	e.sessions = append(e.sessions, s)
	e.mu.Unlock()
	return s, nil
}

// took returns the sessions built so far.
func (e *fakeEngine) took() []*fakeSession {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]*fakeSession(nil), e.sessions...)
}

// only returns the one session a single-request test built.
func (e *fakeEngine) only(t *testing.T) *fakeSession {
	t.Helper()
	got := e.took()
	if len(got) != 1 {
		t.Fatalf("sessions built = %d, want 1", len(got))
	}
	return got[0]
}

// fakeSession records what a request asked for and hands back a stream.
type fakeSession struct {
	eng  *fakeEngine
	spec SessionSpec

	mu     sync.Mutex
	policy tgo.Policy
	msgs   []chat.Message
	prompt string
	closed bool
	stream *fakeStream
}

func (s *fakeSession) Chat(ctx context.Context, msgs []chat.Message, p tgo.Policy) (Stream, error) {
	s.mu.Lock()
	s.msgs = msgs
	s.policy = p
	s.mu.Unlock()
	return s.begin(ctx)
}

func (s *fakeSession) Complete(ctx context.Context, prompt string, p tgo.Policy) (Stream, error) {
	s.mu.Lock()
	s.prompt = prompt
	s.policy = p
	s.mu.Unlock()
	return s.begin(ctx)
}

func (s *fakeSession) begin(ctx context.Context) (Stream, error) {
	if s.eng.chatErr != nil {
		return nil, s.eng.chatErr
	}
	if s.eng.entered != nil {
		s.eng.entered <- struct{}{}
	}
	if s.eng.hold != nil {
		select {
		case <-s.eng.hold:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	st := &fakeStream{
		ctx: ctx, eng: s.eng,
		usage:   tgo.Usage{PromptTokens: s.eng.prompt},
		stopped: make(chan struct{}),
	}
	s.mu.Lock()
	s.stream = st
	s.mu.Unlock()
	return st, nil
}

func (s *fakeSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

func (s *fakeSession) sawPolicy() tgo.Policy {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.policy
}

func (s *fakeSession) sawMessages() []chat.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.msgs
}

func (s *fakeSession) sawPrompt() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.prompt
}

func (s *fakeSession) isClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

func (s *fakeSession) streamOf() *fakeStream {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stream
}

// fakeStream replays the engine's script.
type fakeStream struct {
	ctx context.Context
	eng *fakeEngine

	i     int
	cur   tgo.Event
	usage tgo.Usage
	err   error

	// stopped closes when Next has returned false, whatever the reason. The
	// cancellation test waits on it rather than on a duration.
	stopped  chan struct{}
	stopOnce sync.Once

	// cancelled records that the stream ended because the context did, which
	// is the observation §5's second trap is about.
	cancelled bool
}

func (st *fakeStream) Next() bool {
	if st.i > 0 && st.eng.gate != nil {
		select {
		case <-st.eng.gate:
		case <-st.ctx.Done():
			return st.stop(true)
		}
	}
	if st.ctx.Err() != nil && !st.eng.deaf {
		return st.stop(true)
	}
	if st.i >= len(st.eng.script) {
		st.err = st.eng.streamErr
		return st.stop(false)
	}
	st.cur = st.eng.script[st.i]
	st.i++
	if st.cur.Text != "" {
		st.usage.CompletionTokens++
	}
	// One recorded step per event, so the metrics path has something to
	// report. The durations are fixed rather than measured: a test that
	// asserted a duration would be asserting the clock.
	if r := st.recorder(); r != nil {
		r.Step(bench.Step{Phase: bench.Decode, Tokens: 1, Batch: 1,
			Host: time.Millisecond, Submit: time.Millisecond,
			Device: 3 * time.Millisecond, Readback: 5 * time.Millisecond})
	}
	return true
}

// recorder is the recorder the handler attached, found through the session the
// stream came from.
func (st *fakeStream) recorder() *bench.Recorder {
	st.eng.mu.Lock()
	defer st.eng.mu.Unlock()
	for _, s := range st.eng.sessions {
		if s.streamOf() == st {
			return s.spec.Recorder
		}
	}
	return nil
}

func (st *fakeStream) stop(cancelled bool) bool {
	if cancelled {
		st.cancelled = true
		st.err = st.ctx.Err()
	}
	st.stopOnce.Do(func() { close(st.stopped) })
	return false
}

func (st *fakeStream) Event() tgo.Event { return st.cur }
func (st *fakeStream) Usage() tgo.Usage { return st.usage }
func (st *fakeStream) Err() error       { return st.err }

// text is a script that says one word in one text block, which is the smallest
// well-formed generation.
func text(words ...string) []tgo.Event {
	out := []tgo.Event{{Kind: tgo.BlockStart, Block: chat.BlockText}}
	for _, w := range words {
		out = append(out, tgo.Event{Kind: tgo.TextDelta, Block: chat.BlockText, Text: w})
	}
	return append(out, tgo.Event{Kind: tgo.BlockStop, Block: chat.BlockText})
}

// thinkThenSay is a script that thinks and then answers, which is the shape
// §3.2 exists for.
func thinkThenSay(thought, answer string) []tgo.Event {
	return []tgo.Event{
		{Kind: tgo.BlockStart, Block: chat.BlockThinking},
		{Kind: tgo.ThinkingDelta, Block: chat.BlockThinking, Text: thought},
		{Kind: tgo.BlockStop, Block: chat.BlockThinking},
		{Kind: tgo.BlockStart, Block: chat.BlockText},
		{Kind: tgo.TextDelta, Block: chat.BlockText, Text: answer},
		{Kind: tgo.BlockStop, Block: chat.BlockText},
	}
}

// newTestServer builds a server over a fake engine.
func newTestServer(t *testing.T, eng *fakeEngine, opts ...Option) *Server {
	t.Helper()
	opts = append([]Option{WithNotice(&strings.Builder{})}, opts...)
	s, err := New(eng, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

// post sends one request body to a route and returns the recorded response.
func post(t *testing.T, s *Server, route, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, route, strings.NewReader(body))
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	return w
}

// get sends one GET to a route.
func get(t *testing.T, s *Server, route string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(http.MethodGet, route, nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	return w
}

// routes names the four request surfaces and a minimal body for each, so a
// rule that must hold everywhere can be tested everywhere.
var routes = []struct {
	name string
	path string
	body func(extra string) string

	// dialect is what this route's frontend calls itself, which is the key
	// every table in this package is keyed by.
	dialect ir.Dialect

	stops string // the member that carries stop strings
}{
	{
		name: "openai-chat", path: "/v1/chat/completions", stops: "stop",
		dialect: ir.DialectOpenAIChat,
		body: func(extra string) string {
			return `{"model":"` + fakeName + `","messages":[{"role":"user","content":"hi"}]` +
				extra + `}`
		},
	},
	{
		name: "anthropic", path: "/v1/messages", stops: "stop_sequences",
		dialect: ir.DialectAnthropicMessages,
		body: func(extra string) string {
			return `{"model":"` + fakeName + `","max_tokens":16,` +
				`"messages":[{"role":"user","content":"hi"}]` + extra + `}`
		},
	},
	{
		name: "openai-responses", path: "/v1/responses", stops: "",
		dialect: ir.DialectOpenAIResponses,
		body: func(extra string) string {
			return `{"model":"` + fakeName + `","input":"hi"` + extra + `}`
		},
	},
	{
		name: "openai-completions", path: "/v1/completions", stops: "stop",
		dialect: dialectLegacy,
		body: func(extra string) string {
			return `{"model":"` + fakeName + `","prompt":"hi"` + extra + `}`
		},
	},
}

// inject merges one member into a route's base body.
//
// It exists so a case can send a member the base body already carries -- a
// max_tokens on /v1/messages, where the dialect requires one -- without writing
// the key twice and relying on which of the two a decoder keeps.
func inject(t *testing.T, base, extra string) string {
	t.Helper()
	if extra == "" {
		return base
	}
	var into, from map[string]json.RawMessage
	if err := json.Unmarshal([]byte(base), &into); err != nil {
		t.Fatalf("the base body is not an object: %v: %s", err, base)
	}
	if err := json.Unmarshal([]byte("{"+strings.TrimPrefix(extra, ",")+"}"), &from); err != nil {
		t.Fatalf("the member is not one: %v: %s", err, extra)
	}
	for k, v := range from {
		into[k] = v
	}
	raw, err := json.Marshal(into)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// wantStatus fails unless the response carries this status, printing the body,
// which is where the reason is.
func wantStatus(t *testing.T, w *httptest.ResponseRecorder, status int) {
	t.Helper()
	if w.Code != status {
		t.Fatalf("status = %d, want %d: %s", w.Code, status, w.Body.String())
	}
}

// wantNames fails unless the body names every one of these, which is what "a
// refusal names its field" means from a caller's side.
func wantNames(t *testing.T, w *httptest.ResponseRecorder, names ...string) {
	t.Helper()
	body := w.Body.String()
	for _, n := range names {
		if !strings.Contains(body, n) {
			t.Errorf("the refusal does not name %q: %s", n, body)
		}
	}
}

// errFake is what a fake fails with. It reads as deliberate in a test's output,
// which is the difference between a failure a reader trusts and one they debug.
var errFake = errors.New("the fake engine failed on purpose")
