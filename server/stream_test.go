// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/chat"
)

// §5. Three traps, each of which passes a content test and fails in the field.

// frame is one SSE frame: its event name, which the OpenAI dialects leave
// empty, and its data.
type frame struct {
	name string
	data string
}

// parseSSE reads a body into frames.
func parseSSE(t *testing.T, body string) []frame {
	t.Helper()
	var out []frame
	var cur frame
	var data []string
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			if cur.name != "" || len(data) > 0 {
				cur.data = strings.Join(data, "\n")
				out = append(out, cur)
			}
			cur, data = frame{}, nil
		case strings.HasPrefix(line, "event: "):
			cur.name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			data = append(data, strings.TrimPrefix(line, "data: "))
		default:
			t.Fatalf("an SSE line is neither a name nor data: %q", line)
		}
	}
	if cur.name != "" || len(data) > 0 {
		cur.data = strings.Join(data, "\n")
		out = append(out, cur)
	}
	return out
}

// names lists the frame names in order, with a data-only frame named "".
func names(fs []frame) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.name
	}
	return out
}

// streamBody runs one streaming request and returns the frames.
func streamBody(t *testing.T, s *Server, path, body string) []frame {
	t.Helper()
	w := post(t, s, path, body)
	wantStatus(t, w, http.StatusOK)
	if got := w.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	return parseSSE(t, w.Body.String())
}

func TestEachDialectFramesItsOwnStream(t *testing.T) {
	t.Parallel()
	cases := []struct {
		route int
		// want is a name that must appear, or a data substring for the
		// dialects that send unnamed frames.
		wantNames []string
		wantData  []string
	}{
		{
			route:     0,
			wantData:  []string{`"object":"chat.completion.chunk"`, `"content":"Hello"`, "[DONE]"},
			wantNames: []string{""},
		},
		{
			route: 1,
			wantNames: []string{"message_start", "ping", "content_block_start",
				"content_block_delta", "content_block_stop", "message_delta", "message_stop"},
		},
		{
			route: 2,
			wantNames: []string{"response.created", "response.in_progress",
				"response.output_item.added", "response.output_text.delta",
				"response.output_text.done", "response.output_item.done", "response.completed"},
		},
		{
			route:    3,
			wantData: []string{`"object":"text_completion"`, `"text":"Hello"`, "[DONE]"},
		},
	}
	for _, c := range cases {
		r := routes[c.route]
		t.Run(r.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("Hello"), prompt: 4}
			s := newTestServer(t, eng)
			fs := streamBody(t, s, r.path, r.body(`,"stream":true`))
			if len(fs) == 0 {
				t.Fatal("the stream carried no frames")
			}
			got := strings.Join(names(fs), " ")
			for _, want := range c.wantNames {
				if want != "" && !strings.Contains(got, want) {
					t.Errorf("no %q frame: %v", want, names(fs))
				}
			}
			var all strings.Builder
			for _, f := range fs {
				all.WriteString(f.data)
				all.WriteString("\n")
			}
			for _, want := range c.wantData {
				if !strings.Contains(all.String(), want) {
					t.Errorf("no frame carries %q: %s", want, all.String())
				}
			}
		})
	}
}

// The OpenAI dialects end with [DONE] and the Anthropic one does not: a client
// that waits for a terminator it will never get hangs on every request.
func TestTheOpenAIStreamsEndWithDone(t *testing.T) {
	t.Parallel()
	for _, i := range []int{0, 3} {
		r := routes[i]
		t.Run(r.name, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{script: text("hi")})
			fs := streamBody(t, s, r.path, r.body(`,"stream":true`))
			if last := fs[len(fs)-1]; last.data != "[DONE]" {
				t.Errorf("the last frame is %q, want [DONE]", last.data)
			}
		})
	}
	t.Run("anthropic sends message_stop", func(t *testing.T) {
		t.Parallel()
		s := newTestServer(t, &fakeEngine{script: text("hi")})
		fs := streamBody(t, s, "/v1/messages", routes[1].body(`,"stream":true`))
		if last := fs[len(fs)-1]; last.name != "message_stop" {
			t.Errorf("the last frame is %q, want message_stop", last.name)
		}
		if strings.Contains(strings.Join(names(fs), " "), "[DONE]") {
			t.Error("the Anthropic stream carries an OpenAI terminator")
		}
	})
}

// §5's third trap: a stream that ends without a stop reason and a usage is
// indistinguishable from a dropped connection.
func TestTheTerminalEventCarriesTheStopReasonAndTheUsage(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("a", "b"), prompt: 9}
	s := newTestServer(t, eng)

	t.Run("anthropic", func(t *testing.T) {
		fs := streamBody(t, s, "/v1/messages", routes[1].body(`,"stream":true`))
		var delta map[string]any
		for _, f := range fs {
			if f.name == "message_delta" {
				if err := json.Unmarshal([]byte(f.data), &delta); err != nil {
					t.Fatal(err)
				}
			}
		}
		if delta == nil {
			t.Fatalf("no message_delta: %v", names(fs))
		}
		if got := delta["delta"].(map[string]any)["stop_reason"]; got != "end_turn" {
			t.Errorf("stop_reason = %v, want end_turn", got)
		}
		usage, ok := delta["usage"].(map[string]any)
		if !ok {
			t.Fatalf("message_delta carries no usage: %s", fs[len(fs)-2].data)
		}
		if usage["output_tokens"].(float64) != 2 {
			t.Errorf("output_tokens = %v, want 2", usage["output_tokens"])
		}
	})

	t.Run("openai chat", func(t *testing.T) {
		eng := &fakeEngine{script: text("a", "b"), prompt: 9}
		s := newTestServer(t, eng)
		fs := streamBody(t, s, "/v1/chat/completions", routes[0].body(`,"stream":true`))
		var sawFinish, sawUsage bool
		for _, f := range fs {
			if f.data == "[DONE]" {
				continue
			}
			var chunk map[string]any
			if err := json.Unmarshal([]byte(f.data), &chunk); err != nil {
				t.Fatal(err)
			}
			for _, c := range chunk["choices"].([]any) {
				if c.(map[string]any)["finish_reason"] == "stop" {
					sawFinish = true
				}
			}
			if u, ok := chunk["usage"].(map[string]any); ok {
				sawUsage = true
				if u["prompt_tokens"].(float64) != 9 {
					t.Errorf("prompt_tokens = %v, want 9", u["prompt_tokens"])
				}
			}
		}
		if !sawFinish || !sawUsage {
			t.Errorf("finish_reason seen = %v, usage seen = %v", sawFinish, sawUsage)
		}
	})
}

// A completion cut short by max_tokens says so, which is the difference between
// an answer and a truncation a client cannot see.
//
// Both OpenAI surfaces are checked, because /v1/completions carries tgo's own
// codec rather than llmdialect's and its finish_reason is a second mapping.
func TestReachingMaxTokensIsReportedAsLength(t *testing.T) {
	t.Parallel()
	for _, i := range []int{0, 3} {
		r := routes[i]
		t.Run(r.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("a", "b")}
			s := newTestServer(t, eng)
			w := post(t, s, r.path, inject(t, r.body(""), `,"max_tokens":2`))
			wantStatus(t, w, http.StatusOK)
			if !strings.Contains(w.Body.String(), `"finish_reason":"length"`) {
				t.Errorf("finish_reason is not length: %s", w.Body.String())
			}
		})
	}
	t.Run("anthropic", func(t *testing.T) {
		t.Parallel()
		eng := &fakeEngine{script: text("a", "b")}
		s := newTestServer(t, eng)
		w := post(t, s, "/v1/messages", inject(t, routes[1].body(""), `,"max_tokens":2`))
		wantStatus(t, w, http.StatusOK)
		if !strings.Contains(w.Body.String(), `"stop_reason":"max_tokens"`) {
			t.Errorf("stop_reason is not max_tokens: %s", w.Body.String())
		}
	})
}

// Each output block carries its own ordinal.
//
// ir.Event numbers the blocks and [tgo.Event] does not, which is the one piece
// of state the translation keeps. Two blocks sharing an index is not a
// cosmetic error: an Anthropic client indexes its content array by it, so the
// answer overwrites the thought.
func TestEachBlockCarriesItsOwnIndex(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: thinkThenSay("weighing it", "the answer")}
	s := newTestServer(t, eng)
	fs := streamBody(t, s, "/v1/messages", routes[1].body(`,"stream":true`))

	indices := map[string][]float64{}
	for _, f := range fs {
		switch f.name {
		case "content_block_start", "content_block_stop":
			var ev struct {
				Index float64 `json:"index"`
			}
			if err := json.Unmarshal([]byte(f.data), &ev); err != nil {
				t.Fatalf("%s: %v: %s", f.name, err, f.data)
			}
			indices[f.name] = append(indices[f.name], ev.Index)
		}
	}
	for _, name := range []string{"content_block_start", "content_block_stop"} {
		got := indices[name]
		if len(got) != 2 || got[0] != 0 || got[1] != 1 {
			t.Errorf("%s indices = %v, want [0 1]: two blocks under one index make a "+
				"client overwrite the first with the second", name, got)
		}
	}
}

// §3.2: a thinking block streams as thinking events, not as text. It is the one
// thing a chat UI must know in order to render it.
func TestAThinkingBlockStreamsAsThinking(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: thinkThenSay("weighing it", "the answer")}
	s := newTestServer(t, eng)

	fs := streamBody(t, s, "/v1/messages", routes[1].body(`,"stream":true`))
	var body strings.Builder
	for _, f := range fs {
		body.WriteString(f.data)
	}
	if !strings.Contains(body.String(), `"thinking_delta"`) {
		t.Errorf("no thinking_delta frame: %s", body.String())
	}
	if strings.Contains(body.String(), `{"type":"text_delta","text":"weighing it"}`) {
		t.Error("the thought was streamed as text")
	}

	// And on OpenAI Chat, where the same block becomes reasoning_content.
	eng2 := &fakeEngine{script: thinkThenSay("weighing it", "the answer")}
	s2 := newTestServer(t, eng2)
	w := post(t, s2, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusOK)
	if !strings.Contains(w.Body.String(), `"reasoning_content":"weighing it"`) {
		t.Errorf("the thought is not separated from the answer: %s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"content":"the answer"`) {
		t.Errorf("the answer is not the answer: %s", w.Body.String())
	}
}

// blockingWriter is a client that only lets the engine move when it has been
// flushed.
//
// The fake engine's gate reads the same channel this writer's Flush writes, so
// a handler that buffers its events instead of flushing them stalls the engine
// after the first one and the request never finishes. That is §5's first trap
// asserted rather than described: every test that checks content passes
// without a single Flush call.
type blockingWriter struct {
	hdr     http.Header
	flushes chan struct{}

	// count is every Flush, which is the assertion; flushes is drained by the
	// engine, so its length is what has not been consumed rather than what
	// happened.
	count atomic.Int64

	mu   sync.Mutex
	buf  bytes.Buffer
	code int
}

func newBlockingWriter(capacity int) *blockingWriter {
	return &blockingWriter{hdr: http.Header{}, flushes: make(chan struct{}, capacity)}
}

func (w *blockingWriter) Header() http.Header { return w.hdr }

func (w *blockingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

func (w *blockingWriter) WriteHeader(code int) { w.code = code }

func (w *blockingWriter) Flush() {
	w.count.Add(1)
	w.flushes <- struct{}{}
}

func (w *blockingWriter) body() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.String()
}

func TestEveryEventIsFlushed(t *testing.T) {
	t.Parallel()
	script := text("one", "two", "three")
	w := newBlockingWriter(len(script) + 8)
	eng := &fakeEngine{script: script, gate: w.flushes}
	s := newTestServer(t, eng)

	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(routes[0].body(`,"stream":true`)))
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.ServeHTTP(w, r)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("the handler stalled: it buffered instead of flushing per event. "+
			"Flushes so far: %d, body: %q", w.count.Load(), w.body())
	}
	if !strings.Contains(w.body(), "[DONE]") {
		t.Errorf("the stream did not finish: %q", w.body())
	}
	// One flush per encoded event, and the count is the assertion rather than
	// any duration: a millisecond on Windows is not a millisecond.
	if got := w.count.Load(); got < int64(len(script)) {
		t.Errorf("flushes = %d, want at least one per event (%d)", got, len(script))
	}
}

// §5's second trap. A handler that ignores the request's context keeps the
// session and its KV reservation until max_tokens, and the caller has been gone
// the whole time.
func TestAClientDisconnectCancelsGeneration(t *testing.T) {
	t.Parallel()
	// A script long enough that the client can hang up in the middle of it.
	var script []tgo.Event
	script = append(script, tgo.Event{Kind: tgo.BlockStart, Block: chat.BlockText})
	for i := 0; i < 512; i++ {
		script = append(script, tgo.Event{Kind: tgo.TextDelta, Block: chat.BlockText, Text: "x"})
	}
	script = append(script, tgo.Event{Kind: tgo.BlockStop, Block: chat.BlockText})

	gate := make(chan struct{}, 1)
	eng := &fakeEngine{script: script, gate: gate}
	s := newTestServer(t, eng)
	srv := httptest.NewServer(s)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/v1/chat/completions",
		strings.NewReader(routes[0].body(`,"stream":true`)))
	if err != nil {
		t.Fatal(err)
	}
	gate <- struct{}{} // let the stream reach its second event
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 1)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("reading the first byte: %v", err)
	}
	cancel()
	resp.Body.Close()

	// The stream must stop, and stop because the context did. Waiting on the
	// channel rather than on a sleep is what makes this a test of the
	// cancellation and not of a timer.
	//
	// The generator blocks on the gate between tokens and only reaches its
	// ctx.Err() check after taking one, so it has to keep being offered
	// nudges until it stops -- a loop that stops offering the moment the
	// stream exists can leave it parked forever and report "kept generating"
	// for a stream that never got the chance to notice.
	//
	// # Why the nudger blocks instead of spinning
	//
	// The first version of this offered the nudge with a `default` and called
	// runtime.Gosched, which is a busy-wait against the goroutine it is
	// waiting for. On an idle machine that is merely wasteful; under -race
	// with the whole suite running it is a starvation loop, and it timed out
	// at exactly its deadline while the generator sat runnable behind it.
	//
	// So the nudger is its own goroutine doing a *blocking* send, which parks
	// it whenever the generator is not ready and gives the core back. The test
	// then blocks on the stop channel rather than polling it. Nothing spins.
	var st *fakeStream
	for deadline := time.Now().Add(20 * time.Second); st == nil; {
		if got := eng.took(); len(got) == 1 {
			st = got[0].streamOf()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("no stream was started")
		}
		time.Sleep(time.Millisecond)
	}

	nudged := make(chan struct{})
	go func() {
		defer close(nudged)
		for {
			select {
			case gate <- struct{}{}:
			case <-st.stopped:
				return
			}
		}
	}()
	select {
	case <-st.stopped:
	case <-time.After(20 * time.Second):
		t.Fatal("the stream kept generating after the client hung up")
	}
	<-nudged
	if !st.cancelled {
		t.Error("the stream ended for a reason other than the cancelled context")
	}
	// And the session went back: the reservation §6 counts is released by the
	// close, not by the client.
	sess := eng.only(t)
	for closeBy := time.Now().Add(10 * time.Second); !sess.isClosed() &&
		time.Now().Before(closeBy); {
		time.Sleep(time.Millisecond)
	}
	if !sess.isClosed() {
		t.Error("the session was not closed after the client hung up")
	}
}

// §4's loss report has one channel to the caller, and on a streaming answer it
// is a header written before the first frame.
//
// An httptest.ResponseRecorder keeps its header map readable whatever the
// order, so it cannot tell a header that was sent from one that was set too
// late. This goes over a real connection, where a header set after WriteHeader
// reaches nobody.
func TestTheLossHeaderSurvivesARealStreamingConnection(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("hi")})
	srv := httptest.NewServer(s)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(routes[0].body(`,"stream":true,"user":"u-1"`)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("X-Tgo-Loss"); got != "user" {
		t.Errorf("X-Tgo-Loss over the wire = %q, want %q: the header was set after the "+
			"status line and reached nobody", got, "user")
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "[DONE]") {
		t.Errorf("the stream did not finish: %q", body)
	}
}

// A cancelled request on the non-streaming path is not an empty success.
//
// Nothing has been written yet there, so a bare return would let the runtime
// synthesize 200 with no body -- which a proxy or an SDK reads as a completion
// that produced nothing rather than as a request that was cancelled.
func TestACancelledRequestIsNotAnEmptySuccess(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s := newTestServer(t, &fakeEngine{script: text("never read")})
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(routes[0].body(""))).WithContext(ctx)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	if w.Code == http.StatusOK {
		t.Errorf("a cancelled request answered 200 with %q", w.Body.String())
	}
	if body := get(t, s, "/metrics").Body.String(); !strings.Contains(body,
		`tgo_sessions_rejected_total{reason="client_gone"} 1`) {
		t.Errorf("the cancellation was not counted:\n%s", body)
	}
}

// The handler reads the context itself rather than trusting the engine to.
//
// Passing the request's context into Chat and breaking the loop on it are two
// halves of one requirement: either alone passes a fake-engine test, and only
// both make a real engine stop.
func TestTheHandlerStopsEvenWhenTheEngineIgnoresTheContext(t *testing.T) {
	t.Parallel()
	var script []tgo.Event
	script = append(script, tgo.Event{Kind: tgo.BlockStart, Block: chat.BlockText})
	for range 64 {
		script = append(script, tgo.Event{Kind: tgo.TextDelta, Block: chat.BlockText, Text: "x"})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	eng := &fakeEngine{script: script, deaf: true}
	s := newTestServer(t, eng)

	for _, streaming := range []string{"", `,"stream":true`} {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(routes[0].body(streaming))).WithContext(ctx)
		w := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			defer close(done)
			s.ServeHTTP(w, r)
		}()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatalf("the handler kept generating for a client that had gone (stream=%q)",
				streaming)
		}
		// It stopped before the script ran out, which is the whole claim.
		if strings.Count(w.Body.String(), "x") > 8 {
			t.Errorf("the handler generated past the cancellation: %q", w.Body.String())
		}
	}
}
