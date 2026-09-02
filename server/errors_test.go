// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"latere.ai/x/pkg/llmdialect/ir"

	"github.com/latere-ai/tgo"
)

// §5.1 and 009-D13. Frontend has no error path and ir defines no error type, so
// a refusal, a 429 and a device failure would all reach a client as an abrupt
// close -- which a client cannot tell from a network drop. This is the encoder
// that fills the gap, and the dialects genuinely differ, which is why it is a
// table rather than one shape.

func TestAFailureMidStreamReachesTheClientInItsDialect(t *testing.T) {
	t.Parallel()
	cases := []struct {
		route int
		// wantFrame is the frame name the dialect uses, empty for a data-only
		// frame.
		wantFrame string
		wantData  []string
		wantDone  bool
	}{
		{route: 0, wantFrame: "", wantData: []string{`"error"`, "server_error"}, wantDone: true},
		{route: 1, wantFrame: "error", wantData: []string{`"type":"error"`, "api_error"}},
		{route: 2, wantFrame: "error", wantData: []string{`"type":"error"`, "server_error"}},
		{route: 3, wantFrame: "", wantData: []string{`"error"`, "server_error"}, wantDone: true},
	}
	for _, c := range cases {
		r := routes[c.route]
		t.Run(r.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("half an "), streamErr: errFake}
			s := newTestServer(t, eng)
			fs := streamBody(t, s, r.path, r.body(`,"stream":true`))

			var found *frame
			for i := range fs {
				if strings.Contains(fs[i].data, "error") {
					found = &fs[i]
				}
			}
			if found == nil {
				t.Fatalf("no error frame: %v", names(fs))
			}
			if found.name != c.wantFrame {
				t.Errorf("the error frame is named %q, want %q", found.name, c.wantFrame)
			}
			for _, want := range c.wantData {
				if !strings.Contains(found.data, want) {
					t.Errorf("the error frame does not carry %q: %s", want, found.data)
				}
			}
			if !strings.Contains(found.data, errFake.Error()) {
				t.Errorf("the error frame does not say what failed: %s", found.data)
			}
			if got := fs[len(fs)-1].data == "[DONE]"; got != c.wantDone {
				t.Errorf("ends with [DONE] = %v, want %v", got, c.wantDone)
			}
			// The text produced before the failure was already paid for and is
			// still there: a client that has rendered it does not have to take
			// it back.
			var all strings.Builder
			for _, f := range fs {
				all.WriteString(f.data)
			}
			if !strings.Contains(all.String(), "half an ") {
				t.Errorf("the output before the failure was dropped: %s", all.String())
			}
		})
	}
}

// A block left open by a failure is closed, or a client's parser stays inside
// it forever.
func TestAFailureClosesAnOpenBlock(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{
		script: []tgo.Event{
			{Kind: tgo.BlockStart, Block: "text"},
			{Kind: tgo.TextDelta, Block: "text", Text: "half"},
		},
		streamErr: errFake,
	}
	s := newTestServer(t, eng)
	fs := streamBody(t, s, "/v1/messages", routes[1].body(`,"stream":true`))
	if !strings.Contains(strings.Join(names(fs), " "), "content_block_stop") {
		t.Errorf("the open block was never closed: %v", names(fs))
	}
}

// A failure before anything was written is a status and a body, because the
// status line has not gone out yet and a body is the better answer.
func TestAFailureBeforeTheStreamIsAStatus(t *testing.T) {
	t.Parallel()
	t.Run("a session that could not be built", func(t *testing.T) {
		t.Parallel()
		s := newTestServer(t, &fakeEngine{sessionErr: errFake})
		w := post(t, s, "/v1/chat/completions", routes[0].body(""))
		wantStatus(t, w, http.StatusInternalServerError)
		wantNames(t, w, errFake.Error())
	})
	t.Run("a prompt that does not fit the context", func(t *testing.T) {
		t.Parallel()
		s := newTestServer(t, &fakeEngine{
			chatErr: fmt.Errorf("rendered 9000 tokens: %w", tgo.ErrContextExhausted)})
		w := post(t, s, "/v1/chat/completions", routes[0].body(""))
		// A refusal and never a truncation: dropping the start of a context
		// answers a question the caller did not ask, and nothing downstream can
		// tell that from an answer to the one they did.
		wantStatus(t, w, http.StatusBadRequest)
		wantNames(t, w, "context", "truncated")
	})
	t.Run("a device failure on the non-streaming path", func(t *testing.T) {
		t.Parallel()
		s := newTestServer(t, &fakeEngine{script: text("half"), streamErr: errFake})
		w := post(t, s, "/v1/chat/completions", routes[0].body(""))
		wantStatus(t, w, http.StatusInternalServerError)
		wantNames(t, w, errFake.Error())
	})
}

// The class names each dialect uses. A client switches on these, so getting one
// wrong turns a refusal into an outage in a retry loop.
func TestEachDialectNamesItsErrorClasses(t *testing.T) {
	t.Parallel()
	cases := []struct {
		kind             errKind
		anthropic, other string
		status           int
	}{
		{errInvalidRequest, "invalid_request_error", "invalid_request_error", 400},
		{errNotFound, "not_found_error", "not_found_error", 404},
		{errOverloaded, "overloaded_error", "rate_limit_error", 429},
		{errInternal, "api_error", "server_error", 500},
		{errClientGone, "request_canceled", "request_canceled", 499},
	}
	for _, c := range cases {
		if got := c.kind.name(ir.DialectAnthropicMessages); got != c.anthropic {
			t.Errorf("anthropic names %v %q, want %q", c.kind, got, c.anthropic)
		}
		if got := c.kind.name(ir.DialectOpenAIChat); got != c.other {
			t.Errorf("openai names %v %q, want %q", c.kind, got, c.other)
		}
		if got := c.kind.status(); got != c.status {
			t.Errorf("%v is status %d, want %d", c.kind, got, c.status)
		}
	}
}

// Nothing is written to a client that has gone: there is nobody to read it, and
// writing would only turn a cancelled request into a logged failure.
func TestNothingIsWrittenToAClientThatHasGone(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{})
	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	w.Body.Reset()
	w.Code = 0
	writeError(w, ir.DialectOpenAIChat, &apiError{kind: errClientGone, msg: "gone"})
	if w.Body.Len() != 0 {
		t.Errorf("a body was written to a client that hung up: %q", w.Body.String())
	}
}

// The error body always parses, even when the message does not.
func TestAnErrorBodyIsAlwaysValidJSON(t *testing.T) {
	t.Parallel()
	e := refusal("weird", "tgo: a message with a %q and a newline\n", `"quote"`)
	for _, d := range []ir.Dialect{ir.DialectAnthropicMessages, ir.DialectOpenAIChat,
		ir.DialectOpenAIResponses, dialectLegacy} {
		body := e.body(d)
		if !strings.HasPrefix(string(body), "{") || !strings.HasSuffix(string(body), "}") {
			t.Errorf("%s: %s", d, body)
		}
	}
	if e.Error() == "" {
		t.Error("an apiError with no message")
	}
}
