// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"net/http"
	"strings"
	"testing"
)

// §4's other half: a field whose absence would change the answer is refused,
// and the refusal names it. A caller who has to bisect their own request to
// find out which member was the problem has been told nothing.

// refusedSchema is a schema the grammar compiler will not compile: a numeric
// bound is arithmetic on the value, and the automaton counts characters.
//
// It is the shape a refusal test needs now that a schema is accepted. A
// compilable one -- {"type":"object"} -- runs, which is the point of this wave
// and would make the case below assert the opposite of what it says.
const refusedSchema = `{"type":"integer","minimum":1}`

func TestARefusalNamesItsField(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		route int
		body  string
		field string
	}{
		{
			name: "n greater than one", route: 0, field: "n",
			body: routes[0].body(`,"n":2`),
		},
		{
			name: "an uncompilable schema on chat", route: 0, field: "response_format",
			body: routes[0].body(`,"response_format":{"type":"json_schema","json_schema":` +
				`{"name":"out","schema":` + refusedSchema + `}}`),
		},
		{
			name: "an uncompilable schema on responses", route: 2, field: "text.format",
			body: routes[2].body(`,"text":{"format":{"type":"json_schema","name":"out",` +
				`"schema":` + refusedSchema + `}}`),
		},
		{
			name: "an uncompilable schema on messages", route: 1, field: "output_format",
			body: routes[1].body(`,"output_format":{"type":"json_schema",` +
				`"schema":` + refusedSchema + `}`),
		},
		{
			name: "an image", route: 1, field: "image",
			body: `{"model":"` + fakeName + `","max_tokens":8,"messages":[{"role":"user",
				"content":[{"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}`,
		},
		{
			name: "an image inside a tool result", route: 1, field: "image",
			body: `{"model":"` + fakeName + `","max_tokens":8,"messages":[{"role":"user",
				"content":[{"type":"tool_result","tool_use_id":"t1","content":[
				{"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}]}`,
		},
		{
			name: "a logit_bias id outside the vocabulary", route: 0, field: "logit_bias",
			body: routes[0].body(`,"logit_bias":{"99999":-100}`),
		},
		{
			name: "a negative logit_bias id", route: 0, field: "logit_bias",
			body: routes[0].body(`,"logit_bias":{"-1":-100}`),
		},
		{
			name: "echo", route: 3, field: "echo",
			body: routes[3].body(`,"echo":true`),
		},
		{
			name: "suffix", route: 3, field: "suffix",
			body: routes[3].body(`,"suffix":" the end"`),
		},
		{
			name: "best_of", route: 3, field: "best_of",
			body: routes[3].body(`,"best_of":4`),
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("ok")}
			s := newTestServer(t, eng)
			w := post(t, s, routes[c.route].path, c.body)
			wantStatus(t, w, http.StatusBadRequest)
			wantNames(t, w, c.field)
			// Nothing ran: a request that will not be answered must not first
			// take a session away from one that would.
			if got := eng.took(); len(got) != 0 {
				t.Errorf("sessions built = %d on a refused request, want 0", len(got))
			}
			if body := get(t, s, "/metrics").Body.String(); !strings.Contains(body,
				`tgo_sessions_rejected_total{reason="refused_field"} 1`) {
				t.Errorf("the refusal was not counted:\n%s", body)
			}
		})
	}
}

// A refusal is answered in the caller's dialect, because a client that cannot
// parse the error reports a transport failure for a request that was
// understood perfectly.
func TestARefusalIsShapedByTheDialect(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{})

	anth := post(t, s, "/v1/messages",
		routes[1].body(`,"output_format":{"type":"json_schema","schema":`+refusedSchema+`}`))
	wantStatus(t, anth, http.StatusBadRequest)
	if !strings.Contains(anth.Body.String(), `"type":"error"`) ||
		!strings.Contains(anth.Body.String(), `"type":"invalid_request_error"`) {
		t.Errorf("the Anthropic error body is not Anthropic's: %s", anth.Body.String())
	}

	oai := post(t, s, "/v1/chat/completions", routes[0].body(`,"n":2`))
	wantStatus(t, oai, http.StatusBadRequest)
	if !strings.Contains(oai.Body.String(), `"error":{`) ||
		!strings.Contains(oai.Body.String(), `"param":"n"`) {
		t.Errorf("the OpenAI error body is not OpenAI's: %s", oai.Body.String())
	}
}

// A member the dialect does not define is a stray key, not a refusal.
//
// tgo ignores n on either surface, so §4's own test -- a request with it and a
// request without it produce the same tokens -- makes it advisory there. It
// gets the same answer as every other unrecognized member: the request runs and
// the loss report names it.
func TestAMemberADialectDoesNotDefineIsReportedRatherThanRefused(t *testing.T) {
	t.Parallel()
	for _, route := range []int{1, 2} {
		r := routes[route]
		t.Run(r.name, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{script: text("ok")})
			w := post(t, s, r.path, r.body(`,"n":3`))
			wantStatus(t, w, http.StatusOK)
			if loss := w.Header().Get("X-Tgo-Loss"); !strings.Contains(loss, "n") {
				t.Errorf("X-Tgo-Loss = %q, want it to name the stray member", loss)
			}
		})
	}
}

// A malformed request is wrong rather than unsupported, and is answered as
// such.
func TestAMalformedRequestIsRefused(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		route int
		body  string
		names []string
	}{
		{"not JSON", 0, `{"model":`, []string{"JSON"}},
		{"not an object", 0, `[1,2,3]`, []string{"JSON object"}},
		{"no model", 0, `{"messages":[{"role":"user","content":"hi"}]}`, []string{"model"}},
		{"no messages", 0, `{"model":"` + fakeName + `"}`, []string{"messages"}},
		{"a seed that is not a number", 0, routes[0].body(`,"seed":"soon"`), []string{"seed"}},
		{"a logit_bias key that is not an id", 0, routes[0].body(`,"logit_bias":{"cat":-1}`),
			[]string{"logit_bias", "cat"}},
		{"a logit_bias that is not an object", 0, routes[0].body(`,"logit_bias":[1,2]`),
			[]string{"logit_bias"}},
		{"a negative penalty window", 0, routes[0].body(`,"penalty_window":-4`),
			[]string{"penalty_window"}},
		{"a max_tokens of zero", 0, routes[0].body(`,"max_tokens":0`), []string{"max_tokens"}},
		{"a max_tokens that does not fit an int32", 0,
			routes[0].body(`,"max_tokens":9999999999`), []string{"max_tokens"}},
		{"a top_k that is not a number", 0, routes[0].body(`,"top_k":"many"`), []string{"top_k"}},
		{"an n that is not a number", 0, routes[0].body(`,"n":"two"`), []string{"n"}},
		{"no prompt", 3, `{"model":"` + fakeName + `"}`, []string{"prompt"}},
		{"an empty prompt", 3, `{"model":"` + fakeName + `","prompt":""}`, []string{"prompt"}},
		{"two prompts", 3, `{"model":"` + fakeName + `","prompt":["a","b"]}`,
			[]string{"prompt", "batching"}},
		{"a token id prompt", 3, `{"model":"` + fakeName + `","prompt":[1,2,3]}`,
			[]string{"prompt", "token id"}},
		{"a stop that is not a string", 3, routes[3].body(`,"stop":{"end":true}`),
			[]string{"stop"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{script: text("ok")})
			w := post(t, s, routes[c.route].path, c.body)
			wantStatus(t, w, http.StatusBadRequest)
			wantNames(t, w, c.names...)
		})
	}
}

// A body larger than the bound is refused rather than read into memory.
func TestAnOversizedBodyIsRefused(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{}, WithMaxBodyBytes(32))
	w := post(t, s, "/v1/chat/completions", routes[0].body(`,"user":"`+strings.Repeat("x", 128)+`"`))
	wantStatus(t, w, http.StatusBadRequest)
}

// The IR shapes that carry no payload, which a hand-written client can send.
func TestAnIncompleteBlockIsRefused(t *testing.T) {
	t.Parallel()
	cases := []struct{ name, body string }{
		{
			"a thinking block on a user turn",
			`{"model":"` + fakeName + `","max_tokens":8,"messages":[{"role":"user",
			  "content":[{"type":"thinking","thinking":"not mine"}]}]}`,
		},
		{
			"nothing to answer",
			`{"model":"` + fakeName + `","max_tokens":8,"messages":[{"role":"user",
			  "content":[]}]}`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{script: text("ok")})
			w := post(t, s, "/v1/messages", c.body)
			wantStatus(t, w, http.StatusBadRequest)
		})
	}
}

// A route this server does not have is a 404 from the mux, and a method it
// does not take is a 405: neither reaches a handler.
func TestTheRoutesAreExactlyTheOnesInTheSpec(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{})
	for _, path := range []string{"/v1/chat/completions", "/v1/messages", "/v1/responses",
		"/v1/completions"} {
		if w := get(t, s, path); w.Code != http.StatusMethodNotAllowed {
			t.Errorf("GET %s = %d, want 405", path, w.Code)
		}
	}
	if w := get(t, s, "/v1/embeddings"); w.Code != http.StatusNotFound {
		t.Errorf("GET /v1/embeddings = %d, want 404", w.Code)
	}
}
