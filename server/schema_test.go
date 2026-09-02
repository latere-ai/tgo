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

// §4's schema row, from both sides. A schema tgo can enforce is carried onto
// the policy; one it cannot is a 400 that says why, answered before anything is
// allocated.

// The refusal carries the compiler's own reason, not a generic one.
//
// 015-D4 makes the reason the deliverable: "a numeric bound is arithmetic on
// the value" tells a caller to move the bound into their own validation, while
// "unsupported" sends them to bisect their schema. The three dialects each name
// their own member, and all three carry the same reason.
func TestAnUncompilableSchemaIsAnsweredWithTheCompilersReason(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name                string
		route               int
		body, field, reason string
	}{
		{
			name: "a numeric bound on chat", route: 0, field: "response_format",
			body: routes[0].body(`,"response_format":{"type":"json_schema","json_schema":` +
				`{"name":"out","schema":{"type":"integer","minimum":1}}}`),
			reason: "arithmetic on the value",
		},
		{
			name: "an unknown keyword on messages", route: 1, field: "output_format",
			body: routes[1].body(`,"output_format":{"type":"json_schema","schema":` +
				`{"type":"string","weird":true}}`),
			// The body is JSON, so the compiler's quotes arrive escaped.
			reason: `\"weird\" is not supported`,
		},
		{
			name: "an open object on responses", route: 2, field: "text.format",
			body: routes[2].body(`,"text":{"format":{"type":"json_schema","name":"out",` +
				`"schema":{"type":"object","properties":{},"additionalProperties":true}}}`),
			reason: "only a closed object is regular",
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
			if body := w.Body.String(); !strings.Contains(body, c.reason) {
				t.Errorf("the refusal does not carry the compiler's reason %q: %s",
					c.reason, body)
			}
			// The schema was compiled and the session was not: a request that
			// will not run must not first take a KV reservation from one that
			// would.
			if got := eng.checked(); len(got) != 1 {
				t.Errorf("the engine was asked about %d schemas, want 1", len(got))
			}
			if got := eng.took(); len(got) != 0 {
				t.Errorf("sessions built = %d on a refused schema, want 0", len(got))
			}
		})
	}
}

// A schema the compiler accepts reaches the policy the session generates under,
// which is the whole of the wiring: a grammar compiled and then never bound
// would leave every test above green.
func TestAnAcceptedSchemaReachesThePolicy(t *testing.T) {
	t.Parallel()
	const schema = `{"type":"object","properties":{"ok":{"type":"boolean"}},` +
		`"required":["ok"],"additionalProperties":false}`
	for _, c := range []struct {
		name  string
		route int
		extra string
	}{
		{"chat", 0, `,"response_format":{"type":"json_schema","json_schema":{"name":"out",` +
			`"schema":` + schema + `}}`},
		{"messages", 1, `,"output_format":{"type":"json_schema","schema":` + schema + `}`},
		{"responses", 2, `,"text":{"format":{"type":"json_schema","name":"out","schema":` +
			schema + `}}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("ok")}
			s := newTestServer(t, eng)
			w := post(t, s, routes[c.route].path, routes[c.route].body(c.extra))
			wantStatus(t, w, http.StatusOK)
			if got := string(eng.only(t).sawPolicy().Schema); got != schema {
				t.Errorf("Policy.Schema = %q, want the schema the request carried", got)
			}
			// And it is not reported as dropped, because it was not.
			if loss := w.Header().Get("X-Tgo-Loss"); loss != "" {
				t.Errorf("X-Tgo-Loss = %q on a request whose schema was enforced", loss)
			}
		})
	}
}

// A json_schema format with no schema in it is refused rather than run
// unconstrained: the caller asked for a constraint, and running without one
// answers a different question.
func TestAJSONSchemaFormatWithNoSchemaIsRefused(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	w := post(t, s, routes[0].path,
		routes[0].body(`,"response_format":{"type":"json_schema","json_schema":{"name":"out"}}`))
	wantStatus(t, w, http.StatusBadRequest)
	wantNames(t, w, "response_format")
	// The refusal says the schema is missing rather than that it would not
	// compile. Both are 400s naming the same member, so only the text tells a
	// caller whether to fix their schema or to supply one, and without this
	// assertion the check that produces it can be deleted with every test
	// still green.
	if body := w.Body.String(); !strings.Contains(body, "carries none") {
		t.Errorf("the refusal does not say the schema is missing: %s", body)
	}
	if got := eng.checked(); len(got) != 0 {
		t.Errorf("the engine was asked to compile %d schemas, want 0: there was none to "+
			"compile", len(got))
	}
	if got := eng.took(); len(got) != 0 {
		t.Errorf("sessions built = %d, want 0", len(got))
	}
}

// response_format: {"type": "json_object"} carries no schema at all. It is the
// older spelling and it constrains nothing this package can compile, so it runs
// and is reported, which is §4's advisory half.
func TestJSONObjectModeRunsAndIsReported(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	w := post(t, s, routes[0].path, routes[0].body(`,"response_format":{"type":"json_object"}`))
	wantStatus(t, w, http.StatusOK)
	if got := eng.only(t).sawPolicy().Schema; got != nil {
		t.Errorf("Policy.Schema = %q, and json_object names no schema", got)
	}
	// And it is reported, which is the half this test's name claims. The
	// subtraction table honours `response_format`, and llmdialect files this
	// mode under its own name so that honouring the one does not swallow the
	// other: a caller who asked for JSON and got prose must hear it here.
	if loss := w.Header().Get("X-Tgo-Loss"); !strings.Contains(loss, "response_format.json_object") {
		t.Errorf("X-Tgo-Loss = %q and does not report json_object mode, which nothing "+
			"enforced", loss)
	}
}

// A schema and a stop sequence together are a 400 naming the schema's member,
// answered before a session exists.
//
// The two stop a completion by different rules, and the one that fires first
// wins: a stop string cuts the text where it matched, which is half a document
// on a request whose whole point is a document that parses. Refused here rather
// than left to [github.com/latere-ai/tgo.Policy]'s own check, which runs inside
// Session.start -- after this file has taken the KV reservation it exists to
// protect, and with no field name on the error, so a caller would read a 500
// naming nothing.
func TestASchemaWithAStopSequenceIsRefused(t *testing.T) {
	t.Parallel()
	const schema = `{"type":"object","properties":{"ok":{"type":"boolean"}},` +
		`"required":["ok"],"additionalProperties":false}`
	for _, c := range []struct {
		name  string
		route int
		field string
		extra string
	}{
		{"chat", 0, "response_format", `,"stop":["}"],"response_format":{"type":"json_schema",` +
			`"json_schema":{"name":"out","schema":` + schema + `}}`},
		{"messages", 1, "output_format", `,"stop_sequences":["}"],"output_format":` +
			`{"type":"json_schema","schema":` + schema + `}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("ok")}
			s := newTestServer(t, eng)
			w := post(t, s, routes[c.route].path, routes[c.route].body(c.extra))
			wantStatus(t, w, http.StatusBadRequest)
			wantNames(t, w, c.field)
			if body := w.Body.String(); !strings.Contains(body, "stop") {
				t.Errorf("the refusal does not name the stop sequence: %s", body)
			}
			if got := eng.took(); len(got) != 0 {
				t.Errorf("sessions built = %d on a refused request, want 0", len(got))
			}
		})
	}
}
