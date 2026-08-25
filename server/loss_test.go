// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"maps"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/latere-ai/tgo"
)

// §4 and §4.1. The loss report is llmdialect's, corrected in both directions,
// and each direction has a way of being wrong that the other's test does not
// catch.

// honourCase is one wire member tgo implements, and what it must do.
type honourCase struct {
	// field is the [tgo.Policy] field it sets, which is the key into
	// [honoured].
	field string

	// wire is the member name, which must not appear in X-Tgo-Loss.
	wire string

	route int
	extra string

	// got reads the field back out of the policy the session received.
	got func(tgo.Policy) any

	// want is what got must return.
	want any
}

// honourCases covers every wire name in [honoured]. §4.1 asks for one test per
// Policy field; this is that, plus one per additional spelling, because
// max_output_tokens and max_completion_tokens are different bugs from
// max_tokens.
var honourCases = []honourCase{
	{field: "Temperature", wire: "temperature", route: 0, extra: `,"temperature":0.25`,
		got: func(p tgo.Policy) any { return p.Temperature }, want: float32(0.25)},
	{field: "TopP", wire: "top_p", route: 0, extra: `,"top_p":0.9`,
		got: func(p tgo.Policy) any { return p.TopP }, want: float32(0.9)},
	{field: "TopK", wire: "top_k", route: 0, extra: `,"top_k":40`,
		got: func(p tgo.Policy) any { return p.TopK }, want: 40},
	{field: "TopK", wire: "top_k", route: 1, extra: `,"top_k":40`,
		got: func(p tgo.Policy) any { return p.TopK }, want: 40},
	{field: "Seed", wire: "seed", route: 0, extra: `,"seed":7`,
		got: func(p tgo.Policy) any { return p.Seed }, want: uint64(7)},
	{field: "LogitBias", wire: "logit_bias", route: 0, extra: `,"logit_bias":{"12":-3.5}`,
		got: func(p tgo.Policy) any { return p.LogitBias[12] }, want: float32(-3.5)},
	{field: "PresencePenalty", wire: "presence_penalty", route: 0, extra: `,"presence_penalty":0.5`,
		got: func(p tgo.Policy) any { return p.PresencePenalty }, want: float32(0.5)},
	{field: "FrequencyPenalty", wire: "frequency_penalty", route: 0,
		extra: `,"frequency_penalty":0.75`,
		got:   func(p tgo.Policy) any { return p.FrequencyPenalty }, want: float32(0.75)},
	{field: "RepetitionPenalty", wire: "repetition_penalty", route: 0,
		extra: `,"repetition_penalty":1.125`,
		got:   func(p tgo.Policy) any { return p.RepetitionPenalty }, want: float32(1.125)},
	{field: "PenaltyWindow", wire: "penalty_window", route: 0, extra: `,"penalty_window":64`,
		got: func(p tgo.Policy) any { return p.PenaltyWindow }, want: 64},
	{field: "MaxTokens", wire: "max_tokens", route: 0, extra: `,"max_tokens":13`,
		got: func(p tgo.Policy) any { return p.MaxTokens }, want: 13},
	{field: "MaxTokens", wire: "max_completion_tokens", route: 0,
		extra: `,"max_completion_tokens":13`,
		got:   func(p tgo.Policy) any { return p.MaxTokens }, want: 13},
	{field: "MaxTokens", wire: "max_output_tokens", route: 2, extra: `,"max_output_tokens":13`,
		got: func(p tgo.Policy) any { return p.MaxTokens }, want: 13},
	{field: "Stop", wire: "stop", route: 0, extra: `,"stop":["END"]`,
		got: func(p tgo.Policy) any { return strings.Join(p.Stop, ",") }, want: "END"},
	{field: "Stop", wire: "stop_sequences", route: 1, extra: `,"stop_sequences":["END"]`,
		got: func(p tgo.Policy) any { return strings.Join(p.Stop, ",") }, want: "END"},
}

// TestAnHonouredFieldIsAppliedAndNotReportedAsLost is §4.1 in both directions.
//
// A field parsed and not subtracted reports as unhonoured the knob that was
// honoured. A field subtracted and not parsed reports as honoured the knob that
// was dropped, which is the worse of the two: nothing downstream can tell.
// Each case asserts both.
func TestAnHonouredFieldIsAppliedAndNotReportedAsLost(t *testing.T) {
	t.Parallel()
	for _, c := range honourCases {
		r := routes[c.route]
		t.Run(c.wire+" on "+r.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("ok")}
			s := newTestServer(t, eng)
			w := post(t, s, r.path, r.body(c.extra))
			wantStatus(t, w, http.StatusOK)

			if got := c.got(eng.only(t).sawPolicy()); got != c.want {
				t.Errorf("%s reached Policy.%s as %v, want %v", c.wire, c.field, got, c.want)
			}
			if loss := w.Header().Get("X-Tgo-Loss"); strings.Contains(loss, c.wire) {
				t.Errorf("X-Tgo-Loss reports %q as lost, and it was honoured: %q", c.wire, loss)
			}
			if body := get(t, s, "/metrics").Body.String(); strings.Contains(body,
				`tgo_request_loss_total{field="`+c.wire+`"}`) {
				t.Errorf("the loss counter names %q, and it was honoured", c.wire)
			}
		})
	}
}

// Every Policy field is in the subtraction table.
//
// Without this, adding a sampling knob to the engine silently starts reporting
// it as lost the moment the first caller sends it: the frontend files it as an
// unknown member and nothing takes it back out. §4.1 asks for exactly this
// test.
func TestEveryPolicyFieldIsHonoured(t *testing.T) {
	t.Parallel()
	typ := reflect.TypeFor[tgo.Policy]()
	for i := range typ.NumField() {
		name := typ.Field(i).Name
		if _, ok := honoured[name]; !ok {
			t.Errorf("tgo.Policy.%s is not in the honoured table, so a request that sets it "+
				"would be told the field was dropped", name)
		}
	}
	for name := range honoured {
		if _, ok := typ.FieldByName(name); !ok {
			t.Errorf("the honoured table names %q, which tgo.Policy does not have: the "+
				"subtraction would hide a field nothing applies", name)
		}
	}
}

// Every wire name the table subtracts is a name some case actually sends,
// which is what keeps the subtraction from covering a field nothing parses.
func TestEveryHonouredWireNameIsExercised(t *testing.T) {
	t.Parallel()
	seen := map[string]bool{}
	for _, c := range honourCases {
		seen[c.wire] = true
		if !slices.Contains(honoured[c.field], c.wire) {
			t.Errorf("the case for %q claims Policy.%s, which honoured does not map to it",
				c.wire, c.field)
		}
	}
	for _, wire := range slices.Sorted(maps.Keys(honouredWire)) {
		if !seen[wire] {
			t.Errorf("no case sends %q, so nothing proves it is parsed rather than only "+
				"subtracted", wire)
		}
	}
}

// The subtraction is per dialect, and it is exactly the set of names that
// reached Policy.
//
// This is 009-D12's invariant as a matrix rather than as a list: every honoured
// wire name against every route, asserting that a member was applied if and
// only if it was left out of the loss report. Both diagonals are bugs, and the
// quiet one is the second: max_output_tokens sent to /v1/chat/completions sets
// no bound, and subtracting the name anyway would report an unbounded
// completion as one that honoured its limit.
func TestAWireNameIsSubtractedExactlyWhereItIsApplied(t *testing.T) {
	t.Parallel()
	// The cases come from honourCases so the two cannot drift: one entry per
	// wire name, sent on every route rather than only on its own.
	byWire := map[string]honourCase{}
	for _, c := range honourCases {
		byWire[c.wire] = c
	}
	for _, wire := range slices.Sorted(maps.Keys(honouredWire)) {
		c, ok := byWire[wire]
		if !ok {
			t.Fatalf("no case sends %q, so nothing says which routes apply it", wire)
		}
		for _, r := range routes {
			t.Run(wire+" on "+r.name, func(t *testing.T) {
				t.Parallel()
				eng := &fakeEngine{script: text("ok")}
				s := newTestServer(t, eng)
				w := post(t, s, r.path, inject(t, r.body(""), c.extra))
				wantStatus(t, w, http.StatusOK)

				applied := c.got(eng.only(t).sawPolicy()) == c.want
				reported := slices.Contains(
					strings.Split(w.Header().Get("X-Tgo-Loss"), ", "), wire)

				if want := honouredOn[r.dialect][wire]; applied != want {
					t.Errorf("%s reached Policy.%s = %v on %s, and the table says %v: the "+
						"subtraction is reading a dialect this member does not belong to",
						wire, c.field, applied, r.name, want)
				}
				if applied == reported {
					if applied {
						t.Errorf("X-Tgo-Loss = %q on %s reports %q, which was applied",
							w.Header().Get("X-Tgo-Loss"), r.name, wire)
					} else {
						t.Errorf("%s on %s set nothing and X-Tgo-Loss = %q says nothing: the "+
							"knob was dropped and the caller was told it was honoured",
							wire, r.name, w.Header().Get("X-Tgo-Loss"))
					}
				}
			})
		}
	}
}

// The per-dialect tables and the Policy table describe the same set of names.
//
// A name in honouredOn that Policy does not have would subtract a field nothing
// applies; a name in honoured that no dialect claims would be a knob no route
// can reach, which is a knob that is always reported as lost.
func TestTheDialectTablesAgreeWithThePolicyTable(t *testing.T) {
	t.Parallel()
	union := map[string]bool{}
	for d, names := range honouredOn {
		for n := range names {
			if !honouredWire[n] {
				t.Errorf("%s claims to honour %q, which no tgo.Policy field is mapped to", d, n)
			}
			union[n] = true
		}
	}
	for _, n := range slices.Sorted(maps.Keys(honouredWire)) {
		if !union[n] {
			t.Errorf("no dialect applies %q, so every request carrying it would be told it "+
				"was dropped", n)
		}
	}
	for _, r := range routes {
		if _, ok := honouredOn[r.dialect]; !ok {
			t.Errorf("%s has no subtraction table, so every sampling knob sent to it is "+
				"reported as lost", r.name)
		}
	}
}

// §4's advisory half: the request runs, and the loss is visible.
func TestAnAdvisoryFieldRunsAndIsReported(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		route int
		extra string
		want  string
	}{
		{"service_tier", 0, `,"service_tier":"flex"`, "service_tier"},
		{"user", 0, `,"user":"u-1"`, "user"},
		{"metadata on chat", 0, `,"metadata":{"trace":"t-1"}`, "metadata"},
		{"metadata on responses", 2, `,"metadata":{"trace":"t-1"}`, "metadata"},
		{"user on responses", 2, `,"user":"u-1"`, "user"},
		{"metadata on anthropic", 1, `,"metadata":{"user_id":"u-1"}`, "metadata"},
		{"logprobs", 0, `,"logprobs":true,"top_logprobs":3`, "logprobs"},
		// The legacy route's codec is tgo's own, so a member it accepts and does
		// not act on is one this package has to remember to report: it answers
		// logprobs:null on every choice.
		{"logprobs on completions", 3, `,"logprobs":2`, "logprobs"},
		{"user on completions", 3, `,"user":"u-1"`, "user"},
		{"thinking budget", 1, `,"thinking":{"type":"enabled","budget_tokens":128}`,
			"thinking.budget_tokens"},
		{"reasoning effort", 2, `,"reasoning":{"effort":"low"}`, "reasoning_effort"},
		{"tool_choice", 0, `,"tools":[{"type":"function","function":{"name":"f"}}],` +
			`"tool_choice":"required"`, "tool_choice"},
		{"parallel_tool_calls", 0, `,"tools":[{"type":"function","function":{"name":"f"}}],` +
			`"parallel_tool_calls":false`, "parallel_tool_calls"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			r := routes[c.route]
			eng := &fakeEngine{script: text("ok")}
			s := newTestServer(t, eng)
			w := post(t, s, r.path, r.body(c.extra))
			// It ran. That is the half 009-D2 amended: an advisory field is
			// not a reason to refuse a request that would answer correctly.
			wantStatus(t, w, http.StatusOK)
			loss := w.Header().Get("X-Tgo-Loss")
			if !strings.Contains(loss, c.want) {
				t.Errorf("X-Tgo-Loss = %q, want it to name %q", loss, c.want)
			}
			if body := get(t, s, "/metrics").Body.String(); !strings.Contains(body,
				`tgo_request_loss_total{field="`+c.want+`"} 1`) {
				t.Errorf("the loss counter does not name %q:\n%s", c.want, body)
			}
		})
	}
}

// cache_control is the field 009-D2 was amended for: it would have been a
// refusal, and it is advisory.
func TestCacheControlRunsAndIsReported(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	body := `{"model":"` + fakeName + `","max_tokens":8,"messages":[{"role":"user",
		"content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]}`
	w := post(t, s, "/v1/messages", body)
	wantStatus(t, w, http.StatusOK)
	if loss := w.Header().Get("X-Tgo-Loss"); !strings.Contains(loss, "cache_control") {
		t.Errorf("X-Tgo-Loss = %q, want it to name cache_control", loss)
	}
}

// §4's rule is a test rather than a list: a field is advisory when a request
// with it and a request without it produce the same tokens.
//
// Same seed, same policy, same output. If an advisory field reached the
// sampler, this is where it would show.
func TestAnAdvisoryFieldDoesNotChangeTheTokens(t *testing.T) {
	t.Parallel()
	run := func(extra string) (string, tgo.Policy) {
		eng := &fakeEngine{script: text("deterministic", " words")}
		s := newTestServer(t, eng)
		w := post(t, s, "/v1/chat/completions", routes[0].body(`,"seed":42,"temperature":0.7`+extra))
		wantStatus(t, w, http.StatusOK)
		return w.Body.String(), eng.only(t).sawPolicy()
	}
	plain, plainPolicy := run("")
	for _, extra := range []string{
		`,"user":"u-1"`,
		`,"metadata":{"trace":"t-1"}`,
		`,"service_tier":"flex"`,
	} {
		withIt, withPolicy := run(extra)
		if !reflect.DeepEqual(plainPolicy, withPolicy) {
			t.Errorf("%s changed the sampling policy: %+v vs %+v", extra, plainPolicy, withPolicy)
		}
		// The ids differ per request, so the comparison is on the answer.
		if got, want := completionOf(t, withIt), completionOf(t, plain); got != want {
			t.Errorf("%s changed the answer: %q vs %q", extra, got, want)
		}
	}
}

// completionOf pulls the assistant's words out of an OpenAI Chat body.
func completionOf(t *testing.T, body string) string {
	t.Helper()
	i := strings.Index(body, `"content":"`)
	if i < 0 {
		t.Fatalf("no content in %s", body)
	}
	rest := body[i+len(`"content":"`):]
	j := strings.Index(rest, `"`)
	return rest[:j]
}

// A request with nothing to report carries no header at all, rather than an
// empty one a client has to test for.
func TestACleanRequestCarriesNoLossHeader(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("ok")})
	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusOK)
	if _, ok := w.Header()["X-Tgo-Loss"]; ok {
		t.Errorf("X-Tgo-Loss = %q on a request that lost nothing", w.Header().Get("X-Tgo-Loss"))
	}
}

// The header is a set, comma-separated, and it is set before anything is
// written -- including on a streaming answer, where a header set after the
// first frame reaches nobody.
func TestTheLossHeaderIsSetOnAStreamingAnswerToo(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("ok")})
	w := post(t, s, "/v1/chat/completions", routes[0].body(`,"stream":true,"user":"u-1"`))
	wantStatus(t, w, http.StatusOK)
	if got := w.Header().Get("X-Tgo-Loss"); got != "user" {
		t.Errorf("X-Tgo-Loss = %q, want %q", got, "user")
	}
}

func TestTheHeaderJoinsASet(t *testing.T) {
	t.Parallel()
	if got := header([]string{"a", "b"}); got != "a, b" {
		t.Errorf("header = %q", got)
	}
	if got := header(nil); got != "" {
		t.Errorf("header = %q, want empty", got)
	}
}
