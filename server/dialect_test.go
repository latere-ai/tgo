// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/chat"
)

// §1 and §2: three dialects reach one adapter, and the adapter is a
// translation rather than three code paths. The evidence is that one script
// renders correctly in all of them.

func TestEachDialectRoundTripsOneRequest(t *testing.T) {
	t.Parallel()
	for _, r := range routes {
		t.Run(r.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("Hello", ", world"), prompt: 7}
			s := newTestServer(t, eng)
			w := post(t, s, r.path, r.body(""))
			wantStatus(t, w, http.StatusOK)

			var body map[string]any
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatalf("the answer is not JSON: %v: %s", err, w.Body.String())
			}
			if got := completionText(t, r.name, body); got != "Hello, world" {
				t.Errorf("text = %q, want %q", got, "Hello, world")
			}
			// One session, and it was closed: the KV reservation §6 counts is
			// given back on the way out rather than at garbage-collection time.
			sess := eng.only(t)
			if !sess.isClosed() {
				t.Error("the session was not closed")
			}
		})
	}
}

// completionText digs the assistant's words out of one dialect's answer. The
// four shapes differ, which is the whole reason the frontends exist.
func completionText(t *testing.T, dialect string, body map[string]any) string {
	t.Helper()
	switch dialect {
	case "anthropic":
		var out strings.Builder
		for _, b := range body["content"].([]any) {
			out.WriteString(b.(map[string]any)["text"].(string))
		}
		return out.String()
	case "openai-responses":
		var out strings.Builder
		for _, item := range body["output"].([]any) {
			for _, part := range item.(map[string]any)["content"].([]any) {
				out.WriteString(part.(map[string]any)["text"].(string))
			}
		}
		return out.String()
	case "openai-completions":
		return body["choices"].([]any)[0].(map[string]any)["text"].(string)
	default:
		msg := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
		if msg["content"] == nil {
			return ""
		}
		return msg["content"].(string)
	}
}

// §2: one generation, three wire formats, and the words are the same in each.
func TestOneGenerationRendersInEveryDialect(t *testing.T) {
	t.Parallel()
	const want = "the same answer"
	got := map[string]string{}
	for _, r := range routes {
		eng := &fakeEngine{script: text(want), prompt: 3}
		s := newTestServer(t, eng)
		w := post(t, s, r.path, r.body(""))
		wantStatus(t, w, http.StatusOK)
		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		got[r.name] = completionText(t, r.name, body)
	}
	for name, text := range got {
		if text != want {
			t.Errorf("%s rendered %q, want %q", name, text, want)
		}
	}
}

// §4.1's model row: this server serves one model and says so, rather than
// answering with a model the caller did not ask for.
func TestAnUnknownModelIs404(t *testing.T) {
	t.Parallel()
	for _, r := range routes {
		t.Run(r.name, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{script: text("hi")})
			body := strings.Replace(r.body(""), fakeName, "some-other-model", 1)
			w := post(t, s, r.path, body)
			wantStatus(t, w, http.StatusNotFound)
			wantNames(t, w, "some-other-model", fakeName)
		})
	}
}

// §3.1: a user who types "<think>" gets their own words into the prompt.
//
// The block boundary is structural, so there is no text for the renderer to
// match against and nothing to strip. This is 003-D4's rule seen from the
// server's side: a textual boundary can be forged and a structural one cannot.
func TestAUserMessageContainingThinkSurvives(t *testing.T) {
	t.Parallel()
	const typed = "summarise this: <think>a plan</think> and stop"
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	body, err := json.Marshal(map[string]any{
		"model":    fakeName,
		"messages": []map[string]any{{"role": "user", "content": typed}},
	})
	if err != nil {
		t.Fatal(err)
	}
	w := post(t, s, "/v1/chat/completions", string(body))
	wantStatus(t, w, http.StatusOK)

	msgs := eng.only(t).sawMessages()
	if len(msgs) != 1 || msgs[0].Role != chat.User {
		t.Fatalf("messages = %+v, want one user turn", msgs)
	}
	if len(msgs[0].Blocks) != 1 || msgs[0].Blocks[0].Type != chat.BlockText {
		t.Fatalf("blocks = %+v, want one text block", msgs[0].Blocks)
	}
	if got := msgs[0].Blocks[0].Text; got != typed {
		t.Errorf("the user's text reached the engine as %q, want %q", got, typed)
	}
}

// §3: the IR's system prompt and its tool results land where the template
// looks for them -- position zero and their own Tool turn.
func TestTheConversationIsMappedOntoChatTurns(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	body := `{"model":"` + fakeName + `","max_tokens":8,"system":"be brief","messages":[
		{"role":"user","content":"call it"},
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"I should call it"},
			{"type":"tool_use","id":"tu_1","name":"weather","input":{"city":"Munich"}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"tu_1","content":"12C"},
			{"type":"text","text":"thanks"}]}]}`
	w := post(t, s, "/v1/messages", body)
	wantStatus(t, w, http.StatusOK)

	msgs := eng.only(t).sawMessages()
	var roles []chat.Role
	for _, m := range msgs {
		roles = append(roles, m.Role)
	}
	want := []chat.Role{chat.System, chat.User, chat.Assistant, chat.Tool, chat.User}
	if len(roles) != len(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles = %v, want %v", roles, want)
		}
	}
	if got := msgs[0].Blocks[0].Text; got != "be brief" {
		t.Errorf("system = %q", got)
	}
	if tu := msgs[2].Blocks[1].ToolUse; tu == nil || tu.Name != "weather" || tu.ID != "tu_1" {
		t.Errorf("tool use = %+v", tu)
	}
	if tr := msgs[3].Blocks[0].ToolResult; tr == nil || tr.Text != "12C" {
		t.Errorf("tool result = %+v", tr)
	}
}

// A tool result keeps its place in the turn rather than being sorted to the
// end.
//
// The blocks around it are a caller's sentences, and moving a result past them
// rewrites the conversation the model is asked to continue: the template
// renders turns in order, so an out-of-order result reads as an answer to the
// wrong question.
func TestAToolResultKeepsItsPlaceInTheTurn(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	body := `{"model":"` + fakeName + `","max_tokens":8,"messages":[
		{"role":"user","content":"call it"},
		{"role":"assistant","content":[
			{"type":"tool_use","id":"tu_1","name":"weather","input":{"city":"Munich"}}]},
		{"role":"user","content":[
			{"type":"text","text":"before"},
			{"type":"tool_result","tool_use_id":"tu_1","content":"12C"},
			{"type":"text","text":"after"}]}]}`
	wantStatus(t, post(t, s, "/v1/messages", body), http.StatusOK)

	msgs := eng.only(t).sawMessages()
	var got []string
	for _, m := range msgs {
		var text string
		switch {
		case m.Blocks[0].ToolResult != nil:
			text = m.Blocks[0].ToolResult.Text
		case m.Blocks[0].ToolUse != nil:
			text = m.Blocks[0].ToolUse.Name
		default:
			text = m.Blocks[0].Text
		}
		got = append(got, string(m.Role)+":"+text)
	}
	want := []string{"user:call it", "assistant:weather", "user:before", "tool:12C", "user:after"}
	if len(got) != len(want) {
		t.Fatalf("turns = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("turns = %v, want %v", got, want)
		}
	}
}

// A system prompt sent as several blocks reaches the template's one slot with
// its parts still separated.
//
// Joining them with nothing runs two sentences together, which is a different
// system prompt from the one the caller wrote.
func TestAMultiBlockSystemPromptIsJoinedByABlankLine(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	body := `{"model":"` + fakeName + `","max_tokens":8,"system":[
		{"type":"text","text":"be brief"},{"type":"text","text":"and be kind"}],
		"messages":[{"role":"user","content":"hi"}]}`
	wantStatus(t, post(t, s, "/v1/messages", body), http.StatusOK)

	msgs := eng.only(t).sawMessages()
	if msgs[0].Role != chat.System {
		t.Fatalf("the first turn is %v, want the system turn", msgs[0].Role)
	}
	if got, want := msgs[0].Blocks[0].Text, "be brief\n\nand be kind"; got != want {
		t.Errorf("system = %q, want %q", got, want)
	}
}

// §4.1's tools row: the declarations reach the session, which renders them into
// the system turn.
func TestToolsReachTheSession(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	body := `{"model":"` + fakeName + `","messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"weather","description":"looks it up",
		"parameters":{"type":"object"}}}]}`
	w := post(t, s, "/v1/chat/completions", body)
	wantStatus(t, w, http.StatusOK)

	spec := eng.only(t).spec
	if len(spec.Tools) != 1 || spec.Tools[0].Name != "weather" {
		t.Fatalf("tools = %+v", spec.Tools)
	}
	if string(spec.Tools[0].InputSchema) != `{"type":"object"}` {
		t.Errorf("schema = %s, want it passed through verbatim", spec.Tools[0].InputSchema)
	}
}

// 009-D6: what comes back is what the model emitted, as text.
//
// Nothing has checked that the JSON is well formed, [tgo.Event] carries neither
// a call id nor a name, and every encoder needs both -- so a parsed tool_calls
// array would assert a validity nothing verified. The under-promise is the
// decision, and this is the test that keeps it.
func TestAToolCallComesBackAsTextRatherThanAParsedCall(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: []tgo.Event{
		{Kind: tgo.BlockStart, Block: chat.BlockToolUse},
		{Kind: tgo.ToolArgsDelta, Block: chat.BlockToolUse, Text: `{"name":"weather"`},
		{Kind: tgo.ToolArgsDelta, Block: chat.BlockToolUse, Text: `,"arguments":{}}`},
		{Kind: tgo.BlockStop, Block: chat.BlockToolUse},
	}}
	s := newTestServer(t, eng)
	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusOK)

	body := w.Body.String()
	if strings.Contains(body, "tool_calls") || strings.Contains(body, "tool_use") {
		t.Errorf("the answer claims a parsed call: %s", body)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		t.Fatal(err)
	}
	if got := completionText(t, "openai-chat", parsed); got != `{"name":"weather","arguments":{}}` {
		t.Errorf("text = %q, want the model's own output", got)
	}
}

// §4.1's thinking row: the request's reasoning flag reaches the session, and
// tgo's default is on.
func TestThinkingIsOnUnlessTheRequestTurnsItOff(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		path  string
		body  string
		think bool
	}{
		{"default", "/v1/chat/completions", routes[0].body(""), true},
		{"chat says none", "/v1/chat/completions", routes[0].body(`,"reasoning_effort":"none"`), false},
		{"chat says low", "/v1/chat/completions", routes[0].body(`,"reasoning_effort":"low"`), true},
		{"anthropic disables", "/v1/messages", routes[1].body(`,"thinking":{"type":"disabled"}`), false},
		{"anthropic enables", "/v1/messages",
			routes[1].body(`,"thinking":{"type":"enabled","budget_tokens":64}`), true},
		{"responses says none", "/v1/responses", routes[2].body(`,"reasoning":{"effort":"none"}`), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("ok")}
			s := newTestServer(t, eng)
			w := post(t, s, c.path, c.body)
			wantStatus(t, w, http.StatusOK)
			if got := eng.only(t).spec.Thinking; got != c.think {
				t.Errorf("thinking = %v, want %v", got, c.think)
			}
		})
	}
}

// §4.1's max_tokens row: the bound reaches Policy, under whichever name the
// dialect gave it.
func TestMaxTokensReachesThePolicy(t *testing.T) {
	t.Parallel()
	cases := []struct{ name, path, body string }{
		{"max_tokens", "/v1/chat/completions", routes[0].body(`,"max_tokens":11`)},
		{"max_completion_tokens", "/v1/chat/completions", routes[0].body(`,"max_completion_tokens":11`)},
		{"anthropic max_tokens", "/v1/messages",
			`{"model":"` + fakeName + `","max_tokens":11,"messages":[{"role":"user","content":"hi"}]}`},
		{"max_output_tokens", "/v1/responses", routes[2].body(`,"max_output_tokens":11`)},
		{"legacy max_tokens", "/v1/completions", routes[3].body(`,"max_tokens":11`)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			eng := &fakeEngine{script: text("ok")}
			s := newTestServer(t, eng)
			wantStatus(t, post(t, s, c.path, c.body), http.StatusOK)
			if got := eng.only(t).sawPolicy().MaxTokens; got != 11 {
				t.Errorf("MaxTokens = %d, want 11", got)
			}
		})
	}
}

// The informational routes.
func TestModelsListsTheOneModel(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{})
	w := get(t, s, "/v1/models")
	wantStatus(t, w, http.StatusOK)
	var body struct {
		Object string `json:"object"`
		Data   []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Object != "list" || len(body.Data) != 1 || body.Data[0].ID != fakeName {
		t.Errorf("models = %s", w.Body.String())
	}
}

func TestHealthNamesTheModelAndTheLimit(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{}, WithConcurrency(3))
	w := get(t, s, "/health")
	wantStatus(t, w, http.StatusOK)
	var body struct {
		Status      string `json:"status"`
		Model       string `json:"model"`
		Context     int    `json:"context"`
		Concurrency int    `json:"concurrency"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Status != "ok" || body.Model != fakeName || body.Context != fakeContext ||
		body.Concurrency != 3 {
		t.Errorf("health = %s", w.Body.String())
	}
}
