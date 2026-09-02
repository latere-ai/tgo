// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"testing"

	"github.com/latere-ai/tgo"
)

// chatLogProbs decodes the per-choice logprobs object of a chat completion.
type chatLogProbs struct {
	Content []struct {
		Token       string   `json:"token"`
		LogProb     *float64 `json:"logprob"`
		TopLogProbs []struct {
			Token   string   `json:"token"`
			LogProb *float64 `json:"logprob"`
		} `json:"top_logprobs"`
	} `json:"content"`
}

func logProbEngine() *fakeEngine {
	return &fakeEngine{
		script: text("ab"),
		probs: [][]tgo.TokenProb{{{
			ID: 7, Text: "ab", LogProb: -0.5,
			Top: []tgo.TokenProb{
				{ID: 7, Text: "ab", LogProb: -0.5},
				// 030-D3: a token the policy gave no chance is null on the wire.
				{ID: 9, Text: "zz", LogProb: math.Inf(-1)},
			},
		}}},
	}
}

func checkChatLogProbs(t *testing.T, lp *chatLogProbs) {
	t.Helper()
	if lp == nil {
		t.Fatal("logprobs is null on a route that serves them")
	}
	if len(lp.Content) != 1 || lp.Content[0].Token != "ab" {
		t.Fatalf("content = %+v, want one token ab", lp.Content)
	}
	if v := lp.Content[0].LogProb; v == nil || *v != -0.5 {
		t.Errorf("logprob = %v, want -0.5", v)
	}
	top := lp.Content[0].TopLogProbs
	if len(top) != 2 || top[0].Token != "ab" || top[1].Token != "zz" {
		t.Fatalf("top_logprobs = %+v, want [ab zz]", top)
	}
	if top[1].LogProb != nil {
		t.Errorf("the masked alternative is %v, want null", *top[1].LogProb)
	}
}

// TestChatCompletionsServeLogProbs is specs/030-logprobs.md §4's second route:
// llmdialect's ir carries logprobs on the response, so /v1/chat/completions
// answers them in the dialect's own shape and the loss report does not name
// the member. Before the ir had the field, this route reported it as a loss
// and answered null.
func TestChatCompletionsServeLogProbs(t *testing.T) {
	t.Parallel()
	eng := logProbEngine()
	s := newTestServer(t, eng)
	w := post(t, s, "/v1/chat/completions", inject(t, routes[0].body(""), `,"logprobs":true,"top_logprobs":2`))
	wantStatus(t, w, http.StatusOK)
	if loss := w.Header().Get("X-Tgo-Loss"); strings.Contains(loss, "logprobs") {
		t.Errorf("X-Tgo-Loss = %q names a member this route served", loss)
	}
	p := eng.only(t).sawPolicy()
	if !p.LogProbs || p.TopLogProbs != 2 {
		t.Fatalf("engine was asked for LogProbs=%v TopLogProbs=%d, want true, 2", p.LogProbs, p.TopLogProbs)
	}
	var body struct {
		Choices []struct {
			LogProbs *chatLogProbs `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v\n%s", err, w.Body.String())
	}
	checkChatLogProbs(t, body.Choices[0].LogProbs)
}

// TestChatCompletionsStreamTheSameLogProbs pins that delivery does not change
// the answer: the streamed text delta carries the tokens it decodes from.
func TestChatCompletionsStreamTheSameLogProbs(t *testing.T) {
	t.Parallel()
	eng := logProbEngine()
	s := newTestServer(t, eng)
	w := post(t, s, "/v1/chat/completions", inject(t, routes[0].body(""), `,"stream":true,"logprobs":true,"top_logprobs":2`))
	wantStatus(t, w, http.StatusOK)
	var found *chatLogProbs
	for line := range strings.SplitSeq(w.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") || strings.HasSuffix(line, "[DONE]") {
			continue
		}
		var chunk struct {
			Choices []struct {
				LogProbs *chatLogProbs `json:"logprobs"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("decode chunk: %v\n%s", err, line)
		}
		if len(chunk.Choices) == 1 && chunk.Choices[0].LogProbs != nil {
			found = chunk.Choices[0].LogProbs
		}
	}
	checkChatLogProbs(t, found)
}

// TestChatLogProbsNotAskedAnswerNull keeps the dialect's shape for a caller
// that did not ask: the member is present and null.
func TestChatLogProbsNotAskedAnswerNull(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("ab")})
	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusOK)
	if !strings.Contains(w.Body.String(), `"logprobs":null`) {
		t.Errorf("a request that asked for no logprobs did not answer null: %s", w.Body.String())
	}
}
