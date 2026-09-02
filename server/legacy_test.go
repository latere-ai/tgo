// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"testing"

	tgo "github.com/latere-ai/tgo"
)

// specs/030-logprobs.md §6, the wire half.

// TestLogProbsAreServedOnTheOneRouteThatCanCarryThem is §4.
//
// llmdialect's ir has no logprobs shape at all, so the three dialects it
// encodes cannot express one whatever tgo computes. /v1/completions is tgo's
// own Frontend and can. 030-D5 reports that rather than reaching past a codec
// to append a member to a body tgo did not write.
func TestLogProbsAreServedOnTheOneRouteThatCanCarryThem(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{
		script: text("ab"),
		probs: [][]tgo.TokenProb{{{
			ID: 7, Text: "ab", LogProb: -0.5,
			Top: []tgo.TokenProb{
				{ID: 7, Text: "ab", LogProb: -0.5},
				// A token the policy gave no chance. 030-D3: -Inf is null on
				// the wire, not a floor a consumer would average.
				{ID: 9, Text: "zz", LogProb: math.Inf(-1)},
			},
		}}},
	}
	s := newTestServer(t, eng)
	w := post(t, s, "/v1/completions", inject(t, routes[3].body(""), `,"logprobs":2`))
	wantStatus(t, w, http.StatusOK)

	var body struct {
		Choices []struct {
			LogProbs *struct {
				Tokens        []string              `json:"tokens"`
				TokenLogProbs []*float64            `json:"token_logprobs"`
				TopLogProbs   []map[string]*float64 `json:"top_logprobs"`
				TextOffset    []int                 `json:"text_offset"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("the response does not decode: %v\n%s", err, w.Body.String())
	}
	lp := body.Choices[0].LogProbs
	if lp == nil {
		t.Fatal("logprobs is null on the route that serves them")
	}
	if len(lp.Tokens) != 1 || lp.Tokens[0] != "ab" {
		t.Errorf("tokens = %v, want [ab]", lp.Tokens)
	}
	if len(lp.TokenLogProbs) != 1 || lp.TokenLogProbs[0] == nil || *lp.TokenLogProbs[0] != -0.5 {
		t.Errorf("token_logprobs = %v, want [-0.5]", lp.TokenLogProbs)
	}
	if len(lp.TextOffset) != 1 || lp.TextOffset[0] != 0 {
		t.Errorf("text_offset = %v, want [0]", lp.TextOffset)
	}
	// The four arrays stay parallel, which is what the shape promises and what
	// dropping a -Inf entry would break.
	if len(lp.TopLogProbs) != len(lp.Tokens) {
		t.Fatalf("top_logprobs has %d entries and tokens has %d",
			len(lp.TopLogProbs), len(lp.Tokens))
	}
	alts := lp.TopLogProbs[0]
	if v, ok := alts["zz"]; !ok {
		t.Error("the masked alternative is absent rather than null")
	} else if v != nil {
		t.Errorf(`top_logprobs["zz"] = %v, want null: a token the policy gave no `+
			`chance has a log of -Inf, and a number there is one a consumer averages`, *v)
	}
	if v, ok := alts["ab"]; !ok || v == nil || *v != -0.5 {
		t.Errorf(`top_logprobs["ab"] = %v, want -0.5`, v)
	}
}

// TestARequestThatDidNotAskStillAnswersNull is the shape this route has always
// declared: `logprobs: null`, not an empty object and not a missing member.
func TestARequestThatDidNotAskStillAnswersNull(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("ab")})
	w := post(t, s, "/v1/completions", routes[3].body(""))
	wantStatus(t, w, http.StatusOK)
	if !strings.Contains(w.Body.String(), `"logprobs":null`) {
		t.Errorf("a request that asked for no logprobs did not answer null: %s", w.Body.String())
	}
}

// TestStreamingServesTheSameLogProbs is §4's consistency requirement.
//
// Serving them only when `stream` is false would make a number depend on how
// the caller asked for delivery, and X-Tgo-Loss would call the member honoured
// either way.
func TestStreamingServesTheSameLogProbs(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{
		script: text("a", "b"),
		probs: [][]tgo.TokenProb{
			{{ID: 1, Text: "a", LogProb: -0.25}},
			{{ID: 2, Text: "b", LogProb: -1.5}},
		},
	}
	s := newTestServer(t, eng)
	w := post(t, s, "/v1/completions",
		inject(t, routes[3].body(""), `,"stream":true,"logprobs":0`))
	wantStatus(t, w, http.StatusOK)
	out := w.Body.String()
	for _, want := range []string{`"token_logprobs":[-0.25]`, `"token_logprobs":[-1.5]`} {
		if !strings.Contains(out, want) {
			t.Errorf("the stream does not carry %s:\n%s", want, out)
		}
	}
}
