// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/chat"
)

// The shapes an engine can produce that a naive translation would drop or
// double-count. None of them is hypothetical: a stop string that lands inside a
// block, a device failure between a block's start and its first token, and a
// detokenizer that completes a code point after the block closed all produce
// exactly these.

func TestAnEventStreamWithNoBlockStartStillCarriesItsText(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: []tgo.Event{
		{Kind: tgo.TextDelta, Block: chat.BlockText, Text: "orphaned"},
	}}
	s := newTestServer(t, eng)
	w := post(t, s, "/v1/chat/completions", routes[0].body(""))
	wantStatus(t, w, http.StatusOK)
	if !strings.Contains(w.Body.String(), "orphaned") {
		t.Errorf("the text was dropped: %s", w.Body.String())
	}
}

func TestAStrayBlockStopIsNotFramed(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: []tgo.Event{
		{Kind: tgo.BlockStop, Block: chat.BlockText},
		{Kind: tgo.BlockStart, Block: chat.BlockText},
		{Kind: tgo.TextDelta, Block: chat.BlockText, Text: "hi"},
		{Kind: tgo.BlockStop, Block: chat.BlockText},
	}}
	s := newTestServer(t, eng)
	fs := streamBody(t, s, "/v1/messages", routes[1].body(`,"stream":true`))
	var starts, stops int
	for _, f := range fs {
		switch f.name {
		case "content_block_start":
			starts++
		case "content_block_stop":
			stops++
		}
	}
	if starts != stops {
		t.Errorf("%d block starts against %d stops: a client's parser cannot balance that",
			starts, stops)
	}
}

// An event kind the engine may add later is skipped rather than framed as
// something it is not.
func TestAnUnknownEventKindIsSkipped(t *testing.T) {
	t.Parallel()
	var b blockIndex
	if _, ok := b.translate(tgo.Event{Kind: tgo.EventKind(200)}); ok {
		t.Error("an unknown event kind was translated into an IR event")
	}
	if _, ok := b.closeOpen(); ok {
		t.Error("a block was closed that was never open")
	}
}

// The legacy dialect takes stop as a string as well as an array, which is what
// the OpenAI wire format allows.
func TestALegacyStopMayBeOneString(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	wantStatus(t, post(t, s, "/v1/completions", routes[3].body(`,"stop":"END"`)), http.StatusOK)
	if got := eng.only(t).sawPolicy().Stop; len(got) != 1 || got[0] != "END" {
		t.Errorf("Stop = %v, want [END]", got)
	}
}

// The legacy route sends the prompt verbatim: no template, nothing prepended,
// no control token emitted.
func TestTheLegacyRouteSendsThePromptVerbatim(t *testing.T) {
	t.Parallel()
	const raw = "Once upon a <|im_start|> time"
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	body := `{"model":"` + fakeName + `","prompt":` + quote(raw) + `}`
	wantStatus(t, post(t, s, "/v1/completions", body), http.StatusOK)

	sess := eng.only(t)
	if got := sess.sawPrompt(); got != raw {
		t.Errorf("the prompt reached the engine as %q, want %q", got, raw)
	}
	if got := sess.sawMessages(); got != nil {
		t.Errorf("the legacy route rendered a conversation: %+v", got)
	}
}

// quote is a JSON string literal for a body written by hand.
func quote(s string) string { return fmt.Sprintf("%q", s) }

// The legacy route returns the thought as part of the completion.
//
// No template ran there, so there is no turn structure to separate a thought
// from an answer: what the model produced is the completion the caller asked
// for, and dropping part of it would return a completion of a prompt this
// server did not send.
func TestTheLegacyCompletionCarriesEverythingTheModelProduced(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: thinkThenSay("weighing it", "the answer")}
	s := newTestServer(t, eng)
	w := post(t, s, "/v1/completions", routes[3].body(""))
	wantStatus(t, w, http.StatusOK)
	for _, want := range []string{"weighing it", "the answer"} {
		if !strings.Contains(w.Body.String(), want) {
			t.Errorf("the completion does not carry %q: %s", want, w.Body.String())
		}
	}
}

// n=1 is the ordinary request every OpenAI client sends, and it is not a
// refusal.
func TestNEqualToOneIsFine(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{script: text("ok")})
	wantStatus(t, post(t, s, "/v1/chat/completions", routes[0].body(`,"n":1`)), http.StatusOK)
}

// A null is absent, which is what every dialect's clients send for a knob they
// did not set.
func TestANullMemberIsAbsent(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	extra := `,"seed":null,"logit_bias":null,"top_k":null,"penalty_window":null,` +
		`"presence_penalty":null,"n":null,"stop":null,"thinking":null,"reasoning_effort":null`
	w := post(t, s, "/v1/chat/completions", routes[0].body(extra))
	wantStatus(t, w, http.StatusOK)
	if got := eng.only(t).sawPolicy(); got.Seed != 0 || got.TopK != 0 || got.LogitBias != nil {
		t.Errorf("a null member reached the policy: %+v", got)
	}
}

// An empty logit_bias object is no bias rather than an empty map, so a
// policy comparison against the zero value still holds.
func TestAnEmptyLogitBiasIsNoBias(t *testing.T) {
	t.Parallel()
	eng := &fakeEngine{script: text("ok")}
	s := newTestServer(t, eng)
	wantStatus(t, post(t, s, "/v1/chat/completions", routes[0].body(`,"logit_bias":{}`)),
		http.StatusOK)
	if got := eng.only(t).sawPolicy().LogitBias; got != nil {
		t.Errorf("LogitBias = %v, want nil", got)
	}
}

// A cancelled context is not a device failure, and must not be reported as one:
// a client that hung up gets nothing, and an alert that fires on server_error
// would fire on every cancelled request.
func TestACancelledContextIsNotADeviceFailure(t *testing.T) {
	t.Parallel()
	if got := streamError(context.Canceled).kind; got != errClientGone {
		t.Errorf("a cancelled context is %v, want errClientGone", got)
	}
	if got := streamError(context.DeadlineExceeded).kind; got != errClientGone {
		t.Errorf("a deadline is %v, want errClientGone", got)
	}
	if got := streamError(errFake).kind; got != errInternal {
		t.Errorf("a device failure is %v, want errInternal", got)
	}
}

// Every wire member the extras parser reads is checked, and a value of the
// wrong shape names the member rather than becoming a zero the caller believes
// was applied.
func TestEveryExtraChecksItsValue(t *testing.T) {
	t.Parallel()
	for _, member := range []string{"seed", "presence_penalty", "frequency_penalty",
		"repetition_penalty", "penalty_window", "top_k"} {
		t.Run(member, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{script: text("ok")})
			w := post(t, s, "/v1/chat/completions",
				routes[0].body(`,"`+member+`":{"not":"a number"}`))
			wantStatus(t, w, http.StatusBadRequest)
			wantNames(t, w, member)
		})
	}
}
