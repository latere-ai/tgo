// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/latere-ai/tgo/internal/grammar"
)

// specs/015-structured-output.md wired to a real vocabulary. internal/grammar
// is tested against a synthetic vocabulary it builds itself; everything here is
// about the three joins that package deliberately does not make -- which bytes
// a token stands for, how wide the logits row is, and which ids end a document.

// objectSchema is the fixture schema: two required properties, of different
// types, in a closed object.
//
// Its language is finite, and that is deliberate. An "integer" property would
// make it infinite -- JSON Schema spells a magnitude bound as "maximum", which
// the compiler refuses as arithmetic on the value -- and a checkpoint whose
// weights are arbitrary will type digits for as long as it is allowed to. A
// test that then ran out of budget would be reporting the package's documented
// narrowing as a failure of the mask.
const objectSchema = `{
	"type": "object",
	"properties": {
		"city": {"type": "string", "enum": ["oslo"]},
		"capital": {"type": "boolean"}
	},
	"required": ["city", "capital"],
	"additionalProperties": false
}`

// noWhitespace bans every token that carries an ASCII space, tab, newline or
// carriage return.
//
// It is a property of this fixture rather than of constrained decoding. JSON
// admits whitespace before every token, so the grammar admits it too, and the
// synthetic checkpoint's weights are arbitrary -- it draws spaces as readily as
// it draws braces, and a generation that spends its budget on them says nothing
// about the mask. Banning them leaves the structural tokens as the only
// admissible ones, so the document the model produces is the shortest one the
// schema allows and the test's budget is a real bound rather than a hope.
//
// Negative infinity is [Policy.LogitBias]'s spelling of a ban, and it composes
// with the mask because both are additive.
func noWhitespace(m *Model) map[int]float32 {
	bias := map[int]float32{}
	for id := 0; id < m.cfg.VocabSize; id++ {
		if strings.ContainsAny(string(m.tok.TextBytes(id)), " \t\n\r") {
			bias[id] = float32(math.Inf(-1))
		}
	}
	return bias
}

// answer is what objectSchema admits, decoded.
type answer struct {
	City    string `json:"city"`
	Capital bool   `json:"capital"`
}

// A constrained generation produces a complete document and then stops.
//
// The stopping is the assertion, not a detail of it. grammar.Options.Stop
// carries the ids that may end a document, and a grammar given none returns
// ErrNoToken the moment the document is complete: this language has no trailing
// whitespace, so once the last brace is typed nothing at all is admissible.
// Every constrained generation would then fail on its last step, having
// produced exactly the right answer. So the completion must end below its token
// budget, with no error, on a document that parses.
func TestAConstrainedGenerationCompletesAndStops(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(512))
	const budget = 128
	st, err := s.Complete(t.Context(), "describe a place",
		Policy{MaxTokens: budget, Schema: []byte(objectSchema), LogitBias: noWhitespace(m)})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	text, _ := collect(t, st)
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v, after %q", err, text)
	}
	if !json.Valid([]byte(text)) {
		t.Fatalf("the completion is not valid JSON: %q", text)
	}
	if n := st.Usage().CompletionTokens; n >= budget {
		t.Errorf("the completion used its whole budget of %d tokens, so it was cut off "+
			"rather than stopped: %q", budget, text)
	}
	var got answer
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatalf("decoding %q: %v", text, err)
	}
	if got.City != "oslo" {
		t.Errorf(`city = %q, want "oslo", which is the only member of its enum`, got.City)
	}
	// The property order is the schema's, which is one of the compiler's
	// narrowings and is what a caller reading the raw body sees.
	if i, j := strings.Index(text, `"city"`), strings.Index(text, `"capital"`); i < 0 ||
		j < 0 || i > j {
		t.Errorf("the properties are not in the schema's order: %q", text)
	}
}

// The stop ids are the stream's own, asserted directly rather than inferred
// from a generation that happened to end.
//
// A grammar wired with the wrong set, or with none, is invisible until the last
// step of a request, and it is invisible in exactly the shape a test that only
// checks the text would miss.
func TestTheGrammarStopsOnTheIDsTheStreamStopsOn(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	g, err := m.grammar([]byte(`{"type":"boolean"}`))
	if err != nil {
		t.Fatalf("compiling: %v", err)
	}
	st := g.Start()
	// Nothing has been typed, so the document is not complete and no stop id
	// may fire: a grammar that admitted one here would let the model answer
	// with an empty document.
	for _, id := range m.stopIDs() {
		if allowed(st, id) {
			t.Errorf("stop id %d is admissible before the document has begun", id)
		}
	}
	for _, id := range m.tok.Encode("true", false) {
		if err := st.Advance(id); err != nil {
			t.Fatalf("advancing over %d: %v", id, err)
		}
	}
	if !st.Accepting() {
		t.Fatal(`the grammar does not accept "true" as a complete boolean document`)
	}
	for _, id := range m.stopIDs() {
		if !allowed(st, id) {
			t.Errorf("stop id %d is not admissible on a complete document, so the "+
				"generation cannot end", id)
		}
	}
	// And the set is the one the stream reads, not a second list that agrees
	// today.
	sp := m.special
	if want := []int{sp.imEnd, sp.endOfText}; fmtIDs(m.stopIDs()) != fmtIDs(want) {
		t.Errorf("stopIDs = %v, want %v, which is what decoder.isStop reads",
			m.stopIDs(), want)
	}
	// The decoder and not a Stream: the stop ids are the model's, and reading
	// them through a session was what made the batched path need a second copy
	// of this decision (022 §5).
	fake := &decoder{m: m}
	for _, id := range m.stopIDs() {
		if !fake.isStop(id) {
			t.Errorf("decoder.isStop(%d) is false for an id the grammar ends on", id)
		}
	}
}

// allowed reports whether a state admits an id.
func allowed(st *grammar.State, id int) bool {
	return slices.Contains(st.Allowed(), id)
}

// A token the tokenizer does not hold resolves to -1, and an id outside the
// vocabulary is refused by the compiler. Dropping it is what keeps a checkpoint
// with no <|endoftext|> from refusing every schema.
func TestAMissingStopMarkerIsDroppedRatherThanCompiled(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	m.special.endOfText = -1
	if got := m.stopIDs(); len(got) != 1 || got[0] != m.special.imEnd {
		t.Fatalf("stopIDs = %v, want just %d", got, m.special.imEnd)
	}
	if _, err := m.grammar([]byte(`{"type":"null"}`)); err != nil {
		t.Errorf("compiling with one absent marker: %v", err)
	}
}

// The grammar is compiled against the logits row, and the bytes it reads are
// the tokenizer's real ones.
//
// Both halves are silent failures. A grammar compiled at the tokenizer's width
// is handed a wider row on every step and refuses it; a grammar handed the
// vocabulary file's spelling constrains a different language and says nothing.
func TestTheVocabularyIsTheModelsAndTheBytesAreTheTokenizers(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	v := vocabulary{tok: m.tok, size: m.cfg.VocabSize}
	if v.Size() != synthVocab {
		t.Errorf("Size = %d, want the model's %d", v.Size(), synthVocab)
	}
	if m.tok.VocabSize() >= m.cfg.VocabSize {
		t.Fatalf("the fixture no longer pads the embedding past the tokenizer: %d vs %d; "+
			"this test cannot see the confusion it exists for", m.tok.VocabSize(),
			m.cfg.VocabSize)
	}
	for id := m.tok.VocabSize(); id < m.cfg.VocabSize; id++ {
		if b := v.Bytes(id); b != nil {
			t.Fatalf("padding id %d carries text %q", id, b)
		}
	}
	// A token whose surface form differs from its bytes, which is every token
	// carrying a space in a byte-level vocabulary.
	ids := m.tok.Encode(" and", false)
	if len(ids) != 1 {
		t.Fatalf("Encode(%q) = %v, want one id", " and", ids)
	}
	if got := string(v.Bytes(ids[0])); got != " and" {
		t.Errorf("Bytes(%d) = %q, want %q; the byte-level alphabet reached the grammar",
			ids[0], got, " and")
	}
	// And the width is the one Mask accepts: a row of the model's vocabulary,
	// which is what the session reads back from the device.
	g, err := m.grammar([]byte(`{"type":"boolean"}`))
	if err != nil {
		t.Fatalf("compiling: %v", err)
	}
	row := make([]float32, m.cfg.VocabSize)
	if err := g.Start().Mask(row); err != nil {
		t.Fatalf("masking a row of the model's vocabulary: %v", err)
	}
	// The mask is a statement about tokens: `t` opens `true` and `x` opens
	// nothing.
	yes := m.tok.Encode("t", false)
	no := m.tok.Encode("x", false)
	if len(yes) != 1 || len(no) != 1 {
		t.Fatalf("the fixture holds %q as %v and %q as %v", "t", yes, "x", no)
	}
	if math.IsInf(float64(row[yes[0]]), -1) {
		t.Errorf("token %d (%q) is masked away, and it opens `true`", yes[0], "t")
	}
	if !math.IsInf(float64(row[no[0]]), -1) {
		t.Errorf("token %d (%q) is admissible, and no JSON boolean begins with it",
			no[0], "x")
	}
}

// A schema the compiler refuses is refused at the request, naming the
// construct, and the conversation is left exactly as it was found.
func TestAnUncompilableSchemaIsRefusedBeforeAnythingRuns(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	for _, c := range []struct{ name, schema, want string }{
		{"a numeric bound", `{"type":"integer","minimum":1}`, `"minimum"`},
		{"an unknown keyword", `{"type":"string","weird":true}`, `"weird"`},
		{"a recursive reference", `{"$defs":{"n":{"type":"object","properties":` +
			`{"next":{"$ref":"#/$defs/n"}},"additionalProperties":false}},` +
			`"$ref":"#/$defs/n"}`, "$ref"},
		{"a malformed schema", `{"type":`, "malformed value"},
		{"an empty schema", `{}`, `no "type"`},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			s := session(t, m, WithSessionContext(64))
			_, err := s.Complete(t.Context(), "hello",
				Policy{MaxTokens: 4, Schema: []byte(c.schema)})
			if err == nil {
				t.Fatalf("%s was accepted", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not name %s", err, c.want)
			}
			// Nothing was spent: the session still generates, which it would
			// not if the refusal had rewound a cache first.
			st, err := s.Complete(t.Context(), "hello", greedy(2))
			if err != nil {
				t.Fatalf("after a refusal: %v", err)
			}
			if _, evs := collect(t, st); len(evs) == 0 {
				t.Error("the session produced nothing after a refused schema")
			}
		})
	}
}

// The refusal carries the compiler's own reason, which is what a caller acts
// on: a numeric bound moves out of the schema and into validation, while a
// tuple could be added to the front end. "unsupported" says neither.
func TestARefusalCarriesTheObstruction(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	err := m.CheckSchema([]byte(`{"type":"integer","maximum":10}`))
	if err == nil {
		t.Fatal("a maximum was accepted")
	}
	var un *grammar.UnsupportedError
	if !errors.As(err, &un) {
		t.Fatalf("error %v is not an UnsupportedError, so a server cannot classify it", err)
	}
	if !strings.Contains(un.Why, "arithmetic") {
		t.Errorf("the reason is %q, which does not say what the obstruction is", un.Why)
	}
	if err := m.CheckSchema(nil); err == nil {
		t.Error("an empty schema was accepted")
	}
}

// The compilation is kept, because 015-D1's per-state token sets are only worth
// building if a later request finds them.
func TestACompiledSchemaIsReusedAndTheCacheIsBounded(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	first, err := m.grammar([]byte(objectSchema))
	if err != nil {
		t.Fatalf("compiling: %v", err)
	}
	again, err := m.grammar([]byte(objectSchema))
	if err != nil {
		t.Fatalf("compiling again: %v", err)
	}
	if first != again {
		t.Error("the same schema compiled twice, so every request pays the vocabulary walk")
	}
	// A schema body is caller input arriving over HTTP, so the map that keys on
	// it is bounded. One past the bound is what the bound means.
	for i := 0; i <= schemaCacheMax; i++ {
		if _, err := m.grammar(fmt.Appendf(nil,
			`{"type":"string","enum":["v%d"]}`, i)); err != nil {
			t.Fatalf("compiling variant %d: %v", i, err)
		}
	}
	m.schemaMu.Lock()
	n := len(m.schemas)
	m.schemaMu.Unlock()
	if n > schemaCacheMax {
		t.Errorf("the cache holds %d grammars, above the bound of %d", n, schemaCacheMax)
	}
}

// A draw the grammar does not admit ends the stream with that fact, rather than
// letting a document leave the language.
//
// It is reachable from a request: [Policy.LogitBias] bans a token absolutely
// and the mask is additive, so a caller who bans every admissible token leaves
// the sampler with a row it must still return an argmax of.
func TestATokenTheGrammarRefusesEndsTheStream(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	bias := map[int]float32{}
	for id := 0; id < m.cfg.VocabSize; id++ {
		bias[id] = float32(math.Inf(-1))
	}
	s := session(t, m, WithSessionContext(64))
	st, err := s.Complete(t.Context(), "hello",
		Policy{MaxTokens: 8, Schema: []byte(`{"type":"boolean"}`), LogitBias: bias})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	text, _ := collect(t, st)
	var na *grammar.NotAllowedError
	if !errors.As(st.Err(), &na) {
		t.Fatalf("stream error = %v after %q, want a token the grammar refused",
			st.Err(), text)
	}
	if text != "" {
		t.Errorf("text = %q, want nothing: no admissible token was ever drawn", text)
	}
}

// An unconstrained request is unchanged: no grammar is compiled and no mask is
// applied, so the completion is the one every other test in this package sees.
func TestNoSchemaLeavesTheStreamAlone(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	var runs []string
	for _, p := range []Policy{greedy(6), {MaxTokens: 6, Schema: nil}} {
		s := session(t, m, WithSessionContext(64))
		st, err := s.Complete(t.Context(), "the same prompt", p)
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		text, _ := collect(t, st)
		if err := st.Err(); err != nil {
			t.Fatalf("stream: %v", err)
		}
		runs = append(runs, text)
	}
	if runs[0] != runs[1] {
		t.Errorf("a nil schema changed the completion: %q then %q", runs[0], runs[1])
	}
}

// Two conversations share one compiled grammar, which is 015-D1's whole
// premise: the per-state token sets are worth building because a later request
// finds them.
//
// Sharing is the thing that has to be safe. A [Grammar] is compiled once per
// [Model] and every session reaches it, and the mask runs after the submission
// lock is released -- so two constrained requests genuinely walk one grammar at
// the same time, and both may reach a state whose token set has not been built.
// Nothing else in this package exercises that: every other test opens its own
// Model, so -race sees two grammars rather than one.
func TestTwoConstrainedStreamsShareOneGrammar(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	bias := noWhitespace(m)
	// Cold on purpose: with no CheckSchema first, both goroutines race into the
	// compilation as well as into the per-state caches.
	type result struct {
		text string
		err  error
	}
	const n = 2
	out := make([]result, n)
	// Every session is allocated before any goroutine runs, and all of them
	// are released together. Without the barrier the first request finishes
	// its compilation while the next session is still taking device memory,
	// the two map accesses never overlap, and the test reports a model with no
	// lock at all as safe -- which is how this test was first written and what
	// -race then failed to catch.
	sessions := make([]*Session, n)
	for i := range sessions {
		sessions[i] = session(t, m, WithSessionContext(512))
	}
	release := make(chan struct{})
	var ready, wg sync.WaitGroup
	ready.Add(n)
	for i := range n {
		s := sessions[i]
		wg.Go(func() {
			ready.Done()
			<-release
			st, err := s.Complete(context.Background(), "describe a place",
				Policy{MaxTokens: 64, Schema: []byte(objectSchema), LogitBias: bias})
			if err != nil {
				out[i] = result{err: err}
				return
			}
			var text string
			for st.Next() {
				text += st.Text()
			}
			out[i] = result{text: text, err: st.Err()}
		})
	}
	ready.Wait()
	close(release)
	wg.Wait()
	for i, r := range out {
		if r.err != nil {
			t.Errorf("stream %d: %v", i, r.err)
			continue
		}
		if !json.Valid([]byte(r.text)) {
			t.Errorf("stream %d produced %q, which is not valid JSON", i, r.text)
		}
	}
	// One compilation, however many requests reached it.
	m.schemaMu.Lock()
	got := len(m.schemas)
	m.schemaMu.Unlock()
	if got != 1 {
		t.Errorf("the model holds %d grammars for one schema", got)
	}
}

// The mask composes with the rest of specs/006-sampling.md's order, which is
// 015-D2 and the reason it is applied where it is.
//
// It goes on before [github.com/latere-ai/tgo/sample.Sampler.Next], and the
// penalties and the temperature are inside that call. Both are monotone in the
// logit and negative infinity is a fixed point of both, so a masked token
// cannot be brought back by a penalty that raises it or a temperature that
// flattens the row. Asserted rather than reasoned about: every other
// constrained test here samples greedily with no penalty, which is the one
// configuration where the claim is free.
func TestTheMaskSurvivesThePenaltiesAndTheTemperature(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	bias := noWhitespace(m)
	for _, c := range []struct {
		name string
		pol  Policy
	}{
		{"penalties", Policy{RepetitionPenalty: 1.2, PenaltyWindow: 8,
			PresencePenalty: 0.5, FrequencyPenalty: 0.25}},
		{"temperature and truncation", Policy{Temperature: 1.5, TopK: 20, TopP: 0.9,
			Seed: 11}},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			p := c.pol
			p.MaxTokens, p.Schema, p.LogitBias = 64, []byte(objectSchema), bias
			s := session(t, m, WithSessionContext(512))
			st, err := s.Complete(t.Context(), "describe a place", p)
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			text, _ := collect(t, st)
			if err := st.Err(); err != nil {
				t.Fatalf("stream: %v, after %q", err, text)
			}
			var got answer
			if err := json.Unmarshal([]byte(text), &got); err != nil {
				t.Fatalf("the completion does not match the schema: %v: %q", err, text)
			}
			if got.City != "oslo" {
				t.Errorf(`city = %q, want "oslo"`, got.City)
			}
		})
	}
}

// A control token is never admissible as text, and the seam that makes that
// true is [vocabulary.Bytes].
//
// This is the failure [tokenizer.Tokenizer.TextBytes] exists to prevent, seen
// from the side that matters. A vocabulary adapter written the obvious way --
// decode the id and hand back the string -- gives "<think>" back as its seven
// literal characters, and every one of them is legal inside a JSON string. The
// mask then admits a control token in the middle of a value, the stream reads
// it as a block marker rather than as text, and the document the caller is
// promised loses its closing quote. Nothing errors.
//
// The two stop ids are excluded on purpose: [grammar.Options] admits those
// where a document is complete and nowhere else, which is a different rule and
// is asserted by TestTheGrammarStopsOnTheIDsTheStreamStopsOn.
func TestAControlTokenIsNeverAdmissibleInsideAString(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	v := vocabulary{tok: m.tok, size: m.cfg.VocabSize}
	sp := m.special
	controls := []struct {
		name string
		id   int
	}{
		{"<think>", sp.think[0]},
		{"<think>\n", sp.think[1]},
		{"</think>", sp.thinkEnd},
		{"<tool_call>", sp.toolCall},
		{"</tool_call>", sp.toolEnd},
	}
	g, err := m.grammar([]byte(`{"type":"string"}`))
	if err != nil {
		t.Fatalf("compiling: %v", err)
	}
	st := g.Start()
	quote := m.tok.Encode(`"`, false)
	if len(quote) != 1 {
		t.Fatalf("the fixture holds %q as %v, want one id", `"`, quote)
	}
	if err := st.Advance(quote[0]); err != nil {
		t.Fatalf("opening a string with %d: %v", quote[0], err)
	}
	checked := 0
	for _, c := range controls {
		// The fixture carries the thinking markers and not the tool ones, and
		// -1 is what resolveSpecials reports for a marker a checkpoint does not
		// hold. The count below is what keeps that from making the loop
		// vacuous.
		if c.id < 0 {
			continue
		}
		checked++
		if b := v.Bytes(c.id); b != nil {
			t.Errorf("Bytes(%d) = %q for %q, want nil: the grammar was handed a control "+
				"token's literal characters", c.id, b, c.name)
		}
		if allowed(st, c.id) {
			t.Errorf("%q (id %d) is admissible inside a JSON string, so a constrained "+
				"completion can emit a control token in the middle of a value",
				c.name, c.id)
		}
	}
	// The thinking markers, which every chat checkpoint this package reads
	// holds and the fixture does too. Fewer than three means the loop above
	// asserted nothing.
	if checked < 3 {
		t.Fatalf("only %d control tokens were checked, so this test passed vacuously",
			checked)
	}
}

// A refused schema leaves the conversation's cached prefix where it found it.
//
// [Session.start] rewinds to the position the new prompt shares before it
// builds a stream, and that rewind drops every position past it. A compilation
// placed after it would therefore charge a request that never ran: the next
// turn prefills from scratch, silently, and only a token count says so. So the
// two sessions below run the same three requests apart from the refusal, and
// their reuse must agree.
func TestARefusedSchemaDoesNotSpendTheSessionsPrefix(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	// Distinct prompts on purpose: the refused request must share no leading
	// token with the warm one, which is the case that rewinds furthest.
	const warm, other = "alpha beta gamma delta", "zeta"
	reuseAfter := func(refuse bool) int {
		s := session(t, m, WithSessionContext(cacheCap))
		st, err := s.Complete(t.Context(), warm, greedy(2))
		if err != nil {
			t.Fatalf("the warming request: %v", err)
		}
		collect(t, st)
		if refuse {
			_, err := s.Complete(t.Context(), other,
				Policy{MaxTokens: 2, Schema: []byte(`{"type":"integer","minimum":1}`)})
			if err == nil {
				t.Fatal("the schema was accepted")
			}
		}
		st, err = s.Complete(t.Context(), warm, greedy(2))
		if err != nil {
			t.Fatalf("the second warm request: %v", err)
		}
		collect(t, st)
		return st.Usage().CachedPromptTokens
	}
	control := reuseAfter(false)
	if control == 0 {
		t.Fatal("the control run reused nothing, so this test cannot see a prefix being " +
			"spent; WithPrefixCache is off or the prompt is one token")
	}
	if got := reuseAfter(true); got != control {
		t.Errorf("CachedPromptTokens = %d after a refused schema and %d without one: the "+
			"refusal rewound the conversation before it refused", got, control)
	}
}

// A mask that cannot be applied ends the stream with that fact, rather than
// leaving the step unconstrained.
//
// No request can reach this. [grammar.State.Mask] fails on a row whose width is
// not the vocabulary's -- which [vocabulary] fixes on both sides -- or when no
// token can continue the document, and the fixture's tokenizer holds all 256
// single-byte tokens, so some token is admissible at every state the document
// is not already complete at and the stop ids cover the state where it is. The
// state is therefore built here rather than provoked: the branch is what stands
// between a future width mismatch and a completion that quietly stops being
// constrained while still reporting success.
func TestAMaskThatCannotBeAppliedEndsTheStream(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	// A vocabulary of a different width from the model's, which is the shape
	// the error exists for. 300 is one token per byte value plus room for a
	// stop id, and it is neither the tokenizer's 582 nor the model's 640.
	pieces := make(grammar.Pieces, 300)
	for b := range 256 {
		pieces[b] = []byte{byte(b)}
	}
	const narrowStop = 280
	g, err := grammar.Compile([]byte(`{"type":"boolean"}`), pieces,
		grammar.Options{Stop: []int{narrowStop}})
	if err != nil {
		t.Fatalf("compiling against the narrow vocabulary: %v", err)
	}
	s := session(t, m, WithSessionContext(64))
	st, err := s.Complete(t.Context(), "hello", greedy(4))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	st.gram = g.Start()
	text, _ := collect(t, st)
	if st.Err() == nil {
		t.Fatalf("the stream ended with no error after %q, so a step nothing masked was "+
			"reported as a constrained one", text)
	}
	if !strings.Contains(st.Err().Error(), "masking a constrained step") {
		t.Errorf("stream error = %v, want the masking failure", st.Err())
	}
	if text != "" {
		t.Errorf("text = %q, want nothing: the first step is the one that failed", text)
	}
}

// Two schemas that differ only late get their own grammars.
//
// The cache is keyed on a schema body that arrived over HTTP, and JSON Schema
// puts its structure first: two requests can agree for a hundred bytes and then
// name different enums. A key that is not the whole body hands the second
// request the first one's language, and every symptom of that is a document
// that parses -- against a schema its caller did not send.
func TestTwoSchemasSharingAPrefixGetTheirOwnGrammars(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	const alpha = `{"type":"string","enum":["alpha"]}`
	const bravo = `{"type":"string","enum":["bravo"]}`
	// The shared prefix is the point: anything shorter than the whole body
	// collides.
	if n := len(`{"type":"string","enum":["`); alpha[:n] != bravo[:n] {
		t.Fatalf("the two schemas no longer share a prefix, so this test cannot see the "+
			"collision it exists for: %q and %q", alpha, bravo)
	}
	ga, err := m.grammar([]byte(alpha))
	if err != nil {
		t.Fatalf("compiling %s: %v", alpha, err)
	}
	gb, err := m.grammar([]byte(bravo))
	if err != nil {
		t.Fatalf("compiling %s: %v", bravo, err)
	}
	if ga == gb {
		t.Fatal("the two schemas share one compiled grammar, so one of the two requests " +
			"is constrained to a language its caller did not ask for")
	}
	// And the languages differ where the schemas do: bravo's grammar refuses
	// alpha's only document.
	st := gb.Start()
	var last error
	for _, id := range m.tok.Encode(`"alpha"`, false) {
		if last = st.Advance(id); last != nil {
			break
		}
	}
	if last == nil {
		t.Errorf(`the grammar for %s admits "alpha"`, bravo)
	}
	st = ga.Start()
	for _, id := range m.tok.Encode(`"alpha"`, false) {
		if err := st.Advance(id); err != nil {
			t.Fatalf(`the grammar for %s refuses "alpha" at %d: %v`, alpha, id, err)
		}
	}
	if !st.Accepting() {
		t.Errorf(`the grammar for %s does not accept "alpha" as a complete document`, alpha)
	}
}

// A second constrained request on the same session starts a fresh document.
//
// The grammar position belongs to the [Stream] and not to the [Session]: a
// grammar is compiled once per [Model] and shared, so the walk has to be
// per request. Held on the session instead, the second turn would resume at the
// closing brace of the first -- where nothing but a stop id is admissible -- and
// every multi-turn caller would get one document and then a stream that ends
// immediately.
func TestASecondConstrainedRequestStartsANewDocument(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(512))
	bias := noWhitespace(m)
	const budget = 128
	var texts []string
	for turn := range 2 {
		st, err := s.Complete(t.Context(), "describe a place",
			Policy{MaxTokens: budget, Schema: []byte(objectSchema), LogitBias: bias})
		if err != nil {
			t.Fatalf("turn %d: %v", turn, err)
		}
		text, _ := collect(t, st)
		if err := st.Err(); err != nil {
			t.Fatalf("turn %d: %v, after %q", turn, err, text)
		}
		var got answer
		if err := json.Unmarshal([]byte(text), &got); err != nil {
			t.Fatalf("turn %d produced %q, which does not match the schema: %v",
				turn, text, err)
		}
		if got.City != "oslo" {
			t.Errorf(`turn %d: city = %q, want "oslo"`, turn, got.City)
		}
		if n := st.Usage().CompletionTokens; n >= budget {
			t.Errorf("turn %d used its whole budget of %d tokens: %q", turn, budget, text)
		}
		texts = append(texts, text)
	}
	// Both turns are whole documents. The second one having produced nothing
	// is the failure this test exists for, and an empty string is not valid
	// JSON, so the decode above already caught it -- this says which shape it
	// was.
	if texts[1] == "" {
		t.Error("the second constrained request produced nothing, so its grammar resumed " +
			"where the first one finished")
	}
}

// A stop string and a schema together are refused rather than combined.
//
// [Policy.Stop] cuts the decoded text at the point it matched and ends the
// stream with a nil error, so a stop that fires inside a document hands the
// caller half of one and reports it as a finished answer -- which is exactly
// the failure [Policy.Schema] exists to make impossible. The two also stop for
// different reasons: the grammar ends a generation where the document is
// complete, and a stop string ends it where a substring appeared.
//
// The refusal is asserted next to the behaviour it replaces: without it this
// request returns `{"city":"` with st.Err() nil, which is why silently ignoring
// Stop is not the answer either. A caller who sent one and had it dropped got a
// different request than the one they wrote.
func TestASchemaAndAStopStringAreRefusedTogether(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(512))
	// "o" is inside every document objectSchema admits: the enum's only member
	// is "oslo" and the property name is "city", so a stop that is applied
	// fires before the document closes.
	if !strings.Contains(objectSchema, "oslo") {
		t.Fatalf("the fixture schema no longer admits a document containing %q", "o")
	}
	_, err := s.Complete(t.Context(), "describe a place",
		Policy{MaxTokens: 128, Schema: []byte(objectSchema), LogitBias: noWhitespace(m),
			Stop: []string{"o"}})
	if err == nil {
		t.Fatal("a schema and a stop string were accepted together, so a stop that fires " +
			"inside the document ends the stream on half of one with no error")
	}
	for _, want := range []string{"Schema", "Stop"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal %q does not name %s", err, want)
		}
	}
	// Either one alone still runs, so the refusal is the combination and not a
	// field newly banned.
	st, err := s.Complete(t.Context(), "describe a place",
		Policy{MaxTokens: 128, Schema: []byte(objectSchema), LogitBias: noWhitespace(m)})
	if err != nil {
		t.Fatalf("a schema on its own: %v", err)
	}
	text, _ := collect(t, st)
	if !json.Valid([]byte(text)) {
		t.Errorf("a schema on its own produced %q", text)
	}
	st, err = s.Complete(t.Context(), "describe a place",
		Policy{MaxTokens: 8, Stop: []string{"o"}})
	if err != nil {
		t.Fatalf("a stop string on its own: %v", err)
	}
	collect(t, st)
}
