// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package grammar

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"
)

const personSchema = `{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age":  {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5}
  },
  "required": ["name", "age"],
  "additionalProperties": false
}`

// TestTokenSpansAGrammarBoundary is specs/015-structured-output.md section 2.
//
// The mask is over tokens and a token can carry a string terminator and a
// structural character together. Whether `":` may be typed is not a question
// about the quote or about the colon; it is a question about the pair, and the
// automaton has to answer it by walking both.
func TestTokenSpansAGrammarBoundary(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"object","properties":{"name":{"type":"string"}},`+
		`"required":["name"],"additionalProperties":false}`)

	st := g.Start()
	typeText(t, v, st, `{"name`)

	// Inside the key. A quote here terminates the key, so the token that
	// follows it with a colon is admissible and the one that follows it with a
	// brace is not: a key must be given a value.
	for _, tc := range []struct {
		token string
		want  bool
	}{
		{`":`, true},
		{`":"`, true},
		{`"}`, false},
		{`"},"`, false},
	} {
		if got := admits(st, v.id(t, tc.token)); got != tc.want {
			t.Errorf("after {\"name, admits %q = %v, want %v; allowed: %s",
				tc.token, got, tc.want, v.text(st.Allowed()))
		}
	}

	typeText(t, v, st, `":"Ada`)

	// Inside the value. The same quote now terminates a value, so the pairing
	// that is admissible flips: `"}` closes the string and the object, while
	// `":` would demand a colon after a value.
	for _, tc := range []struct {
		token string
		want  bool
	}{
		{`"}`, true},
		{`":`, false},
		{`":"`, false},
		{`"},"`, false},
		// A comma is an ordinary character inside a string, so this one is
		// admissible for a reason that has nothing to do with separators: it
		// extends the value to "Ada," and then terminates it.
		{`,"`, true},
	} {
		if got := admits(st, v.id(t, tc.token)); got != tc.want {
			t.Errorf("after {\"name\":\"Ada, admits %q = %v, want %v; allowed: %s",
				tc.token, got, tc.want, v.text(st.Allowed()))
		}
	}
}

// TestTokenCarriesHalfARune is the same claim one level down. The alphabet is
// the byte, so a token holding the lead byte of a two-byte rune must leave the
// automaton mid-character, where the string cannot be closed.
func TestTokenCarriesHalfARune(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"string"}`)

	st := g.Start()
	typeText(t, v, st, `"`)
	lead, tail := v.id(t, string([]byte{0xc3})), v.id(t, string([]byte{0xa9}))

	if !admits(st, lead) {
		t.Fatal("the lead byte of a two-byte rune is not admissible at the start of a string")
	}
	if err := st.Advance(lead); err != nil {
		t.Fatalf("advance the lead byte: %v", err)
	}
	if admits(st, v.id(t, `"`)) {
		t.Error("the string can be closed in the middle of a rune")
	}
	if !admits(st, tail) {
		t.Fatal("the continuation byte is not admissible after the lead byte")
	}
	if err := st.Advance(tail); err != nil {
		t.Fatalf("advance the continuation byte: %v", err)
	}
	typeText(t, v, st, `"`)
	if !st.Accepting() {
		t.Error(`"\xc3\xa9" is a complete string and the state does not accept`)
	}
	// A continuation byte on its own is not a character.
	fresh := g.Start()
	typeText(t, v, fresh, `"`)
	if admits(fresh, tail) {
		t.Error("a bare continuation byte is admissible, so the automaton is not checking UTF-8")
	}
}

func TestEveryScalarType(t *testing.T) {
	v := full()
	for _, tc := range []struct {
		schema string
		ok     []string
		bad    []string
	}{
		{`{"type":"string"}`, []string{`"Ada"`, `""`, `"a\nb"`, `"A"`}, []string{`Ada`, `"a`, `'a'`}},
		{`{"type":"number"}`, []string{`0`, `123`, `-4.5e6`, `-0`}, []string{`01`, `+1`, `.5`, `1.`}},
		{`{"type":"integer"}`, []string{`0`, `123`, `-4`}, []string{`1.5`, `007`, `1e2`}},
		{`{"type":"boolean"}`, []string{`true`, `false`}, []string{`True`, `1`, `null`}},
		{`{"type":"null"}`, []string{`null`}, []string{`nil`, `NULL`, `0`}},
	} {
		g := compile(t, v, tc.schema)
		for _, s := range tc.ok {
			st := g.Start()
			typeText(t, v, st, s)
			if !st.Accepting() {
				t.Errorf("%s: %q typed but the document is not complete", tc.schema, s)
			}
		}
		for _, s := range tc.bad {
			st := g.Start()
			if at := rejects(v, st, s); at < 0 && st.Accepting() {
				t.Errorf("%s: %q was admitted whole", tc.schema, s)
			}
		}
	}
}

// TestAcceptingStateStillHasContinuations is the shape that a naive "accepting
// means finished" would get wrong: 1 is a complete number and 12 is another.
func TestAcceptingStateStillHasContinuations(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"integer"}`)
	st := g.Start()
	typeText(t, v, st, `1`)
	if !st.Accepting() {
		t.Fatal("1 is a complete integer")
	}
	if !admits(st, v.id(t, `2`)) {
		t.Error("a digit is not admissible after 1, so the number was closed too early")
	}
	if !admits(st, stopID) {
		t.Error("the stop token is not admissible after a complete integer")
	}
}

func TestNestedObjectsAndArrays(t *testing.T) {
	v := full()
	g := compile(t, v, personSchema)
	for _, s := range []string{
		`{"name":"Ada","age":36}`,
		`{"name":"Ada","age":-4,"tags":["x","y"]}`,
		`{"name":"","age":0,"tags":["a","b","c","d","e"]}`,
	} {
		st := g.Start()
		typeText(t, v, st, s)
		if !st.Accepting() {
			t.Errorf("%q typed but the document is not complete", s)
		}
		var out map[string]any
		if err := json.Unmarshal([]byte(s), &out); err != nil {
			t.Errorf("%q does not parse: %v", s, err)
		}
	}
	for _, s := range []string{
		`{"age":36,"name":"Ada"}`,                                  // declared order is the admitted order
		`{"name":"Ada","age":36,"x":1}`,                            // the object is closed
		`{"name":"Ada","age":36,"tags":["only"]}`,                  // minItems is 2
		`{"name":"Ada","age":36,"tags":["a","b","c","d","e","f"]}`, // maxItems is 5
	} {
		st := g.Start()
		if at := rejects(v, st, s); at < 0 && st.Accepting() {
			t.Errorf("%q was admitted whole", s)
		}
	}
}

func TestRequiredPropertiesCannotBeSkipped(t *testing.T) {
	v := full()
	g := compile(t, v, personSchema)

	st := g.Start()
	typeText(t, v, st, `{`)
	if admits(st, v.id(t, `}`)) {
		t.Error("the object can be closed empty although two properties are required")
	}

	// The optional one may be left out, and the required ones may not.
	st = g.Start()
	typeText(t, v, st, `{"name":"Ada","age":1`)
	if !admits(st, v.id(t, `}`)) {
		t.Error("the object cannot be closed although both required properties are present")
	}

	st = g.Start()
	typeText(t, v, st, `{"name":"Ada"`)
	if admits(st, v.id(t, `}`)) {
		t.Error(`the object can be closed without "age", which is required`)
	}
}

func TestEnumAndConst(t *testing.T) {
	v := newVocab(append(crossers, `red`, `green`, `"red"`, `"green"`, `"blue"`)...)
	g := compile(t, v, `{"type":"object","properties":{`+
		`"colour":{"type":"string","enum":["red","green"]},`+
		`"kind":{"const":"fixed"}},`+
		`"required":["colour","kind"],"additionalProperties":false}`)

	st := g.Start()
	typeText(t, v, st, `{"colour":"red","kind":"fixed"}`)
	if !st.Accepting() {
		t.Error("an enum member and the const were rejected")
	}

	for _, s := range []string{`{"colour":"blue"`, `{"colour":"redd"`, `{"colour":"re"`} {
		st := g.Start()
		if at := rejects(v, st, s); at < 0 {
			t.Errorf("%q was admitted, so the enum is not enforced", s)
		}
	}

	// An enum over non-strings keeps the caller's own spelling.
	g2 := compile(t, v, `{"enum":[1,-4.5e6,true,null,{"a":1}]}`)
	for _, s := range []string{`1`, `-4.5e6`, `true`, `null`, `{"a":1}`} {
		st := g2.Start()
		typeText(t, v, st, s)
		if !st.Accepting() {
			t.Errorf("enum member %q was rejected", s)
		}
	}
}

func TestEmptyObject(t *testing.T) {
	v := full()
	for _, schema := range []string{
		`{"type":"object","properties":{},"additionalProperties":false}`,
		`{"type":"object"}`,
		`{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}`,
	} {
		g := compile(t, v, schema)
		st := g.Start()
		typeText(t, v, st, `{}`)
		if !st.Accepting() {
			t.Errorf("%s: {} is not accepted", schema)
		}
	}
	// With every property optional the single token {} is admissible whole,
	// which is the boundary-crossing case for an empty document.
	g := compile(t, v, `{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}`)
	st := g.Start()
	if !admits(st, v.id(t, `{}`)) {
		t.Error("the token {} is not admissible for an object with no required property")
	}
}

// chain nests one required property depth levels deep, and the document that
// fills it.
func chain(depth int) (schema, doc string) {
	schema, doc = `{"type":"integer"}`, `41`
	for i := 0; i < depth; i++ {
		schema = `{"type":"object","properties":{"n":` + schema + `},"required":["n"],"additionalProperties":false}`
		doc = `{"n":` + doc + `}`
	}
	return schema, doc
}

// arrayChain nests an array with no maxItems depth levels deep. It is the same
// shape as chain one construct over, and it is the shape the element subtree is
// shared for: an array names its element once and reaches it two ways, first in
// the brackets and again after a comma.
func arrayChain(depth int) (schema, doc string) {
	schema, doc = `{"type":"integer"}`, `41`
	for i := 0; i < depth; i++ {
		schema = `{"type":"array","items":` + schema + `}`
		doc = `[` + doc + `]`
	}
	return schema, doc
}

func TestDeeplyNestedSchema(t *testing.T) {
	// Seventeen levels, which is unequal to every other dimension in this file.
	const depth = 17
	schema, doc := chain(depth)
	v := full()
	g := compile(t, v, schema)
	st := g.Start()
	typeText(t, v, st, doc)
	if !st.Accepting() {
		t.Fatalf("a document %d levels deep was not accepted", depth)
	}
	// One brace short is not a document.
	st = g.Start()
	short := strings.TrimSuffix(doc, `}`)
	typeText(t, v, st, short)
	if st.Accepting() {
		t.Error("a document missing its last brace is accepted")
	}
}

// TestCompilingNestingStaysLinear guards the shape of the compile itself.
//
// A property can be reached two ways -- first in its object, or after a comma
// -- and compiling its value once per way doubles the automaton at every level
// of nesting. The growth is invisible at the depths anyone writes by hand and
// fatal at the depths a request body can carry: 2^20 copies of a subtree is
// hundreds of millions of states, built from a caller-supplied field.
func TestCompilingNestingStaysLinear(t *testing.T) {
	v := full()
	shallow, _ := chain(6)
	deep, _ := chain(12)
	small := len(compile(t, v, shallow).n.eps)
	large := len(compile(t, v, deep).n.eps)
	// Doubling per level would put the ratio at 64. Linear growth over a fixed
	// prologue puts it near two.
	if large > 3*small {
		// Fatal rather than an error: the depth-24 build below is what the
		// doubling makes fatal, so a run that has already seen the doubling
		// must not attempt it.
		t.Fatalf("doubling the depth took the automaton from %d states to %d", small, large)
	}
	// The consequence, rather than the measurement: a schema deep enough that
	// the doubling would have been fatal compiles and drives a document.
	schema, doc := chain(24)
	g := compile(t, v, schema)
	st := g.Start()
	typeText(t, v, st, doc)
	if !st.Accepting() {
		t.Error("a document twenty-four levels deep was not accepted")
	}

	// An array reaches its element the same two ways, so it doubles for the
	// same reason and has to be measured separately: the object fix does not
	// reach it. Fifteen levels of array was 1.3 million states before the
	// element subtree was shared.
	shallowArr, _ := arrayChain(6)
	deepArr, _ := arrayChain(12)
	smallArr := len(compile(t, v, shallowArr).n.eps)
	largeArr := len(compile(t, v, deepArr).n.eps)
	if largeArr > 3*smallArr {
		t.Fatalf("doubling the array depth took the automaton from %d states to %d", smallArr, largeArr)
	}
	schema, doc = arrayChain(24)
	g = compile(t, v, schema)
	st = g.Start()
	typeText(t, v, st, doc)
	if !st.Accepting() {
		t.Error("an array twenty-four levels deep was not accepted")
	}
}

// TestArrayWithNoMaximum drives the element loop that TestCompilingNestingStays
// Linear only measures. An array with no "maxItems" is what a hand-written
// schema usually is, and sharing one element subtree between the first element
// and every later one has to leave minItems exactly where it was: the loop
// carries no counter, so the required elements are copies and the copies are
// what the bound is made of.
func TestArrayWithNoMaximum(t *testing.T) {
	v := full()
	for _, tc := range []struct {
		minItems int
		ok, bad  []string
	}{
		{0, []string{`[]`, `[1]`, `[1,2]`, `[1,2,3,4,5,6,7]`}, nil},
		{1, []string{`[1]`, `[1,2]`, `[1,2,3,4,5,6,7]`}, []string{`[]`}},
		{3, []string{`[1,2,3]`, `[1,2,3,4]`, `[1,2,3,4,5,6,7]`}, []string{`[]`, `[1]`, `[1,2]`}},
	} {
		schema := fmt.Sprintf(`{"type":"array","items":{"type":"integer"},"minItems":%d}`, tc.minItems)
		g := compile(t, v, schema)
		for _, s := range tc.ok {
			st := g.Start()
			typeText(t, v, st, s)
			if !st.Accepting() {
				t.Errorf("minItems %d: %q typed but the document is not complete", tc.minItems, s)
			}
		}
		for _, s := range tc.bad {
			st := g.Start()
			if at := rejects(v, st, s); at < 0 && st.Accepting() {
				t.Errorf("minItems %d: %q was admitted whole", tc.minItems, s)
			}
		}
		// The loop must not have made the closing bracket optional: an array
		// left open is not a document however many elements it has.
		st := g.Start()
		typeText(t, v, st, `[1,2,3,4`)
		if st.Accepting() {
			t.Errorf("minItems %d: an unclosed array is accepted", tc.minItems)
		}
		if admits(st, stopID) {
			t.Errorf("minItems %d: the stop token is admissible inside an open array", tc.minItems)
		}
	}
}

// TestStopTokenOnlyWhereTheDocumentIsComplete is the headline guarantee. A
// grammar that let a stop token through mid-document would end a truncated
// document, and "the output parses by construction" would be false exactly
// where it is needed.
func TestStopTokenOnlyWhereTheDocumentIsComplete(t *testing.T) {
	v := full()
	g := compile(t, v, personSchema)
	const doc = `{"name":"Ada","age":36,"tags":["x","y"]}`

	st := g.Start()
	for at := 0; at < len(doc); at++ {
		if admits(st, stopID) {
			t.Fatalf("the stop token is admissible after the prefix %q", doc[:at])
		}
		typeText(t, v, st, doc[at:at+1])
	}
	if !st.Accepting() {
		t.Fatal("the document is complete and the state does not accept")
	}
	if !admits(st, stopID) {
		t.Fatal("the stop token is not admissible at the end of the document")
	}
	if err := st.Advance(stopID); err != nil {
		t.Fatalf("advance the stop token: %v", err)
	}
	if !st.Done() || st.Accepting() {
		t.Fatalf("after the stop token: done %v, accepting %v", st.Done(), st.Accepting())
	}
	if st.Allowed() != nil {
		t.Error("a finished state still admits tokens")
	}
	if err := st.Advance(stopID); !errors.Is(err, ErrDone) {
		t.Errorf("advance after done: %v, want ErrDone", err)
	}
	if err := st.Mask(make([]float32, v.Size())); !errors.Is(err, ErrDone) {
		t.Errorf("mask after done: %v, want ErrDone", err)
	}
}

// TestStopIdWithTextIsNeverTypedGuards the Vocab contract: a stop id is admitted
// by acceptance, never by its bytes. A tokenizer that returned the surface form
// of a control token would otherwise let it be typed inside a string.
func TestStopIdWithTextIsNeverTyped(t *testing.T) {
	v := full()
	talkative := v.id(t, `Ada`)
	g, err := Compile([]byte(`{"type":"string"}`), v, Options{Stop: []int{stopID, talkative, talkative}})
	if err != nil {
		t.Fatal(err)
	}
	st := g.Start()
	typeText(t, v, st, `"`)
	if admits(st, talkative) {
		t.Error("a stop id was admitted for its text")
	}
	typeText(t, v, st, `x"`)
	got := st.Allowed()
	if len(got) != 2 || got[0] != stopID || got[1] != talkative {
		t.Errorf("allowed at acceptance = %v, want the two stop ids deduped and sorted", got)
	}
}

func TestControlAndEmptyTokensAreNeverAdmissible(t *testing.T) {
	v := full()
	v.Pieces = append(v.Pieces, []byte{}) // a zero-length token: consumes nothing
	zero := len(v.Pieces) - 1
	g := compile(t, v, `{"type":"string"}`)
	st := g.Start()
	for _, id := range []int{0, idBase - 1, zero} {
		if admits(st, id) {
			t.Errorf("token %d has no text and is admissible", id)
		}
	}
}

func TestMaskIsAdditiveNegativeInfinity(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"boolean"}`)
	st := g.Start()

	logits := make([]float32, v.Size())
	for i := range logits {
		logits[i] = float32(i%13) - 6 // finite, and both signs
	}
	before := make([]float32, len(logits))
	copy(before, logits)

	if err := st.Mask(logits); err != nil {
		t.Fatalf("mask: %v", err)
	}
	allowed := map[int]bool{}
	for _, id := range st.Allowed() {
		allowed[id] = true
	}
	if len(allowed) == 0 {
		t.Fatal("nothing is admissible at the start of a boolean")
	}
	for i, got := range logits {
		switch {
		case allowed[i] && got != before[i]:
			t.Fatalf("admissible token %d was changed: %v -> %v", i, before[i], got)
		case !allowed[i] && !math.IsInf(float64(got), -1):
			t.Fatalf("masked token %d is %v, want -Inf", i, got)
		}
	}

	// A masked token survives everything specs/006-sampling.md section 3 does
	// after the mask: a penalty is an addition or a division and a temperature
	// is a division, and negative infinity is a fixed point of all of them.
	// This is why 015-D2 puts the mask first.
	masked := float32(math.Inf(-1))
	for _, after := range []float32{masked - 3, masked * 1.1, masked / 0.7} {
		if !math.IsInf(float64(after), -1) {
			t.Errorf("a masked logit was resurrected: %v", after)
		}
	}

	if err := st.Mask(make([]float32, v.Size()-1)); err == nil {
		t.Error("a logits row of the wrong length was accepted")
	}
}

// TestMaskRefusesWhenNothingContinues is the case the whole downstream depends
// on. An all -Inf row is not a mask: sample subtracts the row maximum before
// the exponential, -Inf minus -Inf is NaN, and an argmax with a strict
// comparison then returns token zero and reports nothing.
func TestMaskRefusesWhenNothingContinues(t *testing.T) {
	// A vocabulary with no closing brace and no whitespace: `{` can be typed
	// and nothing can follow it. Byte-live, token-dead -- the hazard a real
	// 152k vocabulary does not have because it contains every byte, and the
	// reason the error path is not dead code.
	v := newVocab()
	for _, s := range []string{`}`, `]`, " ", "\t", "\n", "\r"} {
		v.Pieces[v.byText[s]] = nil
	}

	g := compile(t, v, `{"type":"object","properties":{},"additionalProperties":false}`)
	st := g.Start()
	typeText(t, v, st, `{`)
	if got := st.Allowed(); len(got) != 0 {
		t.Fatalf("allowed = %v, want nothing", v.text(got))
	}
	if err := st.Mask(make([]float32, v.Size())); !errors.Is(err, ErrNoToken) {
		t.Fatalf("mask: %v, want ErrNoToken", err)
	}
}

func TestAdvanceRefusesAnInadmissibleToken(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"boolean"}`)
	st := g.Start()
	var nae *NotAllowedError
	err := st.Advance(v.id(t, `{`))
	if !errors.As(err, &nae) || nae.Token != v.id(t, `{`) {
		t.Fatalf("advance: %v, want a NotAllowedError naming the token", err)
	}
	if !strings.Contains(err.Error(), "not admissible") {
		t.Errorf("message %q does not say what went wrong", err)
	}
}

func TestWhitespaceMayPrecedeATokenAndNotFollowTheDocument(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"object","properties":{"a":{"type":"integer"}},`+
		`"required":["a"],"additionalProperties":false}`)

	st := g.Start()
	typeText(t, v, st, "  {\n\t\"a\" : 1 }")
	if !st.Accepting() {
		t.Fatal("whitespace between tokens was rejected")
	}
	if admits(st, v.id(t, ` `)) {
		t.Error("whitespace is admissible after the document, so the accepting state can be deferred forever")
	}
	// The token that carries a colon and a space at once is the realistic
	// boundary crosser, and it has to work.
	st = g.Start()
	typeText(t, v, st, `{"a`)
	if !admits(st, v.id(t, `": `)) {
		t.Error(`the token '": ' is not admissible where a key ends`)
	}
}

func TestLengthAndItemBounds(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"string","minLength":3,"maxLength":4}`)
	for _, s := range []string{`"abc"`, `"abcd"`} {
		st := g.Start()
		typeText(t, v, st, s)
		if !st.Accepting() {
			t.Errorf("%q is within the length bounds and was not accepted", s)
		}
	}
	for _, s := range []string{`"ab"`, `"abcde"`} {
		st := g.Start()
		if at := rejects(v, st, s); at < 0 && st.Accepting() {
			t.Errorf("%q is outside the length bounds and was accepted", s)
		}
	}
	// The count is in characters, not in bytes: two-byte é is one of the four.
	st := g.Start()
	typeText(t, v, st, "\"abé\"")
	if !st.Accepting() {
		t.Error("a three-character string with a two-byte rune was rejected")
	}
	st = g.Start()
	if at := rejects(v, st, "\"aé\""); at < 0 && st.Accepting() {
		t.Error("a two-character string was accepted below minLength 3")
	}
	// An escape is one character too.
	st = g.Start()
	typeText(t, v, st, `"a\nb"`)
	if !st.Accepting() {
		t.Error(`"a\nb" is three characters and was rejected`)
	}
}

func TestRefIsInlinedAndACycleIsRefused(t *testing.T) {
	v := full()
	g := compile(t, v, `{"$defs":{"Leaf":{"type":"integer"}},`+
		`"type":"object","properties":{"a":{"$ref":"#/$defs/Leaf"},"b":{"$ref":"#/$defs/Leaf"}},`+
		`"required":["a","b"],"additionalProperties":false}`)
	st := g.Start()
	typeText(t, v, st, `{"a":1,"b":2}`)
	if !st.Accepting() {
		t.Error("a schema with two references to one definition was not accepted")
	}
}

// TestTheCacheIsTheDesign is 015-D1. The second request over the same shape
// must add no states and must not rebuild a token set, or the cache is not one.
func TestTheCacheIsTheDesign(t *testing.T) {
	v := full()
	g := compile(t, v, personSchema)
	const doc = `{"name":"Ada","age":36,"tags":["x","y"]}`

	st := g.Start()
	typeText(t, v, st, doc)
	states, builds := g.States(), g.Builds()
	if states < 2 || builds < 2 {
		t.Fatalf("the grammar materialized %d states and %d token sets, which cannot be right", states, builds)
	}
	// A first visit walks the vocabulary; a repeat visit must not. The count
	// of builds is already below the number of steps within one document,
	// because a state repeats inside it.
	if builds >= len(doc) {
		t.Errorf("%d token sets for a %d byte document: no state repeated", builds, len(doc))
	}
	for i := 0; i < 3; i++ {
		st := g.Start()
		typeText(t, v, st, doc)
	}
	if g.States() != states || g.Builds() != builds {
		t.Errorf("after four identical requests: %d states and %d token sets, want %d and %d",
			g.States(), g.Builds(), states, builds)
	}
	// The counts do move when a document reaches somewhere new; otherwise the
	// assertion above would be measuring a grammar that never works.
	deeper := compile(t, v, personSchema)
	st = deeper.Start()
	typeText(t, v, st, doc)
	if deeper.Builds() == 0 {
		t.Error("a fresh grammar built no token set at all")
	}
}

// TestOneGrammarDrivesManyRequests exercises the shared caches. 015-D1 says the
// per-state sets are kept across requests, which makes race-safety a property
// of the design rather than of the implementation.
func TestOneGrammarDrivesManyRequests(t *testing.T) {
	v := full()
	g := compile(t, v, personSchema)
	docs := []string{
		`{"name":"Ada","age":36}`,
		`{"name":"Bo","age":-4,"tags":["x","y"]}`,
		`{"name":"","age":0,"tags":["a","b","c"]}`,
	}
	const workers = 23 // unequal to every other dimension here
	var start sync.WaitGroup
	start.Add(1)
	var done sync.WaitGroup
	fail := make([]string, workers)
	for i := range workers {
		done.Add(1)
		go func() {
			defer done.Done()
			start.Wait()
			doc := docs[i%len(docs)]
			st := g.Start()
			for at := 0; at < len(doc); {
				best, bestLen := -1, 0
				for _, id := range st.Allowed() {
					b := v.Bytes(id)
					if len(b) > bestLen && strings.HasPrefix(doc[at:], string(b)) {
						best, bestLen = id, len(b)
					}
				}
				if best < 0 {
					fail[i] = "stuck at " + doc[:at]
					return
				}
				if err := st.Advance(best); err != nil {
					fail[i] = err.Error()
					return
				}
				at += bestLen
			}
			if !st.Accepting() {
				fail[i] = "not accepting after " + doc
			}
		}()
	}
	start.Done()
	done.Wait()
	for i, f := range fail {
		if f != "" {
			t.Errorf("worker %d: %s", i, f)
		}
	}
}

func TestCompileRefusesABadVocabularyOrStopId(t *testing.T) {
	if _, err := Compile([]byte(`{"type":"null"}`), nil, Options{}); err == nil {
		t.Error("a nil vocabulary was accepted")
	}
	if _, err := Compile([]byte(`{"type":"null"}`), Pieces{}, Options{}); err == nil {
		t.Error("an empty vocabulary was accepted")
	}
	v := full()
	for _, id := range []int{-1, v.Size()} {
		if _, err := Compile([]byte(`{"type":"null"}`), v, Options{Stop: []int{id}}); err == nil {
			t.Errorf("stop id %d outside the vocabulary of %d was accepted", id, v.Size())
		}
	}
}

// A payload of the shape Pydantic emits for a nested model, which is what
// response_format actually carries.
func TestPydanticShapedPayload(t *testing.T) {
	v := full()
	schema := `{
	  "$defs": {"Address": {"type":"object","properties":{
	      "street":{"type":"string","title":"Street"},
	      "zip":{"type":"string","minLength":4,"maxLength":9}},
	      "required":["street","zip"],"title":"Address","additionalProperties":false}},
	  "type":"object",
	  "properties": {
	    "name": {"type":"string","title":"Name"},
	    "age": {"type":"integer","title":"Age"},
	    "home": {"$ref":"#/$defs/Address"},
	    "nick": {"anyOf":[{"type":"string"},{"type":"null"}],"default":null}
	  },
	  "required":["name","age","home"],
	  "title":"Person",
	  "additionalProperties":false}`
	g := compile(t, v, schema)
	for _, doc := range []string{
		`{"name":"Ada","age":36,"home":{"street":"Main","zip":"12345"}}`,
		`{"name":"Ada","age":36,"home":{"street":"","zip":"1234"},"nick":null}`,
		`{"name":"Ada","age":36,"home":{"street":"M","zip":"1234"},"nick":"A"}`,
	} {
		st := g.Start()
		typeText(t, v, st, doc)
		if !st.Accepting() {
			t.Errorf("%q was not accepted", doc)
		}
	}
	// The order the schema declares is the order admitted, and the object is
	// closed: both narrowings, on a payload that exercises them.
	for _, doc := range []string{
		`{"age":36,"name":"Ada"`,
		`{"name":"Ada","age":36,"home":{"zip":"12345"`,
		`{"name":"Ada","age":36,"home":{"street":"M","zip":"123"`,
		`{"name":"Ada","age":36,"extra":1`,
	} {
		if at := rejects(v, g.Start(), doc); at < 0 {
			t.Errorf("%q was admitted", doc)
		}
	}
}

// TestStringEscapes drives the escape alternative, which the boundary tests
// above never reach because a boundary crosser is made of ordinary characters.
//
// The surrogate block is the part worth asserting rather than assuming. A lone
// \ud800 is JSON that json.Valid accepts and that decodes in Go to U+FFFD, so
// admitting one would let the model spell a character the caller cannot get
// back -- and the block is the whole upper half of the d page, D800 through
// DFFF, not just the high surrogates D800 through DBFF.
func TestStringEscapes(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"string"}`)
	// The escapes either side of the surrogate block are admissible: D7FF is the
	// code below it and E000 the code above, so a rejection there would be the
	// exclusion overreaching rather than working.
	for _, s := range []string{
		`"a\nb"`, `"a\"b"`, `"a\\b"`, `"a\/b"`, `"\b\f\r\t"`,
		`"\u00e9"`, `"\u0041\u00FF"`, `"\ud7ff"`, `"\uD7FF"`, `"\ue000"`, `"\uE000"`,
		`"é"`, `"A"`,
	} {
		st := g.Start()
		typeText(t, v, st, s)
		if !st.Accepting() {
			t.Errorf("%s is a complete string and was not accepted", s)
		}
		var out string
		if err := json.Unmarshal([]byte(s), &out); err != nil {
			t.Errorf("%s does not decode: %v", s, err)
		}
	}
	for _, s := range []string{
		`"\ud800"`, `"\uDBFF"`, `"\udc00"`, `"\uDFFF"`, `"\udabc"`, `"\uD9ff"`,
		`"\x41"`, `"\u00"`, `"\uzzzz"`, `"\a"`,
	} {
		if at := rejects(v, g.Start(), s); at < 0 {
			t.Errorf("%s was admitted whole", s)
		}
	}
}

// TestWhitespaceMayPrecedeTheWholeDocument covers the leading run that
// TestWhitespaceMayPrecedeATokenAndNotFollowTheDocument does not: that one
// enters through an object, whose brace carries its own leading whitespace,
// so a scalar document at the root goes down a different branch.
func TestWhitespaceMayPrecedeTheWholeDocument(t *testing.T) {
	v := full()
	for _, tc := range []struct{ schema, doc string }{
		{`{"type":"integer"}`, "  7"},
		{`{"type":"number"}`, "\n\t-4.5e6"},
		{`{"type":"boolean"}`, " \r true"},
		{`{"type":"null"}`, "\tnull"},
		{`{"type":"string"}`, ` "Ada"`},
		{`{"enum":["red","green"]}`, "  \"red\""},
		{`{"type":"array","items":{"type":"integer"}}`, " [ 1 , 2 ]"},
	} {
		g := compile(t, v, tc.schema)
		st := g.Start()
		typeText(t, v, st, tc.doc)
		if !st.Accepting() {
			t.Errorf("%s: %q was not accepted", tc.schema, tc.doc)
		}
	}
}

// TestPropertyNameWithMarkupCharacters guards the one place a Go encoder would
// silently change the language. encoding/json escapes <, > and & by default, so
// a key spelled a<b would compile to the literal a<b -- a spelling the
// model has no reason to type and the caller never wrote.
func TestPropertyNameWithMarkupCharacters(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"object","properties":{"a<b&c>d":{"type":"integer"}},`+
		`"required":["a<b&c>d"],"additionalProperties":false}`)
	st := g.Start()
	typeText(t, v, st, `{"a<b&c>d":1}`)
	if !st.Accepting() {
		t.Error("a property name carrying markup characters was not typed as itself")
	}
	var out map[string]int
	if err := json.Unmarshal([]byte(`{"a<b&c>d":1}`), &out); err != nil || out["a<b&c>d"] != 1 {
		t.Errorf("the document does not decode to the declared key: %v %v", out, err)
	}
}

// TestStopIdsAreMergedInAscendingOrder is the case a synthetic vocabulary hides.
// Here the stop ids sit below every text token, so a merge that simply appended
// them would still come out sorted. A real EOS is an ordinary id in the middle
// of the vocabulary, and Allowed is binary-searched by Advance, so the order is
// load-bearing rather than cosmetic.
func TestStopIdsAreMergedInAscendingOrder(t *testing.T) {
	v := full()
	middle := v.id(t, `Ada`)
	v.Pieces[middle] = nil // now a control token, in the middle of the ids
	top := len(v.Pieces)
	v.Pieces = append(v.Pieces, nil)

	// Handed to Compile out of order, which is what a tokenizer's list of
	// terminators looks like.
	g, err := Compile([]byte(`{"type":"integer"}`), v, Options{Stop: []int{top, stopID, middle}})
	if err != nil {
		t.Fatal(err)
	}
	for _, stop := range []int{stopID, middle, top} {
		st := g.Start()
		typeText(t, v, st, `7`)
		got := st.Allowed()
		if !slices.IsSorted(got) {
			t.Fatalf("Allowed is not ascending: %v", got)
		}
		if !slices.Contains(got, stop) {
			t.Fatalf("stop id %d is not admissible at the end of the document", stop)
		}
		if err := st.Advance(stop); err != nil {
			t.Fatalf("advance stop id %d: %v", stop, err)
		}
		if !st.Done() {
			t.Fatalf("stop id %d did not finish the document", stop)
		}
	}
	// Mid-document the stop ids are gone and the set is still ascending.
	st := g.Start()
	if got := st.Allowed(); !slices.IsSorted(got) || slices.Contains(got, middle) {
		t.Errorf("before the document: sorted %v, admits the middle stop id %v",
			slices.IsSorted(got), slices.Contains(got, middle))
	}
}

// TestTheRefusalNamesTheSameKeywordEveryRun guards 015-D4's usefulness rather
// than its existence. A schema with two offending keywords must name the same
// one on every process, or a caller fixes one refusal and gets a different one
// back from an unchanged schema.
func TestTheRefusalNamesTheSameKeywordEveryRun(t *testing.T) {
	v := full()
	for _, tc := range []struct{ schema, want string }{
		{`{"type":"integer","multipleOf":2,"minimum":1,"maximum":9}`, `"maximum"`},
		{`{"type":"object","properties":{},"minLength":1,"items":{"type":"integer"}}`, `"items"`},
		{`{"type":"string","zebra":1,"aardvark":2}`, `"aardvark"`},
	} {
		for i := 0; i < 50; i++ {
			_, err := Compile([]byte(tc.schema), v, Options{})
			var ue *UnsupportedError
			if !errors.As(err, &ue) {
				t.Fatalf("%s: %v, want an UnsupportedError", tc.schema, err)
			}
			if ue.Construct != tc.want {
				t.Fatalf("%s: run %d named %s, want %s", tc.schema, i, ue.Construct, tc.want)
			}
		}
	}
}

// TestTheStateSetIsSortedBecauseItIsTheCacheKey is the invariant 015-D1 rests
// on. A determinized state is identified by its NFA state set, and the set is
// keyed by its encoding, so two paths that reach the same set must encode it
// the same way. An unsorted set would key each path separately and the cache
// would hold one entry per route rather than one per state.
func TestTheStateSetIsSortedBecauseItIsTheCacheKey(t *testing.T) {
	v := full()
	g := compile(t, v, personSchema)
	for _, doc := range []string{
		`{"name":"Ada","age":36}`,
		`{"name":"a\nb","age":-4,"tags":["x","y","z"]}`,
		`{"name":"","age":0,"tags":["a","b"]}`,
	} {
		typeText(t, v, g.Start(), doc)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.states) < 20 {
		t.Fatalf("only %d states materialized, which cannot exercise the key", len(g.states))
	}
	for key, d := range g.states {
		if !slices.IsSorted(d.set) {
			t.Errorf("the state set %v is not ascending", d.set)
		}
		if encodeSet(d.set) != key || d.key != key {
			t.Errorf("the state keyed %q re-encodes to %q", key, encodeSet(d.set))
		}
	}
}

// TestACompleteDocumentWithNoStopIdRefuses pins the one case where ErrNoToken
// is not a failure. Options.Stop empty and a closed document leaves nothing
// admissible, because this language has no trailing whitespace and there is no
// stop token to admit -- so a caller that reads the error without reading
// Accepting first reports a finished generation as a dead end.
func TestACompleteDocumentWithNoStopIdRefuses(t *testing.T) {
	v := full()
	g, err := Compile([]byte(`{"type":"object","properties":{"a":{"type":"integer"}},`+
		`"required":["a"],"additionalProperties":false}`), v, Options{})
	if err != nil {
		t.Fatal(err)
	}
	st := g.Start()
	typeText(t, v, st, `{"a":1}`)
	if !st.Accepting() {
		t.Fatal("the document is complete and the state does not accept")
	}
	if got := st.Allowed(); len(got) != 0 {
		t.Fatalf("allowed = %s, want nothing", v.text(got))
	}
	if err := st.Mask(make([]float32, v.Size())); !errors.Is(err, ErrNoToken) {
		t.Errorf("mask: %v, want ErrNoToken", err)
	}
	if st.Done() {
		t.Error("no stop token was consumed and the state reports done")
	}
}

// refChain builds a $defs chain in which every level is referenced from two
// sibling properties. It is a few hundred bytes at any depth, and it is the
// shape a generated schema really has: one model reused across fields.
func refChain(depth int) string {
	var defs []string
	defs = append(defs, `"L0":{"type":"integer"}`)
	for i := 1; i <= depth; i++ {
		prev := fmt.Sprintf(`{"$ref":"#/$defs/L%d"}`, i-1)
		defs = append(defs, fmt.Sprintf(
			`"L%d":{"type":"object","properties":{"a":%s,"b":%s},"required":["a","b"],"additionalProperties":false}`,
			i, prev, prev))
	}
	return fmt.Sprintf(`{"$defs":{%s},"$ref":"#/$defs/L%d"}`,
		strings.Join(defs, ","), depth)
}

// TestARefUsedTwiceIsBoundedByTheStateCount is the resource half of the
// nesting guard, and it is the half a count bound cannot make.
//
// c.seen is a cycle detector, not a memo: a $defs entry reached from two
// sibling properties is compiled twice, so a chain of them doubles per level
// while the schema TEXT stays flat. "maxLength" and "maxItems" bound a number
// the caller wrote down; here the caller wrote no number at all, which is why
// the bound has to be on the automaton instead.
//
// The depth below is the point: forty levels is 2^40 states without a bound,
// so a compiler that only reported the size after building it would never
// reach the report. Refusing where the recursion happens is what makes this
// test finish.
func TestARefUsedTwiceIsBoundedByTheStateCount(t *testing.T) {
	v := full()

	// Sharing itself stays legal -- the bound must refuse the blow-up, not the
	// $ref. Three levels is eight leaves and compiles well inside the bound.
	g := compile(t, v, refChain(3))
	st := g.Start()
	typeText(t, v, st, `{"a":{"a":{"a":41,"b":42},"b":{"a":43,"b":44}},"b":{"a":{"a":45,"b":46},"b":{"a":47,"b":48}}}`)
	if !st.Accepting() {
		t.Error("a three-level $ref chain did not accept its own document")
	}
	if n := g.n.size(); n > maxStates {
		t.Errorf("a three-level chain built %d states, past the %d bound", n, maxStates)
	}

	// Forty levels never finishes without the bound.
	_, err := Compile([]byte(refChain(40)), v, Options{})
	var ue *UnsupportedError
	if !errors.As(err, &ue) {
		t.Fatalf("a forty-level $ref chain compiled with %v, want an UnsupportedError", err)
	}
	if !strings.Contains(ue.Why, fmt.Sprint(maxStates)) {
		t.Errorf("the refusal %q does not name the %d states it stopped at", ue.Why, maxStates)
	}
}
