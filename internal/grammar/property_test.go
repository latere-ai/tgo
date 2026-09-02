// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package grammar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

// TestEverythingTheMachineAcceptsIsValid is the property the whole package
// exists for: a document the automaton walks to acceptance parses, and
// validates against the schema it was compiled from.
//
// Checked over generated walks rather than over hand-written examples, because
// a hand-written example tests the case its author thought of. The walk chooses
// randomly among the admissible tokens, so it reaches spellings nobody would
// write down -- an escape in the middle of a key's value, a number with an
// exponent, an array that stops at its minimum -- and each one is put through
// encoding/json and then through an independent validator.
func TestEverythingTheMachineAcceptsIsValid(t *testing.T) {
	// Each schema exercises a different corner, and no two counts in them are
	// equal, so a confusion between two bounds cannot pass.
	schemas := []string{
		personSchema,
		`{"type":"array","items":{"type":"number"},"minItems":1,"maxItems":3}`,
		`{"type":"object","properties":{
			"id":   {"type":"integer"},
			"ok":   {"type":"boolean"},
			"note": {"type":"string","minLength":2,"maxLength":6},
			"kind": {"enum":["a","bb","ccc"]},
			"seen": {"type":"null"}
		 },"required":["id","kind"],"additionalProperties":false}`,
		`{"type":"object","properties":{
			"rows":{"type":"array","minItems":2,"maxItems":4,"items":{
				"type":"object","properties":{"k":{"type":"string","maxLength":3},"v":{"type":"number"}},
				"required":["k"],"additionalProperties":false}}
		 },"required":["rows"],"additionalProperties":false}`,
		`{"anyOf":[{"type":"integer"},{"type":"null"},{"type":"string","maxLength":2}]}`,
		`{"$defs":{"Leaf":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}},
		  "type":"object","properties":{"l":{"$ref":"#/$defs/Leaf"},"r":{"$ref":"#/$defs/Leaf"}},
		  "required":["l"],"additionalProperties":false}`,
		`{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"object",
			"properties":{"c":{"type":"array","items":{"type":"boolean"},"maxItems":2}},
			"required":["c"],"additionalProperties":false}},"required":["b"],"additionalProperties":false}},
		  "required":["a"],"additionalProperties":false}`,
		// Arrays with no maximum, nested. The element subtree is shared between
		// the first element and every later one, so the shape that sharing
		// changes needs the generative property too.
		`{"type":"array","minItems":2,"items":{"type":"array","items":{"type":"integer"},"minItems":3}}`,
		`{"type":"object","properties":{"xs":{"type":"array","items":{"type":"string","maxLength":2}},
			"n":{"type":"integer"}},"required":["xs"],"additionalProperties":false}`,
		`{"const":{"tag":"only","n":[1,2]}}`,
	}

	v := full()
	// A fixed seed, so a failure is reproducible. The two words are unequal,
	// as is the walk count and the budget.
	rng := rand.New(rand.NewPCG(0x5eed, 0xc0ffee))
	const walks = 40

	for i, schema := range schemas {
		g := compile(t, v, schema)
		raw, err := decode([]byte(schema))
		if err != nil {
			t.Fatalf("schema %d does not parse: %v", i, err)
		}
		parsed := raw.(map[string]any)
		o := newOracle(parsed)

		seen := map[string]bool{}
		for range walks {
			doc := randomWalk(t, g, v, rng, 60)
			seen[doc] = true

			if !json.Valid([]byte(doc)) {
				t.Fatalf("schema %d: the machine accepted %q, which is not JSON", i, doc)
			}
			value, err := decode([]byte(doc))
			if err != nil {
				t.Fatalf("schema %d: the machine accepted %q, which does not decode: %v", i, doc, err)
			}
			if err := o.check(parsed, value, ""); err != nil {
				t.Fatalf("schema %d: the machine accepted %q, which the schema rejects: %v", i, doc, err)
			}
		}
		// A generator that produced one document over and over would pass the
		// property and prove nothing. Half the walks distinct is far above
		// what a degenerate generator reaches and far below what these schemas
		// actually produce.
		want := walks / 2
		if i == len(schemas)-1 {
			// A const schema has exactly one member, so the only variation
			// left is the whitespace this language allows before it -- worth
			// asserting rather than counting.
			want = 1
			for doc := range seen {
				if strings.TrimLeft(doc, " \t\n\r") != `{"tag":"only","n":[1,2]}` {
					t.Errorf("the const schema admitted %q", doc)
				}
			}
		}
		if len(seen) < want {
			t.Errorf("schema %d: %d distinct documents over %d walks", i, len(seen), walks)
		}
	}
}

// randomWalk drives a fresh state to acceptance, choosing among the admissible
// tokens, and returns the text it typed.
//
// Past the budget it stops choosing freely and takes a token on a shortest path
// to an accepting state. Without that, an unbounded repetition -- a string
// character, an array with no maximum -- would let a fair coin wander for an
// arbitrarily long time, and the test would be a timeout waiting to happen
// rather than a property.
func randomWalk(t *testing.T, g *Grammar, v *vocab, rng *rand.Rand, budget int) string {
	t.Helper()
	st := g.Start()
	var out []byte
	for step := 0; ; step++ {
		var cands []int
		for _, id := range st.Allowed() {
			if id != stopID {
				cands = append(cands, id)
			}
		}
		// Stopping is forced where the document is complete and nothing may
		// follow it, which is the ordinary end of a closed document: there is
		// no trailing whitespace in this language.
		if st.Accepting() && (step >= budget || len(cands) == 0 || rng.IntN(3) == 0) {
			if !admits(st, stopID) {
				t.Fatalf("accepting after %q and the stop token is not admissible", out)
			}
			if err := st.Advance(stopID); err != nil {
				t.Fatalf("advance the stop token: %v", err)
			}
			return string(out)
		}
		var pick int
		if step >= budget {
			pick = towardAccept(g, st.cur)
			if pick < 0 {
				t.Fatalf("no path to acceptance from %q", out)
			}
		} else if len(cands) == 0 {
			t.Fatalf("nothing admissible after %q", out)
		} else {
			pick = cands[rng.IntN(len(cands))]
		}
		out = append(out, v.Bytes(pick)...)
		if err := st.Advance(pick); err != nil {
			t.Fatalf("advance %q: %v", v.Bytes(pick), err)
		}
		if len(out) > 4096 {
			t.Fatalf("the walk grew past 4096 bytes: %q", out)
		}
	}
}

// towardAccept returns a token that starts a shortest token path from d to an
// accepting state, or -1 when there is none.
func towardAccept(g *Grammar, d *dstate) int {
	type node struct {
		d     *dstate
		first int
	}
	seen := map[string]bool{d.key: true}
	var queue []node
	push := func(from *dstate, first int) int {
		g.tokens(from)
		for i, id := range from.allowed {
			to := from.dest[i]
			if to == nil || seen[to.key] {
				continue
			}
			step := first
			if step < 0 {
				step = id
			}
			if to.acc {
				return step
			}
			seen[to.key] = true
			queue = append(queue, node{to, step})
		}
		return -1
	}
	if got := push(d, -1); got >= 0 {
		return got
	}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		if got := push(n.d, n.first); got >= 0 {
			return got
		}
	}
	return -1
}

// decode parses JSON with numbers left as their literal spelling.
//
// UseNumber rather than the default, and the reason is a real property of this
// grammar rather than a convenience. RFC 8259 puts no bound on a number's
// magnitude and neither does JSON Schema -- a bound there is spelled "minimum"
// or "maximum", which this package refuses to compile -- so the automaton will
// happily walk -4.5e654521, which is JSON that json.Valid accepts and that
// float64 cannot hold. Decoding into float64 here would report the machine as
// broken for emitting something the specification allows. What a caller
// unmarshalling into a float64 field should know is in the package
// documentation instead.
func decode(b []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	return v, nil
}

// oracle validates a decoded value against a decoded schema.
//
// Written here rather than reused from the compiler on purpose: a property
// checked by the machinery under test is not checked. It covers exactly the
// keywords Compile accepts, which is what makes it small enough to read.
type oracle struct{ defs map[string]any }

func newOracle(root map[string]any) *oracle {
	o := &oracle{defs: map[string]any{}}
	for _, key := range []string{"$defs", "definitions"} {
		if d, ok := root[key].(map[string]any); ok {
			for k, v := range d {
				o.defs["#/"+key+"/"+k] = v
			}
		}
	}
	return o
}

func (o *oracle) check(schema map[string]any, v any, path string) error {
	at := func(format string, a ...any) error {
		return fmt.Errorf("%s: %s", or(path, "the document"), fmt.Sprintf(format, a...))
	}
	if ref, ok := schema["$ref"].(string); ok {
		target, ok := o.defs[ref].(map[string]any)
		if !ok {
			return at("the oracle cannot resolve %s", ref)
		}
		return o.check(target, v, path)
	}
	if branches, ok := schema["anyOf"].([]any); ok {
		for _, b := range branches {
			if o.check(b.(map[string]any), v, path) == nil {
				return nil
			}
		}
		return at("matches no branch of anyOf")
	}
	if want, ok := schema["const"]; ok {
		if !reflect.DeepEqual(want, v) {
			return at("%#v is not the const %#v", v, want)
		}
		return nil
	}
	if members, ok := schema["enum"].([]any); ok {
		for _, m := range members {
			if reflect.DeepEqual(m, v) {
				return nil
			}
		}
		return at("%#v is not in the enum", v)
	}

	switch schema["type"].(string) {
	case "string":
		s, ok := v.(string)
		if !ok {
			return at("%#v is not a string", v)
		}
		n := utf8.RuneCountInString(s)
		if lo, ok := num(schema["minLength"]); ok && n < lo {
			return at("%q is %d characters, below minLength %d", s, n, lo)
		}
		if hi, ok := num(schema["maxLength"]); ok && n > hi {
			return at("%q is %d characters, above maxLength %d", s, n, hi)
		}
	case "number":
		if _, ok := v.(json.Number); !ok {
			return at("%#v is not a number", v)
		}
	case "integer":
		n, ok := v.(json.Number)
		if !ok || strings.ContainsAny(string(n), ".eE") {
			return at("%#v is not an integer", v)
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return at("%#v is not a boolean", v)
		}
	case "null":
		if v != nil {
			return at("%#v is not null", v)
		}
	case "object":
		m, ok := v.(map[string]any)
		if !ok {
			return at("%#v is not an object", v)
		}
		props, _ := schema["properties"].(map[string]any)
		if req, ok := schema["required"].([]any); ok {
			for _, r := range req {
				if _, ok := m[r.(string)]; !ok {
					return at("the required property %q is missing", r)
				}
			}
		}
		for k, sub := range m {
			ps, ok := props[k]
			if !ok {
				if closed, _ := schema["additionalProperties"].(bool); !closed {
					return at("%q is not a declared property", k)
				}
				continue
			}
			if err := o.check(ps.(map[string]any), sub, path+"/"+k); err != nil {
				return err
			}
		}
	case "array":
		a, ok := v.([]any)
		if !ok {
			return at("%#v is not an array", v)
		}
		if lo, ok := num(schema["minItems"]); ok && len(a) < lo {
			return at("%d items, below minItems %d", len(a), lo)
		}
		if hi, ok := num(schema["maxItems"]); ok && len(a) > hi {
			return at("%d items, above maxItems %d", len(a), hi)
		}
		items, ok := schema["items"].(map[string]any)
		if !ok {
			return at("the oracle found an array with no items schema")
		}
		for i, e := range a {
			if err := o.check(items, e, fmt.Sprintf("%s/%d", path, i)); err != nil {
				return err
			}
		}
	default:
		return at("the oracle does not know the type %#v", schema["type"])
	}
	return nil
}

func num(v any) (int, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	i, err := n.Int64()
	return int(i), err == nil
}

func or(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}
