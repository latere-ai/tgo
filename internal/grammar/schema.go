// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package grammar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// maxRepeat bounds a length or item count that becomes nested states.
//
// A bounded repetition is regular by counting, and counting means one copy of
// the sub-automaton per unit. "maxLength": 1000000 is therefore a request for a
// million copies of the JSON character automaton, and building it would be
// answered with an out-of-memory rather than with a mask. Refused with the
// bound named, so the caller reads a number rather than a crash.
const maxRepeat = 1024

// maxStates bounds the whole automaton a schema is allowed to build.
//
// maxRepeat bounds a count that a caller wrote down. It does not bound what
// the schema builds when nothing in the text is large: c.seen is a cycle
// detector and not a memo, deliberately, so two sibling properties that both
// $ref one $defs entry compile that entry twice, and a chain of them doubles
// per level -- 315,333 states at depth 12, measured, from a schema of a few
// hundred bytes. A count bound cannot see that, because there is no count.
//
// The schema is caller input arriving over HTTP, so the bound has to be on the
// thing that actually grows. It is checked in value(), the single point the
// compilation recurses through, so every construct that can nest -- $ref,
// arrays, anyOf, and any that nobody has written yet -- is bounded by it
// without knowing it exists.
const maxStates = 1 << 16

// annotations are the keywords that describe a schema without constraining the
// value it admits. Ignoring one cannot narrow or widen the language, which is
// the only reason 015-D4 permits ignoring anything at all.
var annotations = map[string]bool{
	"$schema": true, "$id": true, "$comment": true, "$anchor": true,
	"$defs": true, "definitions": true,
	"title": true, "description": true, "default": true, "examples": true,
	"deprecated": true, "readOnly": true, "writeOnly": true,
}

// unsupported maps a keyword this package knows about, and refuses, to the
// reason. A keyword in neither this map nor an allowlist is refused as unknown.
//
// Each reason is the actual obstruction rather than "not implemented", because
// the caller's next move differs: a numeric bound has to move out of the schema
// and into validation, while a tuple could be added to this front end.
var unsupported = map[string]string{
	"pattern":               `a regular expression front end is specs/015-structured-output.md's EBNF path, which comes second (015-D3)`,
	"format":                `a format asserts something about a value that its spelling does not carry, so no automaton over the spelling can enforce it`,
	"minimum":               `a numeric bound is arithmetic on the value; the automaton counts characters and cannot compare magnitudes`,
	"maximum":               `a numeric bound is arithmetic on the value; the automaton counts characters and cannot compare magnitudes`,
	"exclusiveMinimum":      `a numeric bound is arithmetic on the value; the automaton counts characters and cannot compare magnitudes`,
	"exclusiveMaximum":      `a numeric bound is arithmetic on the value; the automaton counts characters and cannot compare magnitudes`,
	"multipleOf":            `divisibility is arithmetic on the value; the automaton counts characters and cannot divide`,
	"oneOf":                 `exactly-one-of cannot be decided while the document is still a prefix: a prefix may still reach two branches`,
	"allOf":                 `an intersection of schemas needs a product construction over their automata, which this front end does not build`,
	"not":                   `a complement of a narrowed language is not the narrowed complement, so the refusal is safer than the answer`,
	"if":                    `a conditional is decided by the finished document, and a prefix cannot decide it`,
	"then":                  `a conditional is decided by the finished document, and a prefix cannot decide it`,
	"else":                  `a conditional is decided by the finished document, and a prefix cannot decide it`,
	"dependentSchemas":      `a dependency is decided by the finished document, and a prefix cannot decide it`,
	"dependentRequired":     `a dependency is decided by the finished document, and a prefix cannot decide it`,
	"patternProperties":     `a key set described by a regular expression is the EBNF path, which comes second (015-D3)`,
	"propertyNames":         `a key set described by a schema is the EBNF path, which comes second (015-D3)`,
	"uniqueItems":           `uniqueness is a property of the whole array, and the automaton has no memory of what it already emitted`,
	"contains":              `containment is a property of the whole array, and the automaton has no memory of what it already emitted`,
	"minContains":           `containment is a property of the whole array, and the automaton has no memory of what it already emitted`,
	"maxContains":           `containment is a property of the whole array, and the automaton has no memory of what it already emitted`,
	"unevaluatedProperties": `what counts as unevaluated depends on which subschemas applied, which this front end does not track`,
	"unevaluatedItems":      `what counts as unevaluated depends on which subschemas applied, which this front end does not track`,
	"prefixItems":           `a tuple schema is expressible in this machinery but is not built; it is refused rather than approximated`,
	"minProperties":         `a property count is a bound on which properties appear, and this front end fixes the order rather than the count`,
	"maxProperties":         `a property count is a bound on which properties appear, and this front end fixes the order rather than the count`,
	"contentEncoding":       `a content encoding describes what a string means, not which strings are admissible`,
	"contentMediaType":      `a content media type describes what a string means, not which strings are admissible`,
}

// consumed is every keyword some allowlist in value uses. One that appears
// where its allowlist does not is a keyword aimed at the wrong subschema, which
// is worth saying differently from a keyword nobody has heard of.
var consumed = map[string]bool{
	"type": true, "enum": true, "const": true, "anyOf": true, "$ref": true,
	"minLength": true, "maxLength": true,
	"properties": true, "required": true, "additionalProperties": true,
	"items": true, "minItems": true, "maxItems": true,
}

// compiler turns a schema into one NFA fragment. Errors are sticky: the first
// refusal wins and every later builder returns a well-formed empty fragment, so
// the recursion does not need an error branch at every call site.
type compiler struct {
	n    *nfa
	defs map[string]json.RawMessage
	seen map[string]bool
	err  error
}

func (c *compiler) fail(path, construct, why string) frag {
	if c.err == nil {
		c.err = &UnsupportedError{Path: path, Construct: construct, Why: why}
	}
	return c.n.empty()
}

func (c *compiler) bad(path, format string, a ...any) frag {
	if c.err == nil {
		c.err = &SchemaError{Path: path, Reason: fmt.Sprintf(format, a...)}
	}
	return c.n.empty()
}

func (c *compiler) compile(schema []byte) (frag, error) {
	_, m, err := fields(schema, "")
	if err != nil {
		return frag{}, err
	}
	c.defs = map[string]json.RawMessage{}
	for _, key := range []string{"$defs", "definitions"} {
		if raw, ok := m[key]; ok {
			_, d, err := fields(raw, "/"+key)
			if err != nil {
				return frag{}, err
			}
			for k, v := range d {
				c.defs["#/"+key+"/"+k] = v
			}
		}
	}
	f := c.value(schema, "")
	if c.err != nil {
		return frag{}, c.err
	}
	return f, nil
}

// fields reads a JSON object preserving the order its keys appear in the text.
//
// The order matters: it is the order this package requires an object's
// properties to be emitted in (see object below), so it has to come from the
// document rather than from a Go map's iteration.
func fields(raw json.RawMessage, path string) ([]string, map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, &SchemaError{Path: path, Reason: fmt.Sprintf("not a JSON object: %v", err)}
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, nil, &SchemaError{Path: path, Reason: fmt.Sprintf("not a JSON object: found %v", tok)}
	}
	var order []string
	m := map[string]json.RawMessage{}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, nil, &SchemaError{Path: path, Reason: fmt.Sprintf("malformed object: %v", err)}
		}
		k := kt.(string)
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, nil, &SchemaError{Path: path, Reason: fmt.Sprintf("malformed value for %q: %v", k, err)}
		}
		if _, dup := m[k]; dup {
			return nil, nil, &SchemaError{Path: path, Reason: fmt.Sprintf("duplicate key %q", k)}
		}
		order = append(order, k)
		m[k] = v
	}
	if _, err := dec.Token(); err != nil {
		return nil, nil, &SchemaError{Path: path, Reason: fmt.Sprintf("malformed object: %v", err)}
	}
	if dec.More() {
		return nil, nil, &SchemaError{Path: path, Reason: "trailing content after the object"}
	}
	return order, m, nil
}

// value compiles one subschema.
func (c *compiler) value(raw json.RawMessage, path string) frag {
	if c.err != nil {
		return c.n.empty()
	}
	if c.n.size() > maxStates {
		return c.fail(path, "the automaton this schema builds",
			fmt.Sprintf(`a schema of any size may build a state count exponential in its nesting depth, and this one passed the %d states this package builds; a $defs entry referenced from more than one place is compiled once per reference, so flattening the repeated ones is what shrinks it`, maxStates))
	}
	switch strings.TrimSpace(string(raw)) {
	case "true", "false":
		return c.fail(path, "a boolean schema",
			`it admits any JSON value, which may nest without bound and is therefore not a regular language`)
	}
	_, m, err := fields(raw, path)
	if err != nil {
		c.err = err
		return c.n.empty()
	}

	// A keyword this package knows about and refuses is reported before the
	// shape is examined. Otherwise {"oneOf": [...]} -- which has no "type" --
	// would come back as "a schema with no type", which is true and useless:
	// the caller wrote oneOf and needs to be told about oneOf.
	for _, k := range sortedKeys(m) {
		if why, known := unsupported[k]; known {
			return c.fail(path, strconv.Quote(k), why)
		}
	}

	// Pick the mode first, then check that every keyword present belongs to
	// it. Checking before building is what makes the refusal name the
	// caller's keyword rather than whatever the builder tripped over first.
	var allow []string
	switch {
	case m["$ref"] != nil:
		allow = []string{"$ref"}
	case m["anyOf"] != nil:
		allow = []string{"anyOf"}
	case m["const"] != nil:
		allow = []string{"const", "type"}
	case m["enum"] != nil:
		allow = []string{"enum", "type"}
	case m["type"] != nil:
		var t string
		if err := json.Unmarshal(m["type"], &t); err != nil {
			return c.fail(path, `"type" as a list`,
				`a union of types is spelled anyOf here, so that each branch carries its own keywords`)
		}
		switch t {
		case "string":
			allow = []string{"type", "minLength", "maxLength"}
		case "number", "integer", "boolean", "null":
			allow = []string{"type"}
		case "object":
			allow = []string{"type", "properties", "required", "additionalProperties"}
		case "array":
			allow = []string{"type", "items", "minItems", "maxItems"}
		default:
			return c.bad(path, "unknown type %q", t)
		}
	default:
		return c.fail(path, `a schema with no "type"`,
			`it admits any JSON value, which may nest without bound and is therefore not a regular language`)
	}
	if f, bad := c.checkKeywords(m, allow, path); bad {
		return f
	}

	switch allow[0] {
	case "$ref":
		return c.ref(m["$ref"], path)
	case "anyOf":
		return c.anyOf(m["anyOf"], path)
	case "const":
		return c.literals(path, "const", []json.RawMessage{m["const"]}, m["type"])
	case "enum":
		var vals []json.RawMessage
		if err := json.Unmarshal(m["enum"], &vals); err != nil {
			return c.bad(path, `"enum" is not a list: %v`, err)
		}
		if len(vals) == 0 {
			return c.bad(path, `"enum" is empty, so no value is admissible`)
		}
		return c.literals(path, "enum", vals, m["type"])
	}

	var t string
	_ = json.Unmarshal(m["type"], &t)
	switch t {
	case "string":
		lo, hi := c.lengths(m, path, "minLength", "maxLength")
		return c.n.cat(c.ws(), c.jsonString(lo, hi))
	case "number":
		return c.n.cat(c.ws(), c.number())
	case "integer":
		return c.n.cat(c.ws(), c.integer())
	case "boolean":
		return c.n.cat(c.ws(), c.n.alt(c.n.lit("true"), c.n.lit("false")))
	case "null":
		return c.n.cat(c.ws(), c.n.lit("null"))
	case "object":
		return c.object(m, path)
	default: // "array"; every other type was rejected above
		return c.array(m, path)
	}
}

// checkKeywords is 015-D4 as one loop: every keyword present must be one this
// compilation consumes.
func (c *compiler) checkKeywords(m map[string]json.RawMessage, allow []string, path string) (frag, bool) {
	ok := map[string]bool{}
	for _, k := range allow {
		ok[k] = true
	}
	// Sorted so the refusal is the same one on every run: a map's iteration
	// order would make which keyword gets named depend on the process.
	for _, k := range sortedKeys(m) {
		if ok[k] || annotations[k] {
			continue
		}
		if consumed[k] {
			return c.fail(path, strconv.Quote(k),
				`it does not apply to this subschema, and 015-D4 refuses rather than ignores: a keyword dropped here enforces a schema the caller did not write`), true
		}
		return c.fail(path, strconv.Quote(k),
			`it is not a keyword this front end consumes, and 015-D4 refuses rather than ignores: a keyword dropped here enforces a schema the caller did not write`), true
	}
	return frag{}, false
}

func sortedKeys(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return sortedUnique(out)
}

// ref inlines a non-recursive reference.
//
// Inlined rather than refused because every schema generator worth naming --
// Pydantic, zod -- spells a nested model as $defs plus $ref, so a blanket
// refusal would reject most of what response_format actually sends. The visited
// set that makes inlining safe is also the cycle detector: a schema that refers
// to itself has no bound on nesting and is not regular.
func (c *compiler) ref(raw json.RawMessage, path string) frag {
	var ptr string
	if err := json.Unmarshal(raw, &ptr); err != nil {
		return c.bad(path, `"$ref" is not a string: %v`, err)
	}
	if c.seen[ptr] {
		return c.fail(path, "a recursive "+strconv.Quote("$ref"),
			`a schema that contains itself has no bound on nesting, so its language is not regular and this automaton has no stack`)
	}
	target, ok := c.defs[ptr]
	if !ok {
		return c.bad(path, `"$ref" %q resolves to nothing; only #/$defs/NAME and #/definitions/NAME at the root are resolved`, ptr)
	}
	c.seen[ptr] = true
	defer delete(c.seen, ptr)
	return c.value(target, ptr[1:])
}

func (c *compiler) anyOf(raw json.RawMessage, path string) frag {
	var branches []json.RawMessage
	if err := json.Unmarshal(raw, &branches); err != nil {
		return c.bad(path, `"anyOf" is not a list: %v`, err)
	}
	if len(branches) == 0 {
		return c.bad(path, `"anyOf" is empty, so no value is admissible`)
	}
	fs := make([]frag, len(branches))
	for i, b := range branches {
		fs[i] = c.value(b, fmt.Sprintf("%s/anyOf/%d", path, i))
	}
	return c.n.alt(fs...)
}

// literals compiles const and enum to an alternation of exact spellings.
//
// json.Compact rather than a decode-and-re-encode, so the admitted spelling is
// the caller's own: re-encoding would turn 1.0 into 1 and would rewrite the
// escapes inside a string, and the model would then be masked away from the
// literal the schema actually contains.
func (c *compiler) literals(path, kw string, vals []json.RawMessage, typ json.RawMessage) frag {
	var want string
	if typ != nil {
		if err := json.Unmarshal(typ, &want); err != nil {
			return c.fail(path, `"type" as a list`,
				`a union of types is spelled anyOf here, so that each branch carries its own keywords`)
		}
	}
	fs := make([]frag, 0, len(vals))
	for i, v := range vals {
		var buf bytes.Buffer
		if err := json.Compact(&buf, v); err != nil {
			return c.bad(path, `%s[%d] is not valid JSON: %v`, kw, i, err)
		}
		if want != "" {
			if got := jsonKind(buf.Bytes()); !kindMatches(got, want, buf.Bytes()) {
				return c.bad(path, `%s[%d] is %s, which contradicts "type": %q`, kw, i, got, want)
			}
		}
		fs = append(fs, c.n.lit(buf.String()))
	}
	return c.n.cat(c.ws(), c.n.alt(fs...))
}

func jsonKind(b []byte) string {
	switch b[0] {
	case '{':
		return "object"
	case '[':
		return "array"
	case '"':
		return "string"
	case 't', 'f':
		return "boolean"
	case 'n':
		return "null"
	default:
		return "number"
	}
}

func kindMatches(got, want string, b []byte) bool {
	if want == "integer" {
		if got != "number" {
			return false
		}
		f, err := strconv.ParseFloat(string(b), 64)
		return err == nil && f == math.Trunc(f)
	}
	return got == want
}

// lengths reads a pair of count bounds, defaulting to zero and unbounded.
func (c *compiler) lengths(m map[string]json.RawMessage, path, minKW, maxKW string) (int, int) {
	lo, hi := 0, -1
	if raw, ok := m[minKW]; ok {
		if err := json.Unmarshal(raw, &lo); err != nil || lo < 0 {
			c.bad(path, `%q must be a non-negative integer`, minKW)
			return 0, -1
		}
	}
	if raw, ok := m[maxKW]; ok {
		if err := json.Unmarshal(raw, &hi); err != nil || hi < 0 {
			c.bad(path, `%q must be a non-negative integer`, maxKW)
			return 0, -1
		}
	}
	if hi >= 0 && hi < lo {
		c.bad(path, `%q %d is below %q %d`, maxKW, hi, minKW, lo)
		return 0, -1
	}
	for _, over := range []struct {
		kw string
		n  int
	}{{minKW, lo}, {maxKW, hi}} {
		if over.n > maxRepeat {
			c.fail(path, strconv.Quote(over.kw),
				fmt.Sprintf(`a count bound becomes one copy of the sub-automaton per unit, and %d is above the %d this package builds`, over.n, maxRepeat))
			return 0, -1
		}
	}
	return lo, hi
}

// object compiles an object with a fixed property order.
//
// Two narrowings, both sound and both deliberate. Properties must appear in the
// order the schema declares them, because admitting every permutation of n
// properties needs a state per subset already emitted -- 2^n of them -- and the
// exponential is in the compile, not in the request. And an object is closed:
// no property outside the schema is ever emitted. Each narrowing shrinks the
// admitted language, so a document this machine accepts still validates against
// the caller's schema; what is lost is documents the schema would also have
// allowed. An explicit "additionalProperties": true is refused rather than
// narrowed, because that one is the caller stating something this cannot honour.
func (c *compiler) object(m map[string]json.RawMessage, path string) frag {
	if raw, ok := m["additionalProperties"]; ok {
		if strings.TrimSpace(string(raw)) != "false" {
			return c.fail(path, `"additionalProperties" other than false`,
				`an open object admits properties with no schema, whose values may nest without bound; only a closed object is regular`)
		}
	}

	var order []string
	props := map[string]json.RawMessage{}
	if raw, ok := m["properties"]; ok {
		var err error
		order, props, err = fields(raw, path+"/properties")
		if err != nil {
			c.err = err
			return c.n.empty()
		}
	}

	need := map[string]bool{}
	if raw, ok := m["required"]; ok {
		var req []string
		if err := json.Unmarshal(raw, &req); err != nil {
			return c.bad(path, `"required" is not a list of strings: %v`, err)
		}
		for _, name := range req {
			if _, ok := props[name]; !ok {
				return c.bad(path, `"required" names %q, which "properties" does not declare, so it would have no schema`, name)
			}
			need[name] = true
		}
	}

	n := len(order)
	enter := make([]int, n+1)
	after := make([]int, n+1)
	for i := range enter {
		enter[i] = c.n.state()
		after[i] = c.n.state()
	}
	for i, name := range order {
		// One copy of the property, entered two ways: directly when it is the
		// first thing in the object, and through a comma when it is not.
		//
		// Compiling it once per way instead would double the automaton at
		// every level of nesting -- size(d) = 2*size(d-1) -- so a schema
		// twenty objects deep would be hundreds of millions of states built
		// from a caller-supplied request body. The two ways differ by a
		// leading comma and nothing else, so they can share the subtree.
		body := c.property(name, props[name], path+"/properties/"+name)
		comma := c.tok(",")
		c.n.link(enter[i], body.in)
		c.n.link(after[i], comma.in)
		c.n.link(comma.out, body.in)
		c.n.link(body.out, after[i+1])

		if !need[name] {
			c.n.link(enter[i], enter[i+1])
			c.n.link(after[i], after[i+1])
		}
	}
	exit := c.n.state()
	c.n.link(enter[n], exit)
	c.n.link(after[n], exit)
	return c.n.cat(c.tok("{"), frag{enter[0], exit}, c.tok("}"))
}

// property is `"name":value`, without the separator: object above supplies
// that, so this subtree is built once however many ways it can be reached.
func (c *compiler) property(name string, raw json.RawMessage, path string) frag {
	key, err := encodeString(name)
	if err != nil {
		return c.bad(path, "property name is not encodable as JSON: %v", err)
	}
	return c.n.cat(c.tok(key), c.tok(":"), c.value(raw, path))
}

func (c *compiler) array(m map[string]json.RawMessage, path string) frag {
	items, ok := m["items"]
	if !ok {
		return c.fail(path, `an array with no "items"`,
			`its elements would have no schema, so each could be any JSON value and could nest without bound`)
	}
	lo, hi := c.lengths(m, path, "minItems", "maxItems")
	if c.err != nil {
		return c.n.empty()
	}
	if hi == 0 {
		return c.n.cat(c.tok("["), c.tok("]"))
	}
	item := func() frag { return c.value(items, path+"/items") }
	if hi < 0 {
		return c.n.cat(c.tok("["), c.unboundedItems(item, lo), c.tok("]"))
	}
	rest := func() frag { return c.n.cat(c.tok(","), item()) }
	restMin := 0
	if lo > 0 {
		restMin = lo - 1
	}
	body := c.n.cat(item(), c.n.rep(rest, restMin, hi-1))
	if lo == 0 {
		body = c.n.opt(body)
	}
	return c.n.cat(c.tok("["), body, c.tok("]"))
}

// unboundedItems builds item{lo,} separated by commas, with the last required
// copy carrying a back edge through the separator.
//
// The obvious spelling -- an element, then a repetition of "comma then element"
// -- calls the element builder twice, which is object's doubling one construct
// over: size(d) = 2*size(d-1), so an array of arrays fifteen deep is 1.3
// million states built from a caller-supplied request body. The loop shares one
// copy between the first element and every later one, and the growth is linear.
//
// A bounded maxItems still costs one copy per unit, because counting is what a
// bound means and the automaton has no registers; maxRepeat is what bounds it.
// The lo required copies here are that same counting, bounded the same way.
func (c *compiler) unboundedItems(item func() frag, lo int) frag {
	enter, exit := c.n.state(), c.n.state()
	last := item()
	c.n.link(enter, last.in)
	for i := 1; i < lo; i++ {
		sep := c.tok(",")
		f := item()
		c.n.link(last.out, sep.in)
		c.n.link(sep.out, f.in)
		last = f
	}
	// The back edge is what makes the final copy item+ rather than item, so the
	// chain is item{lo-1} item+ = item{lo,} for lo >= 1.
	loop := c.tok(",")
	c.n.link(last.out, loop.in)
	c.n.link(loop.out, last.in)
	c.n.link(last.out, exit)
	if lo == 0 {
		// One copy was built regardless, so the empty array needs its own way
		// past it.
		c.n.link(enter, exit)
	}
	return frag{enter, exit}
}

// encodeString spells a Go string as a JSON string literal, without Go's HTML
// escaping: a model asked for a property called "a<b" will type the angle
// bracket, and < would be a spelling nothing reaches.
func encodeString(s string) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		return "", err
	}
	return strings.TrimRight(buf.String(), "\n"), nil
}
