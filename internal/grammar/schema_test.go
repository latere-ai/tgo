// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package grammar

import (
	"errors"
	"strings"
	"testing"
)

// TestEveryRefusalNamesItsConstruct is specs/015-structured-output.md section 4
// and 015-D4. A keyword this front end does not consume must come back as a
// refusal carrying the caller's own spelling of it, because a keyword silently
// ignored produces a document that validates against a schema the caller did
// not write -- and looks like it worked.
func TestEveryRefusalNamesItsConstruct(t *testing.T) {
	// Each case wraps the keyword in a property, so the refusal has to carry a
	// path as well as a name.
	kw := map[string]string{
		"pattern":               `{"type":"string","pattern":"^a+$"}`,
		"format":                `{"type":"string","format":"email"}`,
		"minLength":             `{"type":"string","minLength":9000}`,
		"maxLength":             `{"type":"string","maxLength":9000}`,
		"minimum":               `{"type":"integer","minimum":1}`,
		"maximum":               `{"type":"integer","maximum":1}`,
		"exclusiveMinimum":      `{"type":"number","exclusiveMinimum":1}`,
		"exclusiveMaximum":      `{"type":"number","exclusiveMaximum":1}`,
		"multipleOf":            `{"type":"number","multipleOf":2}`,
		"oneOf":                 `{"oneOf":[{"type":"integer"}]}`,
		"allOf":                 `{"allOf":[{"type":"integer"}]}`,
		"not":                   `{"not":{"type":"integer"}}`,
		"if":                    `{"type":"integer","if":{"type":"integer"}}`,
		"then":                  `{"type":"integer","then":{"type":"integer"}}`,
		"else":                  `{"type":"integer","else":{"type":"integer"}}`,
		"dependentSchemas":      `{"type":"object","properties":{},"dependentSchemas":{}}`,
		"dependentRequired":     `{"type":"object","properties":{},"dependentRequired":{}}`,
		"patternProperties":     `{"type":"object","properties":{},"patternProperties":{}}`,
		"propertyNames":         `{"type":"object","properties":{},"propertyNames":{}}`,
		"uniqueItems":           `{"type":"array","items":{"type":"integer"},"uniqueItems":true}`,
		"contains":              `{"type":"array","items":{"type":"integer"},"contains":{}}`,
		"minContains":           `{"type":"array","items":{"type":"integer"},"minContains":1}`,
		"maxContains":           `{"type":"array","items":{"type":"integer"},"maxContains":1}`,
		"unevaluatedProperties": `{"type":"object","properties":{},"unevaluatedProperties":false}`,
		"unevaluatedItems":      `{"type":"array","items":{"type":"integer"},"unevaluatedItems":false}`,
		"prefixItems":           `{"type":"array","items":{"type":"integer"},"prefixItems":[]}`,
		"minProperties":         `{"type":"object","properties":{},"minProperties":1}`,
		"maxProperties":         `{"type":"object","properties":{},"maxProperties":1}`,
		"contentEncoding":       `{"type":"string","contentEncoding":"base64"}`,
		"contentMediaType":      `{"type":"string","contentMediaType":"text/plain"}`,
		"wibble":                `{"type":"integer","wibble":1}`,
	}
	v := full()
	for name, sub := range kw {
		schema := `{"type":"object","properties":{"p":` + sub + `},"additionalProperties":false}`
		_, err := Compile([]byte(schema), v, Options{})
		var ue *UnsupportedError
		if !errors.As(err, &ue) {
			t.Errorf("%s: %v, want an UnsupportedError", name, err)
			continue
		}
		if !strings.Contains(ue.Construct, name) {
			t.Errorf("%s: refused %q instead", name, ue.Construct)
		}
		if ue.Path != "/properties/p" {
			t.Errorf("%s: path %q, want /properties/p", name, ue.Path)
		}
		if ue.Why == "" || !strings.Contains(err.Error(), name) {
			t.Errorf("%s: message %q does not name the construct and its reason", name, err)
		}
	}
	// The table above must not fall behind the map it tests.
	for name := range unsupported {
		if _, ok := kw[name]; !ok {
			t.Errorf("%q is refusable and has no case here", name)
		}
	}
}

// TestRefusalsThatAreShapesRatherThanKeywords covers the constructs that have
// no keyword of their own: an unconstrained subschema, an open object, an
// untyped array, a recursive reference. Each one would need a stack, and this
// automaton has none.
func TestRefusalsThatAreShapesRatherThanKeywords(t *testing.T) {
	v := full()
	for _, tc := range []struct{ name, schema, want string }{
		{"a boolean schema", `{"type":"object","properties":{"p":true},"additionalProperties":false}`, "boolean schema"},
		{"no type", `{"type":"object","properties":{"p":{}},"additionalProperties":false}`, `no "type"`},
		{"no type at the root", `{}`, `no "type"`},
		{"an open object", `{"type":"object","properties":{},"additionalProperties":true}`, "additionalProperties"},
		{"an object open to a schema", `{"type":"object","properties":{},"additionalProperties":{"type":"integer"}}`, "additionalProperties"},
		{"an array with no items", `{"type":"array"}`, `"items"`},
		{"a type union", `{"type":["string","null"]}`, `"type" as a list`},
		{"a type union beside an enum", `{"type":["string","null"],"enum":["a"]}`, `"type" as a list`},
		{"a keyword aimed at the wrong type", `{"type":"object","properties":{},"minLength":1}`, "minLength"},
		{"a recursive ref", `{"$defs":{"N":{"type":"object","properties":{"n":{"$ref":"#/$defs/N"}},"additionalProperties":false}},"$ref":"#/$defs/N"}`, "$ref"},
	} {
		_, err := Compile([]byte(tc.schema), v, Options{})
		var ue *UnsupportedError
		if !errors.As(err, &ue) {
			t.Errorf("%s: %v, want an UnsupportedError", tc.name, err)
			continue
		}
		if !strings.Contains(ue.Construct, tc.want) {
			t.Errorf("%s: refused %q, want it to name %q", tc.name, ue.Construct, tc.want)
		}
	}
}

// TestInconsistentSchemasAreReportedAsSchemaErrors separates "this package
// cannot do that" from "that schema contradicts itself". A caller acts on the
// two differently: the first is a feature request, the second is a typo.
func TestInconsistentSchemasAreReportedAsSchemaErrors(t *testing.T) {
	v := full()
	for _, tc := range []struct{ name, schema, want string }{
		{"malformed", `{"type":`, "malformed"},
		{"not an object", `[1]`, "not a JSON object"},
		{"trailing content", `{"type":"null"} {}`, "trailing"},
		{"duplicate key", `{"type":"null","type":"null"}`, "duplicate"},
		{"unknown type", `{"type":"widget"}`, "unknown type"},
		{"required is not a list", `{"type":"object","properties":{},"required":"a"}`, `"required" is not a list`},
		{"required names nothing", `{"type":"object","properties":{"a":{"type":"null"}},"required":["b"]}`, `does not declare`},
		{"empty enum", `{"enum":[]}`, "is empty"},
		{"enum is not a list", `{"enum":1}`, "is not a list"},
		{"enum contradicts type", `{"type":"string","enum":["a",1]}`, `contradicts "type"`},
		{"enum integer contradicts", `{"type":"integer","enum":[1.5]}`, `contradicts "type"`},
		{"const contradicts type", `{"type":"boolean","const":"yes"}`, `contradicts "type"`},
		{"empty anyOf", `{"anyOf":[]}`, "is empty"},
		{"anyOf is not a list", `{"anyOf":{}}`, "is not a list"},
		{"ref is not a string", `{"$ref":3}`, "is not a string"},
		{"ref resolves to nothing", `{"$ref":"#/$defs/Gone"}`, "resolves to nothing"},
		{"negative minLength", `{"type":"string","minLength":-1}`, "non-negative"},
		{"minLength is not a number", `{"type":"string","minLength":"three"}`, "non-negative"},
		{"maxItems below minItems", `{"type":"array","items":{"type":"null"},"minItems":3,"maxItems":2}`, "is below"},
		{"properties is not an object", `{"type":"object","properties":3}`, "not a JSON object"},
		{"defs is not an object", `{"$defs":3,"type":"null"}`, "not a JSON object"},
	} {
		_, err := Compile([]byte(tc.schema), v, Options{})
		var se *SchemaError
		if !errors.As(err, &se) {
			t.Errorf("%s: %v, want a SchemaError", tc.name, err)
			continue
		}
		if !strings.Contains(se.Reason, tc.want) {
			t.Errorf("%s: reason %q, want it to mention %q", tc.name, se.Reason, tc.want)
		}
	}
}

func TestAnnotationsAreIgnoredAndNothingElseIs(t *testing.T) {
	v := full()
	schema := `{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"x","$comment":"c",
	  "title":"Person","description":"a person","default":{},"examples":[{}],
	  "deprecated":false,"readOnly":false,"writeOnly":false,
	  "type":"object","properties":{"a":{"type":"integer","title":"A"}},
	  "required":["a"],"additionalProperties":false}`
	g := compile(t, v, schema)
	st := g.Start()
	typeText(t, v, st, `{"a":1}`)
	if !st.Accepting() {
		t.Error("a schema carrying only annotations beside its constraints was not compiled")
	}
}

// TestTheRefusalReadsAsASentence guards the message rather than the type. A
// refusal a caller cannot act on is a refusal that will be worked around.
func TestTheRefusalReadsAsASentence(t *testing.T) {
	v := full()
	_, err := Compile([]byte(`{"type":"integer","minimum":1}`), v, Options{})
	const want = `grammar: the root schema: "minimum" is not supported: a numeric bound is arithmetic on the value; the automaton counts characters and cannot compare magnitudes`
	if err == nil || err.Error() != want {
		t.Errorf("error =\n  %v\nwant\n  %s", err, want)
	}
	var se *SchemaError
	_, err = Compile([]byte(`{"type":"widget"}`), v, Options{})
	if !errors.As(err, &se) || se.Error() != `grammar: the root schema: unknown type "widget"` {
		t.Errorf("error = %v", err)
	}
}

func TestBoundsAtTheLimitAreBuilt(t *testing.T) {
	v := full()
	// maxRepeat itself compiles; one above it is the refusal above.
	if _, err := Compile([]byte(`{"type":"string","maxLength":1024}`), v, Options{}); err != nil {
		t.Errorf("maxLength at the bound: %v", err)
	}
	g := compile(t, v, `{"type":"array","items":{"type":"integer"},"maxItems":0}`)
	st := g.Start()
	typeText(t, v, st, `[]`)
	if !st.Accepting() {
		t.Error("an array bounded to zero items does not accept the empty array")
	}
	if at := rejects(v, g.Start(), `[1]`); at < 0 {
		t.Error("an array bounded to zero items accepted an element")
	}
}

// TestEnumMembersAreCheckedAgainstTheDeclaredType covers every JSON kind an
// enum member can have. A member that contradicts the type can never be
// admitted, so compiling it would silently drop a value the caller listed.
func TestEnumMembersAreCheckedAgainstTheDeclaredType(t *testing.T) {
	v := full()
	for _, tc := range []struct {
		typ    string
		member string
		ok     bool
	}{
		{"string", `"a"`, true},
		{"string", `{"a":1}`, false},
		{"string", `[1]`, false},
		{"string", `true`, false},
		{"string", `null`, false},
		{"string", `1`, false},
		{"number", `1.5`, true},
		{"integer", `2`, true},
		{"integer", `2.0`, true},
		{"integer", `2.5`, false},
		{"integer", `"2"`, false},
		{"boolean", `false`, true},
		{"null", `null`, true},
		{"object", `{"a":1}`, true},
		{"object", `"a"`, false},
		{"array", `[1,2]`, true},
		{"array", `{"a":1}`, false},
	} {
		schema := `{"type":"` + tc.typ + `","enum":[` + tc.member + `]}`
		_, err := Compile([]byte(schema), v, Options{})
		if tc.ok != (err == nil) {
			t.Errorf("%s: %v", schema, err)
		}
	}
}

func TestAStringBoundedToZeroCharacters(t *testing.T) {
	v := full()
	g := compile(t, v, `{"type":"string","maxLength":0}`)
	st := g.Start()
	typeText(t, v, st, `""`)
	if !st.Accepting() {
		t.Error(`"" is not accepted by a string bounded to zero characters`)
	}
	if at := rejects(v, g.Start(), `"a"`); at < 0 {
		t.Error("a character was admitted into a string bounded to zero")
	}
}
