// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package chat

import (
	"strings"
	"testing"
)

// String is the golden and human form: parts in order, nothing between them.
func TestPromptString(t *testing.T) {
	p := Prompt{Parts: []Part{
		{Control: imStart},
		{Text: "user\nhi"},
		{Control: imEnd},
		{Text: "\n"},
	}}
	const want = "<|im_start|>user\nhi<|im_end|>\n"
	if got := p.String(); got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
	if got := (Prompt{}).String(); got != "" {
		t.Errorf("empty Prompt.String() = %q", got)
	}
}

// A part is a span or a control token, never both, in every prompt this package
// produces. A part carrying both would have no defined encoding: one half must
// be encoded with specials off and the other resolved by id.
func TestPartsAreSpansOrControls(t *testing.T) {
	msgs := []Message{
		system("be terse"),
		user("hi"),
		assistant("thinking", "there", weather("Berlin")),
		toolResult("call_Berlin", "18C"),
		user("thanks"),
	}
	for _, opts := range []Options{
		{Thinking: true, AddGenerationHint: true},
		{AddGenerationHint: true},
		{Thinking: true, Tools: []ToolSpec{{Name: "get_weather", Description: "d"}}, AddGenerationHint: true},
	} {
		p := render(t, msgs, opts)
		for i, part := range p.Parts {
			if part.Text != "" && part.Control != "" {
				t.Errorf("part %d carries both: %+v", i, part)
			}
			if part.Text == "" && part.Control == "" {
				t.Errorf("part %d is empty", i)
			}
		}
	}
}

// Adjacent spans merge, so the part count follows the conversation's structure
// and not the content inside it. Two text blocks are one span; a control token
// always starts a new part.
func TestBuilderMergesAdjacentSpans(t *testing.T) {
	var b builder
	b.text("a")
	b.text("")
	b.text("b")
	b.control(imEnd)
	b.text("c")
	if len(b.parts) != 3 {
		t.Fatalf("parts = %+v, want 3", b.parts)
	}
	if b.parts[0].Text != "ab" {
		t.Errorf("parts[0] = %+v, want text %q", b.parts[0], "ab")
	}
}

func TestChecksum(t *testing.T) {
	// The empty string's SHA-256, so the hex encoding itself is pinned and not
	// only compared against another call of the same function.
	const emptySHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if got := Checksum(""); got != emptySHA256 {
		t.Errorf("Checksum(\"\") = %s, want %s", got, emptySHA256)
	}
	if Checksum("a") == Checksum("b") {
		t.Error("different templates hashed alike")
	}
	if got := Checksum("a"); got != Checksum("a") {
		t.Errorf("Checksum is not deterministic: %s", got)
	}
	if n := len(Checksum("a")); n != 64 {
		t.Errorf("checksum length = %d, want 64 hex digits", n)
	}
}

func TestJSONString(t *testing.T) {
	// The reference renders with ensure_ascii=False and no HTML escaping, so
	// angle brackets, ampersands and non-ASCII stay as they are.
	for in, want := range map[string]string{
		`a<b&c`:   `"a<b&c"`,
		`say "x"`: `"say \"x\""`,
		"tab\t":   `"tab\t"`,
		"грач":    `"грач"`,
	} {
		if got := jsonString(in); got != want {
			t.Errorf("jsonString(%q) = %s, want %s", in, got, want)
		}
	}
	if strings.Contains(jsonString("<"), "u003c") {
		t.Error("jsonString HTML-escaped an angle bracket")
	}
}

// The renderer satisfies the interface the model registry asks for.
var _ Renderer = Qwen3()
