// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package grammar

import (
	"slices"
	"strings"
	"testing"
)

// The synthetic vocabulary.
//
// A real tokenizer is not needed and would hide things: the interesting cases
// are tokens that straddle a grammar boundary, and a synthetic vocabulary can
// contain exactly those and nothing else.
//
// Three properties of it are deliberate.
//
// Token ids start past every byte value the grammar can require, so an id is
// never equal to the byte it stands for. Allowed returns []int and Bytes takes
// an int; a vocabulary where the token for "A" had id 65 would let a confusion
// between the two coincide instead of failing.
//
// Every single character the grammar can require is present as a one-character
// token. Byte-liveness is not token-liveness: a state that requires "}" next is
// alive at the byte level and dead at the token level if the vocabulary only
// has "}}". A real 152k vocabulary contains every byte, so the singles keep the
// synthetic one from tripping over a hazard the real one does not have -- and
// the deliberately holed vocabulary in TestMaskRefusesWhenNothingContinues is
// where that hazard is tested on purpose.
//
// The stop id sits inside the offset block, where Bytes returns nil. That is
// what a control token looks like through the Vocab contract.
const (
	idBase = 137 // no ASCII byte reaches here
	stopID = 11  // a hole in the offset block: no text
)

// crossers are the multi-character tokens. Each one spans a boundary the
// grammar cares about: a key terminator and a colon, a value terminator and a
// separator, a colon and an opening brace.
var crossers = []string{
	`":`, `":"`, `,"`, `"}`, `"]`, `{"`, `["`, `":{"`, `":[`, `"},"`, `"],`,
	`": `, `, "`, `":true`, `":null`, `":123`, `true`, `false`, `null`,
	`}}`, `]}`, `{}`, `[]`, `name`, `Ada`, `123`, `-4.5e6`, "é",
}

type vocab struct {
	Pieces
	byText map[string]int
}

func newVocab(extra ...string) *vocab {
	v := &vocab{Pieces: make(Pieces, idBase), byText: map[string]int{}}
	for b := byte(0x20); b <= 0x7e; b++ {
		v.add(string([]byte{b}))
	}
	for _, s := range []string{"\t", "\n", "\r"} {
		v.add(s)
	}
	for _, s := range extra {
		v.add(s)
	}
	return v
}

// full is the vocabulary nearly every test uses: the printable singles, the
// boundary crossers, and one multi-byte rune cut in half so that a token
// carrying a fragment of UTF-8 is in play.
func full() *vocab {
	v := newVocab(crossers...)
	v.add(string([]byte{0xc3}))
	v.add(string([]byte{0xa9}))
	return v
}

func (v *vocab) add(s string) int {
	if id, ok := v.byText[s]; ok {
		return id
	}
	id := len(v.Pieces)
	v.Pieces = append(v.Pieces, []byte(s))
	v.byText[s] = id
	return id
}

func (v *vocab) id(t *testing.T, s string) int {
	t.Helper()
	id, ok := v.byText[s]
	if !ok {
		t.Fatalf("the vocabulary has no token %q", s)
	}
	return id
}

// text renders an admissible set, for a failure message that a human can read.
func (v *vocab) text(ids []int) string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		if b := v.Bytes(id); len(b) == 0 {
			out = append(out, "<stop>")
		} else {
			out = append(out, string(b))
		}
	}
	return strings.Join(out, " ")
}

func admits(st *State, id int) bool { return slices.Contains(st.Allowed(), id) }

// typeText drives st over the text, at each step taking the longest admissible
// token that the remaining text starts with.
//
// Longest AND admissible, not longest: a helper that picked greedily on text
// alone would report a grammar failure when it had merely chosen badly. Failing
// here means no token can carry the next character, which is the grammar
// rejecting a document it should accept.
func typeText(t *testing.T, v *vocab, st *State, text string) {
	t.Helper()
	for len(text) > 0 {
		best, bestLen := -1, 0
		for _, id := range st.Allowed() {
			b := v.Bytes(id)
			if len(b) > bestLen && strings.HasPrefix(text, string(b)) {
				best, bestLen = id, len(b)
			}
		}
		if best < 0 {
			t.Fatalf("nothing admissible carries %q; allowed here: %s", text, v.text(st.Allowed()))
		}
		if err := st.Advance(best); err != nil {
			t.Fatalf("advance %q: %v", v.Bytes(best), err)
		}
		text = text[bestLen:]
	}
}

// rejects reports where typing the text first became impossible, or -1 when the
// whole text was admissible.
func rejects(v *vocab, st *State, text string) int {
	at := 0
	for at < len(text) {
		best, bestLen := -1, 0
		for _, id := range st.Allowed() {
			b := v.Bytes(id)
			if len(b) > bestLen && strings.HasPrefix(text[at:], string(b)) {
				best, bestLen = id, len(b)
			}
		}
		if best < 0 {
			return at
		}
		if st.Advance(best) != nil {
			return at
		}
		at += bestLen
	}
	return -1
}

func compile(t *testing.T, v Vocab, schema string) *Grammar {
	t.Helper()
	g, err := Compile([]byte(schema), v, Options{Stop: []int{stopID}})
	if err != nil {
		t.Fatalf("compile %s: %v", schema, err)
	}
	return g
}
