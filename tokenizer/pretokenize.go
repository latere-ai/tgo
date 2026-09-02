// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"unicode"
	"unicode/utf8"
)

// The pre-tokenizer, specs/002-tokenizer.md section 4.
//
// The GPT-4 family of split patterns uses negative lookahead -- \s+(?!\S) --
// and Go's RE2 has no lookahead, by design, because the absence of backtracking
// is what buys the linear-time guarantee. 002-D6 chose a hand-written splitter
// per known pattern rather than a backtracking regex dependency, and 002-D7
// says an unrecognised pattern is refused at load rather than approximated: a
// different split silently produces different ids for the same string and there
// is nothing for a human to inspect.
//
// The two registered patterns differ in exactly one alternative, the run of
// digits, so one implementation covers both with a parameter.

const (
	// qwenPattern is the Split regex in Qwen2, Qwen2.5 and Qwen3
	// tokenizer.json. Verified byte for byte against a real Qwen3-0.6B
	// checkpoint; a checkpoint that differs by one character is refused, which
	// is the intended failure.
	qwenPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

	// cl100kPattern is the GPT-4 pattern. It is the Qwen pattern with runs of
	// up to three digits instead of one.
	cl100kPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

// splitPattern is one recognised split pattern and the parameters the shared
// splitter needs to reproduce it.
type splitPattern struct {
	name string
	// maxDigits is the upper bound of the \p{N} alternative: 1 for \p{N},
	// 3 for \p{N}{1,3}.
	maxDigits int
}

// knownPatterns maps the SHA-256 of a pattern string to its splitter. Keying on
// the checksum rather than on a model name is what makes an unknown pattern
// detectable at all: tokenizer.json carries the pattern, not the family.
var knownPatterns = map[string]splitPattern{
	patternDigest(qwenPattern):   {name: "qwen2/qwen3", maxDigits: 1},
	patternDigest(cl100kPattern): {name: "cl100k (gpt-4)", maxDigits: 3},
}

// patternDigest is the checksum 002-D6 keys the registry on.
func patternDigest(pattern string) string {
	sum := sha256.Sum256([]byte(pattern))
	return hex.EncodeToString(sum[:])
}

// split cuts s into pre-tokens, each BPE'd independently.
//
// It reproduces leftmost-first alternation with backtracking, which is what the
// reference regex engine does, rather than a fused character scan: the
// alternatives overlap on whitespace and punctuation, and a scan that decides
// per character gets the overlap wrong in ways that are invisible until ids
// diverge. Pieces are substrings of s, so bytes that are not valid UTF-8 pass
// through untouched.
func (p splitPattern) split(s string) []string {
	var out []string
	for i := 0; i < len(s); {
		n := p.match(s, i)
		if n <= 0 {
			// Unreachable for these patterns: every code point matches some
			// alternative. Kept so a future pattern cannot loop forever.
			_, n = utf8.DecodeRuneInString(s[i:])
		}
		out = append(out, s[i:i+n])
		i += n
	}
	return out
}

// match returns the byte length of the alternative that matches at i, trying
// the alternatives in the order the pattern writes them. Zero means no match.
func (p splitPattern) match(s string, i int) int {
	if n := matchContraction(s, i); n > 0 {
		return n
	}
	if n := matchWord(s, i); n > 0 {
		return n
	}
	if n := matchDigits(s, i, p.maxDigits); n > 0 {
		return n
	}
	if n := matchPunct(s, i); n > 0 {
		return n
	}
	if n := matchNewline(s, i); n > 0 {
		return n
	}
	if n := matchTrailingSpace(s, i); n > 0 {
		return n
	}
	return matchSpaceRun(s, i)
}

// matchContraction implements (?i:'s|'t|'re|'ve|'m|'ll|'d).
//
// This alternative only changes the split when the contraction is followed by
// more letters -- "'sx" is 's + x here and 'sx under the word alternative --
// but that case is common enough in code and in possessives to matter. The
// case folding is Unicode simple folding, not ASCII upper-casing, because the
// reference engine's (?i) is: U+017F, long s, folds to s.
func matchContraction(s string, i int) int {
	if i >= len(s) || s[i] != '\'' {
		return 0
	}
	r1, n1 := utf8.DecodeRuneInString(s[i+1:])
	if n1 == 0 {
		return 0
	}
	switch {
	case foldsTo(r1, 's'), foldsTo(r1, 't'), foldsTo(r1, 'm'), foldsTo(r1, 'd'):
		return 1 + n1
	}
	r2, n2 := utf8.DecodeRuneInString(s[i+1+n1:])
	if n2 == 0 {
		return 0
	}
	if (foldsTo(r1, 'r') && foldsTo(r2, 'e')) ||
		(foldsTo(r1, 'v') && foldsTo(r2, 'e')) ||
		(foldsTo(r1, 'l') && foldsTo(r2, 'l')) {
		return 1 + n1 + n2
	}
	return 0
}

// foldsTo reports whether r is in the simple case folding orbit of the ASCII
// lower-case letter want.
func foldsTo(r rune, want rune) bool {
	for f := unicode.SimpleFold(want); f != want; f = unicode.SimpleFold(f) {
		if f == r {
			return true
		}
	}
	return r == want
}

// matchWord implements [^\r\n\p{L}\p{N}]?\p{L}+ -- a word, optionally carrying
// the single character in front of it. That leading character is how a word
// keeps the space before it, which is why the vocabulary is full of tokens
// beginning with U+0120.
func matchWord(s string, i int) int {
	j := i
	r, n := decodeRune(s, j)
	if n == 0 {
		return 0
	}
	if r != '\r' && r != '\n' && !unicode.IsLetter(r) && !unicode.IsNumber(r) {
		j += n
	}
	start := j
	for {
		r, n := decodeRune(s, j)
		if n == 0 || !unicode.IsLetter(r) {
			break
		}
		j += n
	}
	if j == start {
		// The optional prefix matched but no letter followed. Backtracking to
		// not taking the prefix cannot help: the prefix character is by
		// construction not a letter, so \p{L}+ fails there too.
		return 0
	}
	return j - i
}

// matchDigits implements \p{N} or \p{N}{1,3}.
func matchDigits(s string, i, max int) int {
	j := i
	for range max {
		r, n := decodeRune(s, j)
		if n == 0 || !unicode.IsNumber(r) {
			break
		}
		j += n
	}
	return j - i
}

// matchPunct implements " ?[^\s\p{L}\p{N}]+[\r\n]*".
func matchPunct(s string, i int) int {
	j := i
	if j < len(s) && s[j] == ' ' {
		j++
	}
	start := j
	for {
		r, n := decodeRune(s, j)
		if n == 0 || unicode.IsSpace(r) || unicode.IsLetter(r) || unicode.IsNumber(r) {
			break
		}
		j += n
	}
	if j == start {
		// Same reasoning as matchWord: the optional leading space is itself
		// whitespace, so dropping it cannot make the class match.
		return 0
	}
	for j < len(s) && (s[j] == '\r' || s[j] == '\n') {
		j++
	}
	return j - i
}

// matchNewline implements \s*[\r\n]+.
//
// Written out because the backtracking is not obvious: \s* is greedy, so the
// engine walks back from the end of the whitespace run to the last \r or \n it
// contains, and the match ends there. A run with no line break does not match
// at all, and the whitespace after the last line break is left for the next
// position.
func matchNewline(s string, i int) int {
	run := spaceRun(s, i)
	last := -1
	for j := i; j < i+run; j++ {
		if s[j] == '\r' || s[j] == '\n' {
			last = j
		}
	}
	if last < 0 {
		return 0
	}
	return last + 1 - i
}

// matchTrailingSpace implements \s+(?!\S): whitespace that is not immediately
// followed by a non-space.
//
// This is the alternative Go cannot compile and the reason for 002-D6. Its
// effect is to hand the last character of a whitespace run to the *next* piece,
// so that " hello" stays together, while a run at end of input is taken whole.
func matchTrailingSpace(s string, i int) int {
	run := spaceRun(s, i)
	if run == 0 {
		return 0
	}
	if i+run == len(s) {
		return run
	}
	// The next character is non-space, so the greedy match fails the lookahead
	// and backtracks by one; one character of whitespace has nothing to give up.
	if run == 1 {
		return 0
	}
	return run - lastRuneLen(s[i:i+run])
}

// matchSpaceRun implements the final \s+.
func matchSpaceRun(s string, i int) int { return spaceRun(s, i) }

// spaceRun returns the byte length of the maximal whitespace run at i.
func spaceRun(s string, i int) int {
	j := i
	for {
		r, n := decodeRune(s, j)
		if n == 0 || !unicode.IsSpace(r) {
			return j - i
		}
		j += n
	}
}

// decodeRune returns the rune at i and its width, or width 0 at end of input.
// A byte that is not valid UTF-8 decodes as U+FFFD with width 1, which puts it
// in the "neither letter, digit nor space" class -- the same class the
// reference engine puts an undecodable byte in.
func decodeRune(s string, i int) (rune, int) {
	if i >= len(s) {
		return 0, 0
	}
	return utf8.DecodeRuneInString(s[i:])
}

// lastRuneLen returns the width of the final rune of a non-empty string.
func lastRuneLen(s string) int {
	_, n := utf8.DecodeLastRuneInString(s)
	return n
}
