// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tokenizer

import "unicode/utf8"

// Streaming decode, specs/002-tokenizer.md section 6 and 002-D4.

// Decoder decodes one token at a time and holds back a trailing byte sequence
// that is a valid prefix of a code point but not yet a whole one.
//
// This is not an edge case to be tidy about. A byte-level vocabulary splits
// most non-ASCII characters across several tokens, so rendering each token as
// it arrives would emit U+FFFD in the middle of every CJK character, every
// emoji, and every emoji with a skin-tone or zero-width-joiner modifier.
//
// A Decoder is not safe for concurrent use; one belongs to one stream.
type Decoder struct {
	t *Tokenizer

	// buf holds the bytes that could still become a code point. It is about
	// byte validity and nothing else. 006's stop-string hold-back is a second,
	// separate buffer living with the sampler (002-D8): one holds because a
	// byte sequence is incomplete, the other because a text match might still
	// extend, and a single buffer serving both mis-holds for both.
	buf []byte
}

// NewDecoder returns a streaming decoder over t.
func (t *Tokenizer) NewDecoder() *Decoder { return &Decoder{t: t} }

// Push adds one token and returns the text it completed, which is often "".
//
// The returned bytes are exactly the token bytes that are no longer in doubt,
// so concatenating every Push over a stream that ends on a complete code point
// gives byte-for-byte what Decode gives for the same ids. A byte that cannot
// begin any code point is not held: it is emitted at once, because waiting
// cannot make it valid.
func (d *Decoder) Push(id int) string {
	d.buf = append(d.buf, d.t.bytesFor(id)...)
	n := emitLen(d.buf)
	out := string(d.buf[:n])
	d.buf = d.buf[:copy(d.buf, d.buf[n:])]
	return out
}

// Flush returns whatever is still held back, at end of stream.
//
// What is held at this point is by construction a truncated code point, and a
// truncated code point at end of input is genuinely malformed, so it is
// rendered as one U+FFFD -- the Unicode maximal-subpart replacement, one per
// truncated sequence rather than one per orphaned byte. Returning the raw bytes
// instead would hand the caller a string that is not valid UTF-8, and returning
// "" would silently drop the fact that the stream ended mid-character.
func (d *Decoder) Flush() string {
	if len(d.buf) == 0 {
		return ""
	}
	d.buf = d.buf[:0]
	return string(utf8.RuneError)
}

// emitLen returns the length of the prefix of b that can be emitted now: all of
// it, unless it ends in a valid but incomplete encoding.
//
// utf8.FullRune is the exact predicate wanted here. It reports false only for a
// truncated well-formed sequence; a byte that is invalid outright counts as
// full, because it converts as one width-1 error rune and holding it back would
// never pay off.
func emitLen(b []byte) int {
	for i := 0; i < len(b); {
		if b[i] < utf8.RuneSelf {
			i++
			continue
		}
		if !utf8.FullRune(b[i:]) {
			return i
		}
		_, size := utf8.DecodeRune(b[i:])
		i += size
	}
	return len(b)
}
