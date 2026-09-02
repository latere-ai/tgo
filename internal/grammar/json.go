// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package grammar

// The JSON lexical layer, spelled out over bytes.
//
// Whitespace may precede any JSON token and there is none after the last one.
// That single rule covers every insignificant position RFC 8259 has -- before a
// value, around a colon, around a comma, before a closing brace -- without a
// site-by-site list, and leaving the trailing case out keeps the accepting
// state clean: once the document is complete the only admissible token is a
// stop token, rather than an unbounded run of spaces that defers the decision.

// ws matches a possibly empty run of insignificant whitespace.
func (c *compiler) ws() frag {
	return c.n.star(func() frag {
		return c.n.class(brange{' ', ' '}, brange{'\t', '\t'}, brange{'\n', '\n'}, brange{'\r', '\r'})
	})
}

// tok matches a literal preceded by optional whitespace.
func (c *compiler) tok(s string) frag { return c.n.cat(c.ws(), c.n.lit(s)) }

func (c *compiler) digit() frag { return c.n.class(brange{'0', '9'}) }

// integer matches the JSON integer syntax: an optional sign, then zero or a
// digit string with no leading zero.
//
// Leading zeros are excluded because RFC 8259 excludes them, and a mask that
// admitted 007 would be admitting something encoding/json refuses to parse --
// which is exactly the failure this package exists to make impossible.
func (c *compiler) integer() frag {
	return c.n.cat(
		c.n.opt(c.n.lit("-")),
		c.n.alt(
			c.n.lit("0"),
			c.n.cat(c.n.class(brange{'1', '9'}), c.n.star(c.digit)),
		),
	)
}

// number matches the JSON number syntax: an integer, an optional fraction, an
// optional exponent.
func (c *compiler) number() frag {
	frac := c.n.opt(c.n.cat(c.n.lit("."), c.n.rep(c.digit, 1, -1)))
	exp := c.n.opt(c.n.cat(
		c.n.class(brange{'e', 'e'}, brange{'E', 'E'}),
		c.n.opt(c.n.class(brange{'+', '+'}, brange{'-', '-'})),
		c.n.rep(c.digit, 1, -1),
	))
	return c.n.cat(c.integer(), frac, exp)
}

// jsonString matches a quoted string of between lo and hi characters, hi below
// zero meaning unbounded.
//
// The count is in characters and not in bytes, because that is what JSON
// Schema's minLength and maxLength count. Each repetition of stringChar is
// exactly one code point: the UTF-8 ranges below encode one, and the escape
// alternative admits one \uXXXX at a time with the surrogate block excluded, so
// there is no spelling in this language where two repetitions collapse into one
// character. Admitting surrogate pairs would break that -- 😀 is two
// repetitions and one character -- and would make minLength unsound rather than
// merely conservative. A character outside the basic plane is still reachable:
// it is typed as its own UTF-8 bytes.
func (c *compiler) jsonString(lo, hi int) frag {
	return c.n.cat(c.n.lit(`"`), c.n.rep(c.stringChar, lo, hi), c.n.lit(`"`))
}

func (c *compiler) stringChar() frag {
	return c.n.alt(c.unescaped(), c.escaped())
}

// cont is a UTF-8 continuation byte.
func (c *compiler) cont() frag { return c.n.class(brange{0x80, 0xBF}) }

// unescaped matches one character that may appear in a JSON string as itself:
// any code point except the quote, the backslash and the C0 controls, encoded
// as well-formed UTF-8.
//
// The ranges are Unicode 15 table 3-7 verbatim rather than "0x80..0xFF, several
// times". The difference is not pedantry: the loose version admits overlong
// encodings, surrogates encoded as three bytes, and code points above U+10FFFF,
// all of which are byte strings a BPE vocabulary can produce and none of which
// are text.
func (c *compiler) unescaped() frag {
	return c.n.alt(
		c.n.class(brange{0x20, 0x21}, brange{0x23, 0x5B}, brange{0x5D, 0x7F}),
		c.n.cat(c.n.class(brange{0xC2, 0xDF}), c.cont()),
		c.n.cat(c.n.class(brange{0xE0, 0xE0}), c.n.class(brange{0xA0, 0xBF}), c.cont()),
		c.n.cat(c.n.class(brange{0xE1, 0xEC}), c.cont(), c.cont()),
		c.n.cat(c.n.class(brange{0xED, 0xED}), c.n.class(brange{0x80, 0x9F}), c.cont()),
		c.n.cat(c.n.class(brange{0xEE, 0xEF}), c.cont(), c.cont()),
		c.n.cat(c.n.class(brange{0xF0, 0xF0}), c.n.class(brange{0x90, 0xBF}), c.cont(), c.cont()),
		c.n.cat(c.n.class(brange{0xF1, 0xF3}), c.cont(), c.cont(), c.cont()),
		c.n.cat(c.n.class(brange{0xF4, 0xF4}), c.n.class(brange{0x80, 0x8F}), c.cont(), c.cont()),
	)
}

// escaped matches one backslash escape.
func (c *compiler) escaped() frag {
	short := c.n.class(
		brange{'"', '"'}, brange{'/', '/'}, brange{'\\', '\\'},
		brange{'b', 'b'}, brange{'f', 'f'}, brange{'n', 'n'},
		brange{'r', 'r'}, brange{'t', 't'},
	)
	return c.n.cat(c.n.lit(`\`), c.n.alt(short, c.n.cat(c.n.lit("u"), c.hex4())))
}

// hex4 matches four hex digits whose value is outside the surrogate block
// U+D800..U+DFFF.
//
// The block is exactly the codes whose first digit is d and whose second is 8
// through f -- D800 is the first and DFFF is the last, so the whole upper half
// of the d page is surrogate, high and low alike. Excluding it is therefore two
// alternatives rather than a comparison: a first digit that is not d, or a d
// followed by a second digit in the lower half.
//
// The low half of the block matters as much as the high half. A lone \udc00
// costs nothing to type, is JSON that json.Valid accepts, and decodes in Go to
// U+FFFD -- so a mask that admitted it would let the model spell a character
// the caller cannot get back, which is the failure this package exists to
// make impossible.
func (c *compiler) hex4() frag {
	hex := func() frag {
		return c.n.class(brange{'0', '9'}, brange{'a', 'f'}, brange{'A', 'F'})
	}
	notD := c.n.class(brange{'0', '9'}, brange{'a', 'c'}, brange{'e', 'f'}, brange{'A', 'C'}, brange{'E', 'F'})
	isD := c.n.class(brange{'d', 'd'}, brange{'D', 'D'})
	belowTheBlock := c.n.class(brange{'0', '7'})
	return c.n.alt(
		c.n.cat(notD, hex(), hex(), hex()),
		c.n.cat(isD, belowTheBlock, hex(), hex()),
	)
}
