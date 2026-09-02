// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package grammar compiles a JSON Schema into a per-step mask over a
// tokenizer's vocabulary, so that a constrained generation parses by
// construction.
//
// The guarantee is specs/015-structured-output.md section 1. At each step the
// tokens that cannot continue a valid document are given probability zero:
//
//	p'(v | y<t) is proportional to p(v | y<t) * [ y<t v is a viable prefix of G ]
//
// A retry loop around a model that emits JSON "most of the time" is then
// unnecessary, because there is no other thing for it to emit.
//
// # The mask is over tokens, not over characters
//
// This is section 2 and it is the whole difficulty. A BPE token is a byte
// string that need not align to anything: `":` is one token, and it is a string
// terminator followed by a structural character. So the automaton's alphabet is
// the byte, a token is walked byte by byte, and a token is admissible only if
// EVERY intermediate byte is -- checking the first byte, or the state the token
// lands in, would admit a token whose middle leaves the grammar.
//
// Walking 152k tokens per step is too slow. 015-D1 is the answer and dfa.go is
// where it lives: per automaton state, the admissible set is computed on first
// visit and kept. States repeat heavily -- every string-opening quote in a
// document with six string properties reaches one state -- so the second
// request pays almost nothing, and a Grammar is meant to be shared across them.
// Building the sets eagerly instead would pay the whole vocabulary cross
// product at startup, for states a request may never reach.
//
// # Using it
//
//	g, err := grammar.Compile(schema, vocab, grammar.Options{Stop: eos})
//	st := g.Start()
//	for {
//	    logits := model.Step()
//	    if err := st.Mask(logits); err != nil { return err }  // before penalties
//	    id := sampler.Next(logits, history, policy)
//	    if st.Advance(id) != nil || st.Done() { break }
//	}
//
// Mask is an additive negative infinity applied BEFORE the penalties of
// specs/006-sampling.md section 3 (015-D2), so it composes with that order with
// no special case: a masked token cannot be resurrected by a penalty or a
// temperature, because both are monotone in the logit and -Inf is a fixed point
// of both. Advance is called once per accepted token, after the draw. A
// resumed sequence replays Advance over the tokens already generated.
//
// A stop token is admissible exactly where the document is complete and nowhere
// else. Without that the model could end a truncated document, and "parses by
// construction" would be false in the one case that matters.
//
// # What it refuses, and what it narrows
//
// The front end is an allowlist: every keyword in the schema must be consumed
// by the compilation, and one that is not is refused by name (015-D4). A
// keyword silently ignored produces a document that validates against a schema
// the caller did not write, which is worse than no constrained decoding at all
// because it looks like it worked. The refusal carries the obstruction --
// "a numeric bound is arithmetic on the value" -- rather than "unsupported",
// because the caller's next move differs by reason.
//
// The language is regular. A JSON Schema with no recursive $ref generates
// documents of bounded nesting, so a byte-level NFA with a lazy subset
// determinization is enough, and no pushdown stack is needed. Every construct
// that would break that -- a subschema with no "type", an array with no
// "items", an open object, a $ref cycle -- is refused rather than approximated.
//
// Three narrowings are deliberate, and each one shrinks the admitted language
// rather than widening it, so a document this package accepts still validates
// against the caller's schema:
//
//   - Object properties must appear in the order the schema declares them.
//     Admitting every permutation needs a state per subset already emitted,
//     2^n of them.
//   - Objects are closed. A property outside the schema is never emitted. An
//     explicit "additionalProperties": true is refused rather than narrowed,
//     because that one is the caller stating something this cannot honour.
//   - "integer" admits the plain spelling. 1e2 is an integer to JSON Schema and
//     is not admitted here.
//
// One thing is deliberately NOT narrowed, and a caller has to know it. A
// number's magnitude is unbounded, because RFC 8259 does not bound one and
// JSON Schema spells a bound as "minimum" or "maximum", which this package
// refuses to compile. So 1e999 is admissible: valid JSON that json.Valid
// accepts and that a Go consumer decoding into a float64 field cannot hold.
// A caller who needs a magnitude bound checks it after decoding.
//
// # What is not here
//
// A general EBNF surface, and regex as a special case of it. Same machinery,
// different front end; specs/015-structured-output.md 015-D3 puts it second.
//
// The wiring into sample. This package is deliberately free of any dependency
// on it: Vocab is an interface so no tokenizer is imported, and the mask is a
// method on a row of float32 so no sampler type is either.
package grammar
