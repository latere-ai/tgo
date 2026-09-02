// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tokenizer

import "golang.org/x/text/unicode/norm"

// The normalizer, specs/002-tokenizer.md section 1 and 002-D9.
//
// tokenizer.json declares a normalizer that runs before the pre-tokenizer, and
// every Qwen checkpoint declares NFC. Skipping it gives different ids for text
// that looks identical: "e" followed by U+0301 and "é" are one string to a
// reader and two different pre-tokenizer inputs.
//
// # Why this is the one dependency the core takes
//
// Composing to NFC needs the canonical decomposition table, the canonical
// combining classes, the composition exclusions and the algorithmic Hangul
// rules. None of it is in the standard library. The first implementation left
// this function as the identity rather than take a dependency unilaterally,
// which was the right call to escalate and the wrong state to ship: the loader
// *refused* an NFKC normalizer on the grounds that it "changes ids", then
// accepted NFC and did nothing -- so the file said the text was normalized and
// the ids said it was not. That is the same silent divergence the refusal
// exists to prevent, one step further along.
//
// golang.org/x/text is pure Go, has no cgo anywhere in it, and so does not
// touch specs/000-decisions.md decision 2, which is about cgo rather than about
// dependencies. It is maintained by the Go team, and `unicode/norm` pulls in
// nothing but the standard library. The alternative was being quietly wrong on
// every Qwen checkpoint, which is not a trade worth making to keep a dependency
// count at zero.
//
// Recorded as 002-D10.

// normalizer rewrites input text before the pre-tokenizer sees it.
type normalizer func(string) string

// nfc composes to Unicode Normalization Form C.
//
// norm.NFC.String returns the input unchanged, without allocating, when it is
// already in NFC -- which is the overwhelmingly common case for prompt text, so
// this costs a scan rather than a copy on the hot path.
func nfc(s string) string { return norm.NFC.String(s) }

// nfcImplemented reports whether nfc actually composes. The tests that assert
// NFC behaviour read it so they skip rather than pass vacuously against an
// identity function; it is true now and the constant stays so that a future
// change back to a seam cannot silently turn those tests green.
const nfcImplemented = true
