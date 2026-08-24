// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package conformance is the machinery of specs/010-conformance.md: the tier
// rule every parity test in the tree runs under, the derived tolerance every
// comparison is judged by, and the generator that emits the register.
//
// It is tgo's primary output (010 §1). Two directions meet here. Downward, a
// device result is compared against [github.com/latere-ai/tgo/internal/oracle],
// and a disagreement is a finding against accel or against tgo. Upward, every
// place tgo cannot express something is a row of the register in [Register],
// and [Document] turns those rows into the Markdown 010 §2 publishes.
//
// # The three tiers
//
// A test declares its tier and [Device] decides what happens (010 §4):
//
//	tier    needs                     no device or no weights
//	1       nothing                   cannot happen; the CPU backend is always there
//	2       a Metal device            skip, or fail under TGO_REQUIRE_METAL
//	3       real weights, TGO_MODEL   skip; tier 3 is never in CI (010-D4)
//
// The decision is [decide], a pure function of the tier and what is available,
// so every branch of it is tested on a machine that has neither a Metal device
// nor a checkpoint. A rule that only runs where its inputs happen to be present
// is a rule nobody has checked.
//
// # Tolerances
//
// Nothing here takes a tolerance as a number. A comparison takes [Terms], and
// each constructor is one row of 010 §5.1's table or one ceiling from accel's
// specs/008-numerics.md §6. 010-D3 says a tolerance that had to be raised to
// make a test pass is a finding, and a bound that is computed from the shapes
// and the storage formats cannot be quietly raised: raising it means adding a
// term, and a term that does not exist has to be argued for.
package conformance
