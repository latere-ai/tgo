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
// # The register
//
// [Register] is the register itself and specs/010-conformance.md §2's table is
// its output: [Document] renders the rows and a test fails when the spec and
// the rows have parted. 010-D6 -- a hand-maintained register drifts within one
// milestone, which is the exact failure this project exists to catch in accel,
// so the table cannot be maintained beside the tests that produce it.
//
// Every open row is a named skipping test carrying the reason and the accel
// spec that owns the gap (010-D1). The tests are generated from the rows too,
// so the table cannot claim something no test tracks, and a row leaves the
// register only when its test stops skipping.
//
// [Measurements] is §3: the five numbers tgo reports back, each a question
// accel cannot answer about itself. [Publish] emits the register and the
// numbers as the one generated document §6 asks for.
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
