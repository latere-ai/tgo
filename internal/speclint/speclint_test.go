// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package speclint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// specDir is relative to this package.
const specDir = "../../specs"

func tree(t *testing.T) []Spec {
	t.Helper()
	specs, err := Load(specDir)
	if err != nil {
		t.Fatalf("load the spec tree: %v", err)
	}
	return specs
}

func read(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(specDir, name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(b)
}

// report fails with each finding on its own line, because a linter that
// collapses many problems into one message costs a round trip per problem.
func report(t *testing.T, what string, bad []string) {
	t.Helper()
	for _, b := range bad {
		t.Errorf("%s: %s", what, b)
	}
}

// --- the tree itself ---

func TestTheTreeIsWellFormed(t *testing.T) {
	specs := tree(t)
	report(t, "frontmatter", CheckFrontmatter(specs))
	report(t, "dependencies", CheckDependencies(specs))
	report(t, "acyclicity", CheckAcyclic(specs))
	report(t, "blocked", CheckBlocked(specs))
	report(t, "decision records", CheckDecisionRecords(specs))
	report(t, "index", CheckIndex(specs, read(t, "README.md")))
	report(t, "outcomes", CheckOutcomes(specs, read(t, "011-sequencing.md")))
}

func TestTheRegisterIsWellFormed(t *testing.T) {
	specs := tree(t)
	rows, bad := RegisterRows(read(t, "010-conformance.md"))
	report(t, "register", bad)
	report(t, "register numbering", CheckRegisterNumbering(rows))
	report(t, "register citations", CheckRegisterCitations(specs, rows))

	// The register is the project's primary output. An empty one would make
	// every check above pass while saying nothing.
	if len(rows) == 0 {
		t.Fatal("the register has no rows")
	}
}

// --- negative tests ---
//
// CONTRIBUTING.md requires a checker to be negative-tested: one that passes
// vacuously is the failure it exists to catch. Each case below is a tree that
// must be rejected, so a check silently returning nil is a test failure rather
// than a green build.

func spec(t *testing.T, name, front, body string) Spec {
	t.Helper()
	s, err := Parse(name, "---\n"+front+"\n---\n"+body)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return s
}

const record = "## Decision record\n"

func TestParseRejectsMalformedFrontmatter(t *testing.T) {
	for _, tc := range []struct{ name, src, want string }{
		{"no frontmatter", "# just a heading\n", "no frontmatter"},
		{"unclosed", "---\ntitle: x\n", "not closed"},
		{"orphan list item", "---\n  - a.md\ntitle: x\n---\nbody", "under no key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Parse("x.md", tc.src); err == nil {
				t.Fatal("accepted a malformed spec")
			} else if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error is %q, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestLoadRejectsAnEmptyDirectory(t *testing.T) {
	if _, err := Load(t.TempDir()); err == nil {
		t.Fatal("an empty directory loaded; every check would pass vacuously")
	}
	if _, err := Load(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Fatal("a missing directory loaded")
	}
}

func TestLoadReadsATree(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("README.md", "the index, which Load skips")
	write("notes.txt", "not a spec")
	write("000-a.md", "---\ntitle: A\nstatus: drafted\nlayer: all\n---\nbody")

	specs, err := Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(specs) != 1 || specs[0].File != "000-a.md" {
		t.Fatalf("loaded %v, want only 000-a.md", specs)
	}
	if _, err := Load(filepath.Join(dir)); err != nil {
		t.Fatalf("reload: %v", err)
	}

	write("001-bad.md", "not a spec at all")
	if _, err := Load(dir); err == nil {
		t.Fatal("a directory containing a malformed spec loaded")
	}
}

func TestFrontmatterValuesAreChecked(t *testing.T) {
	bad := CheckFrontmatter([]Spec{
		spec(t, "000-a.md", "title: \nstatus: drafted\nlayer: all", record),
		spec(t, "001-b.md", "title: B\nstatus: invented\nlayer: all", record),
		spec(t, "002-c.md", "title: C\nstatus: drafted\nlayer: nowhere", record),
	})
	if len(bad) != 3 {
		t.Fatalf("found %d problems in three broken specs: %v", len(bad), bad)
	}
}

func TestDependencyEdgesAreChecked(t *testing.T) {
	bad := CheckDependencies([]Spec{
		spec(t, "000-a.md", "title: A\nstatus: drafted\nlayer: all\ndepends_on:\n  - 999-gone.md", record),
		spec(t, "001-b.md", "title: B\nstatus: drafted\nlayer: all\ndepends_on:\n  - 001-b.md", record),
	})
	if len(bad) != 2 {
		t.Fatalf("found %d problems, want a dangling edge and a self-edge: %v", len(bad), bad)
	}
}

func TestACycleIsFound(t *testing.T) {
	cyclic := []Spec{
		spec(t, "000-a.md", "title: A\nstatus: drafted\nlayer: all\ndepends_on:\n  - 001-b.md", record),
		spec(t, "001-b.md", "title: B\nstatus: drafted\nlayer: all\ndepends_on:\n  - 000-a.md", record),
	}
	if bad := CheckAcyclic(cyclic); len(bad) == 0 {
		t.Fatal("a two-node cycle passed")
	}
	acyclic := []Spec{
		spec(t, "000-a.md", "title: A\nstatus: drafted\nlayer: all", record),
		spec(t, "001-b.md", "title: B\nstatus: drafted\nlayer: all\ndepends_on:\n  - 000-a.md", record),
	}
	if bad := CheckAcyclic(acyclic); len(bad) != 0 {
		t.Fatalf("an acyclic tree was rejected: %v", bad)
	}
}

func TestBlockedStatusIsChecked(t *testing.T) {
	bad := CheckBlocked([]Spec{
		spec(t, "000-a.md", "title: A\nstatus: blocked\nlayer: all", record),
		spec(t, "001-b.md", "title: B\nstatus: drafted\nlayer: all\nblocked_on:\n  - something", record),
		// Valid: blocked, with a cause that names an upstream issue.
		spec(t, "002-c.md", "title: C\nstatus: blocked\nlayer: all\nblocked_on:\n  - something (accel#1)", record),
	})
	if len(bad) != 2 {
		t.Fatalf("found %d problems, want blocked-without-cause and cause-without-blocked: %v",
			len(bad), bad)
	}
}

func TestDecisionRecordsAreChecked(t *testing.T) {
	bad := CheckDecisionRecords([]Spec{
		spec(t, "003-a.md", "title: A\nstatus: drafted\nlayer: all", "no record here"),
		spec(t, "004-b.md", "title: B\nstatus: drafted\nlayer: all", record+"| 007-D1 | copied | x | y |\n"),
	})
	if len(bad) != 2 {
		t.Fatalf("found %d problems, want a missing record and a foreign id: %v", len(bad), bad)
	}
	// 000 and 011 are exempt by name, and must stay exempt.
	if bad := CheckDecisionRecords([]Spec{
		spec(t, "000-decisions.md", "title: D\nstatus: normative\nlayer: all", "no record"),
		spec(t, "011-sequencing.md", "title: S\nstatus: living\nlayer: all", "no record"),
	}); len(bad) != 0 {
		t.Fatalf("an exempt spec was flagged: %v", bad)
	}
}

// TestAClaimedOutcomeIsChecked is the rule that makes a status carry
// information again.
//
// Twelve specs sat at `drafted` while their subject shipped, and three sat at
// `implemented` with a body that still described work in the future tense. A
// reader could not tell a spec waiting to be built from one built six waves
// ago, which is the whole reason the lifecycle exists.
func TestAClaimedOutcomeIsChecked(t *testing.T) {
	const seq = "…[013-distribution.md](013-distribution.md)…"
	const open = "## Outcome\n\nit shipped\n\n**Not built.** the queue.\n"
	const closed = "## Outcome\n\nit shipped\n\n**Not built.** Nothing here.\n"

	bad := CheckOutcomes([]Spec{
		// Built and silent about it: the case the rule is for.
		spec(t, "005-a.md", "title: A\nstatus: implemented\nlayer: all", record+"body"),
		// Claims complete, says what happened, but 011 does not carry it —
		// which is precisely what `complete` means and `implemented` does not.
		spec(t, "006-b.md", "title: B\nstatus: complete\nlayer: all", record+closed),
	}, seq)
	if len(bad) != 2 {
		t.Fatalf("found %d problems, want a missing Outcome and an unrecorded "+
			"complete: %v", len(bad), bad)
	}

	if bad := CheckOutcomes([]Spec{
		// Drafted is allowed to describe work nobody has started.
		spec(t, "007-c.md", "title: C\nstatus: drafted\nlayer: all", record+"someday"),
		// Blocked likewise: it is waiting on something.
		spec(t, "008-d.md", "title: D\nstatus: blocked\nlayer: all", record+"waiting"),
		// Built, says what happened and what is left, and 011 links it.
		spec(t, "013-distribution.md", "title: E\nstatus: complete\nlayer: all",
			record+closed),
		// The two exempt by name keep their exemption.
		spec(t, "000-decisions.md", "title: F\nstatus: normative\nlayer: all", "x"),
		spec(t, "011-sequencing.md", "title: G\nstatus: living\nlayer: all", "x"),
	}, seq); len(bad) != 0 {
		t.Fatalf("a spec that is allowed to be silent was flagged: %v", bad)
	}
}

// TestTheTwoBuiltStatusesAreDistinguishable is the half of the outcome rule that
// carries the difference between them.
//
// Without it both statuses check the same two things, so a spec passes at either
// and the pair says nothing. Four auditors reading the same spec split two-two
// on `implemented` against `complete` for exactly that reason: the tree had no
// rule to apply, only a preference. The rule is that `complete` means nothing in
// the spec's own scope is open, which is a claim the "Not built." paragraph
// already makes in prose.
func TestTheTwoBuiltStatusesAreDistinguishable(t *testing.T) {
	const seq = "…[013-a.md](013-a.md)…[014-b.md](014-b.md)…"
	const head = record + "## Outcome\n\nit shipped\n\n"

	for _, tc := range []struct {
		name, status, notBuilt, want string
	}{
		{"complete with open work", "complete",
			"**Not built.** the admission queue, which [021](021-x.md) owns.",
			"either the work moves to a spec that owns it"},
		{"implemented with none", "implemented",
			"**Not built.** Nothing in this spec's scope.",
			"a spec with no open work and a recorded outcome is complete"},
		{"no paragraph at all", "implemented",
			"everything landed and the section stops here.",
			"the part of an outcome that decays"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := CheckOutcomes([]Spec{
				spec(t, "013-a.md", "title: A\nstatus: "+tc.status+"\nlayer: all",
					head+tc.notBuilt),
			}, seq)
			if len(bad) != 1 {
				t.Fatalf("found %d problems, want 1: %v", len(bad), bad)
			}
			if !strings.Contains(bad[0], tc.want) {
				t.Fatalf("message does not say why:\n got %q\nwant it to contain %q",
					bad[0], tc.want)
			}
		})
	}

	// Both statuses pass when the paragraph agrees with them, and the marker is
	// read the same way whichever punctuation the author used.
	for _, body := range []string{
		"**Not built.** Nothing in this spec's scope.",
		"**Not built** Nothing further.",
		"**Not built.**: nothing; the children own the rest.",
	} {
		if bad := CheckOutcomes([]Spec{
			spec(t, "013-a.md", "title: A\nstatus: complete\nlayer: all", head+body),
			spec(t, "014-b.md", "title: B\nstatus: implemented\nlayer: all",
				head+"**Not built.** the batched path."),
		}, seq); len(bad) != 0 {
			t.Fatalf("an outcome that agrees with its status was flagged: %v", bad)
		}
	}
}

func TestIndexDriftIsFound(t *testing.T) {
	specs := []Spec{
		spec(t, "000-a.md", "title: A\nstatus: drafted\nlayer: all", record),
		spec(t, "001-b.md", "title: B\nstatus: complete\nlayer: all", record),
	}
	index := "| [001](001-b.md) | drafted | b |\n"
	bad := CheckIndex(specs, index)
	if len(bad) != 2 {
		t.Fatalf("found %d problems, want an unlinked spec and a stale status: %v", len(bad), bad)
	}
	if bad := CheckIndex(specs, "| [000](000-a.md) | drafted | a |\n| [001](001-b.md) | complete | b |\n"); len(bad) != 0 {
		t.Fatalf("a correct index was rejected: %v", bad)
	}
}

func TestRegisterProblemsAreFound(t *testing.T) {
	rows, bad := RegisterRows("| C1 | a |\n| **C2** | b |\n| C1 | duplicate |\n")
	if len(bad) != 1 {
		t.Fatalf("a duplicate row was not reported: %v", bad)
	}
	if !rows["C1"] || !rows["C2"] {
		t.Fatalf("rows are %v, want C1 and C2 including the bold form", rows)
	}

	if got := CheckRegisterNumbering(map[string]bool{"C1": true, "C3": true}); len(got) == 0 {
		t.Fatal("a gap in the numbering passed")
	}
	if got := CheckRegisterNumbering(map[string]bool{"C1": true, "C2": true}); len(got) != 0 {
		t.Fatalf("contiguous numbering was rejected: %v", got)
	}

	specs := []Spec{
		spec(t, "005-a.md", "title: A\nstatus: drafted\nlayer: all",
			"see [010 C9](010-conformance.md) and [C1](010-conformance.md)"+record),
	}
	if got := CheckRegisterCitations(specs, map[string]bool{"C1": true}); len(got) != 1 {
		t.Fatalf("found %d dangling citations, want 1 (C9): %v", len(got), got)
	}
	if got := CheckRegisterCitations(specs, map[string]bool{"C1": true, "C9": true}); len(got) != 0 {
		t.Fatalf("valid citations were rejected: %v", got)
	}
}

// TestParseAcceptsCRLF covers the Windows checkout. git converts LF to CRLF
// there by default, and a parser matching on "\n" alone rejected every spec in
// the tree -- which CI found and no local run would have.
func TestParseAcceptsCRLF(t *testing.T) {
	const src = "---\r\ntitle: A\r\nstatus: drafted\r\nlayer: all\r\ndepends_on:\r\n  - 000-x.md\r\n---\r\nbody\r\n"
	s, err := Parse("001-a.md", src)
	if err != nil {
		t.Fatalf("a CRLF spec was rejected: %v", err)
	}
	if s.Title != "A" || s.Status != "drafted" || s.Layer != "all" {
		t.Fatalf("parsed %+v, want the fields unchanged by the line ending", s)
	}
	if len(s.DependsOn) != 1 || s.DependsOn[0] != "000-x.md" {
		t.Fatalf("depends_on is %v, want [000-x.md] with no carriage return", s.DependsOn)
	}
	if strings.Contains(s.Body, "\r") {
		t.Fatal("the body kept a carriage return; a body check would then match differently per platform")
	}
}

// TestBlockedSpecsNameAnUpstreamIssue is the check that specs/010-conformance.md
// section 2.3 exists because of: 012 carried status blocked, named an accel spec
// file, and had no issue anywhere, so the blocker was invisible to the project
// that would have to clear it.
func TestBlockedSpecsNameAnUpstreamIssue(t *testing.T) {
	report(t, "blocked", CheckBlocked(tree(t)))

	// Negative: a blocked spec naming only a spec file must be rejected.
	bad := CheckBlocked([]Spec{
		spec(t, "012-x.md", "title: X\nstatus: blocked\nlayer: load\nblocked_on:\n  - \"accel specs/010-kernel-corpus.md\"", record),
	})
	if len(bad) != 1 {
		t.Fatalf("a blocker naming only a file was accepted: %v", bad)
	}
	// Positive: an issue reference is a record.
	if got := CheckBlocked([]Spec{
		spec(t, "012-x.md", "title: X\nstatus: blocked\nlayer: load\nblocked_on:\n  - \"accel specs/010-kernel-corpus.md (accel#15)\"", record),
	}); len(got) != 0 {
		t.Fatalf("a blocker naming an issue was rejected: %v", got)
	}
	// Positive: so is a backticked upstream name, which is what accel chose
	// instead of keeping accel#15 open.
	if got := CheckBlocked([]Spec{
		spec(t, "012-x.md", "title: X\nstatus: blocked\nlayer: load\nblocked_on:\n  - \"accel specs/010-kernel-corpus.md — `quant_matmul_superblock`, not registered\"", record),
	}); len(got) != 0 {
		t.Fatalf("a blocker naming an upstream artifact was rejected: %v", got)
	}
}
