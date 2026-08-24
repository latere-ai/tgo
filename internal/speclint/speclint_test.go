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
		spec(t, "002-c.md", "title: C\nstatus: blocked\nlayer: all\nblocked_on:\n  - something", record),
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
