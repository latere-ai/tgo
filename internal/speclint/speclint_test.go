// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package speclint checks that the spec tree matches the workflow
// specs/README.md documents.
//
// A spec tree nobody checks drifts from the code within one milestone: a
// `depends_on` points at a file somebody renamed, a spec is written and never
// indexed, a `blocked` spec loses the record of what blocks it. Each of those
// is invisible in review and obvious to a test.
package speclint

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// specDir is relative to this package.
const specDir = "../../specs"

// status values specs/README.md defines. A spec using anything else is either a
// typo or a lifecycle state nobody wrote down.
var statuses = map[string]bool{
	"drafted": true, "validated": true, "dispatched": true,
	"implemented": true, "complete": true,
	"blocked": true, "deferred": true, "living": true, "normative": true,
}

// layers are the tiers a spec can belong to.
var layers = map[string]bool{
	"load": true, "text": true, "graph": true, "engine": true, "api": true, "all": true,
}

type spec struct {
	file      string
	title     string
	status    string
	layer     string
	dependsOn []string
	blockedOn []string
	body      string
}

func load(t *testing.T) []spec {
	t.Helper()
	entries, err := os.ReadDir(specDir)
	if err != nil {
		t.Fatalf("read spec directory: %v", err)
	}
	var specs []spec
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") || e.Name() == "README.md" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(specDir, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		specs = append(specs, parse(t, e.Name(), string(b)))
	}
	if len(specs) == 0 {
		t.Fatal("no specs found; the linter would pass vacuously")
	}
	return specs
}

// parse reads the YAML-shaped frontmatter without a YAML dependency. The shape
// is fixed by specs/README.md and is three scalars and two lists, so a parser
// for exactly that is smaller than the dependency and fails on anything else.
func parse(t *testing.T, name, src string) spec {
	t.Helper()
	if !strings.HasPrefix(src, "---\n") {
		t.Fatalf("%s: no frontmatter; specs/README.md requires it", name)
	}
	end := strings.Index(src[4:], "\n---\n")
	if end < 0 {
		t.Fatalf("%s: frontmatter is not closed", name)
	}
	head, body := src[4:4+end], src[4+end+5:]

	s := spec{file: name, body: body}
	var list *[]string
	for _, line := range strings.Split(head, "\n") {
		if strings.HasPrefix(line, "  - ") {
			if list == nil {
				t.Fatalf("%s: list item %q under no key", name, line)
			}
			*list = append(*list, strings.Trim(strings.TrimPrefix(line, "  - "), `"`))
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		value = strings.Trim(strings.TrimSpace(value), `"`)
		list = nil
		switch key {
		case "title":
			s.title = value
		case "status":
			s.status = value
		case "layer":
			s.layer = value
		case "depends_on":
			list = &s.dependsOn
		case "blocked_on":
			list = &s.blockedOn
		}
	}
	return s
}

func TestFrontmatterIsComplete(t *testing.T) {
	for _, s := range load(t) {
		if s.title == "" {
			t.Errorf("%s: no title", s.file)
		}
		if !statuses[s.status] {
			t.Errorf("%s: status %q is not one specs/README.md defines", s.file, s.status)
		}
		if !layers[s.layer] {
			t.Errorf("%s: layer %q is not one specs/README.md defines", s.file, s.layer)
		}
	}
}

func TestDependenciesExist(t *testing.T) {
	specs := load(t)
	known := map[string]bool{}
	for _, s := range specs {
		known[s.file] = true
	}
	for _, s := range specs {
		for _, dep := range s.dependsOn {
			if !known[dep] {
				t.Errorf("%s: depends_on %q, which is not a spec here", s.file, dep)
			}
			if dep == s.file {
				t.Errorf("%s: depends on itself", s.file)
			}
		}
	}
}

// TestDependenciesAreAcyclic keeps the tree a DAG. A cycle makes the reading
// order in specs/README.md a lie and makes "what must land first" unanswerable.
func TestDependenciesAreAcyclic(t *testing.T) {
	specs := load(t)
	edges := map[string][]string{}
	for _, s := range specs {
		edges[s.file] = s.dependsOn
	}
	const (
		unvisited = 0
		active    = 1
		done      = 2
	)
	state := map[string]int{}
	var walk func(string, []string)
	walk = func(n string, path []string) {
		switch state[n] {
		case done:
			return
		case active:
			t.Errorf("dependency cycle: %s -> %s", strings.Join(path, " -> "), n)
			return
		}
		state[n] = active
		for _, d := range edges[n] {
			walk(d, append(path, n))
		}
		state[n] = done
	}
	for _, s := range specs {
		walk(s.file, nil)
	}
}

// TestBlockedSpecsSayWhatBlocksThem is the rule that keeps a blocked spec
// actionable. "Blocked" with no cause is indistinguishable from abandoned.
func TestBlockedSpecsSayWhatBlocksThem(t *testing.T) {
	for _, s := range load(t) {
		if s.status == "blocked" && len(s.blockedOn) == 0 {
			t.Errorf("%s: status is blocked with no blocked_on", s.file)
		}
		if s.status != "blocked" && len(s.blockedOn) > 0 {
			t.Errorf("%s: has blocked_on but status is %q", s.file, s.status)
		}
	}
}

// TestEverySpecHasADecisionRecord enforces specs/README.md's central rule. 000
// is the decision record for the whole project and 011 records outcomes rather
// than decisions, so both are exempt by name.
func TestEverySpecHasADecisionRecord(t *testing.T) {
	exempt := map[string]bool{"000-decisions.md": true, "011-sequencing.md": true}
	for _, s := range load(t) {
		if exempt[s.file] {
			continue
		}
		if !strings.Contains(s.body, "## Decision record") {
			t.Errorf("%s: no Decision record section", s.file)
		}
	}
}

// TestDecisionIdsMatchTheirSpec catches a record copied between specs, which
// silently gives two decisions one id.
func TestDecisionIdsMatchTheirSpec(t *testing.T) {
	id := regexp.MustCompile(`\| (\d{3})-D(\d+) \|`)
	for _, s := range load(t) {
		prefix := s.file[:3]
		for _, m := range id.FindAllStringSubmatch(s.body, -1) {
			if m[1] != prefix {
				t.Errorf("%s: decision id %s-D%s belongs to spec %s", s.file, m[1], m[2], m[1])
			}
		}
	}
}

// TestIndexListsEverySpec keeps specs/README.md from going stale, which is the
// most common way a spec tree stops being navigable.
func TestIndexListsEverySpec(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(specDir, "README.md"))
	if err != nil {
		t.Fatalf("read spec index: %v", err)
	}
	index := string(b)

	var missing []string
	for _, s := range load(t) {
		if !strings.Contains(index, "("+s.file+")") {
			missing = append(missing, s.file)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("specs/README.md does not link: %s", strings.Join(missing, ", "))
	}
}

// TestIndexAgreesOnStatus stops the tree's summary table from claiming a state
// the spec itself has moved past.
func TestIndexAgreesOnStatus(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(specDir, "README.md"))
	if err != nil {
		t.Fatalf("read spec index: %v", err)
	}
	lines := strings.Split(string(b), "\n")
	for _, s := range load(t) {
		for _, line := range lines {
			if !strings.Contains(line, "("+s.file+")") || !strings.HasPrefix(line, "| [") {
				continue
			}
			if !strings.Contains(line, s.status) {
				t.Errorf("%s: status is %q; the index row says %q", s.file, s.status, strings.TrimSpace(line))
			}
		}
	}
}

// The register in 010 is the project's primary output, and other specs cite its
// rows by number. 010-D6 makes it generated from the tests, which cannot happen
// before there are tests -- so until M10 these two checks are what stand in for
// it: the rows are well formed, and nothing cites a row that does not exist.
//
// A dangling "C11" in another spec is the exact drift 010-D1 exists to prevent,
// one level up, and it is invisible in review.

// registerRow matches a row of the table in 010 section 2.
var registerRow = regexp.MustCompile(`(?m)^\| \*{0,2}(C\d+)\*{0,2} \|`)

// registerCite matches a citation of a register row from anywhere in the tree.
var registerCite = regexp.MustCompile(`010 (C\d+)|\[010 (C\d+)\]|\bC(\d+)\]\(010-conformance\.md\)`)

func registerRows(t *testing.T) map[string]bool {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(specDir, "010-conformance.md"))
	if err != nil {
		t.Fatalf("read the conformance spec: %v", err)
	}
	rows := map[string]bool{}
	for _, m := range registerRow.FindAllStringSubmatch(string(b), -1) {
		if rows[m[1]] {
			t.Errorf("010: register row %s appears twice", m[1])
		}
		rows[m[1]] = true
	}
	if len(rows) == 0 {
		t.Fatal("010: the register has no rows; the check would pass vacuously")
	}
	return rows
}

func TestRegisterRowsAreNumberedWithoutGaps(t *testing.T) {
	rows := registerRows(t)
	for i := 1; i <= len(rows); i++ {
		if id := "C" + strconv.Itoa(i); !rows[id] {
			t.Errorf("010: the register has %d rows and no %s; a gap means a row was "+
				"deleted, and a row leaves only when its test stops skipping", len(rows), id)
		}
	}
}

func TestNothingCitesAMissingRegisterRow(t *testing.T) {
	rows := registerRows(t)
	for _, s := range load(t) {
		if s.file == "010-conformance.md" {
			continue
		}
		for _, m := range registerCite.FindAllStringSubmatch(s.body, -1) {
			id := m[1] + m[2]
			if m[3] != "" {
				id = "C" + m[3]
			}
			if id == "" {
				continue
			}
			if !rows[id] {
				t.Errorf("%s cites register row %s, which 010 does not have", s.file, id)
			}
		}
	}
}
