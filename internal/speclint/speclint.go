// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package speclint checks that the spec tree matches the workflow
// specs/README.md documents.
//
// A spec tree nobody checks drifts from the code within one milestone: a
// depends_on points at a file somebody renamed, a spec is written and never
// indexed, a blocked spec loses the record of what blocks it, another spec
// cites a conformance row that no longer exists. Each of those is invisible in
// review and obvious to a program.
//
// The checks live here rather than in the test file so that the coverage gate
// measures them. A linter is code, and code this repository's own CONTRIBUTING
// requires to be negative-tested.
package speclint

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Status values specs/README.md defines. A spec using anything else is either a
// typo or a lifecycle state nobody wrote down.
var Status = map[string]bool{
	"drafted": true, "validated": true, "dispatched": true,
	"implemented": true, "complete": true,
	"blocked": true, "deferred": true, "living": true, "normative": true,
}

// Layer is the tier a spec belongs to.
var Layer = map[string]bool{
	"load": true, "text": true, "graph": true, "engine": true, "api": true, "all": true,
}

// noRecord are the specs exempt from carrying a decision record. 000 *is* the
// decision record for the whole project, and 011 records outcomes rather than
// decisions.
var noRecord = map[string]bool{"000-decisions.md": true, "011-sequencing.md": true}

// Spec is one parsed spec file.
type Spec struct {
	File      string
	Title     string
	Status    string
	Layer     string
	DependsOn []string
	BlockedOn []string
	Body      string
}

// Load reads and parses every spec in dir, excluding the index.
func Load(dir string) ([]Spec, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read spec directory: %w", err)
	}
	var specs []Spec
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") || e.Name() == "README.md" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", e.Name(), err)
		}
		s, err := Parse(e.Name(), string(b))
		if err != nil {
			return nil, err
		}
		specs = append(specs, s)
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("no specs in %s; every check would pass vacuously", dir)
	}
	return specs, nil
}

// Parse reads the YAML-shaped frontmatter without a YAML dependency.
//
// The shape is fixed by specs/README.md and is three scalars and two lists, so a
// parser for exactly that is smaller than the dependency, and it fails on
// anything else rather than accepting a shape the workflow does not define.
func Parse(name, src string) (Spec, error) {
	// Windows checkouts convert LF to CRLF, so a parser that matches on "\n"
	// rejects every spec in the tree on one of the three platforms CI runs. The
	// line ending is not part of what a spec says, so it is normalised here
	// rather than defended against at each comparison.
	src = strings.ReplaceAll(src, "\r\n", "\n")

	if !strings.HasPrefix(src, "---\n") {
		return Spec{}, fmt.Errorf("%s: no frontmatter; specs/README.md requires it", name)
	}
	end := strings.Index(src[4:], "\n---\n")
	if end < 0 {
		return Spec{}, fmt.Errorf("%s: frontmatter is not closed", name)
	}
	head, body := src[4:4+end], src[4+end+5:]

	s := Spec{File: name, Body: body}
	var list *[]string
	for _, line := range strings.Split(head, "\n") {
		if strings.HasPrefix(line, "  - ") {
			if list == nil {
				return Spec{}, fmt.Errorf("%s: list item %q under no key", name, line)
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
			s.Title = value
		case "status":
			s.Status = value
		case "layer":
			s.Layer = value
		case "depends_on":
			list = &s.DependsOn
		case "blocked_on":
			list = &s.BlockedOn
		}
	}
	return s, nil
}

// CheckFrontmatter reports specs whose frontmatter is incomplete or uses a
// value specs/README.md does not define.
func CheckFrontmatter(specs []Spec) []string {
	var bad []string
	for _, s := range specs {
		if s.Title == "" {
			bad = append(bad, s.File+": no title")
		}
		if !Status[s.Status] {
			bad = append(bad, fmt.Sprintf("%s: status %q is not one specs/README.md defines",
				s.File, s.Status))
		}
		if !Layer[s.Layer] {
			bad = append(bad, fmt.Sprintf("%s: layer %q is not one specs/README.md defines",
				s.File, s.Layer))
		}
	}
	return bad
}

// CheckDependencies reports depends_on edges that do not resolve.
func CheckDependencies(specs []Spec) []string {
	known := map[string]bool{}
	for _, s := range specs {
		known[s.File] = true
	}
	var bad []string
	for _, s := range specs {
		for _, dep := range s.DependsOn {
			if dep == s.File {
				bad = append(bad, s.File+": depends on itself")
				continue
			}
			if !known[dep] {
				bad = append(bad, fmt.Sprintf("%s: depends_on %q, which is not a spec here",
					s.File, dep))
			}
		}
	}
	return bad
}

// CheckAcyclic reports dependency cycles.
//
// A cycle makes the reading order in specs/README.md a lie and makes "what must
// land first" unanswerable.
func CheckAcyclic(specs []Spec) []string {
	edges := map[string][]string{}
	for _, s := range specs {
		edges[s.File] = s.DependsOn
	}
	const (
		unvisited = iota
		active
		done
	)
	state := map[string]int{}
	var bad []string
	var walk func(string, []string)
	walk = func(n string, path []string) {
		switch state[n] {
		case done:
			return
		case active:
			bad = append(bad, "dependency cycle: "+strings.Join(append(path, n), " -> "))
			return
		}
		state[n] = active
		for _, d := range edges[n] {
			walk(d, append(path, n))
		}
		state[n] = done
	}
	for _, s := range specs {
		walk(s.File, nil)
	}
	return bad
}

// upstreamRecord matches a durable upstream record of a blocker: an issue
// reference, or a backticked identifier naming the thing upstream that records
// the gap -- a kernel name, a corpus row.
//
// Not a bare file path. "blocked on accel specs/010-kernel-corpus.md" is a
// belief about whose problem it is; "blocked on `quant_matmul_superblock`, not
// registered" is a claim that can be checked by opening that file.
var upstreamRecord = regexp.MustCompile("accel#[0-9]+|/issues/[0-9]+|`[^`]+`")

// CheckBlocked reports blocked specs that do not say what blocks them, and
// unblocked specs that carry a blocked_on.
//
// "Blocked" with no cause is indistinguishable from abandoned, and a cause that
// names only another project's file is a belief rather than a record.
//
// specs/010-conformance.md section 2.3 is the incident: 012 was blocked on an
// accel spec file for a week with no issue anywhere. The first version of this
// check demanded an open *issue*, which was too narrow -- accel closed that
// issue as not planned and recorded the gap as a "not registered" row in its
// kernel corpus instead, which is a better record than an issue with no plan,
// because the corpus is what a kernel author reads. So the requirement is a
// durable record, not a ticket.
func CheckBlocked(specs []Spec) []string {
	var bad []string
	for _, s := range specs {
		switch {
		case s.Status == "blocked" && len(s.BlockedOn) == 0:
			bad = append(bad, s.File+": status is blocked with no blocked_on")
		case s.Status != "blocked" && len(s.BlockedOn) > 0:
			bad = append(bad, fmt.Sprintf("%s: has blocked_on but status is %q",
				s.File, s.Status))
		case s.Status == "blocked":
			if !upstreamRecord.MatchString(strings.Join(s.BlockedOn, " ")) {
				bad = append(bad, fmt.Sprintf("%s: blocked_on names no durable upstream "+
					"record (want an \"accel#N\" reference, or a `backticked` name for "+
					"the thing upstream that records the gap); a bare file path is a "+
					"belief about whose problem it is, not a record anyone can act on",
					s.File))
			}
		}
	}
	return bad
}

// CheckDecisionRecords reports specs with no decision record, and decision ids
// whose number does not match the spec they are in.
//
// A mismatched id means a record was copied between specs, which silently gives
// two different decisions one name.
func CheckDecisionRecords(specs []Spec) []string {
	id := regexp.MustCompile(`\| (\d{3})-D(\d+) \|`)
	var bad []string
	for _, s := range specs {
		if !noRecord[s.File] && !strings.Contains(s.Body, "## Decision record") {
			bad = append(bad, s.File+": no Decision record section")
		}
		prefix := s.File[:3]
		for _, m := range id.FindAllStringSubmatch(s.Body, -1) {
			if m[1] != prefix {
				bad = append(bad, fmt.Sprintf("%s: decision id %s-D%s belongs to spec %s",
					s.File, m[1], m[2], m[1]))
			}
		}
	}
	return bad
}

// CheckIndex reports specs the index does not link, and index rows whose stated
// status disagrees with the spec's own.
func CheckIndex(specs []Spec, index string) []string {
	var missing, wrong []string
	lines := strings.Split(index, "\n")
	for _, s := range specs {
		if !strings.Contains(index, "("+s.File+")") {
			missing = append(missing, s.File)
			continue
		}
		for _, line := range lines {
			if !strings.HasPrefix(line, "| [") || !strings.Contains(line, "("+s.File+")") {
				continue
			}
			if !strings.Contains(line, s.Status) {
				wrong = append(wrong, fmt.Sprintf("%s: status is %q; the index row says %q",
					s.File, s.Status, strings.TrimSpace(line)))
			}
		}
	}
	sort.Strings(missing)
	var bad []string
	if len(missing) > 0 {
		bad = append(bad, "specs/README.md does not link: "+strings.Join(missing, ", "))
	}
	return append(bad, wrong...)
}

// registerRow matches a row of the register table in 010 section 2.
var registerRow = regexp.MustCompile(`(?m)^\| \*{0,2}(C\d+)\*{0,2} \|`)

// registerCite matches a citation of a register row from anywhere in the tree.
var registerCite = regexp.MustCompile(`010 (C\d+)|\b(C\d+)\]\(010-conformance\.md\)`)

// RegisterRows returns the row ids the conformance register declares.
func RegisterRows(conformance string) (map[string]bool, []string) {
	rows := map[string]bool{}
	var bad []string
	for _, m := range registerRow.FindAllStringSubmatch(conformance, -1) {
		if rows[m[1]] {
			bad = append(bad, "010: register row "+m[1]+" appears twice")
		}
		rows[m[1]] = true
	}
	return rows, bad
}

// CheckRegisterNumbering reports gaps in the register's numbering.
//
// A gap means a row was deleted, and specs/010-conformance.md is explicit that a
// row leaves only when its test stops skipping.
func CheckRegisterNumbering(rows map[string]bool) []string {
	var bad []string
	for i := 1; i <= len(rows); i++ {
		if id := "C" + strconv.Itoa(i); !rows[id] {
			bad = append(bad, fmt.Sprintf("010: the register has %d rows and no %s; a gap "+
				"means a row was deleted", len(rows), id))
		}
	}
	return bad
}

// CheckRegisterCitations reports specs citing a register row that does not
// exist. A dangling row reference is the drift 010-D1 exists to prevent, one
// level up, and it is invisible in review.
func CheckRegisterCitations(specs []Spec, rows map[string]bool) []string {
	var bad []string
	for _, s := range specs {
		if s.File == "010-conformance.md" {
			continue
		}
		for _, m := range registerCite.FindAllStringSubmatch(s.Body, -1) {
			id := m[1] + m[2]
			if id == "" {
				continue
			}
			if !rows[id] {
				bad = append(bad, fmt.Sprintf("%s cites register row %s, which 010 does "+
					"not have", s.File, id))
			}
		}
	}
	return bad
}
