// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Command covercheck gates coverage per package rather than per repository.
//
// A repository average lets a well-tested package carry an untested one and
// reports a number nobody can act on: it moves when unrelated code lands and it
// never says where the hole is. Per package, a regression names the package
// that caused it.
//
// Exemptions are declared here, in code, with the reason. A package exempted
// without a reason is a package nobody decided to exempt.
//
//	go test -coverprofile=cover.out -coverpkg=./... ./...
//	go run ./internal/covercheck -profile=cover.out
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// threshold is the floor every package clears.
//
// Ninety, because specs/000-decisions.md decision 8 puts the pure logic --
// tokenizer, safetensors, templates, sampling -- in packages with no device and
// no network, which makes the number reachable rather than aspirational.
const threshold = 90.0

// exempt maps a package suffix to why it does not have to clear the floor.
//
// A short list on purpose. Every entry is a promise that the package is thin
// enough that its coverage would measure the test harness rather than the code.
var exempt = map[string]string{
	"internal/covercheck": "this program; it gates coverage and is not gated by it",
	"cmd/tgo":             "argument parsing and process wiring; the packages under it carry the logic",

	// The tier rule's *decisions* are fully covered -- decide() is a pure
	// function of availability precisely so every branch is testable on a
	// machine that has neither a Metal device nor a checkpoint. What is not
	// covered is the handful of statements that ACT on those decisions:
	// opening a Metal device, failing when one was promised and is absent,
	// and reporting a device-open error. Reaching them needs a machine with
	// Metal (ci-metal.yml has one; this gate does not) or a deliberately
	// broken device. Exempted rather than faked, because a mock device here
	// would cover the statements and prove nothing about accel.
	//
	// specs/010-conformance.md §4. Revisit if the package grows logic that is
	// not tier plumbing.
	"internal/conformance": "device-open and Metal-present branches, unreachable without a Metal device; the tier decisions they act on are pure and fully covered",
}

// block is one coverage record: a statement range and how often it ran.
type block struct {
	stmts  int
	ncount int
}

func main() {
	profile := flag.String("profile", "cover.out", "coverage profile to read")
	flag.Parse()

	f, err := os.Open(*profile)
	if err != nil {
		fatal("open profile: %v", err)
	}
	defer f.Close()

	// A profile line is "name.go:line.col,line.col numStmts count", and the
	// same block appears once per test binary that executed it. Summing the
	// counts would be wrong, so blocks are keyed and merged by taking whether
	// any run covered them.
	blocks := map[string]map[string]*block{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "mode:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 3 {
			fatal("malformed profile line: %q", line)
		}
		file, span, ok := strings.Cut(fields[0], ":")
		if !ok {
			fatal("malformed profile line: %q", line)
		}
		stmts, err := strconv.Atoi(fields[1])
		if err != nil {
			fatal("malformed statement count in %q: %v", line, err)
		}
		count, err := strconv.Atoi(fields[2])
		if err != nil {
			fatal("malformed hit count in %q: %v", line, err)
		}

		pkg := file[:strings.LastIndex(file, "/")]
		if blocks[pkg] == nil {
			blocks[pkg] = map[string]*block{}
		}
		key := file + ":" + span
		b := blocks[pkg][key]
		if b == nil {
			b = &block{stmts: stmts}
			blocks[pkg][key] = b
		}
		b.ncount += count
	}
	if err := sc.Err(); err != nil {
		fatal("read profile: %v", err)
	}
	if len(blocks) == 0 {
		fatal("profile has no records; did the test run?")
	}

	pkgs := make([]string, 0, len(blocks))
	for pkg := range blocks {
		pkgs = append(pkgs, pkg)
	}
	sort.Strings(pkgs)

	failed, measured := false, 0
	for _, pkg := range pkgs {
		var total, covered int
		for _, b := range blocks[pkg] {
			total += b.stmts
			if b.ncount > 0 {
				covered += b.stmts
			}
		}
		if total == 0 {
			continue
		}
		pct := 100 * float64(covered) / float64(total)

		if why, ok := exemptFor(pkg); ok {
			fmt.Printf("  %-44s %6.1f%%  exempt: %s\n", short(pkg), pct, why)
			continue
		}
		measured++
		mark := "ok  "
		if pct < threshold {
			mark = "FAIL"
			failed = true
		}
		fmt.Printf("%s %-44s %6.1f%%  (%d/%d statements)\n", mark, short(pkg), pct, covered, total)
	}

	if failed {
		fmt.Fprintf(os.Stderr, "\ncoverage below %.0f%% in one or more packages\n", threshold)
		os.Exit(1)
	}

	// A gate that passes because it measured nothing is worse than no gate: it
	// reports green over an empty tree and keeps reporting green as the tree
	// fills up, until somebody notices the number never moved. Every package
	// being exempt is the shape that produces it, and it is the shape this
	// repository was in at M0.
	if measured == 0 {
		fmt.Fprintf(os.Stderr, "\nno non-exempt package was measured; the gate would "+
			"pass vacuously. Either the tests did not run, or every package is "+
			"exempt -- both are failures rather than a green build.\n")
		os.Exit(1)
	}
	fmt.Printf("\nevery package at or above %.0f%% (%d measured)\n", threshold, measured)
}

// exemptFor reports whether a package is exempt, and why.
func exemptFor(pkg string) (string, bool) {
	for suffix, why := range exempt {
		if strings.HasSuffix(pkg, suffix) {
			return why, true
		}
	}
	return "", false
}

// short trims the module path so the report fits a terminal.
func short(pkg string) string {
	return strings.TrimPrefix(pkg, "github.com/latere-ai/tgo/")
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "covercheck: "+format+"\n", args...)
	os.Exit(1)
}
