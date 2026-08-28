// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"io"
	"strings"
	"testing"
)

const serverPkg = module + "/server"

// TestTheGateFailsOnAModuleNobodyAllowed is the negative control. A gate is
// only worth its runtime if it can go red, and the failure that matters is the
// one 009-D14 names: a module appears in the build that no decision admits.
func TestTheGateFailsOnAModuleNobodyAllowed(t *testing.T) {
	// The server reaches llmdialect. Allow everything except it, which is the
	// exact shape of an upstream import arriving on a `go get`.
	allow := without(gated[serverPkg].allow, "latere.ai/x/pkg/llmdialect")

	unexpected, _, err := checkAll(serverPkg, allow)
	if err != nil {
		t.Fatalf("checkAll: %v", err)
	}
	if len(unexpected) == 0 {
		t.Fatal("dropping llmdialect from the allowed set left the gate green; " +
			"it reports nothing it is not told to report")
	}
	for _, u := range unexpected {
		if !strings.HasPrefix(u, "latere.ai/x/pkg/llmdialect") {
			t.Errorf("reported %q, which is not the module that was dropped", u)
		}
	}

	var out, errw strings.Builder
	if code := run(map[string]gate{serverPkg: {"009-D14", allow}}, false, &out, &errw); code != 1 {
		t.Errorf("exit code %d on a violation, want 1", code)
	}
	if !strings.Contains(out.String(), "FAIL") ||
		!strings.Contains(out.String(), "llmdialect") {
		t.Errorf("the report does not name what failed:\n%s", out.String())
	}
	// The decision that owns the list is named on the failing line, not in the
	// advice: two packages are gated for two different reasons, and a message
	// naming only one would misdirect half of them.
	if !strings.Contains(out.String(), "009-D14") {
		t.Errorf("the report does not name the decision the list defends:\n%s", out.String())
	}
	if !strings.Contains(errw.String(), "not an upgrade") {
		t.Errorf("the advice does not say a dependency is a decision:\n%s", errw.String())
	}
}

// TestTheShippedListIsTheOneTheBuildNeeds runs the gate as CI runs it. It fails
// when a dependency lands and nobody decided on it, and equally when the list
// keeps a prefix no platform's build reaches -- a stale allowance is a hole
// nobody notices, because it can only ever make the gate greener.
//
// "No platform" and not "this platform" is the correction the first CI run
// forced: purego is in the build on darwin, where accel loads Metal, and not on
// linux, so a host-only check called a live allowance stale and would have
// called a linux-only dependency absent.
func TestTheShippedListIsTheOneTheBuildNeeds(t *testing.T) {
	var out, errw strings.Builder
	if code := run(gated, true, &out, &errw); code != 0 {
		t.Errorf("exit code %d, want 0\nstdout:\n%s\nstderr:\n%s", code, out.String(), errw.String())
	}
	for pkg, g := range gated {
		_, matched, err := checkAll(pkg, g.allow)
		if err != nil {
			t.Fatalf("checkAll %s: %v", pkg, err)
		}
		if g.decision == "" {
			t.Errorf("%s is gated and no decision owns it; a failure would name "+
				"this file rather than the argument the list defends", short(pkg))
		}
		for _, e := range g.allow {
			if matched[e.prefix] == "" {
				t.Errorf("%s reaches %s on none of the %d platforms; the "+
					"allowance is stale, and a prefix no build uses can only "+
					"hide a module that later appears under it",
					short(pkg), e.prefix, len(platforms))
			}
			// -v prints a matched prefix with the reason someone wrote for it,
			// which is the half of the report that says why the cost is worth
			// paying.
			if !strings.Contains(out.String(), e.why) {
				t.Errorf("-v does not report why %s is allowed", e.prefix)
			}
		}
	}
}

// TestAnUnreadableBuildIsNotAViolation pins the third exit code. `go list` on a
// package that does not exist says nothing about the dependency footprint, and
// answering it with the allowlist's advice would send a reader somewhere the
// answer is not.
func TestAnUnreadableBuildIsNotAViolation(t *testing.T) {
	var errw strings.Builder
	code := run(map[string]gate{module + "/no/such/package": {"009-D14", nil}}, false, io.Discard, &errw)
	if code != 2 {
		t.Errorf("exit code %d on an unreadable build, want 2", code)
	}
	if !strings.Contains(errw.String(), "go list -deps") {
		t.Errorf("the error does not name what could not be read:\n%s", errw.String())
	}
}

func without(allow []entry, prefix string) []entry {
	out := make([]entry, 0, len(allow))
	for _, e := range allow {
		if e.prefix != prefix {
			out = append(out, e)
		}
	}
	return out
}

func TestWhatCountsAsADependency(t *testing.T) {
	for _, tc := range []struct {
		pkg  string
		want bool
		why  string
	}{
		{"fmt", false, "the standard library has no dot in its first element"},
		{"internal/abi", false, "a runtime-internal stdlib package, same rule"},
		{module, false, "this module is not a dependency of itself"},
		{module + "/sample", false, "nor is any package in it"},
		{"golang.org/x/text/unicode/norm", true, "another module"},
		{"latere.ai/x/pkg/llmdialect", true, "the module 009-D14 watches"},
	} {
		if got := external(tc.pkg); got != tc.want {
			t.Errorf("external(%q) = %v, want %v: %s", tc.pkg, got, tc.want, tc.why)
		}
	}
}

func TestAPrefixAdmitsItsSubtreeAndNothingElse(t *testing.T) {
	allow := []entry{{"latere.ai/x/pkg/llmdialect", "the three wire dialects"}}
	for _, tc := range []struct {
		pkg  string
		want string
	}{
		{"latere.ai/x/pkg/llmdialect", "latere.ai/x/pkg/llmdialect"},
		{"latere.ai/x/pkg/llmdialect/internal/sse", "latere.ai/x/pkg/llmdialect"},
		// The boundary the trailing separator exists for: a sibling module
		// whose path starts with the same bytes is a different module.
		{"latere.ai/x/pkg/llmdialectextra", ""},
		{"latere.ai/x/pkg/other", ""},
	} {
		if got := allowed(tc.pkg, allow); got != tc.want {
			t.Errorf("allowed(%q) = %q, want %q", tc.pkg, got, tc.want)
		}
	}
}

func TestTheReportNamesThePackageTheSpecDoes(t *testing.T) {
	if got := short(serverPkg); got != "./server" {
		t.Errorf("short(%q) = %q, want ./server", serverPkg, got)
	}
}
