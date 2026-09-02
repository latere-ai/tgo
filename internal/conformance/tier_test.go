// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"errors"
	"strings"
	"testing"
)

// TestDecideCoversEveryTier is the reason decide takes availability as data.
//
// The three branches that matter -- Metal absent and not required, Metal absent
// and required, weights absent -- cannot all be reached on one machine: CI has
// no device and no checkpoint, a laptop has a device and no requirement, and
// the branch deciding whether ci-metal.yml can rot green is reachable on
// neither. So the rule is a pure function and this is the test of it.
func TestDecideCoversEveryTier(t *testing.T) {
	for _, c := range []struct {
		name         string
		tier         Tier
		require      bool
		metal        bool
		model        string
		want         action
		wantContains string
	}{
		{"tier 1 always runs", Tier1, false, false, "", run, ""},
		{"tier 1 ignores the metal requirement", Tier1, true, false, "", run, ""},
		{"tier 2 runs where a device is present", Tier2, false, true, "", run, ""},
		{"tier 2 runs when required and present", Tier2, true, true, "", run, ""},
		{"tier 2 skips where none is promised", Tier2, false, false, "", skip, EnvRequireMetal},

		// The row this whole design exists for: specs/010-conformance.md §4
		// makes a promised backend with no device a FAILURE. A skip here is
		// how a job that claims Metal reports green forever.
		{"tier 2 FAILS where one is promised", Tier2, true, false, "", fail, "failure rather"},

		{"tier 3 skips with no checkpoint", Tier3, false, false, "", skip, EnvModel},
		{"tier 3 runs with one", Tier3, false, false, "/models/qwen3", run, ""},
		{"an unknown tier fails rather than running", Tier(9), false, false, "", fail, "three tiers"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, why := decide(c.tier, c.require, c.metal, c.model)
			if got != c.want {
				t.Fatalf("decide(%v, require=%v, metal=%v, model=%q) = %v, want %v (%s)",
					c.tier, c.require, c.metal, c.model, got, c.want, why)
			}
			if c.wantContains != "" && !strings.Contains(why, c.wantContains) {
				t.Errorf("reason %q does not mention %q; a skip or a failure that does "+
					"not say what would change it sends a reader looking",
					why, c.wantContains)
			}
			if c.want == run && why != "" {
				t.Errorf("a running tier carries the reason %q, which reads as a problem", why)
			}
			if c.want != run && why == "" {
				t.Error("a skip or failure with no reason is indistinguishable from a bug")
			}
		})
	}
}

// TestRequireMetalReadsAnyTruthyValue pins the spelling rule.
//
// A variable silently ignored because its value was not the one spelling the
// code accepts is the same rot the variable exists to prevent: a reader who
// sets it to "yes" meant yes.
func TestRequireMetalReadsAnyTruthyValue(t *testing.T) {
	for _, c := range []struct {
		set  string
		want bool
	}{
		{"", false}, {"0", false}, {"false", false},
		{"1", true}, {"true", true}, {"yes", true}, {"on", true},
	} {
		t.Setenv(EnvRequireMetal, c.set)
		if got := requireMetal(); got != c.want {
			t.Errorf("%s=%q: requireMetal() = %v, want %v", EnvRequireMetal, c.set, got, c.want)
		}
	}
}

// TestTierString keeps the names a diagnostic prints honest.
func TestTierString(t *testing.T) {
	for _, c := range []struct {
		tier Tier
		want string
	}{
		{Tier1, "1"}, {Tier2, "2"}, {Tier3, "3"},
	} {
		if got := c.tier.String(); !strings.Contains(got, c.want) {
			t.Errorf("Tier(%d).String() = %q, want it to mention %q", c.tier, got, c.want)
		}
	}
	if got := Tier(9).String(); got == "" {
		t.Error("an unknown tier stringifies to nothing, so a diagnostic naming it says nothing")
	}
}

// TestDeviceAtTier1 is the one branch that runs everywhere: the CPU backend is
// always present, so tier 1 must return a usable device rather than skipping.
func TestDeviceAtTier1(t *testing.T) {
	dev := Device(t, Tier1)
	if dev == nil {
		t.Fatal("Device(t, Tier1) returned nil; the CPU backend is always available")
	}
	if dev.Queue() == nil {
		t.Error("the tier 1 device has no queue; a device that cannot be " +
			"submitted to is not usable")
	}
}

// TestModelPathSkipsWithoutACheckpoint pins 010-D4: tier 3 never runs in CI.
func TestModelPathSkipsWithoutACheckpoint(t *testing.T) {
	t.Setenv(EnvModel, "")
	fake := &recordingTB{TB: t}
	func() {
		defer func() {
			if p := recover(); p != nil && !isAbort(p) {
				panic(p)
			}
		}()
		ModelPath(fake)
	}()
	if !fake.skipped {
		t.Errorf("ModelPath with %s unset did not skip; tier 3 would run in CI", EnvModel)
	}
}

// errAbort unwinds a recording TB the way t.Skip unwinds a real one. Go 1.27
// makes panic(nil) a runtime error, so the sentinel is explicit.
var errAbort = errors.New("conformance: test aborted by a recording TB")

// isAbort reports whether a recovered value is that sentinel. recover returns
// any, so the value is matched as an error rather than compared directly:
// anything else is a real panic and is re-raised.
func isAbort(p any) bool {
	err, ok := p.(error)
	return ok && errors.Is(err, errAbort)
}

// recordingTB captures whether a helper skipped, without skipping the real test.
type recordingTB struct {
	testing.TB
	skipped bool
}

func (r *recordingTB) Skip(args ...any)                 { r.skipped = true; panic(errAbort) }
func (r *recordingTB) Skipf(format string, args ...any) { r.skipped = true; panic(errAbort) }
func (r *recordingTB) SkipNow()                         { r.skipped = true; panic(errAbort) }
func (r *recordingTB) Helper()                          {}
