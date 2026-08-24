// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// The environment specs/010-conformance.md §4 reads.
const (
	// EnvRequireMetal turns a missing Metal device from a skip into a
	// failure. It is the mechanism accel uses and the reason is the same: a
	// job that promises a backend and skips when it finds none rots green.
	EnvRequireMetal = "TGO_REQUIRE_METAL"

	// EnvModel is the directory a real checkpoint sits in. Unset in CI,
	// always (000 D8, 010-D4).
	EnvModel = "TGO_MODEL"
)

// Tier is which of specs/010-conformance.md §4's three tiers a test belongs to.
//
// The tier is a property of the test, not of the machine: it says what the test
// needs, and [Device] decides what to do when the machine does not have it.
type Tier int

const (
	// Tier1 needs nothing. The CPU backend is a first-class accel backend
	// that is always present, so a tier 1 test has no skip path at all.
	Tier1 Tier = iota + 1

	// Tier2 needs a Metal device.
	Tier2

	// Tier3 needs real weights under EnvModel and never runs in CI.
	Tier3
)

func (t Tier) String() string {
	switch t {
	case Tier1:
		return "tier 1"
	case Tier2:
		return "tier 2"
	case Tier3:
		return "tier 3"
	}
	return fmt.Sprintf("tier %d", int(t))
}

// action is what a tier's requirements say to do with a test.
type action int

const (
	run action = iota
	skip
	fail
)

// decide reports what to do with a test of this tier, and why.
//
// It is a pure function of the tier and of what is available because the three
// interesting branches -- Metal absent and not required, Metal absent and
// required, weights absent -- cannot all be reached on any one machine. CI has
// no Metal device and no checkpoint, a laptop has a device and no requirement,
// and the branch that decides whether ci-metal.yml can rot is reachable on
// neither. Taking availability as data is what lets every branch be a test.
func decide(tier Tier, requireMetal, metalPresent bool, model string) (action, string) {
	switch tier {
	case Tier1:
		return run, ""
	case Tier2:
		if metalPresent {
			return run, ""
		}
		if requireMetal {
			return fail, EnvRequireMetal + " is set and this machine has no Metal " +
				"device. specs/010-conformance.md §4 makes that a failure rather " +
				"than a skip: a job that promises a backend and skips when it finds " +
				"none reports green forever."
		}
		return skip, "no Metal device; set " + EnvRequireMetal + "=1 where one is " +
			"promised, which turns this skip into a failure"
	case Tier3:
		if model == "" {
			return skip, EnvModel + " is unset, so there is no checkpoint to run. " +
				"Tier 3 is run by hand before a release and is never in CI " +
				"(specs/010-conformance.md 010-D4)."
		}
		return run, ""
	}
	return fail, fmt.Sprintf("%v is not one of the three tiers of "+
		"specs/010-conformance.md §4", tier)
}

// requireMetal reports whether EnvRequireMetal asks for a device.
//
// Any value other than the empty string, "0" and "false" requires one. CI sets
// it to 1; a reader who sets it to "yes" meant yes, and a variable that is
// silently ignored because its value was not the one spelling the code accepts
// is the same rot the variable exists to prevent.
func requireMetal() bool {
	v := os.Getenv(EnvRequireMetal)
	return v != "" && v != "0" && !strings.EqualFold(v, "false")
}

// metalPresent reports whether this machine has a Metal device.
//
// Enumerate rather than a build tag: darwin/arm64 without a GPU-capable
// runner is a real configuration, and a build tag would answer the question
// the platform asks rather than the one the test asks.
func metalPresent() bool {
	for _, d := range accel.Enumerate().Devices {
		if d.Backend == accel.BackendMetal {
			return true
		}
	}
	return false
}

// Device opens the device this tier runs on, or skips or fails the test.
//
// The returned device is closed by a cleanup, so a caller never closes it: a
// test that closed it early would be closing a device the runtime still holds.
//
// Tier 1 is the CPU backend, which is the backend a parity test wants anyway --
// a block is a claim about what accel computes, and only accel can settle it.
// Tier 2 is Metal and nothing else: accel's OpenBest never selects the CPU
// backend unless asked, so a tier 2 test cannot silently become a tier 1 test.
// Tier 3 takes the best device that exists including the CPU one, because the
// checkpoint is the requirement there and the backend is not.
func Device(tb testing.TB, tier Tier) *accel.Device {
	tb.Helper()
	switch act, why := decide(tier, requireMetal(), metalPresent(), os.Getenv(EnvModel)); act {
	case skip:
		tb.Skip(why)
	case fail:
		tb.Fatal(why)
	}
	dev, err := open(tier)
	if err != nil {
		tb.Fatalf("open a device for %v: %v", tier, err)
	}
	tb.Cleanup(func() {
		if err := dev.Close(); err != nil {
			tb.Errorf("device close: %v", err)
		}
	})
	return dev
}

// open opens the device a tier runs on. It is separate from [Device] so that
// what cannot be covered on a machine without a GPU is three lines rather than
// the whole rule.
func open(tier Tier) (*accel.Device, error) {
	switch tier {
	case Tier2:
		return accel.OpenBest(accel.Policy{Prefer: []accel.Backend{accel.BackendMetal}})
	case Tier3:
		return accel.OpenBest(accel.Policy{AllowCPU: true})
	}
	return accel.OpenCPU(accel.CPUOptions{})
}

// ModelPath returns the checkpoint directory of a tier 3 test, or skips it.
//
// A path that is set and does not exist fails instead of skipping. Someone who
// exported EnvModel asked for tier 3, and a typo in the path that silently
// skipped would report the release gate as run.
func ModelPath(tb testing.TB) string {
	tb.Helper()
	path := os.Getenv(EnvModel)
	if act, why := decide(Tier3, false, false, path); act == skip {
		tb.Skip(why)
	}
	info, err := os.Stat(path)
	if err != nil {
		tb.Fatalf("%s is %q, which cannot be read: %v", EnvModel, path, err)
	}
	if !info.IsDir() {
		tb.Fatalf("%s is %q, which is not a directory; it names the directory a "+
			"checkpoint's safetensors and config sit in", EnvModel, path)
	}
	return path
}
