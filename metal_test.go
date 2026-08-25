// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"testing"

	"github.com/latere-ai/tgo/internal/conformance"
)

// TestMetalRunsTheForwardPass is the positive form of a test that used to
// assert the opposite, and the history is the point.
//
// specs/004-model-graph.md §3.2 slices the last row before the LM head and
// packs it, because accel refuses a strided operand into MatMul rather than
// copying behind the caller's back. That packing kernel was the only one in
// accel's corpus with no MSL artifact, so **every** tgo graph — synthetic or
// real, prefill or decode — was refused at compile time on Metal:
//
//	kernel Pack carries no MSL artifact, so it cannot run on Metal
//
// tgo filed it as accel#19 and accel lowered it the same day, so the device
// this framework exists to be fast on runs it now. The test that pinned the
// gap said to delete itself when that happened; this replaced it rather than
// vanishing, because a backend that worked once and quietly stopped is exactly
// what specs/010-conformance.md §4's tier 2 exists to catch.
func TestMetalRunsTheForwardPass(t *testing.T) {
	// The tier rule decides whether this machine runs tier 2 at all
	// (specs/010-conformance.md §4): a skip where no device is present, and a
	// failure where TGO_REQUIRE_METAL promises one.
	_ = conformance.Device(t, conformance.Tier2)

	dir := checkpoint{tie: true}.write(t)
	// Through openAt so the model's close is registered before any session's:
	// cleanups run last-in-first-out, and accel closes in order rather than
	// recursively, so a device closed while its buffers live refuses.
	m := openAt(t, dir, WithDevice(Metal))
	s := session(t, m, WithSessionContext(64))

	st, err := s.Complete(t.Context(), "hello", greedy(2))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	got, events := collect(t, st)
	if err := st.Err(); err != nil {
		t.Fatalf("Metal refused a forward pass: %v", err)
	}
	if got == "" || len(events) == 0 {
		t.Errorf("Metal produced %d event(s) and %q; a stream that ends clean with "+
			"nothing in it is indistinguishable from one that never ran",
			len(events), got)
	}
	// The session survives, which is what separates a working backend from one
	// that happens not to have errored yet.
	if err := s.usable(); err != nil {
		t.Errorf("a completed generation left the session unusable: %v", err)
	}
}
