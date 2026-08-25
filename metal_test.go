// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"strings"
	"testing"

	"github.com/latere-ai/tgo/internal/conformance"
)

// TestMetalCannotYetRunTheForwardPass pins an upstream gap rather than working
// around it.
//
// specs/004-model-graph.md §3.2 slices the last row before the LM head and
// calls tensor.Contiguous on the result, because accel refuses a strided
// operand into MatMul rather than copying behind the caller's back. That
// packing kernel carries no MSL artifact, so **every** tgo graph — synthetic or
// real, prefill or decode — is refused at compile time on Metal:
//
//	kernel Pack carries no MSL artifact, so it cannot run on Metal;
//	it is outside the subset specs/021-metal-bringup.md section 5 lowers
//
// The refusal is accel's and it is correct; what it means for tgo is that the
// device this framework exists to be fast on cannot run it at all. It is a
// register row, not a bug in this package, and there is nothing to route around
// — composing the slice out of the graph would mean running the LM head over
// every position, which is specs/004-model-graph.md 004-D4's 1.2 GB.
//
// This test asserts the *current* state, so it fails in either direction: when
// accel lowers Pack the assertion below stops holding and this test should be
// deleted along with [TestRealCheckpointEndToEnd]'s WithDevice(CPU).
func TestMetalCannotYetRunTheForwardPass(t *testing.T) {
	// The tier rule decides whether this machine runs tier 2 at all
	// (specs/010-conformance.md §4); the device it returns is closed by its own
	// cleanup and tgo opens its own.
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
	collect(t, st)
	err = st.Err()
	if err == nil {
		t.Fatal("Metal compiled a tgo forward pass. accel has lowered the packing " +
			"kernel: delete this test and the WithDevice(CPU) in the end-to-end test")
	}
	if !strings.Contains(err.Error(), "Pack") {
		t.Errorf("Metal refused with %v; the known gap is the packing kernel and this is "+
			"a different one", err)
	}
	// And the failure poisons the session, which is §7 holding for a compile
	// refusal as much as for a device fault.
	if err := s.usable(); err == nil {
		t.Error("a refused compile left the session usable")
	}
}
