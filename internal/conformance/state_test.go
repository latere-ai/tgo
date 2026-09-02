// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// The guarantee specs/005-kv-cache.md §1.1 rests its whole ordering argument on.
//
// A [tensor.State] is a *version* of caller-owned storage: a write returns the
// next version, and reading an earlier one is reading what was there before.
// Both versions live in one buffer, so accel cannot serve the old contents and
// refuses instead.
//
// 005 depends on that in a way no other spec does. tgo hand-orders nothing in
// the KV cache — it declares the reads and the writes and lets the version
// chain decide what runs before what. If a stale read were served the new
// contents rather than refused, the ordering would be a comment: a step that
// read the cache after its own scatter would attend to the token it had just
// written, quietly, on every layer. So this asserts the refusal by value rather
// than trusting accel's doc comment, which is 010-D7.

// stateShape is the probe's geometry: two layers of four positions, so a
// per-layer window is a strict subset of the whole and the overlap rule below
// has something to be wrong about.
var stateShape = struct{ layers, positions, width int }{2, 4, 3}

// newStateProbe builds a flat state, [positions, width]: the shape a write
// scatters rows into.
func newStateProbe(t *testing.T, label string) (*Rig, *tensor.State) {
	t.Helper()
	r := New(t, Tier1, Options{Eps: 1e-6, Label: label})
	s := tensor.NewState(r.G.B, tensor.StateDesc{
		Name: "kv", DType: accel.F32,
		Shape: tensor.Shape{stateShape.positions, stateShape.width},
	})
	return r, s
}

// newLayeredProbe builds the shape 005 §2 uses: [layers, positions, width],
// one buffer whose layers LayerState addresses as disjoint windows.
func newLayeredProbe(t *testing.T, label string) (*Rig, *tensor.State) {
	t.Helper()
	r := New(t, Tier1, Options{Eps: 1e-6, Label: label})
	s := tensor.NewState(r.G.B, tensor.StateDesc{
		Name: "kv", DType: accel.F32,
		Shape: tensor.Shape{stateShape.layers, stateShape.positions, stateShape.width},
	})
	return r, s
}

// TestAStaleStateVersionIsRefused is the probe.
func TestAStaleStateVersionIsRefused(t *testing.T) {
	r, s := newStateProbe(t, "kv-stale")

	rows := r.Input("rows", accel.F32, tensor.Shape{1, stateShape.width})
	at := r.Input("at", accel.U32, tensor.Shape{1})

	// The write returns the next version; s is now the previous one.
	next := tensor.ScatterRows(r.G.B, s, rows, at)
	if next == nil {
		t.Fatal("ScatterRows returned no version")
	}
	tensor.ReadState(r.G.B, s)

	err := r.G.Err()
	if err == nil {
		t.Fatal("reading a superseded state version recorded; 005 §1.1 argues that " +
			"tgo never hand-orders a cache write against a cache read, and that " +
			"argument is only sound because this is refused")
	}
	if !strings.Contains(err.Error(), "ReadState") {
		t.Errorf("the refusal does not name the operator: %v", err)
	}
	t.Logf("refused, as 005 §1.1 requires: %v", err)
}

// TestTheCurrentStateVersionIsReadable is the positive control.
//
// Without it the test above passes on an accel that refuses every ReadState,
// which would be a broken cache rather than a guaranteed one.
func TestTheCurrentStateVersionIsReadable(t *testing.T) {
	r, s := newStateProbe(t, "kv-current")

	rows := r.Input("rows", accel.F32, tensor.Shape{1, stateShape.width})
	at := r.Input("at", accel.U32, tensor.Shape{1})

	next := tensor.ScatterRows(r.G.B, s, rows, at)
	if out := tensor.ReadState(r.G.B, next); out == nil {
		t.Fatal("ReadState of the current version returned nothing")
	}
	if err := r.G.Err(); err != nil {
		t.Fatalf("reading the version ScatterRows returned was refused: %v", err)
	}
}

// TestAWriteToOneLayerDoesNotStaleAnother is the half of the rule 005 §2 needs,
// and accel's doc comment states it as an aside.
//
// A per-layer cache is one buffer whose layers are disjoint windows, so accel
// decides staleness by *overlap* rather than by name. Decided by name, every
// layer's read would be stale the moment any layer was written, and 005's
// design — two states sliced per layer, every layer written in one step — would
// not compile at all. tgo's cache depends on the overlap rule specifically, and
// it is the half a reader would assume rather than check.
func TestAWriteToOneLayerDoesNotStaleAnother(t *testing.T) {
	r, s := newLayeredProbe(t, "kv-layers")

	layer0 := tensor.LayerState(r.G.B, s, 0)
	layer1 := tensor.LayerState(r.G.B, s, 1)
	if layer0 == nil || layer1 == nil {
		t.Skip("this accel does not slice a state per layer")
	}

	rows := r.Input("rows", accel.F32, tensor.Shape{1, stateShape.width})
	at := r.Input("at", accel.U32, tensor.Shape{1})

	tensor.ScatterRows(r.G.B, layer0, rows, at)
	tensor.ReadState(r.G.B, layer1)

	if err := r.G.Err(); err != nil {
		t.Fatalf("a write to layer 0 made layer 1's read stale: %v\n"+
			"005 §2 slices one buffer per layer and writes every layer in one "+
			"step, so staleness has to be decided by overlap and not by name",
			err)
	}
}
