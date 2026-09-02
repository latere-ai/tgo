// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	"github.com/latere-ai/tgo/nn"
	"github.com/latere-ai/tgo/weights"
)

// TestInt4WeightsRunTheSameLoop is 004-D6 at the third width: precision is a
// load-time decision and not a graph one, so the int4 path is the same loop
// with different buffers bound.
//
// Three planes where int8 has two, and the third is not decoration. At eight
// bits the codes reach far enough that a scale alone spends them well; at four
// they have to be spent where the weights actually are, so a zero point per
// group is what makes the representation usable at all.
func TestInt4WeightsRunTheSameLoop(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t, WithPrecision(Int4))
	if got := m.Info().Precision; got != Int4 {
		t.Fatalf("Precision = %v, want int4", got)
	}
	// The embedding table is *gathered*, not multiplied, and accel registers
	// no int4 gather -- QuantGatherRows reads a quant plane and a scale plane
	// and has no three-plane form. So the table is pinned to int8 while
	// everything else packs, which is what every quantizer in the ecosystem
	// does anyway: an embedding row is read once per token and never
	// contracted.
	if got := m.stored("embed"); got != nn.FormInt8 {
		t.Fatalf("the embedding is stored as %v; it is gathered rather than "+
			"multiplied and there is no int4 gather, so it is pinned to int8", got)
	}
	// A projection is the thing that packs.
	if got := m.stored("0.wq"); got != nn.FormInt4 {
		t.Fatalf("a projection is stored as %v, want int4", got)
	}

	// All three planes bound, at the counts the packing implies rather than at
	// the matrix's: eight weights to a u32 word, and a scale and a zero per
	// group.
	v, ok := m.set.Get("0.wq")
	if !ok {
		t.Fatal("the query projection did not go through the loader")
	}
	words := (v.Elements + 7) / 8
	groups := (v.Elements + quant.Int4Group - 1) / quant.Int4Group
	for _, p := range []struct {
		name  string
		want  int
		dtype accel.DType
	}{
		{"0.wq", words, accel.U32},
		{"0.wq" + nn.ScaleSuffix, groups, accel.F16},
		{"0.wq" + nn.ZeroSuffix, groups, accel.F16},
	} {
		view, ok := m.weightBind[p.name]
		if !ok {
			t.Fatalf("%q is not bound; an int4 weight is three planes and a missing "+
				"one is a matrix multiplied against another matrix's metadata", p.name)
		}
		if got := view.Count; got != p.want {
			t.Errorf("%q binds %d elements, want %d", p.name, got, p.want)
		}
		if got := view.DType; got != p.dtype {
			t.Errorf("%q is %v, want %v", p.name, got, p.dtype)
		}
	}

	// And it generates. The numerics are bounded where the bound is derivable
	// -- internal/conformance's TestC21Int4IsRepresentableAndComputes, against
	// accel's own quant.Int4ErrorBound -- because a whole forward pass has no
	// closed form for its quantization term and 010-D3 refuses a tolerance
	// somebody tuned until it passed.
	s := session(t, m, WithSessionContext(64))
	st, err := s.Complete(t.Context(), "packed", greedy(4))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	collect(t, st)
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := st.Usage().CompletionTokens; got != 4 {
		t.Fatalf("generated %d tokens, want 4", got)
	}
}

// TestBothInt4MetadataPlanesAreRead is the wiring assertion, and it is stated
// as something that must *change* rather than something that must agree.
//
// The first draft compared int4's tokens against f16's and required them to
// match, on the premise that a fixture this small quantizes closely enough for
// the choice not to move. That premise is false: the fixture's weights are
// multiples of 1/8 over a range of 4, so a 4-bit step is 4/15 and the error is
// real. The tokens diverge for a correct implementation, and a test asserting
// otherwise measures the fixture.
//
// What *is* checkable without a bound: the scales and the zeros are both f16
// and both one per group, so binding them the wrong way round compiles and
// runs. If swapping them changed nothing, neither plane would be reaching the
// kernel -- which is exactly the failure bundling the triple exists to prevent.
//
// The numerics live where the bound is derivable, in
// internal/conformance/TestInt4MatMulMatchesItsReconstruction, against accel's
// own quant.Int4ErrorBound.
func TestBothInt4MetadataPlanesAreRead(t *testing.T) {
	t.Parallel()
	dir := checkpoint{tie: true}.write(t)

	pick := func(m *Model) []int {
		s := session(t, m, WithSessionContext(64))
		st, err := s.Complete(t.Context(), "packed", greedy(4))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		collect(t, st)
		if err := st.Err(); err != nil {
			t.Fatalf("stream: %v", err)
		}
		return append([]int(nil), s.history[len(s.history)-4:]...)
	}

	straight := pick(openAt(t, dir, WithPrecision(Int4)))

	swapped := openAt(t, dir, WithPrecision(Int4))
	crossed := 0
	for _, name := range swapped.set.Names() {
		v, _ := swapped.set.Get(name)
		if v.Zeros == nil {
			continue
		}
		sc, z := swapped.weightBind[name+nn.ScaleSuffix], swapped.weightBind[name+nn.ZeroSuffix]
		swapped.weightBind[name+nn.ScaleSuffix] = z
		swapped.weightBind[name+nn.ZeroSuffix] = sc
		crossed++
	}
	if crossed == 0 {
		t.Fatal("no weight carried a zero plane, so nothing was swapped and this " +
			"test asserts nothing")
	}
	if equalInts(pick(swapped), straight) {
		t.Fatalf("binding %d weights' scale planes as their zero planes and back "+
			"produced the same tokens; the two are the same dtype and the same "+
			"length, so if swapping them changes nothing then neither is read",
			crossed)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestAutoNeverPrefersInt4ToInt8 is the rule that keeps a memory format from
// becoming an accuracy decision nobody made.
//
// accel's own tests show int4 beating int8 on a group of weights clustered away
// from zero and losing on one centred on it. It is not uniformly better, so
// auto reaches for it only where int8 does not fit and the alternative is not
// loading at all.
func TestAutoNeverPrefersInt4ToInt8(t *testing.T) {
	t.Parallel()
	dir := checkpoint{tie: true}.write(t)
	m := openAt(t, dir)
	if got := m.Info().Precision; got == Int4 {
		t.Fatalf("auto chose int4 for a model that fits at %v; narrowing is a last "+
			"resort and int4 is not uniformly more accurate than int8", got)
	}
	// Precision(0) is Inherit; naming it is what the String method is for.
	if got := weights.Precision(0).String(); got != "inherit" {
		t.Errorf("weights.Precision(0) names itself %q, want \"inherit\"", got)
	}
	if got := weights.Int4.String(); !strings.Contains(got, "int4") {
		t.Errorf("weights.Int4 names itself %q", got)
	}
}
