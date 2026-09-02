// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package oracle

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// tolULP is a few units in the last place of a float64.
//
// specs/010-conformance.md §5.1 derives a tolerance from the terms that
// produce it. Nothing in this package stores a value narrower than a float64,
// so the f16 storage term and the f32 accumulation term are both absent and
// the only term left is double rounding, eps_64 = 2^-53 ~ 1.1e-16. A handful
// of ULP covers the two or three roundings between an input and a result.
const tolULP = 1e-15

// mustPanic runs f and fails unless it panics with a message containing want.
// Every refusal in this package is a panic, and specs/010-conformance.md
// requires each refusal to be a test.
//
// The message is asserted, not just the panic. Without a guard, most of these
// calls still panic — out of a slice bound, one or two statements later — so a
// test that accepted any panic passed with the refusal deleted. That makes the
// refusal untested while statement coverage stays at 100%, which is the
// accept-and-silently-wrong class of specs/010-conformance.md §2 wearing a
// green test as a disguise.
func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("no panic; wanted one mentioning %q", want)
			return
		}
		if got := fmt.Sprint(r); !strings.Contains(got, want) {
			t.Errorf("panicked with %q, want a message containing %q", got, want)
		}
	}()
	f()
}

// closeTo reports whether got and want agree to within tol, relative to want
// where want is large enough for a relative comparison to mean anything.
func closeTo(got, want, tol float64) bool {
	d := math.Abs(got - want)
	if scale := math.Abs(want); scale > 1 {
		return d <= tol*scale
	}
	return d <= tol
}

func checkSlice(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if !closeTo(got[i], want[i], tol) {
			t.Errorf("%s[%d] = %.17g, want %.17g", name, i, got[i], want[i])
		}
	}
}

func TestRMSNormHandComputed(t *testing.T) {
	// x = [1,2,3,4]: mean(x^2) = 30/4 = 7.5, sqrt(7.5) = 2.7386127875258306,
	// and the gain scales each channel afterwards. The expected values are
	// computed outside this package from that arithmetic.
	x := []float64{1, 2, 3, 4}
	gain := []float64{1, 2, 0.5, -1}
	want := []float64{0.3651483716701107, 1.4605934866804429, 0.5477225575051661, -1.4605934866804429}
	checkSlice(t, "RMSNorm", RMSNorm(x, gain, 0), want, tolULP)
}

func TestRMSNormEpsIsInsideTheRoot(t *testing.T) {
	// eps sits under the square root with the mean, so it shrinks every
	// channel slightly rather than acting as a floor on the result
	// (specs/004-model-graph.md §2.2).
	x := []float64{1, 2, 3, 4}
	gain := []float64{1, 1, 1, 1}
	want := []float64{0.3651483473268884, 0.7302966946537768, 1.0954450419806652, 1.4605933893075536}
	checkSlice(t, "RMSNorm", RMSNorm(x, gain, 1e-6), want, tolULP)
}

func TestRMSNormNormalizesEachRowIndependently(t *testing.T) {
	// The row count is inferred from len(x)/len(gain), which is what makes
	// Qwen3's per-head QK-norm a reshape (specs/004-model-graph.md §2.4).
	// Normalizing [2, 2] with a width-2 gain must give each row unit RMS
	// separately; normalizing the same eight values as one row of width 4
	// would mix the two heads, which is the bug §2.4 names.
	x := []float64{1, 1, 100, 100}
	gain := []float64{1, 1}
	got := RMSNorm(x, gain, 0)
	want := []float64{1, 1, 1, 1}
	checkSlice(t, "per-head RMSNorm", got, want, tolULP)

	wide := RMSNorm(x, []float64{1, 1, 1, 1}, 0)
	if closeTo(wide[0], 1, 1e-3) {
		t.Errorf("normalizing over the full row gave %.17g for the small channel; the two axes are indistinguishable", wide[0])
	}
}

func TestRMSNormRefusals(t *testing.T) {
	mustPanic(t, "RMSNorm gain is empty", func() { RMSNorm([]float64{1}, nil, 0) })
	mustPanic(t, "RMSNorm x length", func() { RMSNorm([]float64{1, 2, 3}, []float64{1, 1}, 0) })
}

func TestMatMulHandComputed(t *testing.T) {
	// [[1,2,3],[4,5,6]] x [[1,2],[3,4],[5,6]] = [[22,28],[49,64]].
	x := []float64{1, 2, 3, 4, 5, 6}
	w := []float64{1, 2, 3, 4, 5, 6}
	got := MatMul(x, w, 2, 3, 2)
	checkSlice(t, "MatMul", got, []float64{22, 28, 49, 64}, tolULP)
}

func TestMatMulIdentity(t *testing.T) {
	const n = 5
	id := make([]float64, n*n)
	for i := range n {
		id[i*n+i] = 1
	}
	x := []float64{1, -2, 3.5, 0, 7}
	checkSlice(t, "MatMul identity", MatMul(x, id, 1, n, n), x, tolULP)
}

func TestMatMulRowVector(t *testing.T) {
	// M=1 is every decode step (specs/004-model-graph.md §2.1), so it gets its
	// own case rather than being covered only as a degenerate GEMM.
	x := []float64{2, 3}
	w := []float64{1, 10, 100, 1, 10, 100}
	checkSlice(t, "MatMul M=1", MatMul(x, w, 1, 2, 3), []float64{5, 50, 500}, tolULP)
}

func TestMatMulRefusals(t *testing.T) {
	// One case per disjunct of the guard: statement coverage is satisfied by
	// the first one alone, so m, k and n are each exercised.
	mustPanic(t, "MatMul has a negative dimension", func() { MatMul(nil, nil, -1, 1, 1) })
	mustPanic(t, "MatMul has a negative dimension", func() { MatMul(nil, nil, 1, -1, 1) })
	mustPanic(t, "MatMul has a negative dimension", func() { MatMul(nil, nil, 1, 1, -1) })
	// Each length check is exercised from both sides. A too-short operand
	// panics out of a slice bound with or without the guard, so only the
	// too-long case proves the check is an equality rather than a minimum —
	// and an operand longer than its declared shape is a caller that has
	// mistaken the shape, not one with slack to spare.
	mustPanic(t, "MatMul x length", func() { MatMul([]float64{1}, []float64{1}, 2, 1, 1) })
	mustPanic(t, "MatMul x length", func() { MatMul([]float64{1, 2, 3}, []float64{1}, 2, 1, 1) })
	mustPanic(t, "MatMul w length", func() { MatMul([]float64{1, 2}, []float64{1}, 2, 1, 2) })
	mustPanic(t, "MatMul w length", func() { MatMul([]float64{1, 2}, []float64{1, 2, 3}, 2, 1, 2) })
}

func TestSiLU(t *testing.T) {
	got := SiLU([]float64{0, 1, -2})
	// SiLU(0) = 0 exactly; the other two are x/(1+exp(-x)) computed outside
	// this package.
	checkSlice(t, "SiLU", got, []float64{0, 0.7310585786300049, -0.2384058440442351}, tolULP)
}

func TestSiLUSaturates(t *testing.T) {
	// The large-magnitude limits: SiLU(x) -> x for x >> 0 and -> 0 for x << 0.
	// The negative side is where a naive x*sigmoid(x) would produce NaN from
	// an overflowed exponential, so it is asserted rather than assumed.
	got := SiLU([]float64{800, -800})
	if got[0] != 800 {
		t.Errorf("SiLU(800) = %v, want 800", got[0])
	}
	if got[1] != 0 {
		t.Errorf("SiLU(-800) = %v, want 0", got[1])
	}
}

func TestSwiGLUIsSiLUTimesUp(t *testing.T) {
	gate := []float64{0, 1, -2, 3}
	up := []float64{5, -1, 2, 0.5}
	got := SwiGLU(gate, up)
	silu := SiLU(gate)
	want := make([]float64, len(up))
	for i := range up {
		want[i] = silu[i] * up[i]
	}
	checkSlice(t, "SwiGLU", got, want, tolULP)
}

func TestSwiGLUDoesNotAliasItsInput(t *testing.T) {
	gate := []float64{1, 2}
	up := []float64{1, 1}
	_ = SwiGLU(gate, up)
	if gate[0] != 1 || gate[1] != 2 {
		t.Errorf("SwiGLU modified gate: %v", gate)
	}
}

func TestSwiGLURefusal(t *testing.T) {
	mustPanic(t, "SwiGLU gate and up lengths differ", func() { SwiGLU([]float64{1}, []float64{1, 2}) })
}

func TestSoftmaxSumsToOne(t *testing.T) {
	x := []float64{1, 2, 3}
	got := Softmax(x)
	checkSlice(t, "Softmax", got, []float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748218}, tolULP)
	sum := 0.0
	for _, v := range got {
		sum += v
	}
	// The sum of n normalized terms carries n roundings from the division and
	// n-1 from this accumulation: n*eps_64 for n = 3.
	if !closeTo(sum, 1, 3*tolULP) {
		t.Errorf("sum = %.17g, want 1", sum)
	}
}

func TestSoftmaxIsShiftInvariant(t *testing.T) {
	x := []float64{-1, 0.5, 2, 7}
	const shift = 3.0
	shifted := make([]float64, len(x))
	for i, v := range x {
		shifted[i] = v + shift
	}
	// Adding a constant is exact in the algebra but not in float64: x_i+c and
	// max+c each round, so the difference carries |c|*eps_64 into the
	// exponent, amplified by exp'. With |c| = 3 that is a few ULP, hence the
	// same tolULP scale.
	checkSlice(t, "Softmax shift", Softmax(shifted), Softmax(x), 8*tolULP)
}

func TestSoftmaxSurvivesLargeInputs(t *testing.T) {
	// Without the max subtraction exp(1000) is +Inf and every term becomes
	// NaN. specs/004-model-graph.md §2.4's scores are unbounded in principle.
	got := Softmax([]float64{1000, 1000, 1000})
	for i, v := range got {
		if !closeTo(v, 1.0/3.0, tolULP) {
			t.Errorf("Softmax[%d] = %v, want 1/3", i, v)
		}
	}
}

func TestSoftmaxSingleton(t *testing.T) {
	got := Softmax([]float64{-42})
	if got[0] != 1 {
		t.Errorf("Softmax of one score = %v, want exactly 1", got[0])
	}
}

func TestSoftmaxRefusal(t *testing.T) {
	mustPanic(t, "Softmax of an empty vector", func() { Softmax(nil) })
}

func TestGatherRows(t *testing.T) {
	table := []float64{
		0, 1,
		2, 3,
		4, 5,
	}
	got := GatherRows(table, 3, 2, []int{2, 0, 2})
	checkSlice(t, "GatherRows", got, []float64{4, 5, 0, 1, 4, 5}, 0)
}

func TestGatherRowsEmptyIDs(t *testing.T) {
	if got := GatherRows([]float64{1, 2}, 2, 1, nil); len(got) != 0 {
		t.Errorf("len = %d, want 0", len(got))
	}
}

func TestGatherRowsCopiesTheTable(t *testing.T) {
	table := []float64{1, 2}
	got := GatherRows(table, 1, 2, []int{0})
	got[0] = 99
	if table[0] != 1 {
		t.Errorf("GatherRows returned a view into the table: %v", table)
	}
}

func TestGatherRowsRefusals(t *testing.T) {
	mustPanic(t, "GatherRows has a negative dimension", func() { GatherRows(nil, -1, 1, nil) })
	mustPanic(t, "GatherRows has a negative dimension", func() { GatherRows(nil, 1, -1, nil) })
	mustPanic(t, "GatherRows table length", func() { GatherRows([]float64{1, 2, 3}, 2, 2, nil) })
	mustPanic(t, "GatherRows table length", func() { GatherRows([]float64{1, 2, 3, 4, 5}, 2, 2, nil) })
	mustPanic(t, "GatherRows id is out of range", func() { GatherRows([]float64{1, 2}, 2, 1, []int{2}) })
	mustPanic(t, "GatherRows id is out of range", func() { GatherRows([]float64{1, 2}, 2, 1, []int{-1}) })
}
