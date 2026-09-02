// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"math"
	"strings"
	"testing"
)

// TestTermsCannotBeWrittenAsANumber is 010-D3 enforced by the type rather than
// by discipline.
//
// A tolerance that had to be raised to make a test pass is a finding, not a
// fix. The way to make that true in code is to leave no way to write the number
// down: a caller composes the stages its computation actually has, and the
// bound falls out. This test pins the property that makes it work — every
// constructor is monotone in its input and every composition only grows.
func TestTermsCannotBeWrittenAsANumber(t *testing.T) {
	// Composing is monotone: adding a stage never shrinks a budget, so nothing
	// can be widened by removing a term and nothing narrowed by adding one.
	base := AccumF32(64)
	wider := base.And(StoreF16(2))
	if wider.Relative() < base.Relative() {
		t.Errorf("adding an f16 storage term shrank the relative budget: %g -> %g",
			base.Relative(), wider.Relative())
	}

	// The zero value asserts exact equality, which is a real claim about an
	// operation that only moves bytes.
	var exact Terms
	if got := exact.Bound(1000); got != 0 {
		t.Errorf("the zero Terms bounds at %g, want 0; an empty budget must mean "+
			"exact, not unconstrained", got)
	}
}

// TestAccumF32GrowsAsSqrtK checks §5.1's accumulation term against its formula
// rather than against a remembered number.
func TestAccumF32GrowsAsSqrtK(t *testing.T) {
	for _, k := range []int{1, 4, 64, 2560} {
		want := math.Sqrt(float64(k)) * Eps32
		if got := AccumF32(k).Relative(); math.Abs(got-want) > 1e-18 {
			t.Errorf("AccumF32(%d) = %g, want sqrt(%d)·eps32 = %g", k, got, k, want)
		}
	}
	// Quadrupling k doubles the term, which is the shape of the claim.
	if a, b := AccumF32(64).Relative(), AccumF32(256).Relative(); math.Abs(b-2*a) > 1e-18 {
		t.Errorf("AccumF32(256) = %g, want twice AccumF32(64) = %g", b, 2*a)
	}
}

// TestStoreF16DominatesAccumulation is the finding §5.1 states and the reason
// the two terms are separate: the tolerance on a matmul is set by the storage
// format, not by the accumulator, by three orders of magnitude.
func TestStoreF16DominatesAccumulation(t *testing.T) {
	const k = 2560 // Qwen3-4B's hidden size
	accum := AccumF32(k).Relative()
	store := StoreF16(1).Relative()
	if store <= accum {
		t.Fatalf("f16 storage %g does not exceed f32 accumulation over %d terms %g; "+
			"§5.1's claim that storage dominates would be wrong", store, k, accum)
	}
	if ratio := store / accum; ratio < 100 {
		t.Errorf("storage exceeds accumulation by only %.0fx at k=%d; §5.1 says three "+
			"orders of magnitude", ratio, k)
	}
}

// TestBoundAppliesRelativeToMagnitudeAndAbsoluteFlat pins the split.
//
// A quantization bound is an absolute statement about the inputs that produced
// it, so it must not scale with the value being compared; an accumulation error
// is relative, so it must.
func TestBoundAppliesRelativeToMagnitudeAndAbsoluteFlat(t *testing.T) {
	rel := AccumF32(100)
	if small, large := rel.Bound(1), rel.Bound(1000); large <= small {
		t.Errorf("a relative budget did not grow with magnitude: %g at 1, %g at 1000",
			small, large)
	}
	abs := QuantInt8(0.25)
	if small, large := abs.Bound(1), abs.Bound(1000); math.Abs(large-small) > 1e-12 {
		t.Errorf("an absolute budget scaled with magnitude: %g at 1, %g at 1000; "+
			"a quantization bound is a statement about the inputs, not the output",
			small, large)
	}
}

// TestExplainNamesEveryTerm is 010-D3's other half: a tolerance must carry the
// term that produced it, so a reader can tell a derived budget from a tuned one.
func TestExplainNamesEveryTerm(t *testing.T) {
	terms := AccumF32(64).And(StoreF16(2)).And(QuantInt8(0.5)).And(Magnitude(10))
	why := terms.Explain()
	for _, want := range []string{"accum", "f16", "int8"} {
		if !strings.Contains(strings.ToLower(why), want) {
			t.Errorf("Explain() = %q, which does not mention %q; a budget that cannot "+
				"say what produced it is indistinguishable from a tuned constant",
				why, want)
		}
	}
	if Explained := (Terms{}).Explain(); Explained == "" {
		t.Error("the zero Terms explains itself as nothing; exact equality is a claim " +
			"and should say so")
	}
}

// TestPrimitiveTermsAreCeilingsFromAccel checks the constructors that carry
// accel's own numerics ceilings rather than §5.1's.
func TestPrimitiveTermsAreCeilingsFromAccel(t *testing.T) {
	// A ULP ceiling scales with the count of operations, since each rounds.
	one := PrimitiveULP("exp", 2, 1).Relative()
	many := PrimitiveULP("exp", 2, 100).Relative()
	if many <= one {
		t.Errorf("a ULP ceiling over 100 operations (%g) is not above one (%g)", many, one)
	}
	// PrimitiveAbs names an absolute ceiling -- sin and cos at 2^-20 -- and
	// enters the budget RELATIVELY, which looks wrong and is not: the primitive
	// it bounds has unit magnitude, so an error of 2^-20 in a cosine becomes
	// 2^-20 of whatever that cosine multiplies. The term therefore applies to
	// the operand magnitude Magnitude carries. Pinned here because the natural
	// reading is the opposite one, and I had it backwards first.
	abs := PrimitiveAbs("cos", 1.0/(1<<20), 4)
	if got, want := abs.Relative(), 4.0/(1<<20); math.Abs(got-want) > 1e-18 {
		t.Errorf("PrimitiveAbs(cos, 2^-20, 4) contributes %g relatively, want %g", got, want)
	}
	if abs.Absolute() != 0 {
		t.Errorf("PrimitiveAbs contributed %g absolutely; its ceiling scales with "+
			"the operand it multiplies, so it belongs in the relative part",
			abs.Absolute())
	}
	// A quantization bound is the genuinely absolute one.
	if q := QuantInt8(0.5); q.Absolute() <= 0 || q.Relative() != 0 {
		t.Errorf("QuantInt8 = {rel %g, abs %g}, want an absolute-only term",
			q.Relative(), q.Absolute())
	}
}

// TestTermConstructorsRefuseNonsense covers the guards.
//
// A negative count or a negative ceiling is a caller mistake that would
// otherwise produce a *smaller* budget than the honest one — a tolerance
// tightened by arithmetic nobody meant, which fails a correct implementation
// and sends a reader hunting a numerics bug that is not there.
func TestTermConstructorsRefuseNonsense(t *testing.T) {
	for _, c := range []struct {
		name string
		call func()
	}{
		{"AccumF32 with a negative k", func() { AccumF32(-1) }},
		{"RoundF32 with a negative n", func() { RoundF32(-1) }},
		{"StoreF16 with negative operands", func() { StoreF16(-1) }},
		{"QuantInt8 with a negative bound", func() { QuantInt8(-0.5) }},
		{"PrimitiveULP with a negative ceiling", func() { PrimitiveULP("exp", -1, 1) }},
		{"PrimitiveULP with a negative count", func() { PrimitiveULP("exp", 1, -1) }},
		{"PrimitiveAbs with a negative ceiling", func() { PrimitiveAbs("cos", -1, 1) }},
		{"PrimitiveAbs with a negative count", func() { PrimitiveAbs("cos", 1, -1) }},
		{"Magnitude with a negative scale", func() { Magnitude(-1) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s was accepted; it would shrink a budget rather than "+
						"widen it, which fails a correct implementation", c.name)
				}
			}()
			c.call()
		})
	}
}

// TestInt8MatMulBoundRefusesMismatchedShapes covers the guard on the one
// tolerance term that is measured rather than derived.
func TestInt8MatMulBoundRefusesMismatchedShapes(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Int8MatMulBound accepted an x whose length is not m*k; the bound " +
				"it returned would describe a different matrix")
		}
	}()
	Int8MatMulBound(make([]float32, 3), nil, 2, 4, 1)
}

// TestZeroMagnitudeFallsBackToTheReference pins Magnitude's documented default:
// where a caller declares no scale, the bound applies to the reference value
// itself, which is right elementwise and wrong for a cancelling sum — so
// Explain says which was used.
func TestZeroMagnitudeFallsBackToTheReference(t *testing.T) {
	terms := AccumF32(16) // no Magnitude
	if terms.Bound(100) <= terms.Bound(1) {
		t.Error("with no declared magnitude the bound did not scale with the reference")
	}
	if why := terms.Explain(); why == "" {
		t.Error("a budget using the fallback magnitude does not say so")
	}
}
