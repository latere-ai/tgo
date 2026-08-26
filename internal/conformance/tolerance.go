// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance

import (
	"fmt"
	"math"
	"strings"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
)

// The rounding constants of specs/010-conformance.md §5.1.
const (
	// Eps32 is the f32 round-to-nearest bound, 2^-24, which is half a unit in
	// the last place. §5.1 names it as the accumulation term's constant.
	Eps32 = 1.0 / (1 << 24)

	// Eps16 is the f16 round-to-nearest bound, 2^-11, the error a weight
	// suffers by being stored narrow. §5.1: this term dominates the
	// accumulation term by three orders of magnitude, so the tolerance on a
	// matmul is set by the storage format and not by the accumulator.
	Eps16 = 1.0 / (1 << 11)

	// ULP32 is one f32 unit in the last place as a relative bound, 2^-23.
	// accel's specs/008-numerics.md §6 states its primitive ceilings in ULP,
	// and this is the conversion: a value in the binade [2^e, 2^(e+1)) has an
	// ULP of 2^(e-23), which is at most 2^-23 of the value itself.
	ULP32 = 2 * Eps32
)

// Terms is a derived error budget: the sum of the terms
// specs/010-conformance.md §5.1 names, and nothing else.
//
// It is deliberately not a float. 010-D3 says a tolerance that had to be raised
// to make a test pass is a finding rather than a fix, and the way to enforce
// that in code is to make the number unwritable: a caller composes the stages
// its computation actually has, and the bound falls out. Widening one means
// adding a term, which is an argument somebody has to make out loud.
//
// A budget carries three parts. The relative part scales with the magnitude of
// what is being compared; the absolute part does not, because a quantization
// bound is an absolute statement over the inputs that produced it; and the
// magnitude is what the relative part applies to.
//
// The zero Terms is an exact-equality assertion, which is a legitimate thing to
// claim about an operation that only moves bytes.
type Terms struct {
	rel   float64
	abs   float64
	scale float64
	why   []string
}

// AccumF32 is the error of summing k terms in f32: sqrt(k)·eps32.
//
// specs/010-conformance.md §5.1 derives it for a well-conditioned sum with
// random signs and notes that the blocked summation a tiled GEMM performs is
// nearer sqrt(log k), so sqrt(k) is a safe upper bound rather than the truth.
func AccumF32(k int) Terms {
	if k < 0 {
		panic("conformance: AccumF32 over a negative number of terms")
	}
	e := math.Sqrt(float64(k)) * Eps32
	return Terms{rel: e, why: []string{fmt.Sprintf(
		"f32 accumulation over K=%d: sqrt(K)*eps32 = %.3g", k, e)}}
}

// RoundF32 is n elementary f32 roundings, eps32 each.
//
// §5.1's table names stages -- a GEMM, an operand format, a quantizer -- and
// has no row for the single multiply that scales a normalized row or the
// subtraction inside a softmax. This is that row. It is small enough to be
// invisible next to any f16 term and it is not zero, and a budget with a hole
// in it is a budget that gets widened somewhere else.
func RoundF32(n int) Terms {
	if n < 0 {
		panic("conformance: RoundF32 with a negative count")
	}
	e := float64(n) * Eps32
	return Terms{rel: e, why: []string{fmt.Sprintf(
		"%d f32 rounding(s): n*eps32 = %.3g", n, e)}}
}

// StoreF16 is the error of holding n operands in f16, eps16 each.
//
// Per operand and not per matrix: §5.1 charges the term once for each value
// that made the round trip through the narrow format, so a matmul of an f32
// activation against an f16 weight is one and not two.
func StoreF16(operands int) Terms {
	if operands < 0 {
		panic("conformance: StoreF16 with a negative operand count")
	}
	e := float64(operands) * Eps16
	return Terms{rel: e, why: []string{fmt.Sprintf(
		"%d f16 operand(s): n*eps16 = %.3g", operands, e)}}
}

// QuantInt8 admits an absolute bound from [quant.Int8ErrorBound].
//
// Absolute rather than relative because that is what the bound is: it is
// driven by the largest weight in each block and by the activations it
// multiplies, not by the size of the answer. A dot product that cancels to
// near zero still carries the whole of it.
func QuantInt8(bound float64) Terms {
	if bound < 0 {
		panic("conformance: QuantInt8 with a negative bound")
	}
	return Terms{abs: bound, why: []string{fmt.Sprintf(
		"quant.Int8ErrorBound over the actual inputs: %.3g absolute", bound)}}
}

// PrimitiveULP is count calls to an accel primitive whose ceiling
// specs/008-numerics.md §6 states in ULP: rsqrt, exp, log and tanh at 4, a
// division at 2.5, a square root at 1.
//
// specs/010-conformance.md §5.1 has no row for these -- it calls the softmax
// benign, which is a statement about overflow rather than about the accuracy
// of exp -- so the ceiling is taken from the library that promises it. That is
// still a derivation and not a tuning: the number is accel's contract, and a
// result outside it is a finding against accel.
func PrimitiveULP(name string, ulps float64, count int) Terms {
	if ulps < 0 || count < 0 {
		panic("conformance: PrimitiveULP with a negative ceiling or count")
	}
	e := float64(count) * ulps * ULP32
	return Terms{rel: e, why: []string{fmt.Sprintf(
		"%d call(s) to %s at accel's %g ULP ceiling: n*ulps*ULP32 = %.3g",
		count, name, ulps, e)}}
}

// PrimitiveAbs is count calls to a primitive whose ceiling is absolute rather
// than in ULP -- sin and cos, at 2^-20 over accel's declared angle domain.
//
// It enters the budget as a relative term because the primitive it bounds has
// unit magnitude: an error of 2^-20 in a cosine becomes 2^-20 of whatever that
// cosine multiplies, so the term applies to the operand magnitude that
// [Magnitude] carries.
func PrimitiveAbs(name string, bound float64, count int) Terms {
	if bound < 0 || count < 0 {
		panic("conformance: PrimitiveAbs with a negative ceiling or count")
	}
	e := float64(count) * bound
	return Terms{rel: e, why: []string{fmt.Sprintf(
		"%d call(s) to %s at accel's %.3g absolute ceiling: n*bound = %.3g",
		count, name, bound, e)}}
}

// Magnitude declares the size the relative terms apply to.
//
// For a dot product it is sum|x_i·w_i| and not the result: a sum whose terms
// cancel has an absolute error set by the terms that cancelled, and bounding it
// by the near-zero answer would fail every well-behaved implementation. Where
// the caller declares no magnitude the bound falls back to the reference value
// itself, which is right for an elementwise operation and wrong for a
// cancelling sum -- so [Terms.Explain] says which of the two was used.
func Magnitude(scale float64) Terms {
	if scale < 0 {
		panic("conformance: Magnitude with a negative scale")
	}
	return Terms{scale: scale, why: []string{fmt.Sprintf(
		"relative terms apply to a magnitude of %.3g", scale)}}
}

// SoftmaxWeight is the relative error a softmax weight carries because the
// score it exponentiates was accumulated in f32.
//
// The dot product over k terms has a relative error of [AccumF32], so a score
// of magnitude s carries an absolute error of s*AccumF32(k). The exponential
// turns an absolute error in its argument into a *relative* error in its
// result:
//
//	exp(s + d) = exp(s)*exp(d) ~ exp(s)*(1 + d)   for small d
//
// and the normalization by the sum leaves that relative error in place, since a
// factor common to every weight cancels and only the spread survives. So the
// weight's relative error is s*AccumF32(k), where s is the largest score in the
// row, and it applies to the magnitude of the values those weights combine.
//
// It exists because attention is the one place in this project where a relative
// term has to be scaled by an intermediate rather than by the answer, and
// composing that by hand at each call site is how a budget becomes a number
// somebody tuned. maxScore comes from the reference, which computed it.
func SoftmaxWeight(maxScore float64, k int) Terms {
	if maxScore < 0 || k < 0 {
		panic("conformance: SoftmaxWeight with a negative score or width")
	}
	e := maxScore * AccumF32(k).Relative()
	return Terms{rel: e, why: []string{fmt.Sprintf(
		"the softmax weight, whose score of magnitude %.3g accumulated over K=%d: "+
			"s*sqrt(K)*eps32 = %.3g", maxScore, k, e)}}
}

// And composes two stages of one computation.
//
// Relative errors add and absolute ones add. The magnitude is the larger of the
// two, which is the conservative reading when an earlier stage's intermediate
// is larger than the final answer and its error is carried forward through
// everything after it.
//
// Adding two accumulation terms is not the same as accumulating over the sum of
// their lengths -- sqrt(a)+sqrt(b) exceeds sqrt(a+b) -- and the larger of the
// two is the one a chain of stages actually suffers, which is why a chain is
// composed here rather than by adding up K.
func (t Terms) And(u Terms) Terms {
	why := make([]string, 0, len(t.why)+len(u.why))
	why = append(append(why, t.why...), u.why...)
	return Terms{
		rel:   t.rel + u.rel,
		abs:   t.abs + u.abs,
		scale: math.Max(t.scale, u.scale),
		why:   why,
	}
}

// Relative is the budget's relative part, before any magnitude is applied.
func (t Terms) Relative() float64 { return t.rel }

// Absolute is the budget's absolute part.
func (t Terms) Absolute() float64 { return t.abs }

// Bound is how far a device result may sit from the reference value want.
func (t Terms) Bound(want float64) float64 {
	m := math.Abs(want)
	if t.scale > m {
		m = t.scale
	}
	return t.rel*m + t.abs
}

// Explain prints the derivation, one term per line.
//
// It is what a failure message carries. specs/010-conformance.md §5.1 asks for
// a comment on every tolerance constant naming the term that produced it; a
// budget that prints its own terms answers that at the moment somebody is
// deciding whether the number is wrong, which is the moment the comment was
// for.
func (t Terms) Explain() string {
	var b strings.Builder
	for _, w := range t.why {
		b.WriteString("\n\t  " + w)
	}
	if t.scale == 0 {
		b.WriteString("\n\t  no magnitude declared; the relative terms apply to the " +
			"reference value itself, which understates a sum whose terms cancel")
	}
	fmt.Fprintf(&b, "\n\t  total: %.3g relative + %.3g absolute", t.rel, t.abs)
	return b.String()
}

// Int8MatMulBound is [quant.Int8ErrorBound] for the worst element of x·W,
// where W is [k, n] quantized row-major and scales are its per-block scales.
//
// quant.Int8ErrorBound takes one scale per term of a dot product, and the terms
// of column j are W[p*n+j] for p in [0, k) -- strided, so they fall in
// different blocks. Building that per-term array is the call the quant package
// documents as the correct use, and doing it once here is what keeps every
// quantized parity test from reimplementing the stride.
//
// The maximum over every output element rather than a bound per element: the
// comparison applies one budget to the whole output, so the budget has to be
// the largest of them. It is still derived from the values that were used.
func Int8MatMulBound(x []float32, scales []accel.Float16, m, k, n int) float64 {
	if len(x) != m*k {
		panic("conformance: Int8MatMulBound x length does not match m*k")
	}
	worst := 0.0
	terms := make([]accel.Float16, k)
	for j := 0; j < n; j++ {
		for p := 0; p < k; p++ {
			terms[p] = scales[(p*n+j)/quant.Int8Block]
		}
		for i := 0; i < m; i++ {
			if b := quant.Int8ErrorBound(x[i*k:(i+1)*k], terms); b > worst {
				worst = b
			}
		}
	}
	return worst
}
