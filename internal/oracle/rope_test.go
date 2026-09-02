// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package oracle

import (
	"math"
	"math/rand/v2"
	"testing"
)

// permute reorders each row's rotary segment from half-split channel order
// into interleaved channel order:
//
//	y[2i] = x[i], y[2i+1] = x[i+rotaryDim/2]
//
// This is exactly the load-time permutation of
// specs/004-model-graph.md §2.5.2, written here from the spec rather than
// called from the loader, so that the test below is a statement about the
// mathematics and not about tgo's code. Channels past rotaryDim are copied.
func permute(x []float64, rows, width, rotaryDim int) []float64 {
	half := rotaryDim / 2
	out := make([]float64, len(x))
	copy(out, x)
	for r := range rows {
		off := r * width
		for i := range half {
			out[off+2*i] = x[off+i]
			out[off+2*i+1] = x[off+i+half]
		}
	}
	return out
}

// unpermute inverts permute.
func unpermute(y []float64, rows, width, rotaryDim int) []float64 {
	half := rotaryDim / 2
	out := make([]float64, len(y))
	copy(out, y)
	for r := range rows {
		off := r * width
		for i := range half {
			out[off+i] = y[off+2*i]
			out[off+i+half] = y[off+2*i+1]
		}
	}
	return out
}

// TestHalfSplitIsInterleavedUnderThePermutation is the test this package
// exists for.
//
// specs/004-model-graph.md 004-D9 handles Qwen3's half-split rotary convention
// by permuting the q and k projection output channels at load and letting
// accel's interleaved kernel do the rotation. Nothing refuses the mismatch if
// that permutation is wrong, missing, or inverted: every shape still checks
// and the model still produces fluent text. This property is what makes the
// permutation verifiable without any reference data.
//
// The claim is an identity, not an approximation. Both styles rotate the same
// pair ordinals by the same angles through the same two multiplies and one
// add each, so permuting, rotating interleaved, and unpermuting must reproduce
// the half-split result bit for bit. Equality is asserted exactly: a
// disagreement in the last place would mean the two styles derive theta
// differently, which is the indexing bug the test is looking for.
func TestHalfSplitIsInterleavedUnderThePermutation(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	cases := []struct {
		name                   string
		rows, width, rotaryDim int
		base                   float64
	}{
		{"qwen3 head, whole head rotated", 6, 128, 128, 1e6},
		{"llama base", 3, 64, 64, 1e4},
		{"partial rotation leaves a tail", 4, 12, 8, 1e6},
		{"one pair", 2, 2, 2, 1e6},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			x := make([]float64, c.rows*c.width)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			pos := make([]int, c.rows)
			for i := range pos {
				pos[i] = rng.IntN(4096)
			}

			direct := RoPE(x, c.rows, c.width, c.rotaryDim, c.base, pos, StyleHalfSplit)
			viaAccel := unpermute(
				RoPE(permute(x, c.rows, c.width, c.rotaryDim), c.rows, c.width, c.rotaryDim, c.base, pos, StyleInterleaved),
				c.rows, c.width, c.rotaryDim,
			)
			for i := range direct {
				if direct[i] != viaAccel[i] {
					t.Fatalf("channel %d: half-split %.17g, permuted-interleaved %.17g", i, direct[i], viaAccel[i])
				}
			}
		})
	}
}

// TestStylesDiffer keeps the permutation property above from being vacuous.
// If the two styles were the same computation, permuting and unpermuting a
// symmetric operation would also agree and prove nothing.
func TestStylesDiffer(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	pos := []int{5}
	inter := RoPE(x, 1, 8, 8, 1e6, pos, StyleInterleaved)
	half := RoPE(x, 1, 8, 8, 1e6, pos, StyleHalfSplit)
	same := true
	for i := range inter {
		if inter[i] != half[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("the two rotary conventions produced identical output; one of them is not implemented")
	}
}

func TestRoPEAtPositionZeroIsTheIdentity(t *testing.T) {
	// m = 0 makes every angle 0, and cos(0) is exactly 1 while sin(0) is
	// exactly 0, so x*1 - y*0 = x holds in float64 with no rounding. Exact
	// equality is therefore the right assertion.
	rng := rand.New(rand.NewPCG(3, 5))
	x := make([]float64, 4*16)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	pos := []int{0, 0, 0, 0}
	for _, style := range []Style{StyleInterleaved, StyleHalfSplit} {
		got := RoPE(x, 4, 16, 16, 1e6, pos, style)
		for i := range x {
			if got[i] != x[i] {
				t.Fatalf("%v: channel %d changed at position 0: %.17g != %.17g", style, i, got[i], x[i])
			}
		}
	}
}

func TestRoPEPreservesEachPairNorm(t *testing.T) {
	// A rotation is orthogonal, so every rotated pair keeps its length. The
	// only error is cos^2+sin^2-1, which is bounded by about 2*eps_64 from the
	// two roundings in Sincos.
	rng := rand.New(rand.NewPCG(13, 17))
	const rows, width, rotaryDim = 5, 32, 32
	x := make([]float64, rows*width)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	pos := []int{0, 1, 17, 512, 4095}
	for _, style := range []Style{StyleInterleaved, StyleHalfSplit} {
		got := RoPE(x, rows, width, rotaryDim, 1e6, pos, style)
		for r := range rows {
			for i := range rotaryDim / 2 {
				lo, hi := pair(style, i, rotaryDim/2)
				before := x[r*width+lo]*x[r*width+lo] + x[r*width+hi]*x[r*width+hi]
				after := got[r*width+lo]*got[r*width+lo] + got[r*width+hi]*got[r*width+hi]
				if !closeTo(after, before, 4*tolULP) {
					t.Errorf("%v row %d pair %d: norm^2 %.17g -> %.17g", style, r, i, before, after)
				}
			}
		}
	}
}

func TestRoPELeavesChannelsPastRotaryDimAlone(t *testing.T) {
	// Qwen3 rotates the whole head so nothing in the model exercises this,
	// but rotaryDim is a parameter and a partial rotation must not touch the
	// tail.
	x := []float64{1, 2, 3, 4, 5, 6}
	got := RoPE(x, 1, 6, 4, 1e6, []int{9}, StyleHalfSplit)
	if got[4] != 5 || got[5] != 6 {
		t.Errorf("tail channels changed: %v", got[4:])
	}
	if got[0] == 1 && got[1] == 2 {
		t.Errorf("rotated channels did not change: %v", got[:4])
	}
}

func TestRoPERotatesEachRowByItsOwnPosition(t *testing.T) {
	// accel takes one position per row and a caller repeats each token's
	// position H times (specs/004-model-graph.md §3 row 12). Two rows holding
	// the same vector at different positions must therefore differ, and two
	// rows at the same position must agree.
	x := []float64{1, 2, 1, 2, 1, 2}
	got := RoPE(x, 3, 2, 2, 1e6, []int{0, 1, 1}, StyleInterleaved)
	if got[2] != got[4] || got[3] != got[5] {
		t.Errorf("rows at the same position differ: %v", got)
	}
	if got[0] == got[2] && got[1] == got[3] {
		t.Errorf("rows at different positions agree: %v", got)
	}
}

func TestRoPEAngleMatchesTheFormula(t *testing.T) {
	// One hand-computed rotation, so that a sign flip or a transposed rotation
	// matrix fails here and not only in a property. theta_0 = base^0 = 1, so
	// the first pair at position m turns by m radians.
	const m = 2.0
	x := []float64{1, 0, 1, 0}
	got := RoPE(x, 1, 4, 4, 1e6, []int{int(m)}, StyleInterleaved)
	wantCos, wantSin := math.Cos(m), math.Sin(m)
	if !closeTo(got[0], wantCos, tolULP) || !closeTo(got[1], wantSin, tolULP) {
		t.Errorf("pair 0 = (%.17g, %.17g), want (%.17g, %.17g)", got[0], got[1], wantCos, wantSin)
	}
	// theta_1 = base^(-2/4) = 1/sqrt(base), a much slower rotation.
	theta1 := math.Pow(1e6, -0.5)
	if !closeTo(got[2], math.Cos(m*theta1), tolULP) || !closeTo(got[3], math.Sin(m*theta1), tolULP) {
		t.Errorf("pair 1 = (%.17g, %.17g), want (%.17g, %.17g)", got[2], got[3], math.Cos(m*theta1), math.Sin(m*theta1))
	}
}

// TestRoPEThetaDividesByRotaryDimNotWidth pins the one reading of
// specs/004-model-graph.md §2.5 that the spec text leaves open.
//
// §2.5 writes theta_i = base^(-2i/d_h) and §2.5.2 says Qwen3 rotates the whole
// head, so for Qwen3 the rotated width and the row width are the same number
// and no test on a Qwen3 shape can tell the two divisors apart. accel settles
// it: its kernel computes
//
//	exponent := float32(-2) * float32(k) / float32(p.RotaryDim)
//
// (internal/testkernels/elementwise.go, RoPE). Since this oracle exists to be
// compared against that kernel, dividing by width instead would put the two out
// of parity on every partial rotation while every Qwen3 test still passed.
// width 8 with rotaryDim 4 is the smallest shape that discriminates: pair 1
// turns by base^(-2/4), and the wrong divisor would turn it by base^(-2/8).
func TestRoPEThetaDividesByRotaryDimNotWidth(t *testing.T) {
	const base, m = 1e6, 2.0
	// Only the rotated segment matters; the tail is checked elsewhere.
	x := []float64{1, 0, 1, 0, 9, 9, 9, 9}
	got := RoPE(x, 1, 8, 4, base, []int{int(m)}, StyleInterleaved)

	wantTheta := math.Pow(base, -2.0/4.0)
	wrongTheta := math.Pow(base, -2.0/8.0)
	wantCos, wantSin := math.Cos(m*wantTheta), math.Sin(m*wantTheta)
	if !closeTo(got[2], wantCos, tolULP) || !closeTo(got[3], wantSin, tolULP) {
		t.Errorf("pair 1 = (%.17g, %.17g), want (%.17g, %.17g) from base^(-2/rotaryDim)",
			got[2], got[3], wantCos, wantSin)
	}
	// Stated as an inequality too, so that the failure names the confusion
	// rather than only a number.
	if closeTo(got[2], math.Cos(m*wrongTheta), 1e-6) {
		t.Errorf("pair 1 turned by base^(-2/width); theta must divide by rotaryDim")
	}
}

func TestStyleString(t *testing.T) {
	for _, c := range []struct {
		s    Style
		want string
	}{
		{StyleInterleaved, "interleaved"},
		{StyleHalfSplit, "half-split"},
		{invalidStyle, "invalid"},
		{Style(99), "invalid"},
	} {
		if got := c.s.String(); got != c.want {
			t.Errorf("Style(%d).String() = %q, want %q", int(c.s), got, c.want)
		}
	}
}

func TestRoPERefusals(t *testing.T) {
	ok := []float64{1, 2, 3, 4}
	// Both disjuncts of the negative-dimension guard, not just rows.
	mustPanic(t, "RoPE has a negative dimension", func() { RoPE(nil, -1, 2, 2, 1e6, nil, StyleInterleaved) })
	mustPanic(t, "RoPE has a negative dimension", func() { RoPE(nil, 1, -1, 2, 1e6, []int{0}, StyleInterleaved) })
	// Both sides of each length check: a short slice panics out of a bound
	// anyway, so only a long one shows the check is an equality.
	mustPanic(t, "RoPE x length", func() { RoPE(ok, 3, 2, 2, 1e6, []int{0, 0, 0}, StyleInterleaved) })
	mustPanic(t, "RoPE x length", func() { RoPE(ok, 1, 2, 2, 1e6, []int{0}, StyleInterleaved) })
	mustPanic(t, "RoPE positions length", func() { RoPE(ok, 2, 2, 2, 1e6, []int{0}, StyleInterleaved) })
	mustPanic(t, "RoPE positions length", func() { RoPE(ok, 2, 2, 2, 1e6, []int{0, 0, 0}, StyleInterleaved) })
	mustPanic(t, "RoPE rotaryDim is not a positive even number", func() { RoPE(ok, 1, 4, 3, 1e6, []int{0}, StyleInterleaved) })
	mustPanic(t, "RoPE rotaryDim is not a positive even number", func() { RoPE(ok, 1, 4, 0, 1e6, []int{0}, StyleInterleaved) })
	mustPanic(t, "RoPE rotaryDim exceeds the row width", func() { RoPE(ok, 1, 4, 6, 1e6, []int{0}, StyleInterleaved) })
	// The zero Style is neither convention on purpose: a default that silently
	// meant one of them is the failure specs/004-model-graph.md §2.5.1
	// describes.
	mustPanic(t, "RoPE style must be", func() { RoPE(ok, 1, 4, 4, 1e6, []int{0}, invalidStyle) })
}
