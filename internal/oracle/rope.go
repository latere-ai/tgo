// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package oracle

import "math"

// Style is the rotary pairing convention: which two channels of a head form
// the plane that a rotation turns.
//
// specs/004-model-graph.md §2.5.1 is the reason this type exists. The rotation
// formula in §2.5 names the pair (x_2i, x_2i+1), which is one of two
// conventions in use, and the other one is what Qwen3 and every HF
// Llama-family checkpoint were trained with. Rotating the wrong pairs is not
// refused by anything: every shape checks, the model stays fluent, and it
// loses long-range coherence. Both conventions are implemented here so that
// 004-D9's load-time permutation has something to be verified against.
type Style int

// The two rotary pairing conventions.
//
// The zero value is deliberately neither of them. A default that silently
// meant one convention would reintroduce exactly the failure 004 §2.5.1
// describes at every call site that forgot the argument, so a caller must say
// which convention it means.
const (
	invalidStyle Style = iota

	// StyleInterleaved pairs (x[2i], x[2i+1]) — the GPT-J convention, and the
	// one accel's RoPE kernel implements. tensor.RoPE has no style option.
	StyleInterleaved

	// StyleHalfSplit pairs (x[i], x[i+rotaryDim/2]) — the NeoX convention,
	// which is what Qwen3 uses: vLLM builds it with is_neox_style=True and
	// ollama calls MLX's RoPEWithBase with traditional=false.
	StyleHalfSplit
)

// String returns the convention's name, as it appears in
// specs/004-model-graph.md §2.5.1's table.
func (s Style) String() string {
	switch s {
	case StyleInterleaved:
		return "interleaved"
	case StyleHalfSplit:
		return "half-split"
	default:
		return "invalid"
	}
}

// RoPE applies rotary position embedding to the first rotaryDim channels of
// each row of x, which is [rows, width].
//
// Per specs/004-model-graph.md §2.5, the i-th channel pair of the row at
// position m is turned by m·theta_i with
//
//	theta_i = base^(-2i/rotaryDim), 0 <= i < rotaryDim/2
//
// which pair that is, is what style selects. Channels in [rotaryDim, width)
// are copied through unrotated.
//
// pos carries one absolute position per row and its length must be rows. That
// mirrors accel, whose RoPE takes a u32 positions tensor and refuses one whose
// length is not the row count (004 §2.5.2). Since a head-major reshape makes
// rows = T·H, a caller rotating a whole step repeats each token's position H
// times, which is 004 §3 row 12's formula.
//
// Two readings the spec text leaves open, and where each one is settled:
//
//   - theta's exponent divides by rotaryDim, not by width. The spec writes
//     base^(-2i/d_h) and says Qwen3 rotates the whole head, so the two
//     coincide for Qwen3 and only a partial rotation tells them apart. accel's
//     kernel decides it, not convention: it computes the exponent as
//     -2*k/p.RotaryDim (internal/testkernels/elementwise.go, RoPE). An oracle
//     that divided by width would be out of parity with the device on every
//     partial rotation. TestRoPEThetaDividesByRotaryDimNotWidth pins it.
//   - the half-split partner is i+rotaryDim/2, not i+width/2, so a partial
//     rotation splits the rotated segment and leaves the tail alone. Nothing
//     upstream decides this one — accel has no half-split kernel — and Qwen3
//     never reaches it, since it rotates the whole head. It stays a reported
//     reading.
func RoPE(x []float64, rows, width, rotaryDim int, base float64, pos []int, style Style) []float64 {
	if rows < 0 || width < 0 {
		panic("oracle: RoPE has a negative dimension")
	}
	if len(x) != rows*width {
		panic("oracle: RoPE x length does not match rows*width")
	}
	if len(pos) != rows {
		panic("oracle: RoPE positions length does not match the row count")
	}
	if rotaryDim <= 0 || rotaryDim%2 != 0 {
		// accel's RoPE refuses an odd rotaryDim because it rotates pairs, and
		// specs/004-model-graph.md §7 refuses the config field that would
		// produce one.
		panic("oracle: RoPE rotaryDim is not a positive even number")
	}
	if rotaryDim > width {
		panic("oracle: RoPE rotaryDim exceeds the row width")
	}
	if style != StyleInterleaved && style != StyleHalfSplit {
		panic("oracle: RoPE style must be StyleInterleaved or StyleHalfSplit")
	}

	out := make([]float64, len(x))
	copy(out, x)
	half := rotaryDim / 2
	for r := 0; r < rows; r++ {
		off := r * width
		m := float64(pos[r])
		for i := 0; i < half; i++ {
			// theta is computed once, here, for both styles. The two
			// conventions differ only in which channels the i-th pair lives
			// in, so deriving theta from a channel index in one branch and
			// from the pair ordinal in the other would make the permutation
			// identity in rope_test.go hold to a rounding error instead of
			// exactly, and that test is worth more when it is exact.
			angle := m * math.Pow(base, -2*float64(i)/float64(rotaryDim))
			sin, cos := math.Sincos(angle)
			lo, hi := pair(style, i, half)
			rotate(out, off+lo, off+hi, cos, sin)
		}
	}
	return out
}

// pair returns the two channel indices of the i-th rotated pair, relative to
// the start of the row. half is rotaryDim/2.
func pair(style Style, i, half int) (lo, hi int) {
	if style == StyleInterleaved {
		return 2 * i, 2*i + 1
	}
	return i, i + half
}

// rotate turns the plane spanned by two channels of a row in place.
//
// Both styles go through this one function so that they perform the same
// arithmetic in the same order on the same values, which is what makes the
// permutation identity exact rather than approximate.
func rotate(v []float64, lo, hi int, cos, sin float64) {
	lv, hv := v[lo], v[hi]
	v[lo] = lv*cos - hv*sin
	v[hi] = lv*sin + hv*cos
}
