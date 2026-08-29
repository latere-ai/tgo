// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package weights

import (
	"encoding/binary"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	"github.com/latere-ai/tgo/safetensors"
)

// bf16Plane encodes f32 values as the bf16 bytes a checkpoint holds. Widening
// is exact, so this is a lossy step only where the value needs more than eight
// mantissa bits; every value the tests below feed it is chosen to survive.
func bf16Plane(vals ...float32) []byte {
	out := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[2*i:], accel.ToBFloat16(v).Bits())
	}
	return out
}

func TestDecodeBF16IsAShift(t *testing.T) {
	// specs/001-weights.md §3: f32bits = bf16bits << 16, exactly, with no table.
	// The patterns are chosen to cover the sign, the exponent extremes and the
	// two special encodings.
	cases := []struct {
		bits uint16
		want uint32
	}{
		{0x0000, 0x00000000}, // +0
		{0x8000, 0x80000000}, // -0
		{0x3f80, 0x3f800000}, // 1
		{0xbf80, 0xbf800000}, // -1
		{0x4049, 0x40490000}, // ~3.14
		{0x7f80, 0x7f800000}, // +Inf
		{0xff80, 0xff800000}, // -Inf
		{0x0001, 0x00010000}, // smallest positive bf16 subnormal
		{0x7f7f, 0x7f7f0000}, // largest finite bf16
	}
	src := make([]byte, 2*len(cases))
	for i, c := range cases {
		binary.LittleEndian.PutUint16(src[2*i:], c.bits)
	}
	got := make([]float32, len(cases))
	if err := decodeF32(safetensors.BF16, src, got); err != nil {
		t.Fatalf("decodeF32: %v", err)
	}
	for i, c := range cases {
		if bits := math.Float32bits(got[i]); bits != c.want {
			t.Errorf("bf16 %#04x widened to %#08x, want %#08x", c.bits, bits, c.want)
		}
	}
}

func TestDecodeEveryBF16Pattern(t *testing.T) {
	// All 65536 encodings, because the shift has no interesting subset.
	src := make([]byte, 2*65536)
	for i := range 65536 {
		binary.LittleEndian.PutUint16(src[2*i:], uint16(i))
	}
	got := make([]float32, 65536)
	if err := decodeF32(safetensors.BF16, src, got); err != nil {
		t.Fatalf("decodeF32: %v", err)
	}
	for i := range 65536 {
		if want := uint32(i) << 16; math.Float32bits(got[i]) != want {
			t.Fatalf("bf16 %#04x widened to %#08x, want %#08x", i, math.Float32bits(got[i]), want)
		}
	}
}

func TestDecodeF16AndF32(t *testing.T) {
	f16 := make([]byte, 6)
	for i, v := range []float32{1, -2.5, 0.5} {
		binary.LittleEndian.PutUint16(f16[2*i:], accel.ToFloat16(v).Bits())
	}
	got := make([]float32, 3)
	if err := decodeF32(safetensors.F16, f16, got); err != nil {
		t.Fatalf("f16: %v", err)
	}
	if got[0] != 1 || got[1] != -2.5 || got[2] != 0.5 {
		t.Errorf("f16 plane decoded to %v", got)
	}

	f32 := make([]byte, 12)
	for i, v := range []float32{3, -4, 1e30} {
		binary.LittleEndian.PutUint32(f32[4*i:], math.Float32bits(v))
	}
	if err := decodeF32(safetensors.F32, f32, got); err != nil {
		t.Fatalf("f32: %v", err)
	}
	if got[0] != 3 || got[1] != -4 || got[2] != 1e30 {
		t.Errorf("f32 plane decoded to %v", got)
	}
}

func TestDecodeRefusals(t *testing.T) {
	// An integer plane read as weights is a well-shaped tensor of nonsense, so
	// it is refused rather than reinterpreted (001-D6).
	if err := decodeF32(safetensors.I64, make([]byte, 8), make([]float32, 1)); err == nil {
		t.Error("decodeF32 accepted an I64 plane")
	}
	if err := decodeF32(safetensors.DType("Q4_K"), nil, make([]float32, 1)); err == nil {
		t.Error("decodeF32 accepted an unknown dtype")
	}
	if err := decodeF32(safetensors.BF16, make([]byte, 3), make([]float32, 2)); err == nil {
		t.Error("decodeF32 accepted a plane whose length disagrees with the element count")
	}
}

func TestF16RoundsTiesToEven(t *testing.T) {
	// f16 has ten mantissa bits, so above 1 the spacing is 2^-10 and a value
	// exactly halfway between two encodings is a tie. Ties go to the even
	// significand, which is the rule the hardware uses and therefore the rule
	// that makes this conversion agree with a device Cast (001-D1).
	const ulp = 1.0 / 1024
	cases := []struct {
		in   float32
		want float32
	}{
		{1 + ulp/2, 1},           // tie down to an even significand
		{1 + ulp*3/2, 1 + ulp*2}, // tie up to an even significand
		{1 + ulp*5/2, 1 + ulp*2}, // tie down
		{1 + ulp*7/2, 1 + ulp*4}, // tie up
		{1 + ulp*0.6, 1 + ulp},   // above the tie, rounds up
		{1 + ulp*0.4, 1},         // below the tie, rounds down
		{-1 - ulp/2, -1},         // the sign does not change the rule
		{2 + 2*ulp/2, 2},         // ties at the next binade
	}
	for _, c := range cases {
		bits, sat := f32ToF16Bits(c.in)
		if sat {
			t.Errorf("%v saturated", c.in)
		}
		if got := accel.Float16FromBits(bits).F32(); got != c.want {
			t.Errorf("f32ToF16Bits(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestF16SaturatesAndCounts(t *testing.T) {
	// 001-D2: overflow lands on ±65504 and is counted, never on ±Inf. One
	// infinity in a weight matrix makes every output of that row NaN and the
	// failure surfaces many layers later as gibberish rather than as an error.
	cases := []struct {
		in  float32
		sat bool
	}{
		{65504, false}, // the largest finite f16, exactly representable
		{-65504, false},
		{65505, true}, // just over: §3's rule counts it
		{70000, true},
		{3.4e38, true},
		{float32(math.Inf(1)), true},
		{float32(math.Inf(-1)), true},
	}
	for _, c := range cases {
		bits, sat := f32ToF16Bits(c.in)
		if sat != c.sat {
			t.Errorf("f32ToF16Bits(%v) saturated=%v, want %v", c.in, sat, c.sat)
		}
		got := accel.Float16FromBits(bits).F32()
		if math.IsInf(float64(got), 0) {
			t.Errorf("f32ToF16Bits(%v) produced %v; an infinite weight is what this rule exists to prevent", c.in, got)
		}
		if c.sat && math.Abs(float64(got)) != maxF16 {
			t.Errorf("f32ToF16Bits(%v) = %v, want ±65504", c.in, got)
		}
		if want := float32(math.Copysign(1, float64(c.in))); math.Copysign(1, float64(got)) != float64(want) {
			t.Errorf("f32ToF16Bits(%v) = %v, which changed sign", c.in, got)
		}
	}
}

func TestF16FlushesSubnormalsKeepingSign(t *testing.T) {
	// Below 2^-14 a value is an f16 subnormal, under 6e-8, and contributes
	// nothing to a dot product against activations of order 1.
	for _, in := range []float32{minNormalF16 / 2, 1e-8, -1e-8, 5.96e-8, 0, float32(math.Copysign(0, -1))} {
		bits, sat := f32ToF16Bits(in)
		if sat {
			t.Errorf("%v saturated", in)
		}
		if bits&0x7fff != 0 {
			t.Errorf("f32ToF16Bits(%v) = %#04x, want a zero", in, bits)
		}
		wantSign := uint16(math.Float32bits(in) >> 16 & 0x8000)
		if bits != wantSign {
			t.Errorf("f32ToF16Bits(%v) = %#04x, want sign %#04x preserved", in, bits, wantSign)
		}
	}
	// 2^-14 itself is the smallest normal and survives.
	bits, _ := f32ToF16Bits(minNormalF16)
	if got := accel.Float16FromBits(bits).F32(); got != minNormalF16 {
		t.Errorf("the smallest normal became %v", got)
	}
}

func TestF16NaNIsCanonicalAndNotCounted(t *testing.T) {
	bits, sat := f32ToF16Bits(float32(math.NaN()))
	if sat {
		t.Error("a NaN was counted as a saturation; it is not an overflow")
	}
	if !math.IsNaN(float64(accel.Float16FromBits(bits).F32())) {
		t.Errorf("a NaN became %#04x", bits)
	}
}

func TestF16AgreesWithADeviceCastEverywhereItCan(t *testing.T) {
	// 001-D1 claims this conversion agrees with a device Cast. accel.ToFloat16
	// is that rounding, so the claim is checkable exhaustively over every bf16
	// pattern — which is every f32 a bf16 checkpoint can hold. The two disagree
	// exactly on the sets §3 carves out: overflow, where accel produces Inf and
	// this saturates, and f16 subnormals, which this flushes.
	var overflow, flushed, agreed int
	for i := range 65536 {
		x := math.Float32frombits(uint32(i) << 16)
		bits, sat := f32ToF16Bits(x)
		ref := accel.ToFloat16(x).Bits()
		abs := math.Abs(float64(x))
		switch {
		case math.IsNaN(abs):
			if bits != bitsQuietNaNF16 {
				t.Fatalf("NaN pattern %#04x became %#04x", i, bits)
			}
		case abs > maxF16:
			overflow++
			if !sat {
				t.Fatalf("%v did not saturate", x)
			}
			if ref&0x7fff != 0x7c00 {
				t.Fatalf("accel.ToFloat16(%v) = %#04x, expected an infinity", x, ref)
			}
		case abs < minNormalF16:
			flushed++
			if bits&0x7fff != 0 {
				t.Fatalf("%v did not flush", x)
			}
		default:
			agreed++
			if bits != ref {
				t.Fatalf("f32ToF16Bits(%v) = %#04x, accel.ToFloat16 = %#04x", x, bits, ref)
			}
			if sat {
				t.Fatalf("%v saturated inside f16's range", x)
			}
		}
	}
	// A sanity floor on the partition: if any set were empty the agreement
	// above would be proving less than it looks like it proves.
	if overflow == 0 || flushed == 0 || agreed == 0 {
		t.Fatalf("partition is degenerate: %d overflow, %d flushed, %d agreed", overflow, flushed, agreed)
	}
}

func TestToF16CountsPerPlane(t *testing.T) {
	plane := []float32{1, 1e30, -1e30, 0.5}
	dst := make([]byte, 2*len(plane))
	if got := toF16(plane, dst); got != 2 {
		t.Errorf("toF16 counted %d saturations, want 2", got)
	}
	if got := accel.Float16FromBits(binary.LittleEndian.Uint16(dst[2:])).F32(); got != maxF16 {
		t.Errorf("the second element became %v", got)
	}
}

func TestTranspose(t *testing.T) {
	// [2,3] -> [3,2].
	src := []float32{1, 2, 3, 4, 5, 6}
	dst := make([]float32, 6)
	transpose(src, dst, 2, 3)
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("transpose = %v, want %v", dst, want)
		}
	}
	// Transposing back is the identity, which is the property that matters:
	// the conversion has to be a relabelling and not a reshuffle.
	back := make([]float32, 6)
	transpose(dst, back, 3, 2)
	for i := range src {
		if back[i] != src[i] {
			t.Fatalf("transpose twice = %v, want %v", back, src)
		}
	}
}

func TestPermuteHeadsMovesChannels(t *testing.T) {
	// Two heads of four channels, one row. y[2i] = x[i], y[2i+1] = x[i+d/2].
	x := []float32{0, 1, 2, 3, 10, 11, 12, 13}
	if err := permuteHeads(x, 8, 4); err != nil {
		t.Fatalf("permuteHeads: %v", err)
	}
	want := []float32{0, 2, 1, 3, 10, 12, 11, 13}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("permuteHeads = %v, want %v", x, want)
		}
	}
}

func TestPermuteHeadsRefusals(t *testing.T) {
	if err := permuteHeads(make([]float32, 8), 8, 3); err == nil {
		t.Error("an odd head dim was accepted; RoPE rotates pairs")
	}
	if err := permuteHeads(make([]float32, 8), 8, 0); err == nil {
		t.Error("a zero head dim reached the permutation")
	}
	if err := permuteHeads(make([]float32, 8), 6, 4); err == nil {
		t.Error("an output axis that does not divide into heads was accepted")
	}
}

// ropeInterleaved is accel's convention, from internal/testkernels/elementwise.go:
// pair k is (x[2k], x[2k+1]) and rotates by pos * base^(-2k/d).
func ropeInterleaved(x []float32, pos float64, base float64) []float32 {
	d := len(x)
	out := make([]float32, d)
	copy(out, x)
	for k := range d / 2 {
		theta := pos * math.Pow(base, -2*float64(k)/float64(d))
		c, s := math.Cos(theta), math.Sin(theta)
		lo, hi := 2*k, 2*k+1
		a, b := float64(x[lo]), float64(x[hi])
		out[lo] = float32(a*c - b*s)
		out[hi] = float32(a*s + b*c)
	}
	return out
}

// ropeHalfSplit is Qwen3's convention: pair k is (x[k], x[k+d/2]) at the same
// angle. vLLM's is_neox_style=True and MLX's traditional=false name this one.
func ropeHalfSplit(x []float32, pos float64, base float64) []float32 {
	d := len(x)
	out := make([]float32, d)
	copy(out, x)
	for k := range d / 2 {
		theta := pos * math.Pow(base, -2*float64(k)/float64(d))
		c, s := math.Cos(theta), math.Sin(theta)
		lo, hi := k, k+d/2
		a, b := float64(x[lo]), float64(x[hi])
		out[lo] = float32(a*c - b*s)
		out[hi] = float32(a*s + b*c)
	}
	return out
}

func TestPermuteMakesInterleavedRoPEComputeHalfSplit(t *testing.T) {
	// The property the permutation exists for (004-D9):
	//
	//	interleavedRoPE(permute(x)) == permute(halfSplitRoPE(x))
	//
	// per head, because RoPE runs on a plane reshaped to [rows, d_h] and every
	// head rotates independently. The permutation stays on the output, and that
	// is correct: attention computes q·kᵀ, which is invariant under a
	// permutation applied identically to q and to k, so nothing downstream
	// undoes it.
	//
	// Both conventions are computed here rather than taken from a shared
	// helper. Two derivations agreeing is evidence; one compared against itself
	// is not (specs/010-conformance.md §5).
	const base = 1e6
	for _, c := range []struct{ headDim, heads int }{
		{2, 1}, {4, 1}, {4, 2}, {64, 1}, {128, 16}, {128, 8},
	} {
		width := c.headDim * c.heads
		x := make([]float32, width)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*1.7)) * 2
		}
		for _, pos := range []float64{0, 1, 7, 4095} {
			left := make([]float32, width)
			copy(left, x)
			if err := permuteHeads(left, width, c.headDim); err != nil {
				t.Fatalf("permuteHeads: %v", err)
			}
			left = perHead(left, c.headDim, func(h []float32) []float32 {
				return ropeInterleaved(h, pos, base)
			})

			right := perHead(x, c.headDim, func(h []float32) []float32 {
				return ropeHalfSplit(h, pos, base)
			})
			if err := permuteHeads(right, width, c.headDim); err != nil {
				t.Fatalf("permuteHeads: %v", err)
			}

			for i := range width {
				// Both sides are the same two products and one sum of f64
				// intermediates rounded once to f32, so the difference is one
				// f32 rounding: 2^-24 relative, scaled by the largest term.
				// specs/010-conformance.md §5.1's ε₃₂ term, K = 2.
				const tol = 4 * 6e-8 * 2
				if diff := math.Abs(float64(left[i] - right[i])); diff > tol {
					t.Fatalf("d_h=%d heads=%d pos=%v channel %d: interleaved(permute(x))=%v, permute(halfSplit(x))=%v",
						c.headDim, c.heads, pos, i, left[i], right[i])
				}
			}
		}
	}
}

// perHead applies f to each headDim-wide segment of x and returns the joined
// result, which is how RoPE sees a projection's output: a plane reshaped to
// [rows, d_h], one rotation per row.
func perHead(x []float32, headDim int, f func([]float32) []float32) []float32 {
	out := make([]float32, len(x))
	for off := 0; off < len(x); off += headDim {
		copy(out[off:off+headDim], f(x[off:off+headDim]))
	}
	return out
}

func TestPermuteAppliesPerHeadNotPerRow(t *testing.T) {
	// A projection's post-transpose plane is [in, H*d_h], so the permutation
	// must run inside each head of each row. Doing it over the whole row would
	// pass the single-head test above and silently mix heads here.
	const dh, heads, rows = 8, 3, 2
	x := make([]float32, rows*heads*dh)
	for i := range x {
		x[i] = float32(i)
	}
	got := make([]float32, len(x))
	copy(got, x)
	if err := permuteHeads(got, heads*dh, dh); err != nil {
		t.Fatalf("permuteHeads: %v", err)
	}
	for seg := range rows * heads {
		off := seg * dh
		for i := range dh / 2 {
			if got[off+2*i] != x[off+i] {
				t.Fatalf("segment %d channel %d: got %v, want %v", seg, 2*i, got[off+2*i], x[off+i])
			}
			if got[off+2*i+1] != x[off+i+dh/2] {
				t.Fatalf("segment %d channel %d: got %v, want %v", seg, 2*i+1, got[off+2*i+1], x[off+i+dh/2])
			}
		}
	}
}

func TestQuantizeIntoMatchesTheWholeMatrixCall(t *testing.T) {
	// quantizeInto windows the matrix so the host holds one block rather than a
	// second copy of the tensor. quant blocks from offset zero in runs of
	// Int8Block, so the windowed output must be byte-identical to the
	// whole-matrix call — and if it ever stops being, the int8 path is
	// quantizing against scales the kernel will not use.
	for _, n := range []int{1, 31, 32, 33, 64, 1000} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%7)
		}
		quants := make([]byte, n)
		scales := make([]byte, blocks(n)*2)
		quantizeInto(src, quants, scales)

		wantQ, wantS := quant.Int8Quantize(src)
		for i := range wantQ {
			if int8(quants[i]) != wantQ[i] {
				t.Fatalf("n=%d quant %d: got %d, want %d", n, i, int8(quants[i]), wantQ[i])
			}
		}
		if len(wantS) != blocks(n) {
			t.Fatalf("n=%d: quant returned %d scales, blocks says %d", n, len(wantS), blocks(n))
		}
		for i := range wantS {
			if got := binary.LittleEndian.Uint16(scales[2*i:]); got != wantS[i].Bits() {
				t.Fatalf("n=%d scale %d: got %#04x, want %#04x", n, i, got, wantS[i].Bits())
			}
		}
	}
}

func TestQuantizeRoundTripIsWithinTheMeasuredBound(t *testing.T) {
	// 001-D5: the bound is measured on the blocks that actually exist, not
	// tuned. quant.Int8ErrorBound gives the distance a dot product may be from
	// the unquantized result, so the assertion is derived and cannot be quietly
	// raised.
	const n = 256
	w := make([]float32, n)
	x := make([]float32, n)
	for i := range w {
		w[i] = float32(math.Cos(float64(i)*0.11)) * 0.7
		x[i] = float32(math.Sin(float64(i) * 0.23))
	}
	quants := make([]byte, n)
	scales := make([]byte, blocks(n)*2)
	quantizeInto(w, quants, scales)

	q := make([]int8, n)
	for i, v := range quants {
		q[i] = int8(v)
	}
	s := make([]accel.Float16, blocks(n))
	for i := range s {
		s[i] = accel.Float16FromBits(binary.LittleEndian.Uint16(scales[2*i:]))
	}
	back := quant.Int8Dequantize(q, s)

	var exact, approx float64
	for i := range n {
		exact += float64(x[i]) * float64(w[i])
		approx += float64(x[i]) * float64(back[i])
	}
	// Int8ErrorBound wants one scale per term, which is the scale of the block
	// that term's weight lives in.
	termScales := make([]accel.Float16, n)
	for i := range n {
		termScales[i] = s[i/quant.Int8Block]
	}
	bound := quant.Int8ErrorBound(x, termScales)
	if diff := math.Abs(exact - approx); diff > bound {
		t.Fatalf("dot product is %v from the unquantized result, past the measured bound %v", diff, bound)
	}
	if bound <= 0 {
		t.Fatalf("the bound is %v, so the assertion above proves nothing", bound)
	}
}
