// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/weights"
)

// The re-audit of every register row still open, run when accel's tracker went
// to zero open issues.
//
// specs/010-conformance.md §2.3 rule 1 says an open row cites an *open* issue,
// and every issue tgo filed is now closed while four rows are not. §2.2.1 is why
// that is a question and not an answer: on 2026-08-24 accel closed ten issues
// and four of the capabilities were still absent, because each fix matched its
// issue's title and a title is a summary where a register row is a capability.
//
// So each of these probes asks what the row asks, by value.

// TestC21Int4IsRepresentableAndComputes re-audits C21, the row that blocks a
// target model: at int8 a 27B model is 25.1 GiB against a 24 GiB device.
//
// accel closed [#22](https://github.com/golang-design/accel/issues/22) with a
// representation and two kernels. The value question is whether a 4-bit matrix
// reconstructs and multiplies to within its own declared bound at a shape a
// transformer has.
func TestC21Int4IsRepresentableAndComputes(t *testing.T) {
	// A transformer's shape rather than a square: K is the activation width and
	// a multiple of the group, N is the output width.
	const m, k, n = 3, 2 * quant.Int4Group, 5

	w := spread(k*n, 17)
	codes, scales, zeros := quant.Int4Quantize(w)
	back := quant.Int4Dequantize(codes, scales, zeros, k*n)
	a := spread(m*k, 23)

	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c21-int4"})
	at := r.Input("a", accel.F32, tensor.Shape{m, k})
	r.F32("a", a)
	ct := r.Input("codes", accel.U32, tensor.Shape{len(codes)})
	r.bind("codes", accel.U32, len(codes), codes)
	st := r.Input("scales", accel.F16, tensor.Shape{len(scales)})
	r.bind("scales", accel.F16, len(scales), bits16(scales))
	zt := r.Input("zeros", accel.F16, tensor.Shape{len(zeros)})
	r.bind("zeros", accel.F16, len(zeros), bits16(zeros))

	out := tensor.Int4MatMul(r.G.B, at, tensor.Int4{
		Codes: ct, Scales: st, Zeros: zt, Weights: k * n,
	})
	if err := r.G.Err(); err != nil {
		t.Fatalf("a 4-bit matrix at a transformer's shape was refused: %v", err)
	}
	got, plan := r.Run(out)

	// The reference multiplies the *reconstructed* weights, so the comparison
	// is charged for the accumulation and not a second time for the storage:
	// what the kernel reads is what this computes from.
	want := make([]float64, m*n)
	mag := 0.0
	for i := range m {
		for j := range n {
			acc := 0.0
			for p := range k {
				x, y := float64(a[i*k+p]), float64(back[p*n+j])
				acc += x * y
				mag += math.Abs(x * y)
			}
			want[i*n+j] = acc
		}
	}
	// The accumulation only: the reference multiplies the *reconstructed*
	// weights, so what the kernel reads is what this computes from and the
	// storage error is not charged twice.
	Compare(t, got, want, AccumF32(k).And(Magnitude(mag/float64(m*n))),
		"a 4-bit matrix multiply")

	var kernel string
	for _, s := range plan.Selections() {
		if s.Op == "Int4MatMul" {
			kernel = s.Kernel
		}
	}
	if kernel == "" {
		t.Fatal("no Int4MatMul selection")
	}
	t.Logf("C21: %s, %d weights in %d words plus %d groups of scale and zero",
		kernel, k*n, len(codes), len(scales))
}

// TestC21TgoStoresInt4Weights is what the re-audit's finding turned into.
//
// The row closed on accel's half while tgo's loader still named only f16 and
// int8, so this asserted the gap. It asserts the capability now, and the
// arithmetic that made the gap worth naming: a form at 0.53125 bytes per weight
// rather than 1.0625 is what decides whether a 27B-class model fits hardware
// people own.
//
// [C8](#2-the-register) is why the two halves were tracked separately. accel
// answered "MatMul is f16-only" with an f32 GEMM, the report was accepted, and
// the casts stayed — because a fix that matches an issue's title need not
// remove the cost the row is about.
func TestC21TgoStoresInt4Weights(t *testing.T) {
	if got := weights.Int4.String(); got != "int4" {
		t.Fatalf("weights.Int4 names itself %q", got)
	}
	// The footprint specs/001-weights.md §5.1 states, checked against the
	// loader rather than restated: eight codes to a u32 word, and an f16 scale
	// and an f16 zero per group.
	const n = 8 * quant.Int4Group
	words, groups := (n+7)/8, (n+quant.Int4Group-1)/quant.Int4Group
	want := int64(words*4 + groups*2*2)
	got := weights.Value{Precision: weights.Int4, Elements: n}.Bytes()
	if got != want {
		t.Fatalf("%d weights at int4 cost %d bytes, want %d", n, got, want)
	}
	if per := float64(got) / n; per != 0.53125 {
		t.Fatalf("int4 costs %.6f bytes per weight, want 0.53125", per)
	}
	// And it is **exactly** half of int8, which is not a coincidence and is the
	// point of accel grouping 128 rather than 32.
	//
	// The payload halves, 1 byte per weight to 0.5. The metadata would double
	// as a *share* of a payload half the size -- 6.2% becoming 12.5% -- so the
	// group doubles too, and 2 bytes per 32 becomes 4 bytes per 128: 0.0625 to
	// 0.03125. Both terms halve, so the total does.
	i8 := weights.Value{Precision: weights.Int8, Elements: n}.Bytes()
	if got*2 != i8 {
		t.Fatalf("int4 costs %d bytes and int8 costs %d; the payload halves and the "+
			"group doubles so the metadata halves with it, which makes the total "+
			"exactly half", got, i8)
	}
	t.Logf("C21: %.5f bytes per weight against int8's %.5f, so a 27B checkpoint is "+
		"13.4 GiB rather than 26.7", float64(got)/n, float64(i8)/n)
}

// TestC7ABf16GemmIsStillAbsent re-audits C7.
//
// The row was narrowed once already: `Cast` widens bf16 to f32, so only the
// GEMM is missing, and a weight would not want a per-step cast anyway. Issue 14
// is closed and this is what closed against it — a mixed GEMM, not a bf16 one.
func TestC7ABf16GemmIsStillAbsent(t *testing.T) {
	const m, k, n = 2, 4, 3
	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c7-bf16"})
	x := r.Input("x", accel.F32, tensor.Shape{m, k})
	w := r.Input("w", accel.BF16, tensor.Shape{k, n})
	tensor.MatMul(r.G.B, x, w)
	err := r.G.Err()
	if err == nil {
		t.Fatal("a bf16 weight matrix multiplied without complaint; C7 is closed " +
			"and the register should say so")
	}
	if !strings.Contains(err.Error(), "bf16") {
		t.Fatalf("the refusal does not name the dtype: %v", err)
	}
	// And the narrowing still holds: widening bf16 to f32 is expressible, which
	// is what makes the workaround a host-side cast at load rather than a wall.
	r2 := New(t, Tier1, Options{Eps: 1e-6, Label: "c7-cast"})
	b := r2.Input("b", accel.BF16, tensor.Shape{k})
	tensor.Cast(r2.G.B, b, accel.F32)
	if err := r2.G.Err(); err != nil {
		t.Fatalf("Cast from bf16 to f32 was refused, so C7's workaround is gone "+
			"too and the row is wider than it says: %v", err)
	}
	t.Logf("C7: %v", err)
}

// bits16 is a scale plane as the port reads it.
func bits16(v []accel.Float16) []uint16 {
	out := make([]uint16, len(v))
	for i, x := range v {
		out[i] = x.Bits()
	}
	return out
}

// TestC17NoSuperBlockRepresentation re-audits the one row that stays open.
//
// Issue 15 is closed **as not planned**, and §2.3 accepted that: accel recorded
// the gap as a `quant_matmul_superblock` row in its kernel corpus, carrying the
// layout, the formula and both workarounds, which is a better record than an
// issue with no plan because the corpus is what someone adding a kernel reads.
//
// So the row's citation is a closed issue on purpose, and what has to be
// checked is the capability rather than the tracker: `quant` registers the
// representations, and a K-quant super-block is not one of them.
func TestC17NoSuperBlockRepresentation(t *testing.T) {
	// A Q4_K super-block is 256 weights as eight sub-blocks of 32, each with a
	// 6-bit scale and a 6-bit minimum over two fp16 super-scales:
	//
	//	w_i = d·s_j·q_i - d_min·m_j,   j = floor(i/32)
	//
	// Two levels of scale and a per-sub-block minimum. What quant registers is
	// one level and, at int4, a zero point per group -- which is a different
	// shape, not a smaller one.
	if quant.Int4Group != 128 {
		t.Errorf("the int4 group is %d; §2's row quotes 128", quant.Int4Group)
	}
	// int8 is symmetric: a scale per block and no minimum.
	w := spread(64, 5)
	q8, s8 := quant.Int8Quantize(w)
	if len(q8) != len(w) || len(s8) == 0 {
		t.Fatalf("int8 quantized %d weights into %d codes and %d scales",
			len(w), len(q8), len(s8))
	}
	// int4 has a zero point, which is the second thing a super-block needs and
	// the only one it has: still one level of scale, and no per-sub-block
	// minimum under a super-scale.
	_, s4, z4 := quant.Int4Quantize(w)
	if len(s4) != len(z4) {
		t.Fatalf("int4 produced %d scales and %d zeros", len(s4), len(z4))
	}
	if want := (len(w) + quant.Int4Group - 1) / quant.Int4Group; len(s4) != want {
		t.Fatalf("int4 grouped %d weights into %d scales, want %d; a super-block "+
			"would be two levels of scale over eight sub-blocks", len(w), len(s4), want)
	}
	t.Logf("C17: quant registers int8 (one scale per block) and int4 (a scale and "+
		"a zero per %d), and a Q4_K super-block is two levels of scale over eight "+
		"sub-blocks with a minimum each; nothing reads one", quant.Int4Group)
}

// TestInt4MatMulIsInsideAccelsOwnBound is the other half of C21's numerics, and
// the one that says the representation is good enough rather than that the
// kernel implements it.
//
// The comparison above reconstructs the weights and charges only the f32
// accumulation, which asks "does the device compute what the format defines".
// This compares against the **original** weights and charges
// [quant.Int4ErrorBound], which asks "is what the format defines close enough
// to what was quantized".
//
// # What a green result here does not say
//
// [§3](../../specs/010-conformance.md) asks for quantization error on *real*
// blocks, and these are synthetic. `quant.Int4Quantize` spends a group's codes
// over that group's range, so weights drawn from one distribution — with no
// outlier channel to stretch the range — flatter the scheme. Trained
// transformer weights have outliers, which is the entire reason mixed-precision
// schemes exist. The number that decides whether int4 is usable for a model is
// a tier-3 measurement against a checkpoint, and it has not been taken.
func TestInt4MatMulIsInsideAccelsOwnBound(t *testing.T) {
	const m, k, n = 3, 2 * quant.Int4Group, 5

	w := spread(k*n, 17)
	codes, scales, zeros := quant.Int4Quantize(w)
	a := spread(m*k, 23)

	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c21-int4-bound"})
	at := r.Input("a", accel.F32, tensor.Shape{m, k})
	r.F32("a", a)
	ct := r.Input("codes", accel.U32, tensor.Shape{len(codes)})
	r.bind("codes", accel.U32, len(codes), codes)
	st := r.Input("scales", accel.F16, tensor.Shape{len(scales)})
	r.bind("scales", accel.F16, len(scales), bits16(scales))
	zt := r.Input("zeros", accel.F16, tensor.Shape{len(zeros)})
	r.bind("zeros", accel.F16, len(zeros), bits16(zeros))

	got, _ := r.Run(tensor.Int4MatMul(r.G.B, at, tensor.Int4{
		Codes: ct, Scales: st, Zeros: zt, Weights: k * n,
	}))

	// The reference multiplies the weights the checkpoint held, so the whole
	// quantization error is in the comparison and the bound has to cover it.
	want := make([]float64, m*n)
	worst := 0.0
	for i := range m {
		for j := range n {
			acc := 0.0
			// One group range per term, because a per-group figure bounds a dot
			// product only where the caller says which group each term came
			// from -- accel's own precondition, and the reason the bound takes
			// the inputs rather than being a constant.
			// The bound is over the *stored* f16 scale and zero of each
			// term's group since accel 2026-09-02 (its specs/048): the exact
			// range over thirty was not a bound, because the step actually
			// used is the rounded one.
			x := make([]float32, k)
			termScales := make([]accel.Float16, k)
			termZeros := make([]accel.Float16, k)
			for p := range k {
				acc += float64(a[i*k+p]) * float64(w[p*n+j])
				x[p] = a[i*k+p]
				g := (p*n + j) / quant.Int4Group
				termScales[p] = scales[g]
				termZeros[p] = zeros[g]
			}
			want[i*n+j] = acc
			if b := quant.Int4ErrorBound(x, termScales, termZeros); b > worst {
				worst = b
			}
		}
	}
	Compare(t, got, want, AccumF32(k).And(QuantInt4(worst)),
		"a 4-bit matrix against the weights it was quantized from")

	// The margin, reported rather than asserted.
	//
	// A derived bound is an upper bound and is expected to be loose; what a
	// reader needs to know is *how* loose, because a bound the result sits far
	// under is one that would also admit a wrong answer. Asserting a ratio
	// would be 010-D3's tuned tolerance wearing a different hat, so this
	// prints the number and the swap below is what makes the test able to
	// fail.
	seen := 0.0
	for i := range want {
		if d := math.Abs(float64(got[i]) - want[i]); d > seen {
			seen = d
		}
	}
	t.Logf("C21: worst %.3g against a bound of %.3g, a margin of %.1fx, over %d groups",
		seen, worst, worst/seen, (k*n+quant.Int4Group-1)/quant.Int4Group)

	// The control: the scales and the zeros are the same dtype and the same
	// length, so binding them the wrong way round is a mistake the shapes
	// cannot catch. It has to leave the bound.
	r2 := New(t, Tier1, Options{Eps: 1e-6, Label: "c21-int4-swapped"})
	a2 := r2.Input("a", accel.F32, tensor.Shape{m, k})
	r2.F32("a", a)
	c2 := r2.Input("codes", accel.U32, tensor.Shape{len(codes)})
	r2.bind("codes", accel.U32, len(codes), codes)
	s2 := r2.Input("scales", accel.F16, tensor.Shape{len(scales)})
	r2.bind("scales", accel.F16, len(scales), bits16(zeros))
	z2 := r2.Input("zeros", accel.F16, tensor.Shape{len(zeros)})
	r2.bind("zeros", accel.F16, len(zeros), bits16(scales))
	crossed, _ := r2.Run(tensor.Int4MatMul(r2.G.B, a2, tensor.Int4{
		Codes: c2, Scales: s2, Zeros: z2, Weights: k * n,
	}))
	over := 0
	for i := range want {
		if math.Abs(float64(crossed[i])-want[i]) > worst {
			over++
		}
	}
	if over == 0 {
		t.Fatalf("multiplying with the scale and zero planes swapped stayed inside "+
			"the bound at every one of %d elements; a bound nothing violates is a "+
			"bound this test cannot fail", len(want))
	}
}

// TestC24APagedPrefillReadsAnF16Cache closes C24, which blocked the f16 block
// pool for the six hours between filing it and accel shipping the kernel.
//
// The row was [C5](#2-the-register)'s pattern a third time. C5 closed on
// "ScatterRows, prefill and paged decode all take f16" and each of those is
// true; the *combination* was not, because `Attention` selected the f16 prefill
// kernel and then overwrote the selection whenever a page table was supplied.
// A pool is paged by construction, so the narrow cache was reachable for a
// contiguous single sequence and for nobody who shares blocks — the opposite of
// who wants it, since concurrency is what forces paging.
//
// accel chose the pair rather than a fourth kernel name: the width and the
// paging select together now.
func TestC24APagedPrefillReadsAnF16Cache(t *testing.T) {
	const qHeads, kvHeads, headDim, block, maxPages = 4, 2, 4, 4, 4
	const rows, tokens = block * maxPages, 3

	kf := spread(rows*kvHeads*headDim, 13)
	vf := spread(rows*kvHeads*headDim, 29)
	qf := spread(tokens*qHeads*headDim, 7)

	// The f16 cache, and the values widened back: what the kernel reads is what
	// the f32 run must be compared against, or the comparison is charged for
	// the storage error twice.
	half := func(v []float32) ([]uint16, []float32) {
		bits := make([]uint16, len(v))
		back := make([]float32, len(v))
		for i, x := range v {
			h := accel.ToFloat16(x)
			bits[i], back[i] = h.Bits(), h.F32()
		}
		return bits, back
	}
	kb, kw := half(kf)
	vb, vw := half(vf)

	run := func(t *testing.T, dt accel.DType) []float32 {
		t.Helper()
		r := New(t, Tier1, Options{Eps: 1e-6, Label: "c24"})
		shape := tensor.Shape{rows, kvHeads, headDim}
		k := tensor.NewState(r.G.B, tensor.StateDesc{Name: "k", DType: dt, Shape: shape})
		v := tensor.NewState(r.G.B, tensor.StateDesc{Name: "v", DType: dt, Shape: shape})
		if dt == accel.F16 {
			r.bind("k", accel.F16, len(kb), kb)
			r.bind("v", accel.F16, len(vb), vb)
		} else {
			r.F32("k", kw)
			r.F32("v", vw)
		}
		q := r.Input("q", accel.F32, tensor.Shape{tokens, qHeads, headDim})
		r.F32("q", qf)
		pages := r.Input("pages", accel.U32, tensor.Shape{1, maxPages})
		r.U32("pages", []uint32{0, 1, 2, 3})
		lengths := r.Input("lengths", accel.U32, tensor.Shape{1})
		r.U32("lengths", []uint32{tokens})
		r.ScalarF32("scale", 0.5)
		r.ScalarU32("base", 0)
		got, plan := r.Run(tensor.Attention(r.G.B, q, k, v, tensor.AttentionOptions{
			Lengths: lengths, Pages: pages, Block: block,
			ScaleName: "scale", BaseName: "base",
		}))
		for _, s := range plan.Selections() {
			if s.Op == "Attention" {
				t.Logf("C24: a %v cache selected %s", dt, s.Kernel)
			}
		}
		return got
	}

	var narrow, wide []float32
	t.Run("an f16 cache", func(t *testing.T) { narrow = run(t, accel.F16) })
	t.Run("the same values at f32", func(t *testing.T) { wide = run(t, accel.F32) })

	// The f32 run reads the *widened* f16 values, so the two see the same
	// numbers and differ only in how they were stored. What is left is the
	// accumulation order, which is the same kernel shape either way.
	if len(narrow) != len(wide) {
		t.Fatalf("%d values narrow and %d wide", len(narrow), len(wide))
	}
	Compare(t, narrow, f64(wide), AccumF32(headDim).And(AccumF32(tokens)).
		And(Magnitude(magnitudeOf(vw))), "a paged prefill over an f16 cache")
}

// f64 widens a device result so it can stand as a reference.
func f64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// magnitudeOf is the largest absolute value a weighted sum could combine, which
// is what the relative terms apply to when the sum itself can cancel.
func magnitudeOf(v []float32) float64 {
	worst := 0.0
	for _, x := range v {
		if a := math.Abs(float64(x)); a > worst {
			worst = a
		}
	}
	return worst
}

// TestC25AReshapedResultReachesItsOutputPort closes C25, and it is the row this
// register exists for: it was **accept-and-silently-wrong**, not a refusal.
//
// Declaring a reshaped result as a graph output produced all zeros — correct
// shape, no error, and `Contiguous` in front did not help. It cost an hour
// inside a recurrence that was correct, and it would have cost a wrong answer
// in production rather than a red build.
func TestC25AReshapedResultReachesItsOutputPort(t *testing.T) {
	sum := func(t *testing.T, wrap func(*tensor.Builder, *tensor.Tensor) *tensor.Tensor) []float32 {
		t.Helper()
		r := New(t, Tier1, Options{Eps: 1e-6, Label: "c25"})
		x := r.Input("x", accel.F32, tensor.Shape{3, 4})
		y := r.Input("y", accel.F32, tensor.Shape{3, 4})
		r.F32("x", spread(12, 7))
		r.F32("y", spread(12, 13))
		got, _ := r.Run(wrap(r.G.B, tensor.Add(r.G.B, x, y)))
		return got
	}

	var direct, reshaped, packed []float32
	t.Run("direct", func(t *testing.T) {
		direct = sum(t, func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor { return x })
	})
	t.Run("reshaped", func(t *testing.T) {
		reshaped = sum(t, func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
			return tensor.Reshape(b, x, tensor.Shape{2, 6})
		})
	})
	t.Run("reshaped then made contiguous", func(t *testing.T) {
		packed = sum(t, func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
			return tensor.Contiguous(b, tensor.Reshape(b, x, tensor.Shape{2, 6}))
		})
	})

	// A reshape moves no element, so all three hold the same values in the same
	// order whatever shape the port declares. Asserted elementwise rather than
	// as "not all zeros", because zeros were the symptom and agreeing with the
	// direct output is the property.
	for _, c := range []struct {
		what string
		got  []float32
	}{{"a reshaped output", reshaped}, {"a reshaped output made contiguous", packed}} {
		if len(c.got) != len(direct) {
			t.Fatalf("%s produced %d values and the direct one %d",
				c.what, len(c.got), len(direct))
		}
		for i := range direct {
			if c.got[i] != direct[i] {
				t.Fatalf("%s: element %d is %v and the direct output is %v; a reshape "+
					"moves no element", c.what, i, c.got[i], direct[i])
			}
		}
	}
	t.Logf("C25: the same sum is %v direct and %v reshaped", direct[1], reshaped[1])
}
