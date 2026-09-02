// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 The tgo Authors. All rights reserved.

package nn_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// The weights below are multiples of 1/4 and smaller than 8, so f16 holds them
// exactly: an f16 mantissa is 11 bits and 1/4 is a power of two. That is what
// lets a reference computed in f64 be compared against the graph without a
// rounding term in the tolerance -- the only error left is the f32 accumulation
// the kernel does.
var (
	linX = []float32{1, -2, 0.5, 0.25, 3, -1}
	linW = []float32{
		0.5, -0.25, 1, 0.75,
		2, 0.5, -1.5, 0.25,
		-0.75, 1, 0.25, -2,
	}
)

// matmul is the reference: three nested loops from the definition of a product,
// sharing no structure with the kernel.
func matmul(x, w []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		for j := range n {
			var acc float64
			for p := range k {
				acc += x[i*k+p] * w[p*n+j]
			}
			out[i*n+j] = acc
		}
	}
	return out
}

func f64s(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func TestLinearMultipliesF32ActivationsByF16Weights(t *testing.T) {
	r := newRig(t, 1e-6)
	x := r.input("x", accel.F32, tensor.Shape{2, 3})
	r.f32("x", linX)
	w := r.g.Weight("wq", tensor.Shape{3, 4})
	r.f16("block.wq", linW)

	got, plan := r.run(nn.Linear(r.g, x, w))

	want := matmul(f64s(linX), f64s(linW), 2, 3, 4)
	// By hand, the first output: 1*0.5 + (-2)*2 + 0.5*(-0.75) = -3.875.
	if want[0] != -3.875 {
		t.Fatalf("the reference is %v, and by hand it is -3.875", want[0])
	}
	// 1e-6 relative: the f32 accumulation over K=3, the only inexact term --
	// the weights are exact in f16 and the activations are exact in f32.
	closeTo(t, got, want, 1e-6, "linear")

	// No cast in front of the projection: specs/010-conformance.md C8 closed,
	// and a Cast here would be one dispatch per projection buying nothing.
	for _, s := range plan.Selections() {
		if s.Op == "Cast" {
			t.Fatalf("the graph casts: %v", s)
		}
	}
	sel := selected(t, plan, "MatMul")
	if !strings.Contains(sel.Reason, "f32 activations against f16 weights") {
		t.Fatalf("MatMul became %q because %q; want the mixed GEMM",
			sel.Kernel, sel.Reason)
	}
}

// A decode step is M=1, and specs/004-model-graph.md section 2.1 says MatMul
// selects the matrix-vector kernel there. It does not, for the widths a
// transformer actually has: accel's matrix-vector kernel reads f16 on *both*
// operands and tgo's activations are f32, so an f16 weight takes the tile and
// the selection says how many of its rows are idle. Asserted here because the
// spec's claim is the one a reader would rely on.
func TestLinearAtOneRowTakesTheTileForF16Weights(t *testing.T) {
	r := newRig(t, 1e-6)
	x := r.input("x", accel.F32, tensor.Shape{1, 3})
	r.f32("x", linX[:3])
	w := r.g.Weight("wq", tensor.Shape{3, 4})
	r.f16("block.wq", linW)

	got, plan := r.run(nn.Linear(r.g, x, w))

	closeTo(t, got, matmul(f64s(linX[:3]), f64s(linW), 1, 3, 4), 1e-6, "linear")
	sel := selected(t, plan, "MatMul")
	if len(sel.Rejected) == 0 || !strings.Contains(strings.Join(sel.Rejected, " "), "matrix-vector") {
		t.Fatalf("MatMul at M=1 became %q rejecting %v; the matrix-vector kernel should "+
			"appear as rejected", sel.Kernel, sel.Rejected)
	}
}

// The int8 path is the one a large model takes, and it does have an M=1
// specialization: specs/010-conformance.md C15.
func TestLinearAtOneRowTakesTheMatrixVectorKernelForInt8Weights(t *testing.T) {
	r := newRig(t, 1e-6)
	r.g.Stored = func(string) nn.Form { return nn.FormInt8 }
	x := r.input("x", accel.F32, tensor.Shape{1, 3})
	r.f32("x", linX[:3])
	w := r.g.Weight("wq", tensor.Shape{3, 4})
	dequantized := r.quantize("block.wq", linW)

	got, plan := r.run(nn.Linear(r.g, x, w))

	// The reference multiplies the *dequantized* weights, which is what the
	// kernel reads. Asserting against the original weights would be asserting
	// the quantizer's error budget, which specs/027-quantization.md owns.
	want := matmul(f64s(linX[:3]), f64s(dequantized), 1, 3, 4)
	// 1e-5 relative: the f32 accumulation over K=3 against dequantized
	// weights, whose f16 scales are exact only to f16.
	closeTo(t, got, want, 1e-5, "quantized linear")

	sel := selected(t, plan, "QuantMatMul")
	if !strings.Contains(sel.Reason, "M is 1") {
		t.Fatalf("QuantMatMul at M=1 became %q because %q; want the matrix-vector kernel",
			sel.Kernel, sel.Reason)
	}
	if _, ok := r.bufs["block.wq"+nn.ScaleSuffix]; !ok {
		t.Fatal("the scale plane was never declared under its name")
	}
}

func TestRMSNormScalesEachRowByItsOwnRootMeanSquare(t *testing.T) {
	const eps = 1e-6
	r := newRig(t, eps)
	// Two rows of deliberately different magnitude: a normalization that used a
	// shared statistic would agree with this reference on equal rows.
	values := []float32{1, 2, 3, 4, 10, -20, 30, -40}
	gain := []float32{0.5, 1, 1.5, 2}
	x := r.input("x", accel.F32, tensor.Shape{2, 4})
	r.f32("x", values)
	g := r.g.Gain("norm", 4)
	r.f32("block.norm", gain)

	got, _ := r.run(nn.RMSNorm(r.g, x, g))

	want := make([]float64, len(values))
	for row := range 2 {
		var sq float64
		for i := range 4 {
			v := float64(values[row*4+i])
			sq += v * v
		}
		scale := 1 / math.Sqrt(sq/4+eps)
		for i := range 4 {
			want[row*4+i] = float64(values[row*4+i]) * scale * float64(gain[i])
		}
	}
	// 1e-6 relative: the f32 reciprocal square root, which is the only step
	// where the kernel and an f64 reference differ.
	closeTo(t, got, want, 1e-6, "rmsnorm")
}

func TestSwiGLUMLPGatesTheUpProjectionAndProjectsBack(t *testing.T) {
	r := newRig(t, 1e-6)
	x := r.input("x", accel.F32, tensor.Shape{2, 3})
	r.f32("x", linX)
	gateW := []float32{0.5, -1, 0.25, 1.5, 0.75, -0.5} // [3, 2]
	upW := []float32{1, 0.5, -0.25, 2, 0.5, -1}        // [3, 2]
	downW := []float32{0.25, -0.5, 1, 1.5, 0.75, -2}   // [2, 3]
	gate := r.g.Weight("wgate", tensor.Shape{3, 2})
	up := r.g.Weight("wup", tensor.Shape{3, 2})
	down := r.g.Weight("wdown", tensor.Shape{2, 3})
	r.f16("block.wgate", gateW)
	r.f16("block.wup", upW)
	r.f16("block.wdown", downW)

	got, _ := r.run(nn.SwiGLUMLP(r.g, x, gate, up, down))

	// The composed SiLU-then-Mul form is the reference the fused kernel is
	// checked against (004-D5).
	g := matmul(f64s(linX), f64s(gateW), 2, 3, 2)
	u := matmul(f64s(linX), f64s(upW), 2, 3, 2)
	act := make([]float64, len(g))
	for i := range g {
		act[i] = g[i] / (1 + math.Exp(-g[i])) * u[i]
	}
	want := matmul(act, f64s(downW), 2, 2, 3)
	// 1e-5 relative: the f32 exponential inside SiLU, which is the widest term
	// here; the two projections contribute the f32 accumulation of K=3 and K=2.
	closeTo(t, got, want, 1e-5, "swiglu mlp")
}

func TestAnOperandCarriesExactlyOneForm(t *testing.T) {
	r := newRig(t, 1e-6)
	x := r.input("x", accel.F32, tensor.Shape{1, 3})
	r.refuses(nn.Linear(r.g, x, nn.Operand{}), "an int8 pair, or an int4 triple")
}

// TestAHalfBuiltOperandIsRefused covers every incomplete form, and the third
// one is why the check is worth having.
//
// A code plane bound against another matrix's metadata compiles, runs, and
// produces noise -- which is what bundling the planes into one struct is meant
// to prevent, and what a caller assembling one by hand can still get wrong.
func TestAHalfBuiltOperandIsRefused(t *testing.T) {
	plane := func(r *rig, name string, dt accel.DType, shape tensor.Shape) *tensor.Tensor {
		return tensor.Weight(r.g.B, tensor.ValueDesc{Name: name, DType: dt, Shape: shape})
	}
	for _, c := range []struct {
		name string
		of   func(*rig) nn.Operand
	}{
		{"int8 quants with no scales", func(r *rig) nn.Operand {
			return nn.Operand{Quant: tensor.Quantized{
				Quants: plane(r, "q", accel.I8, tensor.Shape{3, 4})}}
		}},
		{"int4 codes with no scales or zeros", func(r *rig) nn.Operand {
			return nn.Operand{Packed: tensor.Int4{
				Codes: plane(r, "c", accel.U32, tensor.Shape{2}), Weights: 12}}
		}},
		{"int4 with no zero plane", func(r *rig) nn.Operand {
			return nn.Operand{Packed: tensor.Int4{
				Codes:  plane(r, "c", accel.U32, tensor.Shape{2}),
				Scales: plane(r, "s", accel.F16, tensor.Shape{1}), Weights: 12}}
		}},
		{"int4 that declares no weight count", func(r *rig) nn.Operand {
			return nn.Operand{Packed: tensor.Int4{
				Codes:  plane(r, "c", accel.U32, tensor.Shape{2}),
				Scales: plane(r, "s", accel.F16, tensor.Shape{1}),
				Zeros:  plane(r, "z", accel.F16, tensor.Shape{1})}}
		}},
		{"two forms at once", func(r *rig) nn.Operand {
			return nn.Operand{
				Dense: plane(r, "d", accel.F16, tensor.Shape{3, 4}),
				Quant: tensor.Quantized{
					Quants: plane(r, "q", accel.I8, tensor.Shape{3, 4}),
					Scales: plane(r, "qs", accel.F16, tensor.Shape{1})},
			}
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := newRig(t, 1e-6)
			x := r.input("x", accel.F32, tensor.Shape{1, 3})
			r.refuses(nn.Linear(r.g, x, c.of(r)), "exactly one of them")
		})
	}
}

func TestAWeightPortNeedsAPositiveExtent(t *testing.T) {
	r := newRig(t, 1e-6)
	r.g.Weight("wq", tensor.Shape{})
	if err := r.g.Err(); err == nil || !strings.Contains(err.Error(), "positive extent") {
		t.Fatalf("the diagnostic is %v; it should name the extent", err)
	}
}

func TestAWeightStoredAsSomethingElseIsRefused(t *testing.T) {
	r := newRig(t, 1e-6)
	r.g.Stored = func(string) nn.Form { return nn.Form(99) }
	r.g.Weight("wq", tensor.Shape{3, 4})
	if err := r.g.Err(); err == nil || !strings.Contains(err.Error(), "block.wq") {
		t.Fatalf("the diagnostic is %v; it should name the port", err)
	}
}

func TestAGainIsOneValuePerFeature(t *testing.T) {
	r := newRig(t, 1e-6)
	r.g.Gain("norm", 0)
	if err := r.g.Err(); err == nil || !strings.Contains(err.Error(), "one value per feature") {
		t.Fatalf("the diagnostic is %v; it should name the width", err)
	}
}

// A gain of the wrong width is accel's refusal, and it is worth a test here
// because it is the diagnostic a per-head norm produces when its reshape is
// missing: the gain is headDim values and the row would be H*headDim wide.
func TestAGainOfTheWrongWidthIsRefused(t *testing.T) {
	r := newRig(t, 1e-6)
	x := r.input("x", accel.F32, tensor.Shape{2, 4})
	g := r.g.Gain("norm", 3)
	r.refuses(nn.RMSNorm(r.g, x, g), "gain", "one value per feature")
}

func TestAnEmptyGraphReportsNoError(t *testing.T) {
	var g nn.Graph
	if err := g.Err(); err != nil {
		t.Fatalf("a graph with no builder and no refusal reports %v", err)
	}
	g.B = newRig(t, 1e-6).g.B       // the same rig's builder: a second device would
	if err := g.Err(); err != nil { // outlive nothing and close in the wrong order
		t.Fatalf("a fresh builder reports %v", err)
	}
}

func TestAnOperandKnowsWhichFormItCarries(t *testing.T) {
	r := newRig(t, 1e-6)
	dense := r.g.Weight("wq", tensor.Shape{3, 4})
	if dense.IsQuant() {
		t.Fatal("an f16 operand reports itself quantized")
	}
	r.g.Stored = func(string) nn.Form { return nn.FormInt8 }
	if !r.g.Weight("wk", tensor.Shape{3, 4}).IsQuant() {
		t.Fatal("an int8 operand reports itself dense")
	}
	// Either plane on its own is still the quantized form, half built. An
	// operand that reported itself dense here would send a caller looking for
	// a dense plane that was never going to arrive, and Linear's diagnostic
	// names the missing half instead.
	half := nn.Operand{Quant: tensor.Quantized{
		Scales: r.g.Weight("wk"+nn.ScaleSuffix, tensor.Shape{1}).Quant.Quants,
	}}
	if !half.IsQuant() {
		t.Fatal("an operand carrying only the scale plane reports itself dense")
	}
}

// A refusal from nn and one from accel are both reported, because a caller who
// checked only one would compile a graph that had already failed.
func TestErrReportsBothItsOwnRefusalsAndAccels(t *testing.T) {
	r := newRig(t, 1e-6)
	r.g.Prefix = "" // and a graph with no prefix still says where it was
	x := r.input("x", accel.F32, tensor.Shape{2, 4})
	nn.Linear(r.g, x, nn.Operand{})                     // nn's
	tensor.RMSNorm(r.g.B, x, r.g.Gain("norm", 3), 1e-6) // accel's
	err := r.g.Err()
	if err == nil {
		t.Fatal("two refusals were reported as none")
	}
	for _, want := range []string{"<no prefix>", "accel/tensor"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("the diagnostic is %q, which does not mention %q", err, want)
		}
	}
}

// rms_norm_eps is a per-model constant and [nn.Graph] is where it lives, so a
// block that used a constant of its own would agree with every other test here
// -- they all build a graph at 1e-6, which is the value it would have picked.
//
// The values are small on purpose: eps is added to the mean square, so it is
// visible only when the mean square is of its order. Here the mean square is
// 2.5e-4 against an eps of 0.25, and the two divisors differ by a factor of 32.
func TestRMSNormUsesTheGraphsEps(t *testing.T) {
	const eps = 0.25
	values := []float32{0.01, 0.02, -0.01, 0.02}
	gain := []float32{1, 1, 1, 1}

	r := newRig(t, eps)
	x := r.input("x", accel.F32, tensor.Shape{1, 4})
	r.f32("x", values)
	g := r.g.Gain("norm", 4)
	r.f32("block.norm", gain)
	got, _ := r.run(nn.RMSNorm(r.g, x, g))

	reference := func(eps float64) []float64 {
		var sq float64
		for _, v := range values {
			sq += float64(v) * float64(v)
		}
		scale := 1 / math.Sqrt(sq/4+eps)
		out := make([]float64, len(values))
		for i, v := range values {
			out[i] = float64(v) * scale * float64(gain[i])
		}
		return out
	}
	// 1e-6 relative: the f32 reciprocal square root, as above.
	closeTo(t, got, reference(eps), 1e-6, "rmsnorm at eps 0.25")
	// And it is not the 1e-6 every other graph in this file is built at:
	// measured maximum relative deviation between the two divisors, 0.54.
	differs(t, got, reference(1e-6), 1e-2, "rmsnorm at eps 1e-6")
}

// The scale plane's port name is a contract with the loader, which uploads two
// buffers for one checkpoint tensor and has nothing but this to name the second
// by. Pinned as a literal here, and end to end: the graph below compiles only
// if the port it declared is the name the rig bound.
func TestTheScalePlaneIsDeclaredUnderTheWeightsNameAndSuffix(t *testing.T) {
	if nn.ScaleSuffix != ".scales" {
		t.Fatalf("ScaleSuffix is %q; the loader writes %q", nn.ScaleSuffix, ".scales")
	}
	r := newRig(t, 1e-6)
	r.g.Stored = func(string) nn.Form { return nn.FormInt8 }
	x := r.input("x", accel.F32, tensor.Shape{1, 3})
	r.f32("x", linX[:3])
	w := r.g.Weight("wq", tensor.Shape{3, 4})

	quants, scales := quant.Int8Quantize(linW)
	r.bind("block.wq", accel.I8, len(quants), quants)
	bits := make([]uint16, len(scales))
	for i, s := range scales {
		bits[i] = s.Bits()
	}
	// The literal name, not nn.ScaleSuffix: a test that spelled the constant
	// would agree with any value the constant took.
	r.bind("block.wq.scales", accel.F16, len(bits), bits)

	got, _ := r.run(nn.Linear(r.g, x, w))
	// 1e-5 relative, as in the quantized projection above: the f32
	// accumulation over K=3 against weights whose scales are f16.
	closeTo(t, got, matmul(f64s(linX[:3]), f64s(quant.Int8Dequantize(quants, scales)), 1, 3, 4),
		1e-5, "quantized linear")
}

// TestAnF16CacheNarrowsTheScatteredRows is tgo's half of C24, built and
// checked while accel's half is filed.
//
// One kernel reads the rows and writes the state, so the two share a dtype and
// accel refuses the pair split apart. The projections that produce the rows are
// f32, so an f16 cache needs a Cast — which this block records now, and did not
// before, and which is the reason model.GraphSpec stopped refusing f16
// outright.
//
// The dtype is a config field and not something read off the State because a
// tensor.State does not report its own. The caller knows: it allocated it.
func TestAnF16CacheNarrowsTheScatteredRows(t *testing.T) {
	for _, c := range []struct {
		name  string
		cache accel.DType
		casts int
	}{
		{"an f32 cache scatters what the projections produced", accel.F32, 0},
		{"an f16 cache narrows the key and the value", accel.F16, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := newParts(t)
			p.cfg.Cache = c.cache
			p.k = tensor.NewState(p.r.g.B, tensor.StateDesc{
				Name: "k16", DType: c.cache,
				Shape: tensor.Shape{attCap, attKVHeads, attHeadDim},
			})
			p.v = tensor.NewState(p.r.g.B, tensor.StateDesc{
				Name: "v16", DType: c.cache,
				Shape: tensor.Shape{attCap, attKVHeads, attHeadDim},
			})
			out := p.record()
			if err := p.r.g.Err(); err != nil {
				t.Fatalf("recording over a %v cache: %v", c.cache, err)
			}
			plan := p.r.compile(t, out)
			casts := 0
			for _, s := range plan.Selections() {
				if s.Op == "Cast" {
					casts++
				}
			}
			if casts != c.casts {
				t.Fatalf("a %v cache recorded %d casts, want %d; one kernel reads the "+
					"rows and writes the state, so an f16 state needs f16 rows and an "+
					"f32 state must not pay for a conversion it does not need",
					c.cache, casts, c.casts)
			}
		})
	}
}
