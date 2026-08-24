// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance_test

import (
	"errors"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/internal/conformance"
	"github.com/latere-ai/tgo/internal/oracle"
	"github.com/latere-ai/tgo/nn"
)

// These are specs/010-conformance.md §5's parity checks, one per nn block, on
// the CPU backend at tier 1.
//
// They test the harness by using it, which is the only way a harness is worth
// anything: a Rig nobody ran proves nothing about the tolerances it computes.
// Each budget is composed from the stages the computation actually has, so
// nothing here can be widened without adding a term (010-D3).
//
// The package is conformance_test rather than conformance because these import
// nn, and nn is a consumer of the harness rather than a part of it.

// seeded builds a deterministic plane, exact in f16 so that a comparison is
// charged for the arithmetic rather than for the storage format.
func seeded(n int, salt uint32) []float32 {
	v := make([]float32, n)
	for i := range v {
		s := uint32(i)*2654435761 + salt
		s ^= s << 13
		s ^= s >> 17
		s ^= s << 5
		v[i] = float32(int32(s%32)-16) / 8
	}
	return v
}

func f64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func TestRMSNormParity(t *testing.T) {
	const rows, width = 3, 32
	const eps = 1e-6

	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: eps, Label: "rmsnorm"})
	x := seeded(rows*width, 1)
	gain := seeded(width, 2)

	in := r.Input("x", accel.F32, tensor.Shape{rows, width})
	r.F32("x", x)
	out := nn.RMSNorm(r.G, in, r.G.Gain("gain", width))
	r.F32("gain", gain)

	// One sum of width terms, then a reciprocal square root and a scale. The
	// magnitude is the input's, since a normalization neither cancels nor
	// amplifies beyond its gain.
	terms := conformance.AccumF32(width).
		And(conformance.PrimitiveULP("rsqrt", 2, 1)).
		And(conformance.Magnitude(2))

	r.Parity(out, func() []float64 {
		want := make([]float64, 0, rows*width)
		for i := range rows {
			row := oracle.RMSNorm(f64(x[i*width:(i+1)*width]), f64(gain), eps)
			want = append(want, row...)
		}
		return want
	}, terms)
}

func TestSwiGLUParity(t *testing.T) {
	const n = 48

	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: 1e-6, Label: "swiglu"})
	gate, up := seeded(n, 3), seeded(n, 4)

	g := r.Input("gate", accel.F32, tensor.Shape{1, n})
	u := r.Input("up", accel.F32, tensor.Shape{1, n})
	r.F32("gate", gate)
	r.F32("up", up)
	out := tensor.SwiGLU(r.G.B, g, u)

	// SiLU is a sigmoid and a multiply: one exp at accel's ULP ceiling, then
	// two roundings. No accumulation -- it is elementwise.
	terms := conformance.PrimitiveULP("exp", 2, 1).
		And(conformance.RoundF32(2)).
		And(conformance.Magnitude(4))

	r.Parity(out, func() []float64 {
		return oracle.SwiGLU(f64(gate), f64(up))
	}, terms)
}

func TestMatMulParity(t *testing.T) {
	const m, k, n = 4, 64, 16

	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: 1e-6, Label: "matmul"})
	x, w := seeded(m*k, 5), seeded(k*n, 6)

	xt := r.Input("x", accel.F32, tensor.Shape{m, k})
	r.F32("x", x)
	wt := r.Input("w", accel.F32, tensor.Shape{k, n})
	r.F32("w", w)
	out := tensor.MatMul(r.G.B, xt, wt)

	// A dot product of k terms accumulated in f32. The magnitude is
	// sum|x_i·w_i| rather than the result: a sum whose terms cancel has an
	// error set by the terms that cancelled, and bounding it by a near-zero
	// answer would fail a correct implementation (Magnitude's doc).
	scale := 0.0
	for i := range m {
		for j := range n {
			s := 0.0
			for p := range k {
				s += math.Abs(float64(x[i*k+p]) * float64(w[p*n+j]))
			}
			scale = math.Max(scale, s)
		}
	}
	terms := conformance.AccumF32(k).And(conformance.Magnitude(scale))

	r.Parity(out, func() []float64 {
		return oracle.MatMul(f64(x), f64(w), m, k, n)
	}, terms)
}

// TestMatMulF16WeightsChargesTheStorageTerm is §5.1's finding made concrete:
// the tolerance on a matmul against narrow weights is set by the storage
// format, not by the accumulator.
func TestMatMulF16WeightsChargesTheStorageTerm(t *testing.T) {
	const m, k, n = 2, 32, 8

	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: 1e-6, Label: "matmul-f16"})
	x := seeded(m*k, 7)
	w := seeded(k*n, 8)
	// Perturb off the f16 grid so the storage term is real rather than zero.
	for i := range w {
		w[i] += float32(i%3) * 1e-4
	}

	xt := r.Input("x", accel.F32, tensor.Shape{m, k})
	r.F32("x", x)
	wt := r.Input("w", accel.F16, tensor.Shape{k, n})
	stored := r.F16("w", w) // what the device actually holds

	out := tensor.MatMul(r.G.B, xt, wt)

	scale := 0.0
	for i := range m {
		for j := range n {
			s := 0.0
			for p := range k {
				s += math.Abs(float64(x[i*k+p]) * float64(stored[p*n+j]))
			}
			scale = math.Max(scale, s)
		}
	}
	terms := conformance.AccumF32(k).And(conformance.Magnitude(scale))

	// The reference uses the STORED weights, not the originals: charging the
	// comparison for a rounding the device never performed would make the
	// tolerance say something about the fixture rather than about accel.
	r.Parity(out, func() []float64 {
		return oracle.MatMul(f64(x), f64(stored), m, k, n)
	}, terms)
}

// TestCompareFailsOutsideTheBudget is the negative half: a harness that cannot
// fail is not a harness. Compare is exercised through a recording TB so the
// failure is observed rather than propagated.
func TestCompareFailsOutsideTheBudget(t *testing.T) {
	got := []float32{1.0, 2.0}
	want := []float64{1.0, 2.5} // 20% out
	terms := conformance.AccumF32(4).And(conformance.Magnitude(2))

	rec := &failTB{TB: t}
	func() {
		defer func() {
			if p := recover(); p != nil && p != errAbort {
				panic(p)
			}
		}()
		conformance.Compare(rec, got, want, terms, "deliberate mismatch")
	}()
	if !rec.failed {
		t.Fatalf("Compare accepted a 20%% deviation under a budget of %g; a harness "+
			"that cannot fail proves nothing", terms.Bound(2))
	}

	// And the positive half, so the failure above is not simply "always fails".
	conformance.Compare(t, []float32{1, 2}, []float64{1, 2}, terms, "exact agreement")
}

// errAbort unwinds a recording TB the way t.Fatalf unwinds a real one. Go 1.27
// makes panic(nil) a runtime error, so the sentinel is explicit.
var errAbort = errors.New("conformance: test aborted by a recording TB")

type failTB struct {
	testing.TB
	failed bool
}

func (f *failTB) Errorf(format string, args ...any) { f.failed = true }
func (f *failTB) Fatalf(format string, args ...any) { f.failed = true; panic(errAbort) }
func (f *failTB) Helper()                           {}

// TestQuantMatMulParity is the int8 path, whose budget is the one place
// specs/010-conformance.md §3 says a measurement beats an assumption.
//
// quant.Int8ErrorBound is driven by the LARGEST weight in each block of 32, so
// a bound measured on the blocks that were actually built is a different number
// from one measured on noise. Int8MatMulBound computes it over the real scales,
// which is why this test can assert rather than tolerate.
func TestQuantMatMulParity(t *testing.T) {
	const m, k, n = 2, 64, 8

	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: 1e-6, Label: "quantmatmul"})
	x := seeded(m*k, 11)
	w := seeded(k*n, 12)
	// An outlier per block, which is what makes a real weight matrix quantize
	// worse than synthetic noise: the block's scale is set by its largest
	// magnitude and every other weight in it is represented against that.
	for i := 0; i < len(w); i += 32 {
		w[i] *= 7
	}

	xt := r.Input("x", accel.F32, tensor.Shape{m, k})
	r.F32("x", x)
	// I8 binds both planes a quantized port declares and returns what the
	// device actually holds.
	dequant, scales := r.I8("w", w)
	wq := tensor.Quantized{
		Quants: r.Input("w", accel.I8, tensor.Shape{k, n}),
		Scales: r.Input("w"+nn.ScaleSuffix, accel.F16, tensor.Shape{len(scales)}),
	}
	out := tensor.QuantMatMul(r.G.B, xt, wq)

	// The int8 term is absolute and measured on the blocks that exist, plus
	// the f32 accumulation the product still performs.
	scale := 0.0
	for i := range m {
		for j := range n {
			s := 0.0
			for p := range k {
				s += math.Abs(float64(x[i*k+p]) * float64(dequant[p*n+j]))
			}
			scale = math.Max(scale, s)
		}
	}
	terms := conformance.QuantInt8(conformance.Int8MatMulBound(x, scales, m, k, n)).
		And(conformance.AccumF32(k)).
		And(conformance.Magnitude(scale))

	// The reference multiplies the DEQUANTIZED weights: the quantization error
	// is already carried by the QuantInt8 term, and charging it twice would
	// make the budget describe the fixture rather than accel.
	r.Parity(out, func() []float64 {
		return oracle.MatMul(f64(x), f64(dequant), m, k, n)
	}, terms)
}

// TestScalarAndU32Bindings covers the two binding kinds the parity tests above
// do not reach, through the operator that needs both: RoPE takes its frequency
// base as a scalar and its positions as a u32 tensor (accel 043).
func TestScalarAndU32Bindings(t *testing.T) {
	const rows, width = 4, 16

	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: 1e-6, Label: "rope"})
	x := seeded(rows*width, 13)

	in := r.Input("x", accel.F32, tensor.Shape{rows, width})
	r.F32("x", x)
	pos := []uint32{0, 1, 2, 3}
	p := r.Input("pos", accel.U32, tensor.Shape{rows})
	r.U32("pos", pos)
	const base = 10000.0
	r.ScalarF32("rope_base", base)

	out := tensor.RoPE(r.G.B, in, width, "rope_base", p)

	// A rotation is a pair of multiplies and an add per channel, with sin and
	// cos at accel's absolute ceiling.
	terms := conformance.PrimitiveAbs("sincos", 1.0/(1<<20), 2).
		And(conformance.RoundF32(3)).
		And(conformance.Magnitude(2))

	ints := make([]int, len(pos))
	for i, v := range pos {
		ints[i] = int(v)
	}
	r.Parity(out, func() []float64 {
		// Interleaved, because that is accel's kernel; the half-split
		// permutation is the loader's and never reaches a graph
		// (specs/004-model-graph.md §2.5.2).
		return oracle.RoPE(f64(x), rows, width, width, base, ints, oracle.StyleInterleaved)
	}, terms)
}

// TestScalarU32Binds covers the binding kind the parity tests above do not
// reach: a prefill's causal base is a u32 scalar (specs/004-model-graph.md §3),
// and Attention is the operator that takes one.
//
// The assertion is that it compiles and runs: accel refuses a declared-but-
// unbound scalar and a bound-but-undeclared one alike, so a graph that reaches
// a result has proved the pairing.
func TestScalarU32Binds(t *testing.T) {
	const heads, kvHeads, headDim, capacity = 2, 1, 8, 4

	r := conformance.New(t, conformance.Tier1, conformance.Options{Eps: 1e-6, Label: "scalar-u32"})
	q := r.Input("q", accel.F32, tensor.Shape{2, heads, headDim})
	r.F32("q", seeded(2*heads*headDim, 21))
	lengths := r.Input("lengths", accel.U32, tensor.Shape{1})
	r.U32("lengths", []uint32{2})

	k := tensor.NewState(r.G.B, tensor.StateDesc{
		Name: "k", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim}})
	v := tensor.NewState(r.G.B, tensor.StateDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim}})
	r.F32("k", make([]float32, capacity*kvHeads*headDim))
	r.F32("v", make([]float32, capacity*kvHeads*headDim))

	r.ScalarF32("scale", 1/float32(math.Sqrt(headDim)))
	r.ScalarU32("base", 0) // the binding under test

	out := tensor.Attention(r.G.B, q, k, v, tensor.AttentionOptions{
		Lengths: lengths, ScaleName: "scale", BaseName: "base"})

	got, plan := r.Run(out)
	if plan == nil {
		t.Fatal("a graph binding a u32 scalar did not compile")
	}
	if len(got) != 2*heads*headDim {
		t.Fatalf("attention returned %d values, want %d", len(got), 2*heads*headDim)
	}
}
