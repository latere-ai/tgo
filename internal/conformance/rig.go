// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// Options are the per-model constants a recorded graph needs.
type Options struct {
	// Eps is rms_norm_eps. accel refuses a zero one rather than treating an
	// unset field as a default, so a graph with a normalization in it has to
	// say.
	Eps float32

	// Prefix is the weight-name prefix blocks declare their ports under.
	Prefix string

	// Label names the plan in accel's diagnostics.
	Label string
}

// Rig records one graph, compiles it on the tier's device, submits it and
// compares the result against the oracle.
//
// The device is the real one and not a mock, at every tier: a block is a claim
// about what accel computes, and only accel can settle it.
// specs/000-decisions.md decision 8 keeps the real checkpoint out of the
// default run, not the real device.
type Rig struct {
	tb   testing.TB
	tier Tier
	opts Options

	// Dev is the open device, closed by a cleanup.
	Dev *accel.Device

	// RT is the tensor runtime over Dev.
	RT *tensor.Runtime

	// G is the graph blocks record into.
	G *nn.Graph

	views   map[string]accel.BufferView
	buffers map[string]*accel.Buffer
	scalars map[string]tensor.ScalarValue
}

// New opens the tier's device and returns a rig recording into a fresh graph.
//
// It skips or fails the test when the tier's requirements are not met, so a
// caller writes no environment check of its own: that is the point of there
// being one rule (010 §4).
func New(tb testing.TB, tier Tier, opts Options) *Rig {
	tb.Helper()
	dev := Device(tb, tier)
	rt, err := tensor.NewRuntime(dev)
	if err != nil {
		tb.Fatalf("tensor runtime: %v", err)
	}
	tb.Cleanup(func() {
		if err := rt.Close(); err != nil {
			tb.Errorf("runtime close: %v", err)
		}
	})
	label := opts.Label
	if label == "" {
		label = "conformance"
	}
	return &Rig{
		tb: tb, tier: tier, opts: opts, Dev: dev, RT: rt,
		G: &nn.Graph{
			B:      rt.NewBuilder(label),
			Eps:    opts.Eps,
			Prefix: opts.Prefix,
		},
		views:   map[string]accel.BufferView{},
		buffers: map[string]*accel.Buffer{},
		scalars: map[string]tensor.ScalarValue{},
	}
}

// bind allocates a device buffer under a port name and fills it.
func (r *Rig) bind(name string, dtype accel.DType, count int, data any) {
	r.tb.Helper()
	buf, err := r.Dev.NewBuffer(accel.BufferDescriptor{
		DType: dtype, Count: count, Label: name,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		r.tb.Fatalf("buffer %s: %v", name, err)
	}
	r.tb.Cleanup(func() { _ = buf.Close() })
	if data != nil {
		if err := r.Dev.Queue().WriteBuffer(buf, 0, data); err != nil {
			r.tb.Fatalf("write %s: %v", name, err)
		}
	}
	view, err := buf.View(0, count)
	if err != nil {
		r.tb.Fatalf("view %s: %v", name, err)
	}
	r.views[name] = view
	r.buffers[name] = buf
}

// F32 binds an f32 buffer under a port name.
func (r *Rig) F32(name string, v []float32) { r.bind(name, accel.F32, len(v), v) }

// U32 binds a u32 buffer: positions, slots, lengths and ids.
func (r *Rig) U32(name string, v []uint32) { r.bind(name, accel.U32, len(v), v) }

// F16 binds a weight narrowed to f16 and returns it as the kernel will read
// it, widened back.
//
// The widened values are what a reference computes from. Comparing against the
// values before narrowing would charge the comparison for the storage error
// twice: once in the eps16 term of the budget, and once again in a reference
// that used numbers the device never saw.
func (r *Rig) F16(name string, v []float32) []float32 {
	bits := make([]uint16, len(v))
	back := make([]float32, len(v))
	for i, x := range v {
		h := accel.ToFloat16(x)
		bits[i] = h.Bits()
		back[i] = h.F32()
	}
	r.bind(name, accel.F16, len(bits), bits)
	return back
}

// I8 binds a weight as the two planes a quantized port declares and returns
// the dequantized weights and the block scales.
//
// The scales come back because [Int8MatMulBound] needs them: the budget for a
// quantized product is computed from the blocks that were actually built, not
// from a bound somebody remembered.
func (r *Rig) I8(name string, v []float32) ([]float32, []accel.Float16) {
	quants, scales := quant.Int8Quantize(v)
	r.bind(name, accel.I8, len(quants), quants)
	bits := make([]uint16, len(scales))
	for i, s := range scales {
		bits[i] = s.Bits()
	}
	r.bind(name+nn.ScaleSuffix, accel.F16, len(bits), bits)
	return quant.Int8Dequantize(quants, scales), scales
}

// ScalarF32 declares a named f32 runtime value and binds it in one call,
// because a declared-but-unbound scalar and a bound-but-undeclared one are both
// errors nobody makes on purpose.
func (r *Rig) ScalarF32(name string, v float32) {
	tensor.Scalar(r.G.B, tensor.ScalarDesc{Name: name, Kind: tensor.ScalarF32})
	r.scalars[name] = tensor.F32(v)
}

// ScalarU32 declares and binds a named u32 runtime value.
func (r *Rig) ScalarU32(name string, v uint32) {
	tensor.Scalar(r.G.B, tensor.ScalarDesc{Name: name, Kind: tensor.ScalarU32})
	r.scalars[name] = tensor.U32(v)
}

// Input declares an input port.
func (r *Rig) Input(name string, dtype accel.DType, shape tensor.Shape) *tensor.Tensor {
	return tensor.Input(r.G.B, tensor.ValueDesc{Name: name, DType: dtype, Shape: shape})
}

// Run compiles the graph with out as its only output, submits it once and
// returns the result together with the plan its selections can be read from.
func (r *Rig) Run(out *tensor.Tensor) ([]float32, *tensor.Plan) {
	r.tb.Helper()
	tensor.Output(r.G.B, "out", out)
	if err := r.G.Err(); err != nil {
		r.tb.Fatalf("graph: %v", err)
	}
	label := r.opts.Label
	if label == "" {
		label = "conformance"
	}
	plan, err := r.G.B.Compile(r.RT, tensor.CompileOptions{Label: label})
	if err != nil {
		r.tb.Fatalf("compile: %v", err)
	}
	r.tb.Cleanup(func() {
		if err := plan.Close(); err != nil {
			r.tb.Errorf("plan close: %v", err)
		}
	})
	n := out.Shape().Elements()
	r.F32("out", make([]float32, n))
	fence := plan.Submit(r.Dev.Queue(), tensor.Bindings{Buffers: r.views, Scalars: r.scalars})
	if err := fence.Wait(); err != nil {
		r.tb.Fatalf("submit: %v", err)
	}
	got := make([]float32, n)
	if err := r.Dev.Queue().ReadBuffer(r.buffers["out"], 0, got); err != nil {
		r.tb.Fatalf("readback: %v", err)
	}
	return got, plan
}

// Parity runs the recorded graph, runs the oracle, and fails unless every
// element agrees within the budget.
//
// This is specs/010-conformance.md §5 in one call. The oracle is a function
// rather than a slice so that it is evaluated after the graph has been
// recorded and bound -- a reference built from what the device was actually
// given, not from what the test meant to give it.
//
// On disagreement the oracle is presumed right (010-D5), so the message says
// the device is out of parity rather than that the reference is wrong, and it
// prints the derivation of the budget: the next move is to find the term that
// is missing or to file the finding, and never to widen the number.
func (r *Rig) Parity(out *tensor.Tensor, ref func() []float64, terms Terms) ([]float32, *tensor.Plan) {
	r.tb.Helper()
	got, plan := r.Run(out)
	Compare(r.tb, got, ref(), terms, r.opts.Label)
	return got, plan
}

// Compare fails tb unless every element of got is within the budget of want.
//
// It is exported separately from [Rig.Parity] because a result that did not
// come from a rig -- a readback taken during an engine step, a probe that binds
// its own buffers -- is judged by the same rule.
func Compare(tb testing.TB, got []float32, want []float64, terms Terms, what string) {
	tb.Helper()
	if what == "" {
		what = "the device result"
	}
	if len(got) != len(want) {
		tb.Fatalf("%s: %d values, and the oracle produced %d", what, len(got), len(want))
	}
	worst, at, over := 0.0, -1, 0
	for i := range want {
		d := math.Abs(float64(got[i]) - want[i])
		if d <= terms.Bound(want[i]) {
			continue
		}
		over++
		if excess := d - terms.Bound(want[i]); excess > worst {
			worst, at = excess, i
		}
	}
	if over == 0 {
		return
	}
	tb.Fatalf("%s is out of parity with the oracle at %d of %d elements.\n"+
		"\tworst: element %d is %v and the oracle says %v, a difference of %.3g "+
		"against a bound of %.3g.\n"+
		"\tthe bound is derived from:%s\n"+
		"\tspecs/010-conformance.md 010-D3: a tolerance that had to be raised to "+
		"make a test pass is a finding, not a fix. Either a term above is missing, "+
		"or the device disagrees with the mathematics and the oracle is presumed "+
		"right (010-D5).",
		what, over, len(want), at, got[at], want[at],
		math.Abs(float64(got[at])-want[at]), terms.Bound(want[at]), terms.Explain())
}
