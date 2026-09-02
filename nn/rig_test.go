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

// rig records one graph, compiles it on the CPU backend and submits it.
//
// The CPU backend rather than a mock: a block is a claim about what accel
// computes, and only accel can settle it. specs/000-decisions.md decision 8
// keeps the real checkpoint out of the default run, not the real device.
type rig struct {
	t       *testing.T
	dev     *accel.Device
	rt      *tensor.Runtime
	g       *nn.Graph
	bufs    map[string]accel.BufferView
	buffers map[string]*accel.Buffer
	scalars map[string]tensor.ScalarValue
}

func newRig(t *testing.T, eps float32) *rig {
	t.Helper()
	dev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open CPU: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Errorf("device close: %v", err)
		}
	})
	rt, err := tensor.NewRuntime(dev)
	if err != nil {
		t.Fatalf("runtime: %v", err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("runtime close: %v", err)
		}
	})
	return &rig{
		t: t, dev: dev, rt: rt,
		g:       &nn.Graph{B: rt.NewBuilder("nn"), Eps: eps, Prefix: "block."},
		bufs:    map[string]accel.BufferView{},
		buffers: map[string]*accel.Buffer{},
		scalars: map[string]tensor.ScalarValue{},
	}
}

// reuse records a second graph against the SAME device, runtime and bound
// buffers, which is what lets a second step read a state the first one wrote.
//
// The two plans are different shapes and the state is caller-owned, so sharing
// it across submissions is exactly what tensor.State is for.
func (r *rig) reuse(t *testing.T, keep ...string) *rig {
	t.Helper()
	next := &rig{
		t: t, dev: r.dev, rt: r.rt,
		bufs:    map[string]accel.BufferView{},
		buffers: map[string]*accel.Buffer{},
		scalars: map[string]tensor.ScalarValue{},
	}
	// Only the named buffers, which in practice is the state. accel refuses a
	// binding for a port the plan does not declare, so carrying the first
	// step's inputs into a second plan of a different shape is an error rather
	// than a harmless extra.
	for _, k := range keep {
		if v, ok := r.bufs[k]; ok {
			next.bufs[k] = v
		}
		if v, ok := r.buffers[k]; ok {
			next.buffers[k] = v
		}
	}
	next.g = &nn.Graph{B: r.rt.NewBuilder("nn"), Eps: r.g.Eps, Prefix: r.g.Prefix}
	return next
}

func (r *rig) bind(name string, dtype accel.DType, count int, data any) {
	r.t.Helper()
	buf, err := r.dev.NewBuffer(accel.BufferDescriptor{
		DType: dtype, Count: count, Label: name,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		r.t.Fatalf("buffer %s: %v", name, err)
	}
	// The close error is reported and not discarded. A buffer with a batched
	// write still outstanding refuses to close, and swallowing that left every
	// test which binds a buffer and never submits leaking it -- surfacing as
	// accel refusing to close the device, from a cleanup, about something the
	// test was not testing.
	r.t.Cleanup(func() {
		if err := buf.Close(); err != nil {
			r.t.Errorf("closing %q: %v", name, err)
		}
	})
	if data != nil {
		if err := r.dev.Queue().WriteBuffer(buf, 0, data); err != nil {
			r.t.Fatalf("write %s: %v", name, err)
		}
	}
	// The write is batched, so it is finished here. A test that submits gets
	// this for free from the submission; one that only records does not, and
	// the buffer it bound cannot be closed until the write lands.
	if data != nil {
		if err := r.dev.Queue().Flush().Wait(); err != nil {
			r.t.Fatalf("staging %s: %v", name, err)
		}
	}
	view, err := buf.View(0, count)
	if err != nil {
		r.t.Fatalf("view %s: %v", name, err)
	}
	r.bufs[name] = view
	r.buffers[name] = buf
}

func (r *rig) f32(name string, v []float32) { r.bind(name, accel.F32, len(v), v) }
func (r *rig) u32(name string, v []uint32)  { r.bind(name, accel.U32, len(v), v) }

func (r *rig) f16(name string, v []float32) {
	bits := make([]uint16, len(v))
	for i, x := range v {
		bits[i] = accel.ToFloat16(x).Bits()
	}
	r.bind(name, accel.F16, len(bits), bits)
}

// quantize binds a weight as the two planes a quantized port declares, and
// returns the weights as the kernel will see them: dequantizing here is what
// lets a test assert the product rather than the quantizer's error budget.
func (r *rig) quantize(name string, v []float32) []float32 {
	r.t.Helper()
	quants, scales := quant.Int8Quantize(v)
	r.bind(name, accel.I8, len(quants), quants)
	bits := make([]uint16, len(scales))
	for i, s := range scales {
		bits[i] = s.Bits()
	}
	r.bind(name+nn.ScaleSuffix, accel.F16, len(bits), bits)
	return quant.Int8Dequantize(quants, scales)
}

// scalar declares and binds a named runtime value in one call, because a
// declared-but-unbound scalar and a bound-but-undeclared one are both errors
// nobody makes on purpose.
func (r *rig) scalarF32(name string, v float32) {
	tensor.Scalar(r.g.B, tensor.ScalarDesc{Name: name, Kind: tensor.ScalarF32})
	r.scalars[name] = tensor.F32(v)
}

func (r *rig) scalarU32(name string, v uint32) {
	tensor.Scalar(r.g.B, tensor.ScalarDesc{Name: name, Kind: tensor.ScalarU32})
	r.scalars[name] = tensor.U32(v)
}

func (r *rig) input(name string, dtype accel.DType, shape tensor.Shape) *tensor.Tensor {
	return tensor.Input(r.g.B, tensor.ValueDesc{Name: name, DType: dtype, Shape: shape})
}

// run compiles the graph with out as its only output, submits it once, and
// returns the result and the plan the selections can be read from.
// compile records out and compiles, without binding or submitting.
//
// For a test whose question is which kernels the graph selected rather than
// what they computed -- a Cast that should or should not be there is visible in
// Selections and needs no buffers behind it.
func (r *rig) compile(t *testing.T, out *tensor.Tensor) *tensor.Plan {
	t.Helper()
	tensor.Output(r.g.B, "out", out)
	if err := r.g.Err(); err != nil {
		t.Fatalf("graph: %v", err)
	}
	plan, err := r.g.B.Compile(r.rt, tensor.CompileOptions{Label: "nn"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	t.Cleanup(func() {
		if err := plan.Close(); err != nil {
			t.Errorf("plan close: %v", err)
		}
	})
	return plan
}

func (r *rig) run(out *tensor.Tensor) ([]float32, *tensor.Plan) {
	r.t.Helper()
	tensor.Output(r.g.B, "out", out)
	if err := r.g.Err(); err != nil {
		r.t.Fatalf("graph: %v", err)
	}
	plan, err := r.g.B.Compile(r.rt, tensor.CompileOptions{Label: "nn"})
	if err != nil {
		r.t.Fatalf("compile: %v", err)
	}
	r.t.Cleanup(func() {
		if err := plan.Close(); err != nil {
			r.t.Errorf("plan close: %v", err)
		}
	})
	n := out.Shape().Elements()
	r.f32("out", make([]float32, n))
	fence := plan.Submit(r.dev.Queue(), tensor.Bindings{Buffers: r.bufs, Scalars: r.scalars})
	if err := fence.Wait(); err != nil {
		r.t.Fatalf("submit: %v", err)
	}
	got := make([]float32, n)
	if err := r.dev.Queue().ReadBuffer(r.buffers["out"], 0, got); err != nil {
		r.t.Fatalf("readback: %v", err)
	}
	return got, plan
}

// refuses compiles a graph expected to fail and asserts the diagnostic names
// what went wrong. It reads [nn.Graph.Err] rather than only the builder's,
// because a block that refuses records on the graph.
func (r *rig) refuses(out *tensor.Tensor, want ...string) {
	r.t.Helper()
	tensor.Output(r.g.B, "out", out)
	err := r.g.Err()
	if err == nil {
		if _, cerr := r.g.B.Compile(r.rt, tensor.CompileOptions{}); cerr != nil {
			err = cerr
		}
	}
	if err == nil {
		r.t.Fatal("the graph compiled; it was expected to be refused")
	}
	for _, w := range want {
		if !strings.Contains(err.Error(), w) {
			r.t.Fatalf("the diagnostic is %q, which does not mention %q", err, w)
		}
	}
}

// selected reports the kernel an operator became, and fails if the operator ran
// more than once or not at all.
func selected(t *testing.T, plan *tensor.Plan, op string) tensor.KernelSelection {
	t.Helper()
	var found []tensor.KernelSelection
	for _, s := range plan.Selections() {
		if s.Op == op {
			found = append(found, s)
		}
	}
	if len(found) != 1 {
		t.Fatalf("%d selections for %s, want 1: %v", len(found), op, plan.Selections())
	}
	return found[0]
}

// close compares against a reference computed in the test, with a tolerance the
// caller names a term for.
func closeTo(t *testing.T, got []float32, want []float64, tol float64, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > tol*(1+math.Abs(want[i])) {
			t.Fatalf("%s: element %d is %v, want about %v", what, i, got[i], want[i])
		}
	}
}

// differs asserts a reference is *not* what the graph computed, which is what
// makes an ordering test a test: two orders that agree everywhere would let a
// wrong one pass.
func differs(t *testing.T, got []float32, want []float64, tol float64, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	worst := 0.0
	for i := range want {
		if d := math.Abs(float64(got[i])-want[i]) / (1 + math.Abs(want[i])); d > worst {
			worst = d
		}
	}
	if worst <= tol {
		t.Fatalf("%s: the graph agrees with a reference it must not, to %v; the test "+
			"cannot distinguish the two and would pass either way", what, worst)
	}
}
