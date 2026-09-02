// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"fmt"
	"maps"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/internal/oracle"
	"github.com/latere-ai/tgo/nn"
)

// synthetic is the config every graph test below is recorded at:
// specs/004-model-graph.md §8's "2-layer, small" model, with config_test.go's
// good() dimensions, which collide nowhere. d=80, L=2, H=8, H_kv=2, d_h=48,
// f=176, V=112, so H·d_h = 384 and H_kv·d_h = 96 are distinct from every other
// extent and a weight map that confused two of them does not compile.
func synthetic(t *testing.T) *qwen3 {
	t.Helper()
	b, err := New(raw(t, good()))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return b.(*qwen3)
}

// f16exact snaps a value to a multiple of 1/8 in [-2, 2), which f16 holds
// exactly: its mantissa is 11 bits and 1/8 needs 3.
//
// That is what removes the ε₁₆ = 2⁻¹¹ storage term from every tolerance in
// this file (specs/010-conformance.md §5.1). With arbitrary weights the
// tolerance on a projection would be set by the storage format at 4.9e-4 and
// would say nothing about the graph; with exact weights what is left is the
// f32 accumulation, √K·ε₃₂, which is what the numbers below are derived from.
func f16exact(seed uint32) float32 {
	// A 32-bit xorshift, so the fixture is the same on every machine and does
	// not depend on math/rand's stream staying put across Go releases.
	seed ^= seed << 13
	seed ^= seed >> 17
	seed ^= seed << 5
	return float32(int32(seed%32)-16) / 8
}

// synthWeights builds one deterministic plane per graph port.
//
// The port names are spelled here the way §3's node table reads them, not
// taken from Weights(): the point of TestGraphPortsAreTheWeightMapsPorts below
// is that the two agree, and a fixture derived from one of them could not
// check it.
func synthWeights(c *Config) map[string][]float32 {
	w := map[string][]float32{}
	add := func(name string, n int) {
		v := make([]float32, n)
		for i := range v {
			v[i] = f16exact(uint32(len(w)*7919 + i*104729 + 1))
		}
		w[name] = v
	}
	add(portEmbed, c.VocabSize*c.HiddenSize)
	for l := range c.NumLayers {
		p := fmt.Sprintf("%d.", l)
		add(p+"attn_norm", c.HiddenSize)
		add(p+"wq", c.HiddenSize*c.QWidth())
		add(p+"wk", c.HiddenSize*c.KVWidth())
		add(p+"wv", c.HiddenSize*c.KVWidth())
		add(p+"wo", c.QWidth()*c.HiddenSize)
		add(p+"qnorm", c.HeadDim)
		add(p+"knorm", c.HeadDim)
		add(p+"ffn_norm", c.HiddenSize)
		add(p+"wgate", c.HiddenSize*c.IntermediateSize)
		add(p+"wup", c.HiddenSize*c.IntermediateSize)
		add(p+"wdown", c.IntermediateSize*c.HiddenSize)
	}
	add(portFinalNorm, c.HiddenSize)
	add(portLMHead, c.HiddenSize*c.VocabSize)
	return w
}

// isGain reports whether a port is a norm gain, which binds f32 and never
// quantizes: a gain is one value per feature, so it is the smallest thing in
// the checkpoint and the one whose rounding a normalization carries into every
// row it scales.
func isGain(name string) bool {
	for _, s := range []string{"norm", portFinalNorm} {
		if strings.HasSuffix(name, s) {
			return true
		}
	}
	return false
}

// rig records one graph, compiles it on the CPU backend and submits it.
//
// The CPU backend rather than a mock: the forward pass is a claim about what
// accel computes and only accel can settle it. specs/000-decisions.md decision
// 8 keeps the real checkpoint out of the default run, not the real device.
type rig struct {
	t       *testing.T
	dev     *accel.Device
	rt      *tensor.Runtime
	b       *tensor.Builder
	g       *nn.Graph
	in      Inputs
	plan    *tensor.Plan
	bufs    map[string]accel.BufferView
	buffers map[string]*accel.Buffer
	scalars map[string]tensor.ScalarValue
}

func newRig(t *testing.T) *rig {
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
		bufs:    map[string]accel.BufferView{},
		buffers: map[string]*accel.Buffer{},
		scalars: map[string]tensor.ScalarValue{},
	}
}

// reuse records a second plan against the SAME device, runtime and bound
// buffers, which is what lets a decode step read the cache a prefill wrote.
//
// The two plans are different shapes -- accel reads q's rank as the phase
// (specs/004-model-graph.md §3.1) -- and the cache is caller-owned, so sharing
// it across submissions is exactly what tensor.State is for. Weight buffers are
// shared too, so the decode is the same model rather than a second one.
func (r *rig) reuse(m *qwen3, s GraphSpec) *rig {
	r.t.Helper()
	next := &rig{
		t: r.t, dev: r.dev, rt: r.rt,
		bufs:    map[string]accel.BufferView{},
		buffers: map[string]*accel.Buffer{},
		scalars: map[string]tensor.ScalarValue{},
	}
	// Carry the weights and the cache; the per-step ports are rebound by step.
	maps.Copy(next.bufs, r.bufs)
	maps.Copy(next.buffers, r.buffers)
	// The logits buffer is per plan: a decode writes one row and reusing the
	// prefill's would leave the previous step's values under a short write.
	delete(next.bufs, PortLogits)
	delete(next.buffers, PortLogits)
	next.record(m, s)
	return next
}

// record builds the whole forward pass for one step and compiles it.
func (r *rig) record(m *qwen3, s GraphSpec) *tensor.Plan {
	r.t.Helper()
	r.b = r.rt.NewBuilder("qwen3")
	g, in, err := Record(r.b, m, s)
	if err != nil {
		r.t.Fatalf("Record: %v", err)
	}
	r.g, r.in = g, in
	plan, err := r.b.Compile(r.rt, tensor.CompileOptions{Label: "qwen3"})
	if err != nil {
		r.t.Fatalf("compile: %v", err)
	}
	r.t.Cleanup(func() {
		if err := plan.Close(); err != nil {
			r.t.Errorf("plan close: %v", err)
		}
	})
	r.plan = plan
	return plan
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
	r.t.Cleanup(func() { _ = buf.Close() })
	if data != nil {
		if err := r.dev.Queue().WriteBuffer(buf, 0, data); err != nil {
			r.t.Fatalf("write %s: %v", name, err)
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

// weights binds every port of synthWeights: gains f32, matrices f16.
func (r *rig) weights(w map[string][]float32) {
	r.t.Helper()
	for name, v := range w {
		if isGain(name) {
			r.f32(name, v)
			continue
		}
		r.f16(name, v)
	}
}

// step binds §3's five input ports and three scalars for one forward pass.
func (r *rig) step(c *Config, ids []uint32, first int) Step {
	r.t.Helper()
	s, err := NewStep(c, first, len(ids))
	if err != nil {
		r.t.Fatalf("NewStep: %v", err)
	}
	r.u32(PortIDs, ids)
	r.u32(PortPosQ, s.PosQ)
	r.u32(PortPosK, s.PosK)
	r.u32(PortSlots, s.Slots)
	r.u32(PortLengths, s.Lengths)
	r.scalars[ScalarRoPEBase] = tensor.F32(c.RoPETheta)
	r.scalars[ScalarScale] = tensor.F32(float32(1 / math.Sqrt(float64(c.HeadDim))))
	if len(ids) > 1 {
		r.scalars[ScalarBase] = tensor.U32(s.Base)
	}
	return s
}

// cache binds the two states. They are caller-owned, so the same buffers can
// serve a prefill plan and a decode plan, which is what
// TestDecodeAtOneTokenEqualsThePrefillsLastRow does.
func (r *rig) cache(c *Config, capacity int) {
	r.t.Helper()
	n := c.NumLayers * capacity * c.NumKVHeads * c.HeadDim
	r.f32(PortKeys, make([]float32, n))
	r.f32(PortValues, make([]float32, n))
}

// submit runs the compiled plan and reads the logits back.
func (r *rig) submit(c *Config) []float32 {
	r.t.Helper()
	if _, ok := r.buffers[PortLogits]; !ok {
		r.f32(PortLogits, make([]float32, c.VocabSize))
	}
	fence := r.plan.Submit(r.dev.Queue(), tensor.Bindings{Buffers: r.bufs, Scalars: r.scalars})
	if err := fence.Wait(); err != nil {
		r.t.Fatalf("submit: %v", err)
	}
	got := make([]float32, c.VocabSize)
	if err := r.dev.Queue().ReadBuffer(r.buffers[PortLogits], 0, got); err != nil {
		r.t.Fatalf("readback: %v", err)
	}
	return got
}

// qkOrder selects which arrangement of QK-norm the reference computes.
type qkOrder int

const (
	normBeforeRoPE qkOrder = iota // what Qwen3 does (§2.4)
	normAfterRoPE                 // the ordering mistake
)

// referenceForward is specs/004-model-graph.md §3 in float64, composed from
// internal/oracle.
//
// It shares no code with [qwen3.Forward]: every stage is an oracle call
// written from §3's node table, the cache is two Go slices indexed by slot,
// and the shapes are plain ints. Two derivations of the same mathematics
// agreeing is evidence; one derivation compared against itself is not
// (010-D2).
//
// The rotary style is interleaved because that is accel's kernel, and
// §2.5.2's half-split-to-interleaved permutation is the loader's, applied to
// bytes before they reach any graph. A fixture generated in accel's basis is
// therefore the right one to compare a *graph* against; the permutation itself
// is tgo/weights' test.
func referenceForward(c *Config, w map[string][]float32, ids []int, order qkOrder) []float64 {
	t := len(ids)
	f := func(name string) []float64 { return f64s(w[name]) }
	eps := float64(c.RMSNormEps)
	base := float64(c.RoPETheta)
	dh, hq, hkv := c.HeadDim, c.NumHeads, c.NumKVHeads

	// One position per row, each token's position repeated once per head:
	// §3 row 12's formula, written out rather than taken from NewStep so that
	// a bug there cannot cancel against a bug here.
	posQ := make([]int, 0, t*hq)
	posK := make([]int, 0, t*hkv)
	for p := range t {
		for range hq {
			posQ = append(posQ, p)
		}
		for range hkv {
			posK = append(posK, p)
		}
	}

	h := oracle.GatherRows(f(portEmbed), c.VocabSize, c.HiddenSize, ids)
	for l := range c.NumLayers {
		p := fmt.Sprintf("%d.", l)
		n := oracle.RMSNorm(h, f(p+"attn_norm"), eps)

		q := oracle.MatMul(n, f(p+"wq"), t, c.HiddenSize, c.QWidth())
		k := oracle.MatMul(n, f(p+"wk"), t, c.HiddenSize, c.KVWidth())
		v := oracle.MatMul(n, f(p+"wv"), t, c.HiddenSize, c.KVWidth())

		// Per head, over head_dim: RMSNorm's row count is len(x)/len(gain), so
		// a d_h-wide gain over a T·H·d_h vector is exactly the reshape of
		// rows 9 to 11 and never the pooled H·d_h norm.
		switch order {
		case normBeforeRoPE:
			q = oracle.RoPE(oracle.RMSNorm(q, f(p+"qnorm"), eps),
				t*hq, dh, dh, base, posQ, oracle.StyleInterleaved)
			k = oracle.RoPE(oracle.RMSNorm(k, f(p+"knorm"), eps),
				t*hkv, dh, dh, base, posK, oracle.StyleInterleaved)
		case normAfterRoPE:
			q = oracle.RMSNorm(oracle.RoPE(q, t*hq, dh, dh, base, posQ,
				oracle.StyleInterleaved), f(p+"qnorm"), eps)
			k = oracle.RMSNorm(oracle.RoPE(k, t*hkv, dh, dh, base, posK,
				oracle.StyleInterleaved), f(p+"knorm"), eps)
		}

		// The cache, at slots 0..T-1, which is what a prefill from an empty
		// cache writes: a contiguous cache addresses a token by its position.
		scale := 1 / math.Sqrt(float64(dh))
		a := oracle.Attention(q, k, v, t, hq, hkv, dh, t, scale, 0)
		h = add(h, oracle.MatMul(a, f(p+"wo"), t, c.QWidth(), c.HiddenSize))

		n2 := oracle.RMSNorm(h, f(p+"ffn_norm"), eps)
		gate := oracle.MatMul(n2, f(p+"wgate"), t, c.HiddenSize, c.IntermediateSize)
		up := oracle.MatMul(n2, f(p+"wup"), t, c.HiddenSize, c.IntermediateSize)
		h = add(h, oracle.MatMul(oracle.SwiGLU(gate, up), f(p+"wdown"),
			t, c.IntermediateSize, c.HiddenSize))
	}

	// Row 26: the last position only.
	last := h[(t-1)*c.HiddenSize:]
	return oracle.MatMul(oracle.RMSNorm(last, f(portFinalNorm), eps), f(portLMHead),
		1, c.HiddenSize, c.VocabSize)
}

func add(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
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

// worst returns the largest relative deviation between a device result and a
// reference, with the +1 that keeps a near-zero reference from dividing.
func worst(got []float32, want []float64) float64 {
	m := 0.0
	for i := range want {
		if d := math.Abs(float64(got[i])-want[i]) / (1 + math.Abs(want[i])); d > m {
			m = d
		}
	}
	return m
}

func closeTo(t *testing.T, got []float32, want []float64, tol float64, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	if w := worst(got, want); w > tol {
		t.Fatalf("%s: worst relative deviation is %.3g, tolerance is %.3g", what, w, tol)
	}
}

// differs asserts a reference is *not* what the graph computed, which is what
// makes an ordering test a test: two orders that agreed everywhere would let a
// wrong one pass.
func differs(t *testing.T, got []float32, want []float64, tol float64, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	if w := worst(got, want); w <= tol {
		t.Fatalf("%s: the graph agrees with a reference it must not, to %.3g; the test "+
			"cannot distinguish the two and would pass either way", what, w)
	}
}

// TestPrefillMatchesTheOracle is the test the rest of this file exists for.
//
// specs/010-conformance.md §5: the device forward pass against a float64
// reference derived from the mathematics rather than from the graph code. A
// forward pass that compiles proves the shapes agree; only this proves the
// arithmetic does.
func TestPrefillMatchesTheOracle(t *testing.T) {
	m := synthetic(t)
	c := m.Config()
	w := synthWeights(c)
	ids := []uint32{3, 17, 42, 8}

	r := newRig(t)
	r.record(m, GraphSpec{Tokens: len(ids), Capacity: 16, Cache: accel.F32})
	r.weights(w)
	r.cache(c, 16)
	r.step(c, ids, 0)
	got := r.submit(c)

	want := referenceForward(c, w, u32sToInts(ids), normBeforeRoPE)
	closeTo(t, got, want, prefillTol(c), "prefill logits")
}

// TestQKNormRunsBeforeRoPE fails if §2.4's ordering moves.
//
// It is the ordering that produces a model which emits plausible tokens and
// loses coherence after a few sentences, so nothing downstream would report
// it. The reference is computed both ways and the device must match one and
// not the other; asserting only the match would pass if the two orderings
// happened to agree at this fixture, which `differs` rules out.
func TestQKNormRunsBeforeRoPE(t *testing.T) {
	m := synthetic(t)
	c := m.Config()
	w := synthWeights(c)
	ids := []uint32{5, 11, 29}

	r := newRig(t)
	r.record(m, GraphSpec{Tokens: len(ids), Capacity: 8, Cache: accel.F32})
	r.weights(w)
	r.cache(c, 8)
	r.step(c, ids, 0)
	got := r.submit(c)

	tol := prefillTol(c)
	closeTo(t, got, referenceForward(c, w, u32sToInts(ids), normBeforeRoPE), tol,
		"logits against QK-norm before RoPE")
	differs(t, got, referenceForward(c, w, u32sToInts(ids), normAfterRoPE), tol,
		"logits against QK-norm after RoPE")
}

// TestDecodeAtOneTokenEqualsThePrefillsLastRow is §3.1: prefill and decode are
// the same graph at different T, and accel reads q's rank as the phase, so the
// two are genuinely different plans over one cache.
//
// Prefilling T tokens and then decoding the (T+1)th must give what prefilling
// T+1 tokens gives for its last row. If it does not, the cache and the
// positions disagree, which is the failure that looks like a model with a
// short memory.
func TestDecodeAtOneTokenEqualsThePrefillsLastRow(t *testing.T) {
	m := synthetic(t)
	c := m.Config()
	w := synthWeights(c)
	const capacity = 16
	ids := []uint32{7, 23, 4, 19}

	// The whole sequence in one prefill.
	whole := newRig(t)
	whole.record(m, GraphSpec{Tokens: len(ids), Capacity: capacity, Cache: accel.F32})
	whole.weights(w)
	whole.cache(c, capacity)
	whole.step(c, ids, 0)
	oneShot := whole.submit(c)

	// The same sequence as a prefill of the first T-1 followed by a decode of
	// the last, over one cache that both plans bind.
	split := newRig(t)
	split.record(m, GraphSpec{Tokens: len(ids) - 1, Capacity: capacity, Cache: accel.F32})
	split.weights(w)
	split.cache(c, capacity)
	split.step(c, ids[:len(ids)-1], 0)
	split.submit(c)

	dec := split.reuse(m, GraphSpec{Tokens: 1, Capacity: capacity, Cache: accel.F32})
	dec.step(c, ids[len(ids)-1:], len(ids)-1)
	stepwise := dec.submit(c)

	// Both are f32 sums over the same values in a different grouping, so the
	// budget is the accumulation term twice rather than once: specs/010 §5.1's
	// √K·ε₃₂ applied to each of the two paths.
	closeTo(t, stepwise, f64s(oneShot), 2*prefillTol(c),
		"decode of the last token against the prefill's last row")
}

// u32sToInts is the reference's id type, which is plain ints because the
// oracle takes no tgo types (010 §5).
func u32sToInts(v []uint32) []int {
	out := make([]int, len(v))
	for i, x := range v {
		out[i] = int(x)
	}
	return out
}

// prefillTol is specs/010-conformance.md §5.1's budget for this graph, derived
// rather than tuned. 010-D3: a tolerance that had to be raised to make a test
// pass is a finding, so it is computed from the terms.
//
// The ε₁₆ storage term is absent by construction — f16exact makes every weight
// exact in f16 — so what is left is f32 accumulation, √K·ε₃₂ with
// ε₃₂ = 2⁻²⁴, over the longest chain a logit depends on: the LM head's K is
// the hidden size, and each layer contributes its own projections. Summed
// conservatively over the depth rather than composed, and scaled by the
// magnitude the reference actually reaches.
func prefillTol(c *Config) float64 {
	const eps32 = 1.0 / (1 << 24)
	k := float64(c.HiddenSize)
	perLayer := math.Sqrt(k) * eps32
	// Two projections deep per layer plus the head, and a residual that adds
	// rather than multiplies error.
	chain := perLayer * float64(2*c.NumLayers+1)
	// The logits are O(10) at this fixture, so an absolute budget is the
	// relative one times that scale, with a factor of eight of headroom for
	// the softmax and the norms, which are stable but not exact.
	return 8 * chain * 10
}

// TestLogitsAreTheLastRowOnly is §3.2, which is a *memory* claim rather than
// an arithmetic one.
//
// Running the LM head over all T positions instead of the last costs T×V f32
// values: at Qwen3-4B and T=2000 that is 1.2 GB of logits nobody reads, larger
// than the int8 weights of the model producing them.
//
// Measuring transient bytes does NOT catch it, and finding that out is the
// reason this comment is long. PortLogits is an *output* port, so its buffer is
// the caller's and never appears in Plan.Memory().TransientBytes -- a head over
// every position widens the port and leaves transients flat. Verified by
// mutation: slicing [0,t) instead of [t-1,t) left transients identical at
// 38912 bytes across a 36× vocabulary change.
//
// So the claim is checked where it lives: the declared shape of the output
// port. A port of [T, V] is the bug, whatever it costs.
func TestLogitsAreTheLastRowOnly(t *testing.T) {
	m := synthetic(t)
	c := m.Config()

	for _, tokens := range []int{1, 2, 8} {
		r := newRig(t)
		r.record(m, GraphSpec{Tokens: tokens, Capacity: 32, Cache: accel.F32})

		var logits *tensor.PortDesc
		for _, p := range r.plan.Ports() {
			if p.Name == PortLogits {
				port := p
				logits = &port
			}
		}
		if logits == nil {
			t.Fatalf("T=%d: the plan declares no %q port", tokens, PortLogits)
		}
		want := tensor.Shape{1, c.VocabSize}
		if !logits.Shape.Equal(want) {
			t.Errorf("T=%d: %s is %v, want %v; the LM head is running over every "+
				"position rather than the last, which costs T×V f32 values and is "+
				"specs/004-model-graph.md §3.2's single largest avoidable cost",
				tokens, PortLogits, logits.Shape, want)
		}
	}
}
