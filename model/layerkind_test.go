// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// specs/023-cache-kinds.md §2 and §10, over the declaration rather than over a
// forward pass: what a hybrid graph declares, and how many rows of each.

// hybridSchedule is `full_attention_interval: 4` in miniature: one full layer
// in four, over a stack whose length is not a multiple of the interval so that
// the last layers are gated-delta rather than falling on a boundary.
func hybridSchedule(layers int) LayerSchedule {
	s := make(LayerSchedule, layers)
	for i := range s {
		if i%4 == 3 {
			s[i] = LayerFullAttention
			continue
		}
		s[i] = LayerGatedDelta
	}
	return s
}

// hybridConfig is a stack of 10 layers: 2 full-attention and 8 gated-delta.
//
// No two extents are equal and none is a multiple of another where the
// confusion would be silent, which is the rule the fixtures in this package
// already follow: a head count equal to a layer count is the identity for every
// confusion between the two.
func hybridConfig() *Config {
	return &Config{
		Architecture: "hybrid-fixture", HiddenSize: 96, NumLayers: 10,
		NumHeads: 6, NumKVHeads: 3, HeadDim: 16, IntermediateSize: 176,
		VocabSize: 640, RMSNormEps: 1e-6, RoPETheta: 1e6,
		LayerTypes: hybridSchedule(10),
		Recurrent: &Recurrent{
			Heads: 2, KeyDim: 4, ValueDim: 12, Taps: 3, ConvWidth: 28,
		},
	}
}

// TestKindLocalLayerIndexing is §2's rule: the row a layer addresses is its
// ordinal **within its own kind**, not its index in the stack.
//
// A schedule that passed the model index into a state sized by kind reads
// another layer's state for most layers and past the allocation for the last
// ones, which is why this is asserted over a stack long enough for the model
// index to exceed the kind's count.
func TestKindLocalLayerIndexing(t *testing.T) {
	s := hybridSchedule(10)
	if got, want := s.Count(LayerFullAttention), 2; got != want {
		t.Errorf("%d full-attention layers, want %d", got, want)
	}
	if got, want := s.Count(LayerGatedDelta), 8; got != want {
		t.Errorf("%d gated-delta layers, want %d", got, want)
	}
	// Layer 3 and layer 7 are the two full-attention layers, and they are rows
	// 0 and 1 of a two-row state — not rows 3 and 7 of a ten-row one.
	for _, c := range []struct{ layer, ordinal int }{
		{0, 0}, {1, 1}, {2, 2}, {3, 0}, {4, 3}, {5, 4}, {6, 5}, {7, 1}, {8, 6}, {9, 7},
	} {
		if got := s.Ordinal(c.layer); got != c.ordinal {
			t.Errorf("layer %d (%v) is ordinal %d, want %d", c.layer, s.Kind(c.layer),
				got, c.ordinal)
		}
	}
	// Every ordinal is inside its own kind's count, which is the property that
	// makes the allocation safe rather than merely small.
	for i := range s {
		if got, want := s.Ordinal(i), s.Count(s.Kind(i)); got >= want {
			t.Errorf("layer %d is ordinal %d of %d %v layers", i, got, want, s.Kind(i))
		}
	}
	// A dense stack is the nil schedule, and every layer of it is full
	// attention at its own index.
	var dense LayerSchedule
	if dense.Hybrid() {
		t.Error("a nil schedule is hybrid")
	}
	if got := dense.Kind(5); got != LayerFullAttention {
		t.Errorf("a nil schedule's layer 5 is %v", got)
	}
}

// declare records a graph's ports on a fresh builder.
func declare(t *testing.T, c *Config, s GraphSpec) (Inputs, error) {
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
	return Declare(rt.NewBuilder("declare"), c, s)
}

// TestAHybridDeclaresThreeStatesSizedByKind is §2 and [023-D3]: the key/value
// state is allocated at the **full-attention** layer count, and the two
// recurrent states at the gated-delta one.
//
// Sizing the key/value state at NumLayers and indexing it by the model layer
// would allocate four times the state for rows nothing writes — and, worse,
// would let a scheduler price a block over layers that cache nothing, which is
// wrong by the same factor in the direction that refuses admissions a device
// has room for.
func TestAHybridDeclaresThreeStatesSizedByKind(t *testing.T) {
	c := hybridConfig()
	in, err := declare(t, c, GraphSpec{Tokens: 5, Capacity: 64, Batch: 2, Block: 32,
		Cache: accel.F32})
	if err != nil {
		t.Fatalf("Declare: %v", err)
	}
	if got, want := in.FullLayers, 2; got != want {
		t.Errorf("the key/value state has %d layers, want %d", got, want)
	}
	if got, want := in.LinearLayers, 8; got != want {
		t.Errorf("the recurrent states have %d layers, want %d", got, want)
	}
	if in.Recurrent == nil || in.ConvWindow == nil {
		t.Fatal("a hybrid declared no recurrent state or no convolution window")
	}
	// R = B(K-1) + T.
	if got, want := in.ConvRows, 2*(3-1)+5; got != want {
		t.Errorf("the window is %d rows, want %d = B(K-1) + T", got, want)
	}
	if len(in.ConvTaps) != c.Recurrent.Taps {
		t.Errorf("%d tap index ports for %d taps", len(in.ConvTaps), c.Recurrent.Taps)
	}
	if in.ConvWrite == nil || in.ConvCarry == nil || in.ConvCarryWrite == nil {
		t.Error("the window has no write, carry or carry-write port")
	}

	// A dense stack declares neither, and its key/value state covers every
	// layer.
	dense := hybridConfig()
	dense.LayerTypes, dense.Recurrent = nil, nil
	in, err = declare(t, dense, GraphSpec{Tokens: 5, Capacity: 64, Batch: 2, Block: 32,
		Cache: accel.F32})
	if err != nil {
		t.Fatalf("Declare, dense: %v", err)
	}
	if got, want := in.FullLayers, dense.NumLayers; got != want {
		t.Errorf("a dense stack's key/value state has %d layers, want %d", got, want)
	}
	if in.Recurrent != nil || in.ConvWindow != nil || in.ConvTaps != nil {
		t.Error("a dense stack declared a recurrent state")
	}
}

// TestAHybridDeclaresExtentsAtBatchOne is §2.1's consequence, and it is a
// refusal upstream rather than a preference here: tensor.LinearAttention
// requires QueryExtents at every batch size, where softmax attention over one
// sequence does not need them.
func TestAHybridDeclaresExtentsAtBatchOne(t *testing.T) {
	for _, c := range []struct {
		name   string
		cfg    *Config
		extent bool
	}{
		{"a hybrid at batch one", hybridConfig(), true},
		{"a dense stack at batch one", func() *Config {
			d := hybridConfig()
			d.LayerTypes, d.Recurrent = nil, nil
			return d
		}(), false},
	} {
		t.Run(c.name, func(t *testing.T) {
			in, err := declare(t, c.cfg, GraphSpec{Tokens: 1, Capacity: 64,
				Batch: 1, Cache: accel.F32})
			if err != nil {
				t.Fatalf("Declare: %v", err)
			}
			if got := in.Extents != nil; got != c.extent {
				t.Errorf("extents declared = %v, want %v", got, c.extent)
			}
			// Last stays a batch port under both: it is what a ragged step
			// reads the final row of each sequence from, and one sequence's
			// final row is the last row.
			if in.Last != nil {
				t.Error("a batch of one declared the last-row port")
			}
		})
	}
}

// TestAHybridDeclarationRefusesWhatItCannotBuild: each names the number.
func TestAHybridDeclarationRefusesWhatItCannotBuild(t *testing.T) {
	spec := GraphSpec{Tokens: 5, Capacity: 64, Batch: 2, Block: 32, Cache: accel.F32}
	for _, c := range []struct {
		name string
		edit func(*Config)
		want string
	}{
		{"a schedule shorter than the stack", func(c *Config) {
			c.LayerTypes = c.LayerTypes[:9]
		}, "names 9 layers and the config has 10"},
		{"a kind this graph does not build", func(c *Config) {
			c.LayerTypes[2] = LayerKind(9)
		}, "kind 9"},
		{"a hybrid with no geometry", func(c *Config) {
			c.Recurrent = nil
		}, "carries no recurrent geometry"},
		{"no key heads", func(c *Config) { c.Recurrent.Heads = 0 }, "0 key heads"},
		{"a value width that is not whole key widths", func(c *Config) {
			c.Recurrent.ValueDim = 10
		}, "not a whole number of key widths"},
		{"one tap", func(c *Config) { c.Recurrent.Taps = 1 }, "at least two"},
		{"no convolution channels", func(c *Config) {
			c.Recurrent.ConvWidth = 0
		}, "runs over 0 channels"},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := hybridConfig()
			c.edit(cfg)
			_, err := declare(t, cfg, spec)
			if err == nil {
				t.Fatal("Declare accepted a stack it cannot build")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("the refusal %q does not say %q", err, c.want)
			}
		})
	}
}

// TestTheRecurrentArithmeticMultipliesOut is §6's two per-slot formulas, which
// [023-D6] keeps as their own numbers rather than folding into
// CacheBytesPerSession: a breakdown that does not multiply out to the number
// beside it is worse than no breakdown.
func TestTheRecurrentArithmeticMultipliesOut(t *testing.T) {
	// Qwen3.8-27B's geometry, so the table in §6 is what this checks against.
	r := Recurrent{Heads: 16, KeyDim: 128, ValueDim: 384, Taps: 4, ConvWidth: 10240}

	// H_lin · d_v · d_k · 4 = 3 MiB per slot per layer.
	if got, want := r.StateBytes(), int64(3<<20); got != want {
		t.Errorf("StateBytes = %d, want %d = 3 MiB", got, want)
	}
	// 48 layers at B = 8 is 1.13 GiB, charged whether the slots are busy or
	// idle, which is the ceiling §4 describes.
	if got := r.StateBytes() * 48 * 8; got < 1<<30 || got > 5<<28 {
		t.Errorf("48 layers at 8 slots is %d bytes, want about 1.13 GiB", got)
	}

	// L_lin · (B(K-1) + T) · C_conv · 4. The carry alone, at T = 0, is 45 MiB;
	// a 512-token chunk is nearly a gigabyte, which is what makes the chunk a
	// memory parameter for a hybrid.
	carry := r.WindowBytes(48, 8, 0)
	if got, want := carry, int64(48*8*3*10240*4); got != want {
		t.Errorf("the carry is %d bytes, want %d", got, want)
	}
	chunk := r.WindowBytes(48, 8, 512)
	if chunk <= carry*10 {
		t.Errorf("a 512-token chunk's window is %d bytes against a carry of %d; §6's "+
			"reading is that the window is dominated by the step and not the carry",
			chunk, carry)
	}
}
