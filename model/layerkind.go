// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import "fmt"

// A cache per layer kind.
//
// specs/023-cache-kinds.md. [Declare] has one cache shape for every layer,
// which holds for a dense transformer where every layer caches the same thing.
// A hybrid holds **three** things at once and one forward pass touches all
// three: the key/value cache of its softmax-attention layers, and — per
// gated-delta layer — a recurrent matrix and the rolling window of the
// depthwise causal convolution in front of it.
//
// This file is what the graph declares. Which layer is which comes from the
// checkpoint, and reading it is specs/024-qwen3-5-architecture.md's.

// LayerKind is what one layer of the stack holds between steps.
type LayerKind uint8

// The kinds a stack can mix.
const (
	// LayerFullAttention is softmax attention over a key/value cache: the only
	// kind a dense model has, and the only kind that writes a KV block.
	LayerFullAttention LayerKind = iota

	// LayerGatedDelta is a linear-attention recurrence with a depthwise causal
	// convolution in front of it. It holds two states and no positions, so it
	// has no cache a block can be reserved for
	// (specs/018-hybrid-models.md §2.1).
	LayerGatedDelta
)

// String names a LayerKind.
func (k LayerKind) String() string {
	switch k {
	case LayerFullAttention:
		return "full_attention"
	case LayerGatedDelta:
		return "linear_attention"
	}
	return "unknown"
}

// LayerSchedule is what each layer of the stack is, in model order.
//
// A nil schedule is a dense stack: every layer is [LayerFullAttention], which
// is what every model in the registry was until a hybrid arrived.
type LayerSchedule []LayerKind

// Kind is what layer i of the stack is.
func (s LayerSchedule) Kind(i int) LayerKind {
	if i < 0 || i >= len(s) {
		return LayerFullAttention
	}
	return s[i]
}

// Count is how many layers of a kind the stack has.
func (s LayerSchedule) Count(k LayerKind) int {
	n := 0
	for _, x := range s {
		if x == k {
			n++
		}
	}
	return n
}

// Ordinal is layer i's index **within its own kind**, which is the row of its
// own state it addresses.
//
// specs/023-cache-kinds.md §2: the leading axis of each state is the kind-local
// ordinal and not the model layer index. The 47th gated-delta layer is row 47
// of a 48-row state, not row 62 of one sized at 64 — and sizing the axis at 64
// to index by the model layer would allocate 4x the key/value state and 1.33x
// each recurrent state for rows nothing reads.
//
// The failure the other way round is worse than waste: a schedule that passed
// the model index into a state sized by kind reads another layer's state for
// most layers and past the allocation for the last ones.
func (s LayerSchedule) Ordinal(i int) int {
	if i < 0 || i >= len(s) {
		return i
	}
	n := 0
	for j := range i {
		if s[j] == s[i] {
			n++
		}
	}
	return n
}

// Hybrid reports whether the stack mixes kinds, which is the one question
// [Declare] asks the schedule before it decides what to declare.
func (s LayerSchedule) Hybrid() bool { return s.Count(LayerGatedDelta) > 0 }

// check reports whether a schedule describes the stack the config names.
func (s LayerSchedule) check(layers int) error {
	if len(s) == 0 {
		return nil
	}
	if len(s) != layers {
		return fmt.Errorf("model: the layer schedule names %d layers and the config "+
			"has %d; a schedule that does not cover the stack leaves the rest to a "+
			"default, and a layer built as the wrong kind runs and produces fluent "+
			"different text (specs/024-qwen3-5-architecture.md 024-D1)", len(s), layers)
	}
	for i, k := range s {
		if k != LayerFullAttention && k != LayerGatedDelta {
			return fmt.Errorf("model: layer %d is kind %d, which this graph does not "+
				"build", i, k)
		}
	}
	return nil
}

// Recurrent is the geometry of the gated-delta layers, which is one shape for
// all of them.
//
// specs/023-cache-kinds.md §2.2: `linear_num_key_heads: 16` against
// `linear_num_value_heads: 48` is **one** head count of 16 with a value width
// three times the key width, because the recurrence is row-separable in the
// value dimension — three value heads sharing one key head are three disjoint
// row bands of one state, and stacking them is an identity rather than an
// approximation.
type Recurrent struct {
	// Heads is H_lin, the key heads, and is the head count of the **state**.
	Heads int

	// ValueHeads is H_v, the value heads, which is a whole multiple of Heads.
	//
	// It is carried and not derived as ValueDim/KeyDim, because the two are
	// the same number only under §2.2's folding and the checkpoint states it.
	// Deriving it would be getting it right by luck — and the output norm is
	// one gain per *value* head (`linear_attn.norm.weight` is `[128]`, not
	// `[384]`), so a graph that had only the folded geometry would broadcast
	// that gain over three value heads at once
	// (specs/024-qwen3-5-architecture.md §4.5).
	ValueHeads int

	// KeyDim is d_k and ValueDim is d_v, and d_v is a whole multiple of d_k
	// when several value heads share a key head.
	KeyDim, ValueDim int

	// Taps is K, the depthwise convolution's kernel width.
	Taps int

	// ConvWidth is C_conv, the width the convolution runs over: the
	// concatenated q, k and v projections. It is read from the checkpoint's
	// conv weight rather than derived (024's), because every byte of the
	// window scales linearly with it.
	ConvWidth int
}

// check reports whether the geometry is one a graph can be built from.
func (r Recurrent) check() error {
	switch {
	case r.Heads < 1:
		return fmt.Errorf("model: the recurrence has %d key heads", r.Heads)
	case r.KeyDim < 1 || r.ValueDim < 1:
		return fmt.Errorf("model: the recurrence is %d x %d; a state has a key and a "+
			"value width", r.ValueDim, r.KeyDim)
	case r.ValueDim%r.KeyDim != 0:
		return fmt.Errorf("model: the value width %d is not a whole number of key "+
			"widths of %d; value heads sharing a key head are disjoint row bands of "+
			"one state, and a partial band is not one (specs/023-cache-kinds.md §2.2)",
			r.ValueDim, r.KeyDim)
	case r.ValueHeads != 0 && r.ValueHeads%r.Heads != 0:
		return fmt.Errorf("model: %d value heads over %d key heads do not group; "+
			"value heads sharing a key head are disjoint row bands of one state",
			r.ValueHeads, r.Heads)
	case r.Taps < 2:
		return fmt.Errorf("model: the convolution is %d taps; a causal convolution "+
			"has at least two, and one tap is an elementwise scale", r.Taps)
	case r.ConvWidth < 1:
		return fmt.Errorf("model: the convolution runs over %d channels", r.ConvWidth)
	}
	return nil
}

// StateBytes is what one slot's recurrent state costs, per gated-delta layer.
//
// specs/023-cache-kinds.md §6: H_lin · d_v · d_k · 4. f32 and not a parameter,
// because tensor.LinearAttention refuses any other dtype — and the reason
// behind that refusal is the one [023-D4] gives: the state is an
// **accumulator**, decayed and rewritten once per token, so at a quarter of a
// million tokens an f16 state has no bits left that describe the early prefix.
func (r Recurrent) StateBytes() int64 {
	return int64(r.Heads) * int64(r.ValueDim) * int64(r.KeyDim) * 4
}

// WindowBytes is what the whole convolution window costs over every slot and
// every gated-delta layer, for a step of rows token rows.
//
// specs/023-cache-kinds.md §6: L_lin · (B(K-1) + T) · C_conv · 4. Only B(K-1)
// of those rows persist; the rest is scratch proportional to the step, which is
// what makes the prefill chunk a memory parameter for a hybrid where it is a
// latency parameter for a dense model.
func (r Recurrent) WindowBytes(layers, slots, rows int) int64 {
	return int64(layers) * int64(slots*(r.Taps-1)+rows) * int64(r.ConvWidth) * 4
}
