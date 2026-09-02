// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// The ports and scalars specs/004-model-graph.md §3 declares, by name.
//
// They are constants rather than strings written twice because the caller who
// binds a buffer and the graph that reads it are in different packages, and a
// typo between them is a submission-time "not bound" rather than a compile
// error.
const (
	// PortIDs is the step's tokens, [T] u32.
	PortIDs = "ids"

	// PortPosQ is the rotary position of every *query row*, [T·H] u32: each
	// token's position repeated H times, which is §3 row 12's formula.
	PortPosQ = "posq"

	// PortPosK is the same for the key rows, [T·H_kv] u32.
	//
	// A separate tensor from PortPosQ, not a second reading of it: under GQA
	// the two row counts differ, so one positions tensor cannot serve both.
	PortPosK = "posk"

	// PortSlots is the cache row each token is written to, [T] u32.
	PortSlots = "slots"

	// PortLengths is how much of the cache holds real tokens, [1] u32.
	PortLengths = "lengths"

	// PortPages is the page table, [1, MaxPages] u32: entry [0][i] is the
	// physical block holding this sequence's i-th logical block.
	//
	// Declared only when [GraphSpec.Block] is set. A contiguous cache is the
	// same thing with an identity table and a block of one, and accel says so
	// -- but the indirection is a real cost in the innermost loop of decode,
	// so a session that shares nothing does not pay it and does not declare
	// this port.
	PortPages = "pages"

	// PortExtents is how many query tokens each sequence contributes to this
	// step, [B] u32. Declared only when [GraphSpec.Batch] is above one.
	//
	// It is what makes the step ragged: q is every sequence's tokens end to
	// end, so a 512-token prefill chunk and three decodes are one dispatch
	// (specs/008-scheduler.md §2).
	PortExtents = "extents"

	// PortLast is the flat row each sequence's logits are taken from, [B] u32.
	//
	// One sequence's last token is at a different flat index from the next
	// one's, because the sequences contribute different counts, so the row is
	// data rather than an offset the graph can compute. §3.2's "the last
	// position only" becomes "one position per sequence", and the same
	// argument holds: running the head over every row would cost T*V values
	// nobody reads.
	PortLast = "last"

	// PortKeys and PortValues are the two cache states, [L, C, H_kv, d_h]
	// each: one allocation per role for the whole model, sliced per layer
	// (specs/005-kv-cache.md §2.1).
	PortKeys   = "k"
	PortValues = "v"

	// PortRecurrent is the gated-delta layers' recurrent state,
	// [L_lin, B, H_lin, d_v, d_k] f32, sliced per layer by its kind-local
	// ordinal (specs/023-cache-kinds.md §2).
	//
	// f32 and not a choice: tensor.LinearAttention refuses any other dtype, and
	// [023-D4] is why the refusal is right — the state is an accumulator
	// decayed and rewritten once per token, not an operand read once.
	PortRecurrent = "rec"

	// PortConvWindow is the depthwise convolution's rolling window,
	// [L_lin, R, C_conv] f32 with R = B(K-1) + T. It is flat: the slot is
	// arithmetic in the index ports below, because a state has one row axis and
	// a slot axis in front of it would make one row a whole sequence
	// ([023-D2]).
	PortConvWindow = "conv"

	// PortConvWrite, PortConvTap, PortConvCarry and PortConvCarryWrite are the
	// window's index ports. They are one set for the whole stack rather than
	// one per layer: every gated-delta layer's window has the same layout, so
	// the indices are the same numbers and binding them once is one upload
	// instead of forty-eight.
	PortConvWrite      = "conv_write"
	PortConvTap        = "conv_tap"
	PortConvCarry      = "conv_carry"
	PortConvCarryWrite = "conv_carry_to"

	// PortLogits is the graph's output, [1, V] f32 — the last position only
	// (§3.2).
	PortLogits = "logits"

	// ScalarRoPEBase is rope_theta, f32.
	ScalarRoPEBase = "rope_base"

	// ScalarScale is 1/sqrt(head_dim), f32.
	ScalarScale = "scale"

	// ScalarBase is a prefill's first position, u32.
	//
	// Declared only when the step scores more than one token. accel refuses a
	// BaseName on a decode — "a decode attends over the whole cache its
	// Lengths names" — and a declared scalar must be bound at submission even
	// where nothing reads it, so a decode plan that declared this would demand
	// a value for a question it does not ask.
	ScalarBase = "base"
)

// The graph ports outside the layer stack, matching the Port field of the
// [WeightSpec] rows [Builder.Weights] returns. The loader files a value under
// that name and the graph declares a port under this one; they are the same
// string or nothing binds.
const (
	portEmbed     = "embed"
	portFinalNorm = "final_norm"
	portLMHead    = "lm_head"
)

// GraphSpec is what one recorded graph needs beyond the config.
//
// Tokens is what buckets the plan: prefill and decode are separate plans
// (004-D3) and they differ by this number, plus the rank q reaches attention
// with, which [Builder.Forward] derives from it.
type GraphSpec struct {
	// Tokens is T, the tokens this step scores. One is a decode.
	Tokens int

	// Capacity is C, the cache capacity in positions. It is a session
	// parameter and not a model one: max_position_embeddings is advisory
	// (005-D2), and a graph sized from it would reserve the whole trained
	// context before the first token.
	Capacity int

	// Cache is the dtype the key and value states hold. The zero value is
	// [accel.F32], which is the only one this graph records.
	//
	// f32 is the default because the projections are f32 and the block writes
	// them to the cache unconverted; an f16 cache halves the largest
	// allocation a serving process has after the weights and needs a Cast on
	// the scattered rows that nothing here records yet
	// (specs/005-kv-cache.md §3).
	Cache accel.DType

	// Batch is how many sequences step together, and zero or one is a single
	// sequence.
	//
	// Fixed for the life of a plan, and 008-D1 is why: batch is a leading
	// dimension on every port, so a step that changed it would be a different
	// graph and a different compile. Idle slots are parked on a zero-length
	// sequence instead, which costs one row of arithmetic and no plan.
	//
	// [GraphSpec.Tokens] is then the *total* across the batch rather than one
	// sequence's, because a ragged step's q is flat.
	Batch int

	// Block is how many positions one physical block holds, and zero means
	// the cache is contiguous.
	//
	// It is a graph parameter and not a step one because accel folds it into
	// the plan's attributes: two block sizes are two plans, the same way two
	// token counts are. Capacity must be a whole number of blocks, so that a
	// page table addresses exactly the cache that exists rather than a prefix
	// of it.
	Block int

	// Stored reports how the loader stored the weight of a given full port
	// name. A nil func means every weight is f16.
	Stored func(name string) nn.Form
}

// Inputs is the set of ports one forward pass reads, as [Declare] recorded
// them.
//
// It is a struct of tensors rather than a map so that a missing port is a
// compile error in the caller rather than a nil operand three operators later.
type Inputs struct {
	// IDs, PosQ, PosK, Slots and Lengths are §3's five input ports.
	IDs, PosQ, PosK, Slots, Lengths *tensor.Tensor

	// Keys and Values are the two cache states, whole: [L, C, H_kv, d_h].
	// [Builder.Forward] takes each layer's window with [tensor.LayerState].
	Keys, Values *tensor.State

	// Pages is the page table port, or nil when the cache is contiguous.
	Pages *tensor.Tensor

	// Block mirrors [GraphSpec.Block] so that [Builder.Forward] has both
	// halves of the paged binding from one value. A table with no block size
	// addresses nothing, and accel refuses the pair split apart.
	Block int

	// Extents and Last are the batch's two ports, nil for a single sequence.
	Extents, Last *tensor.Tensor

	// Batch mirrors [GraphSpec.Batch] and Cache mirrors [GraphSpec.Cache].
	Batch int
	Cache accel.DType

	// Base is the name of the prefill's first-position scalar, and is empty on
	// a decode step. It is carried rather than assumed so that a hand-built
	// Inputs says which of the two plans it is for.
	Base string

	// FullLayers is how many layers of the stack have a key/value cache, which
	// is the leading extent of Keys and Values. For a dense model it is every
	// layer; for a hybrid it is one in four (specs/023-cache-kinds.md §2).
	FullLayers int

	// LinearLayers is how many gated-delta layers the stack has, and is zero
	// for a dense model. It is the leading extent of Recurrent and ConvWindow.
	LinearLayers int

	// Recurrence is the geometry those two states were declared from, carried
	// so that a caller sizing or binding them does not re-read the config.
	Recurrence Recurrent

	// Recurrent and ConvWindow are the two states a gated-delta layer holds,
	// whole. [tensor.LayerState] takes each layer's window by its kind-local
	// ordinal, which [LayerSchedule.Ordinal] is.
	Recurrent, ConvWindow *tensor.State

	// ConvRows is R, the window's row count: B(K-1) + T.
	ConvRows int

	// ConvWrite, ConvTaps, ConvCarry and ConvCarryWrite are the window's index
	// ports, which [github.com/latere-ai/tgo/nn.ConvIndex] fills. One set for
	// the whole stack: every gated-delta layer's window has the same layout.
	ConvWrite, ConvCarry, ConvCarryWrite *tensor.Tensor
	ConvTaps                             []*tensor.Tensor
}

// Declare records specs/004-model-graph.md §3's ports, scalars and cache
// states on b and returns them.
//
// It is separate from [Builder.Forward] because the ports are a property of
// the config and the step, not of the architecture: two models with the same
// shapes declare the same ports, and a caller that wants to bind a cache it
// allocated elsewhere needs the tensors before the graph exists.
func Declare(b *tensor.Builder, c *Config, s GraphSpec) (Inputs, error) {
	if b == nil {
		return Inputs{}, fmt.Errorf("model: Declare: the builder is nil")
	}
	if c == nil {
		return Inputs{}, fmt.Errorf("model: Declare: the config is nil")
	}
	if err := s.check(); err != nil {
		return Inputs{}, err
	}
	batch := max(s.Batch, 1)
	in := Inputs{
		Batch:   batch,
		Cache:   s.Cache,
		IDs:     input(b, PortIDs, accel.U32, s.Tokens),
		PosQ:    input(b, PortPosQ, accel.U32, s.Tokens*c.NumHeads),
		PosK:    input(b, PortPosK, accel.U32, s.Tokens*c.NumKVHeads),
		Slots:   input(b, PortSlots, accel.U32, s.Tokens),
		Lengths: input(b, PortLengths, accel.U32, batch),
	}
	if s.Block > 0 {
		in.Block = s.Block
		// One row per sequence. A single sequence is the same port with one
		// row, not a different port.
		in.Pages = tensor.Input(b, tensor.ValueDesc{Name: PortPages, DType: accel.U32,
			Shape: tensor.Shape{batch, s.Capacity / s.Block}})
	}
	if err := c.LayerTypes.check(c.NumLayers); err != nil {
		return Inputs{}, err
	}
	hybrid := c.LayerTypes.Hybrid()
	// tensor.LinearAttention requires QueryExtents at every batch size, where
	// softmax attention over one sequence does not need them. So a hybrid
	// declares the port at B = 1 as well (specs/023-cache-kinds.md §2.1).
	if batch > 1 || hybrid {
		in.Extents = input(b, PortExtents, accel.U32, batch)
	}
	if batch > 1 {
		in.Last = input(b, PortLast, accel.U32, batch)
	}
	// The leading axis is the count of the layers that *have* a key/value
	// cache, and a gated-delta layer has none. Sizing it at NumLayers would
	// allocate four times the state for rows nothing writes ([023-D3]).
	full := c.NumLayers
	if hybrid {
		full = c.LayerTypes.Count(LayerFullAttention)
	}
	in.FullLayers = full
	shape := tensor.Shape{full, s.Capacity, c.NumKVHeads, c.HeadDim}
	in.Keys = tensor.NewState(b, tensor.StateDesc{Name: PortKeys, DType: s.Cache, Shape: shape})
	in.Values = tensor.NewState(b, tensor.StateDesc{Name: PortValues, DType: s.Cache, Shape: shape})
	if hybrid {
		if err := declareRecurrent(b, c, s, batch, &in); err != nil {
			return Inputs{}, err
		}
	}

	tensor.Scalar(b, tensor.ScalarDesc{Name: ScalarRoPEBase, Kind: tensor.ScalarF32})
	tensor.Scalar(b, tensor.ScalarDesc{Name: ScalarScale, Kind: tensor.ScalarF32})
	if s.Tokens > 1 && batch == 1 {
		tensor.Scalar(b, tensor.ScalarDesc{Name: ScalarBase, Kind: tensor.ScalarU32})
		in.Base = ScalarBase
	}
	return in, nil
}

// declareRecurrent declares the two states a gated-delta layer holds and the
// index ports that address the second of them.
//
// Three states and not one union. tensor.LinearAttention checks the state's
// shape against [slots, heads, valueDim, keyDim] exactly and its dtype against
// f32, so a union carrying a tag and the widest shape would be reshaped and
// re-typed at every call site — and a tag read at record time is a branch the
// graph does not need, because the layer schedule is known when the graph is
// recorded ([023-D1]).
func declareRecurrent(b *tensor.Builder, c *Config, s GraphSpec, batch int,
	in *Inputs) error {

	if c.Recurrent == nil {
		return fmt.Errorf("model: the stack has %d gated-delta layer(s) and the "+
			"config carries no recurrent geometry; a layer built without one has no "+
			"state shape (specs/023-cache-kinds.md §2)",
			c.LayerTypes.Count(LayerGatedDelta))
	}
	if err := c.Recurrent.check(); err != nil {
		return err
	}
	r := *c.Recurrent
	lin := c.LayerTypes.Count(LayerGatedDelta)
	in.LinearLayers, in.Recurrence = lin, r

	// The slot count is the plan's batch exactly, and not a pool B sequences
	// index into: LinearAttention requires the state's leading extent to equal
	// the number of entries in QueryExtents (§2.1). So a hybrid's concurrency
	// ceiling is B, and 1.13 GiB of recurrent state at B = 8 is charged whether
	// the slots are busy or idle — which is the cost §4 describes.
	in.Recurrent = tensor.NewState(b, tensor.StateDesc{
		Name: PortRecurrent, DType: accel.F32,
		Shape: tensor.Shape{lin, batch, r.Heads, r.ValueDim, r.KeyDim},
	})
	rows := batch*(r.Taps-1) + s.Tokens
	in.ConvRows = rows
	in.ConvWindow = tensor.NewState(b, tensor.StateDesc{
		Name: PortConvWindow, DType: accel.F32,
		Shape: tensor.Shape{lin, rows, r.ConvWidth},
	})
	in.ConvWrite = input(b, PortConvWrite, accel.U32, s.Tokens)
	in.ConvCarry = input(b, PortConvCarry, accel.U32, batch*(r.Taps-1))
	in.ConvCarryWrite = input(b, PortConvCarryWrite, accel.U32, batch*(r.Taps-1))
	for i := range r.Taps {
		in.ConvTaps = append(in.ConvTaps,
			input(b, fmt.Sprintf("%s%d", PortConvTap, i), accel.U32, s.Tokens))
	}
	return nil
}

// NewPagedStep is [NewStep] over a cache addressed through a page table.
//
// The positions are unchanged -- a token's rotary position is where it sits in
// the *sequence*, and paging moves where its key and value are *stored*. Only
// the slots differ, and they differ by exactly one indirection: logical
// position p lives at row pages[p/Block]*Block + p%Block
// (specs/005-kv-cache.md §2.2).
//
// The table is the sequence's own row and not the pool: a table of n entries
// addresses n*Block positions, and a position past that is a step the caller
// has not allocated blocks for. Refusing it here is the difference between a
// diagnostic and a write into whatever block sits after this sequence's last
// one -- which is another sequence's cache, and which reads back as a fluent
// answer to somebody else's prompt.
func NewPagedStep(c *Config, first, tokens int, pages []int, block int) (Step, error) {
	s, err := NewStep(c, first, tokens)
	if err != nil {
		return Step{}, err
	}
	if block <= 0 {
		return Step{}, fmt.Errorf("model: NewPagedStep: block is %d; it is positions "+
			"per block and a paged step has at least one", block)
	}
	if need := (first + tokens + block - 1) / block; need > len(pages) {
		return Step{}, fmt.Errorf("model: NewPagedStep: %d positions over blocks of "+
			"%d need %d page table entries and the table has %d; the missing blocks "+
			"are ones nothing has allocated", first+tokens, block, need, len(pages))
	}
	for i := range s.Slots {
		p := first + i
		b := pages[p/block]
		if b < 0 {
			return Step{}, fmt.Errorf("model: NewPagedStep: page table entry %d is %d; "+
				"a physical block is an index", p/block, b)
		}
		s.Slots[i] = uint32(b*block + p%block)
	}
	return s, nil
}

// input declares one u32 vector port.
func input(b *tensor.Builder, name string, dt accel.DType, n int) *tensor.Tensor {
	return tensor.Input(b, tensor.ValueDesc{Name: name, DType: dt, Shape: tensor.Shape{n}})
}

// check refuses a step the graph cannot record. Each names the field.
//
// specs/004-model-graph.md §7's remaining rows are config fields and are
// refused by [ParseConfig]; these two are the step's own, and they are refused
// here rather than left to accel because accel sees a shape and this sees the
// number a caller passed.
func (s GraphSpec) check() error {
	if s.Tokens <= 0 {
		return fmt.Errorf("model: Tokens is %d; a step scores at least one token", s.Tokens)
	}
	if s.Batch < 0 {
		return fmt.Errorf("model: Batch is %d; it is how many sequences step "+
			"together, and zero or one is a single sequence", s.Batch)
	}
	if s.Batch > 1 && s.Block <= 0 {
		return fmt.Errorf("model: Batch is %d and Block is %d; sequences that step "+
			"together have different lengths, so a contiguous cache would pad every "+
			"one of them to the longest -- which is the allocation paging exists to "+
			"avoid (specs/008-scheduler.md §2)", s.Batch, s.Block)
	}
	if s.Batch > 1 && s.Tokens < s.Batch {
		return fmt.Errorf("model: Batch is %d and Tokens is %d; Tokens is the total "+
			"across the batch, and a step with fewer rows than sequences has at "+
			"least one contributing nothing -- which is legal, but not expressible "+
			"as a plan this small", s.Batch, s.Tokens)
	}
	if s.Block < 0 {
		return fmt.Errorf("model: Block is %d; it is positions per block, and zero "+
			"is a contiguous cache", s.Block)
	}
	if s.Block > 0 && s.Capacity%s.Block != 0 {
		return fmt.Errorf("model: Capacity is %d over blocks of %d; a page table "+
			"addresses whole blocks, so a capacity that is not a multiple of the "+
			"block size names positions no entry reaches", s.Capacity, s.Block)
	}
	if s.Capacity <= 0 {
		return fmt.Errorf("model: Capacity is %d; a cache holds at least one position",
			s.Capacity)
	}
	if s.Tokens > s.Capacity {
		return fmt.Errorf("model: Tokens is %d and Capacity is %d; every token of a step "+
			"is written to the cache, so a step cannot be longer than the cache holds",
			s.Tokens, s.Capacity)
	}
	switch s.Cache {
	case accel.F32, accel.F16:
		// f16 was refused here until 2026-08-27, and the refusal named what was
		// missing rather than the dtype: ScatterRows reads the rows and writes
		// the state with one kernel, so the two share a dtype, and the
		// projections that produce the rows are f32. nn.Attention records the
		// Cast now, so both widths are reachable.
	default:
		return fmt.Errorf("model: Cache is %v; a key/value cache is f32 or f16 "+
			"(specs/005-kv-cache.md §3)", s.Cache)
	}
	return nil
}

// Validate reports whether these ports are the ones a config's forward pass
// reads, in the shapes it reads them.
//
// [Declare] builds an Inputs that satisfies this by construction. It is
// exported because [Builder.Forward] takes an Inputs a caller may have
// assembled by hand — binding a cache that outlives one plan is the case — and
// a wrong extent there is otherwise reported by whichever operator reaches it
// first, in accel's terms rather than in the model's.
func (in Inputs) Validate(c *Config) error {
	if c == nil {
		return fmt.Errorf("model: Inputs.Validate: the config is nil")
	}
	t, err := vector(PortIDs, in.IDs, 0)
	if err != nil {
		return err
	}
	if _, err := vector(PortPosQ, in.PosQ, t*c.NumHeads); err != nil {
		return err
	}
	if _, err := vector(PortPosK, in.PosK, t*c.NumKVHeads); err != nil {
		return err
	}
	if _, err := vector(PortSlots, in.Slots, t); err != nil {
		return err
	}
	batch := max(in.Batch, 1)
	if _, err := vector(PortLengths, in.Lengths, batch); err != nil {
		return err
	}
	if batch > 1 {
		for _, e := range []struct {
			name string
			x    *tensor.Tensor
		}{{PortExtents, in.Extents}, {PortLast, in.Last}} {
			if _, err := vector(e.name, e.x, batch); err != nil {
				return err
			}
		}
	}
	if in.Keys == nil || in.Values == nil {
		return fmt.Errorf("model: the %s or %s cache state is missing", PortKeys, PortValues)
	}
	// A batched step is ragged, and a ragged step derives every token's
	// position from its sequence's length and count. So there is no single
	// first position for a base to name, and accel refuses one -- which makes
	// the two rules below a single-sequence question.
	if batch > 1 {
		if in.Base != "" {
			return fmt.Errorf("model: Base is %q on a %d-sequence step; a ragged step "+
				"derives each token's position from its own sequence, so a base is a "+
				"value nothing reads", in.Base, batch)
		}
		return nil
	}
	// A decode names no base. accel refuses a BaseName on a one-token step,
	// and a plan that declared the scalar anyway would demand a binding for a
	// value nothing reads.
	if t == 1 && in.Base != "" {
		return fmt.Errorf("model: Base is %q on a one-token step; a decode has no causal "+
			"mask to place and accel refuses the scalar", in.Base)
	}
	if t > 1 && in.Base == "" {
		return fmt.Errorf("model: Base is empty on a %d-token step; a prefill's causal "+
			"limit is its first position plus the query's offset", t)
	}
	return nil
}

// vector checks one port's rank and extent, returning the extent. want of 0
// accepts any positive length, which is how the token count is read from ids
// before anything else can be compared against it.
func vector(name string, x *tensor.Tensor, want int) (int, error) {
	if x == nil {
		return 0, fmt.Errorf("model: the %s port is missing", name)
	}
	s := x.Shape()
	if len(s) != 1 || s[0] <= 0 {
		return 0, fmt.Errorf("model: the %s port is %v; it is a vector", name, s)
	}
	if want != 0 && s[0] != want {
		return 0, fmt.Errorf("model: the %s port is %d long and this step needs %d",
			name, s[0], want)
	}
	return s[0], nil
}

// Record is the whole of specs/004-model-graph.md §3 in one call: it declares
// the ports, records the forward pass, and records Output([PortLogits]).
//
// The [nn.Graph] it returns carries the diagnostics, so a caller checks
// g.Err() once rather than after every line. The [Inputs] it returns are the
// ports, for a caller that binds them by tensor rather than by name.
func Record(b *tensor.Builder, m Builder, s GraphSpec) (*nn.Graph, Inputs, error) {
	if m == nil {
		return nil, Inputs{}, fmt.Errorf("model: Record: the builder is nil")
	}
	c := m.Config()
	in, err := Declare(b, c, s)
	if err != nil {
		return nil, Inputs{}, err
	}
	g := &nn.Graph{B: b, Eps: c.RMSNormEps, Stored: s.Stored}
	logits := m.Forward(g, in)
	tensor.Output(b, PortLogits, logits)
	return g, in, g.Err()
}

// Step is the runtime data one forward pass binds beyond the token ids: the
// contents of §3's posq, posk, slots and lengths ports.
//
// It is here rather than in the caller because row 12's formula — each token's
// position repeated once per head — is the kind of thing that is written per
// token by mistake, produces a correctly shaped tensor for a single-head model,
// and rotates every head but the first by the wrong angle for every other one.
type Step struct {
	// PosQ and PosK are one position per row, T·H and T·H_kv long.
	PosQ, PosK []uint32

	// Slots is the cache row each token is written to, T long.
	Slots []uint32

	// Lengths is how much of the cache holds real tokens after this step, as
	// the one-element tensor AttentionOptions.Lengths binds.
	Lengths []uint32

	// Base is the value of the [ScalarBase] scalar: the first position this
	// step scores. It is unread on a decode.
	Base uint32
}

// NewStep builds the position data for tokens at consecutive positions
// first..first+tokens-1 of a contiguous cache.
//
// Contiguous, so a token's slot is its position. [NewPagedStep] is the same
// step over a cache addressed through a page table.
func NewStep(c *Config, first, tokens int) (Step, error) {
	if c == nil {
		return Step{}, fmt.Errorf("model: NewStep: the config is nil")
	}
	if tokens <= 0 {
		return Step{}, fmt.Errorf("model: NewStep: tokens is %d; a step scores at least one",
			tokens)
	}
	if first < 0 {
		return Step{}, fmt.Errorf("model: NewStep: first is %d; a position is not negative",
			first)
	}
	s := Step{
		PosQ:    make([]uint32, 0, tokens*c.NumHeads),
		PosK:    make([]uint32, 0, tokens*c.NumKVHeads),
		Slots:   make([]uint32, tokens),
		Lengths: []uint32{uint32(first + tokens)},
		Base:    uint32(first),
	}
	for i := range tokens {
		p := uint32(first + i)
		s.Slots[i] = p
		for range c.NumHeads {
			s.PosQ = append(s.PosQ, p)
		}
		for range c.NumKVHeads {
			s.PosK = append(s.PosK, p)
		}
	}
	return s, nil
}
