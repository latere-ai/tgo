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

	// PortKeys and PortValues are the two cache states, [L, C, H_kv, d_h]
	// each: one allocation per role for the whole model, sliced per layer
	// (specs/005-kv-cache.md §2.1).
	PortKeys   = "k"
	PortValues = "v"

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

	// Stored reports how the loader stored the weight of a given full port
	// name, [accel.F16] or [accel.I8]. A nil func means every weight is f16.
	Stored func(name string) accel.DType
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

	// Base is the name of the prefill's first-position scalar, and is empty on
	// a decode step. It is carried rather than assumed so that a hand-built
	// Inputs says which of the two plans it is for.
	Base string
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
	in := Inputs{
		IDs:     input(b, PortIDs, accel.U32, s.Tokens),
		PosQ:    input(b, PortPosQ, accel.U32, s.Tokens*c.NumHeads),
		PosK:    input(b, PortPosK, accel.U32, s.Tokens*c.NumKVHeads),
		Slots:   input(b, PortSlots, accel.U32, s.Tokens),
		Lengths: input(b, PortLengths, accel.U32, 1),
	}
	shape := tensor.Shape{c.NumLayers, s.Capacity, c.NumKVHeads, c.HeadDim}
	in.Keys = tensor.NewState(b, tensor.StateDesc{Name: PortKeys, DType: s.Cache, Shape: shape})
	in.Values = tensor.NewState(b, tensor.StateDesc{Name: PortValues, DType: s.Cache, Shape: shape})

	tensor.Scalar(b, tensor.ScalarDesc{Name: ScalarRoPEBase, Kind: tensor.ScalarF32})
	tensor.Scalar(b, tensor.ScalarDesc{Name: ScalarScale, Kind: tensor.ScalarF32})
	if s.Tokens > 1 {
		tensor.Scalar(b, tensor.ScalarDesc{Name: ScalarBase, Kind: tensor.ScalarU32})
		in.Base = ScalarBase
	}
	return in, nil
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
	case accel.F32:
	case accel.F16:
		// accel takes an f16 cache -- C5 closed, and halving the largest
		// allocation after the weights is what specs/005-kv-cache.md §3 is
		// written for. What is missing is one layer down: ScatterRows reads
		// the rows and writes the state with one kernel, so the two share a
		// dtype, and the projections that produce the rows are f32. Reaching
		// an f16 cache needs a Cast on the scattered rows inside nn.Attention,
		// which does not record one. Refused here, naming that, rather than
		// left to surface as accel's "Cast the rows" from a call this package
		// does not make.
		return fmt.Errorf("model: Cache is f16 and the graph writes f32 projections to " +
			"it; tgo/nn's attention block records no Cast on the scattered rows, so an " +
			"f16 cache is not reachable yet (specs/005-kv-cache.md §3)")
	default:
		return fmt.Errorf("model: Cache is %v; a key/value cache is f32 "+
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
	if _, err := vector(PortLengths, in.Lengths, 1); err != nil {
		return err
	}
	if in.Keys == nil || in.Values == nil {
		return fmt.Errorf("model: the %s or %s cache state is missing", PortKeys, PortValues)
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
// Contiguous, so a token's slot is its position: the paged case maps a logical
// position through a page table (specs/005-kv-cache.md §2.2) and binds
// AttentionOptions.Pages, which this graph does not record.
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
