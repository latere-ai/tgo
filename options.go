// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"fmt"
	"github.com/latere-ai/tgo/bench"

	"github.com/latere-ai/tgo/chat"
)

// Device names the accelerator a model runs on.
type Device int

// The devices a caller can ask for.
const (
	// AutoDevice is the zero value: the best device present, falling back to
	// the CPU backend. It is the default because a caller who has not said
	// otherwise wants the model to run.
	AutoDevice Device = iota

	// CPU is accel's CPU backend. It is always present, which is what makes it
	// the tier-1 device every test in this tree runs on.
	CPU

	// Metal is the Metal backend, and is refused where there is none rather
	// than falling back: a caller who names a device has a reason to.
	Metal
)

// String names a Device for an error message and for [Info].
func (d Device) String() string {
	switch d {
	case AutoDevice:
		return "auto"
	case CPU:
		return "cpu"
	case Metal:
		return "metal"
	}
	return fmt.Sprintf("Device(%d)", int(d))
}

// Precision is the form the weights take on the device.
type Precision int

// The precisions a caller can ask for. They are the policy
// [github.com/latere-ai/tgo/weights] resolves, restated here so the public
// surface does not export the loader (007-D7).
const (
	// AutoPrecision is the zero value: f16 where the weights fit the device's
	// budget and int8 where they do not, printed either way.
	AutoPrecision Precision = iota

	// F16 stores one 16-bit float per weight.
	F16

	// Int8 stores one int8 quant per weight plus one f16 scale per block.
	Int8
)

// String names a Precision for [Info] and for error text.
func (p Precision) String() string {
	switch p {
	case AutoPrecision:
		return "auto"
	case F16:
		return "f16"
	case Int8:
		return "int8"
	}
	return fmt.Sprintf("Precision(%d)", int(p))
}

// DefaultContext is the KV capacity a session takes when nothing says
// otherwise: specs/005-kv-cache.md 005-D2's 4096, and not the model's
// max_position_embeddings, which for Qwen3 would reserve gigabytes before the
// first token.
const DefaultContext = 4096

// Option configures [Open].
type Option func(*options)

// options is what the [Option] list resolves to.
type options struct {
	device    Device
	precision Precision
	context   int
}

// defaults is every option's zero meaning, spelled out so a field's default is
// read here rather than inferred from a zero value three files away.
func defaults() options {
	return options{device: AutoDevice, precision: AutoPrecision, context: DefaultContext}
}

// WithDevice selects the accelerator. The default is [AutoDevice].
func WithDevice(d Device) Option { return func(o *options) { o.device = d } }

// WithPrecision selects how the weights are stored. The default is
// [AutoPrecision].
func WithPrecision(p Precision) Option { return func(o *options) { o.precision = p } }

// WithContext sets the KV capacity a session gets, in positions. The default is
// [DefaultContext].
//
// It is a per-session number and not a model constant (005-D2). Raising it
// prints what the cache will cost before anything is allocated (005-D3),
// because a caller who asks for a 32k context should learn the price when they
// ask rather than from an out-of-memory error.
func WithContext(n int) Option { return func(o *options) { o.context = n } }

// SessionOption configures [Model.NewSession].
type SessionOption func(*sessionOptions)

// sessionOptions is what the [SessionOption] list resolves to.
type sessionOptions struct {
	context  int
	thinking bool
	tools    []chat.ToolSpec
	recorder *bench.Recorder
}

// WithRecorder instruments this session's loop, one [bench.Step] per prefill
// and per decode step.
//
// specs/017-benchmarks.md 017-D1 makes the host/submit/device/readback
// breakdown the deliverable rather than a detail: a single throughput number
// cannot say whether a regression is tgo's or accel's, which is the question
// this project exists to answer. The engine was already recording the four
// terms and had no way to hand them out, so `tgo bench` could report wall-clock
// throughput and had to print the breakdown as missing -- exactly the number
// 017-D1 says is not enough on its own.
//
// The recorder is the caller's, so its lifetime and its window are the caller's
// too, and a nil one costs the loop one branch per step (017-D3).
func WithRecorder(r *bench.Recorder) SessionOption {
	return func(o *sessionOptions) { o.recorder = r }
}

// WithSessionContext overrides this session's KV capacity.
//
// specs/007-engine.md §1 lists no session options and specs/005-kv-cache.md
// 005-D2 says capacity is a session parameter, which is a contradiction only a
// caller can resolve: one long conversation beside twenty short ones is the
// case the paged cache exists for, and until [016] lands the only way to spend
// less on the short ones is to ask for less.
//
// [016]: https://github.com/latere-ai/tgo/blob/main/specs/016-prefix-cache.md
func WithSessionContext(n int) SessionOption {
	return func(o *sessionOptions) { o.context = n }
}

// WithThinking says whether the assistant may open a thinking block.
//
// The default is true, which is what a Qwen3 checkpoint does unprompted: the
// renderer emits no pre-closed block and the model opens its own. False emits a
// pre-closed one, which is how the template turns thinking off — omitting it
// would leave the model free to open one anyway (specs/003-chat-template.md §3).
//
// It is a session option rather than a field of [Policy] because [Policy] is
// specs/007-engine.md §1's and adding to it would change a published struct.
func WithThinking(v bool) SessionOption {
	return func(o *sessionOptions) { o.thinking = v }
}

// WithTools declares the functions the model may call. They are rendered into
// the system turn, which is where the Qwen3 template puts them.
func WithTools(specs ...chat.ToolSpec) SessionOption {
	return func(o *sessionOptions) { o.tools = specs }
}
