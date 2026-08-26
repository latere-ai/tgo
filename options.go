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

	// cacheScope and cachePositions are [WithPrefixCache]'s. The zero scope is
	// [CacheOff], so a Model opened with no option prefills every prompt whole.
	cacheScope     CacheScope
	cachePositions int
}

// defaults is every option's zero meaning, spelled out so a field's default is
// read here rather than inferred from a zero value three files away.
func defaults() options {
	return options{device: AutoDevice, precision: AutoPrecision, context: DefaultContext,
		cacheScope: CacheOff}
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
	salt     string
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
// WithCacheSalt bounds what a conversation may match in a shared block pool.
//
// Under [CacheProcess] a block is reachable from any session, and a hit is
// faster than a miss, so timing makes the cache a membership oracle over other
// conversations' prompts (016 §7). The salt is mixed into every block hash, so
// two conversations share only when the layer in front says they may.
//
// The empty string is a key of its own and shares with nobody rather than with
// everybody: a caller who supplies nothing gets the safe answer, not the fast
// one. tgo has no notion of a tenant (009 §7), so what belongs here is whatever
// the layer in front uses to tell them apart — the server puts a request's
// cache_salt in it.
func WithCacheSalt(v string) SessionOption {
	return func(o *sessionOptions) { o.salt = v }
}

func WithThinking(v bool) SessionOption {
	return func(o *sessionOptions) { o.thinking = v }
}

// WithTools declares the functions the model may call. They are rendered into
// the system turn, which is where the Qwen3 template puts them.
func WithTools(specs ...chat.ToolSpec) SessionOption {
	return func(o *sessionOptions) { o.tools = specs }
}

// CacheScope bounds what a request may reuse another request's key/value state
// for, and is specs/016-prefix-cache.md §7's table as a type.
//
// A cache hit is faster than a miss and that timing is observable, so
// cross-request reuse is a membership oracle over other requests' prompts. The
// scope is what makes the default safe for a deployment; it is a decision the
// operator makes rather than one tgo makes for them (016-D7).
type CacheScope int8

// The scopes a caller can ask for.
const (
	// CacheOff is the zero value: nothing is looked up and nothing is kept.
	// Every prompt is prefilled whole, which is the cold baseline §7 names and
	// what a [Model] does when [WithPrefixCache] is not passed.
	CacheOff CacheScope = iota

	// CacheSession reuses within one conversation only. It is safe under
	// multi-tenancy — no block ever crosses a [Session] — and it still
	// captures the multi-turn win, which approaches 1-1/n by turn n and is
	// most of the value.
	CacheSession

	// CacheProcess shares across every session in the process, which is the
	// scope an agent runtime or a single-tenant server wants: two conversations
	// that begin with the same system prompt prefill it once between them.
	//
	// It costs a page table in the innermost loop of every decode, and it makes
	// a hit observable across conversations — so it is a deployment's decision
	// and never a default (016-D7). [WithPrefixCache]'s salt is what narrows it
	// back where the deployment is not single-tenant.
	CacheProcess
)

// String names a CacheScope.
func (s CacheScope) String() string {
	switch s {
	case CacheOff:
		return "off"
	case CacheSession:
		return "session"
	case CacheProcess:
		return "process"
	}
	return fmt.Sprintf("CacheScope(%d)", int8(s))
}

// WithPrefixCache reuses the key/value state a conversation has already paid
// for, instead of prefilling the whole prompt again.
//
// The key/value state at position t is a function of tokens 0..t and the
// weights alone, so a prompt that begins with tokens this session already
// scored begins with key/value state this session already holds. Reusing it is
// not an approximation; it is declining to recompute a pure function
// (specs/016-prefix-cache.md §1). positions caps how many leading positions may
// be reused, and is clamped to the session's capacity.
//
// It is off by default. Turning it on changes what a request costs and,
// measurably, what it produces: the reused prefix was computed under a
// different prefill shape and floating point is not associative, so a warm
// answer equals a cold one in distribution rather than bit for bit (016-D6).
//
// # Reuse stops one token short of the prompt, always
//
// The cache holds key/value state, not logits. Sampling the next token needs
// the logits at the last prompt position, and those come from a forward pass
// over it, so the reusable run is capped at len(ids)-1 and an identical prompt
// resubmitted still prefills one token (016-D10). In chat this is invisible,
// because a rendered prompt always ends with a fresh assistant opener.
//
// # What positions means, and it is not the same number in both scopes
//
// Under [CacheSession] it caps how many leading positions one conversation may
// reuse of its own cache, which each session holds privately.
//
// Under [CacheProcess] it is **the shared pool's size**, in positions, rounded
// down to whole blocks of [CacheBlock]. That is the memory the process spends
// on key/value state in total — one allocation for every session rather than
// one each — so a server's cache footprint stops scaling with concurrency and
// starts being a number the operator chose. A prefix longer than the pool
// cannot be held whatever the cap said, so the pool is also the cap.
//
// # [CacheProcess] used to be refused, and what changed
//
// Sharing across sessions means one session attending to key/value rows another
// session wrote, which means addressing the cache through a page table. For as
// long as specs/004-model-graph.md §3 declared no such port, nn.Attention could
// bind no tensor.AttentionOptions.Pages and the kernels read the cache
// contiguously, taking a row's index as its position. The port exists now, and
// the scope with it.
func WithPrefixCache(scope CacheScope, positions int) Option {
	return func(o *options) { o.cacheScope, o.cachePositions = scope, positions }
}

// checkCache refuses a prefix-cache configuration that cannot be honoured.
func (o options) checkCache() error {
	switch o.cacheScope {
	case CacheOff:
		return nil
	case CacheSession:
		if o.cachePositions <= 0 {
			return fmt.Errorf("tgo: WithPrefixCache asks to reuse %d positions; a cache "+
				"holds at least one", o.cachePositions)
		}
		return nil
	case CacheProcess:
		// The pool is measured in blocks, so a size below one block is a
		// configuration that would allocate nothing and share nothing.
		if o.cachePositions < CacheBlock {
			return fmt.Errorf("tgo: WithPrefixCache(process) asks for a pool of %d "+
				"positions and a block holds %d; the pool is the memory every "+
				"session shares, so it is at least one block "+
				"(specs/016-prefix-cache.md §3)", o.cachePositions, CacheBlock)
		}
		return nil
	}
	return fmt.Errorf("tgo: WithPrefixCache scope is %v; it is one of %v, %v or %v",
		o.cacheScope, CacheOff, CacheSession, CacheProcess)
}
