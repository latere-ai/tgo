// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
)

// Engine is everything this package needs from a model.
//
// It is an interface and not [tgo.Model] so that the HTTP surface can be tested
// against a scripted token stream, with no device and no weights
// (specs/009-server.md 009-D4). [Wrap] is the only implementation that ships,
// and it forwards.
type Engine interface {
	// Name is the model id a request must ask for. A request naming anything
	// else is a 404: this package serves one model (009-D5).
	Name() string

	// Context is the KV capacity a new session gets, in positions.
	Context() int

	// VocabSize is how many token ids exist. A logit_bias key outside
	// [0, VocabSize) is refused rather than dropped: a ban that lands on no
	// token changes the answer without saying so.
	VocabSize() int

	// CacheBytesPerSession is what one session's key and value states cost,
	// which is what [WithKVBudget] divides into an admission limit.
	CacheBytesPerSession() int64

	// CheckSchema reports whether a requested JSON schema can be compiled to
	// a per-step token mask over this model's vocabulary.
	//
	// It is on the engine and not in adapt.go because the compilation needs
	// the vocabulary, and it is separate from generation because a schema the
	// compiler refuses is a 400 that must be answered before a session is
	// allocated. The error is the compiler's, naming the construct and the
	// obstruction (015-D4).
	CheckSchema(schema []byte) error

	// NewSession takes one conversation for one request. One per in-flight
	// request.
	//
	// The context is the request's. It is here because an engine that pools
	// its sessions ([WrapPool]) waits for a free one rather than allocating,
	// and a client that hangs up while waiting must stop waiting. An engine
	// that allocates per request ([Wrap]) never blocks and never reads it.
	NewSession(ctx context.Context, spec SessionSpec) (Session, error)
}

// SessionSpec is what a request needs its session built with.
//
// Tools and thinking are session options in tgo rather than fields of
// [tgo.Policy], because they are rendered into the prompt and Policy is one
// request's sampling configuration. A server that made one session and reused
// it could not honour either.
type SessionSpec struct {
	// Tools are the functions the model may call, rendered into the system
	// turn.
	Tools []chat.ToolSpec

	// Thinking says whether the assistant may open a thinking block.
	Thinking bool

	// Recorder instruments the loop. It is the server's, one per request, and
	// is what feeds tgo_decode_step_seconds and tgo_logits_readback_seconds.
	Recorder *bench.Recorder

	// Key bounds what this request may reuse of another request's key/value
	// state, and is the request's cache_salt
	// (specs/016-prefix-cache.md §7.1).
	//
	// It is honoured only by a pooled engine, where it decides which pooled
	// session a request may be routed to, and it fails closed: a request with
	// no key matches only sessions whose last request had none
	// (specs/019-session-affinity.md 019-D3). tgo has no notion of a tenant
	// (009 §7), so the key is whatever the layer in front supplies.
	Key string
}

// Session is one conversation, and is used by exactly one request.
type Session interface {
	// Chat renders messages through the model's template and generates.
	Chat(ctx context.Context, msgs []chat.Message, p tgo.Policy) (Stream, error)

	// Complete generates from raw text with no template, which is what
	// /v1/completions serves.
	Complete(ctx context.Context, prompt string, p tgo.Policy) (Stream, error)

	// Close ends the request's hold on the session.
	//
	// For an engine that allocates per request ([Wrap]) that releases the
	// session's device memory, which is the KV reservation the admission
	// semaphore counted. For a pooled engine ([WrapPool]) it returns the
	// session to the pool with its history intact, which is what makes the
	// next request able to reuse it, and the memory is held until the pool is
	// closed (019-D2).
	Close() error
}

// Stream yields a completion as it is produced. It is [tgo.Stream]'s surface,
// minus Text, which is Event().Text.
type Stream interface {
	Next() bool
	Event() tgo.Event
	Usage() tgo.Usage
	Err() error

	// StopReason and StopSequence are why generation ended. They are read once,
	// after Next has returned false: a stream still running reports
	// [tgo.StopRunning], and so does one that failed, whose answer is Err.
	StopReason() tgo.StopReason
	StopSequence() string

	// LogProbs is the tokens the last Next produced, and is empty unless the
	// request asked for them. It is valid until the next Next
	// (specs/030-logprobs.md §2), so a handler that keeps them appends.
	LogProbs() []tgo.TokenProb
}

// Wrap adapts a loaded model to [Engine], serving it under the id name.
//
// Every method forwards. The mapping that is not a forward -- an ir.Request to
// messages and a [tgo.Policy] -- is in adapt.go, and it is the only place the
// two vocabularies meet (009-D10).
func Wrap(m *tgo.Model, name string) Engine { return &modelEngine{m: m, name: name} }

// modelEngine is [Wrap]'s implementation.
type modelEngine struct {
	m    *tgo.Model
	name string
}

func (e *modelEngine) Name() string                { return e.name }
func (e *modelEngine) Context() int                { return e.m.Info().Context }
func (e *modelEngine) VocabSize() int              { return e.m.Info().VocabSize }
func (e *modelEngine) CacheBytesPerSession() int64 { return e.m.Info().CacheBytesPerSession }

func (e *modelEngine) CheckSchema(schema []byte) error { return e.m.CheckSchema(schema) }

func (e *modelEngine) NewSession(_ context.Context, spec SessionSpec) (Session, error) {
	opts := []tgo.SessionOption{
		tgo.WithThinking(spec.Thinking),
		// The request's cache_salt, which this engine used to drop.
		//
		// A session of its own shares nothing with another session, so under
		// [tgo.CacheSession] the salt reached nothing and dropping it was
		// invisible. Under [tgo.CacheProcess] every session draws from one
		// block pool, and a salt that does not reach it means two tenants with
		// the same system prompt seed identically: the second one's first token
		// arrives fast, which is a membership test over the first one's prompt
		// (016 §7.1). §4's loss report told both of them it had been honoured.
		tgo.WithCacheSalt(spec.Key),
	}
	if len(spec.Tools) > 0 {
		opts = append(opts, tgo.WithTools(spec.Tools...))
	}
	if spec.Recorder != nil {
		opts = append(opts, tgo.WithRecorder(spec.Recorder))
	}
	s, err := e.m.NewSession(opts...)
	if err != nil {
		return nil, err
	}
	return &modelSession{s: s}, nil
}

// modelSession forwards to a [tgo.Session].
type modelSession struct {
	s  *tgo.Session
	st *tgo.Stream
}

func (s *modelSession) Chat(ctx context.Context, msgs []chat.Message, p tgo.Policy) (Stream, error) {
	// The nil check is not ceremony: a typed nil in an interface is not nil,
	// and a caller that checked the stream rather than the error would read a
	// method on it.
	st, err := s.s.Chat(ctx, msgs, p)
	if err != nil {
		return nil, err
	}
	s.st = st
	return st, nil
}

func (s *modelSession) Complete(ctx context.Context, prompt string, p tgo.Policy) (Stream, error) {
	st, err := s.s.Complete(ctx, prompt, p)
	if err != nil {
		return nil, err
	}
	s.st = st
	return st, nil
}

// Reused is how many leading prompt positions came from a cache rather than
// from a forward pass, the same quantity [leasedSession.Reused] reports.
//
// Both engines answer it, because it is the number that says whether a request
// was isolated from another one's work, and an engine that could not be asked
// was an engine whose isolation could not be tested. server.Wrap dropped the
// request's cache_salt for a week and nothing noticed.
func (s *modelSession) Reused() int {
	if s.st == nil {
		return 0
	}
	return s.st.Usage().CachedPromptTokens
}

func (s *modelSession) Close() error { return s.s.Close() }

// WrapPool adapts a loaded model to [Engine] with a pool of n sessions behind
// it, so a request can be routed to the session already holding the longest
// matching prefix.
//
// This is what makes [github.com/latere-ai/tgo.WithPrefixCache] reachable from
// a server. Without it every request gets its own session and closes it on the
// way out, so a session never sees a second turn and there is no own-prefix to
// reuse (specs/019-session-affinity.md §1).
//
// What it costs is stated where the operator will read it: every one of the n
// sessions' key/value cache is allocated here and held until [PoolEngine.Close],
// so a process that served one request holds n sessions' cache for its life
// (019-D2). A device that cannot hold it fails here, at startup, rather than
// under load.
//
// n is also the concurrency: pass it to [WithConcurrency] so that the admission
// semaphore and the pool are the same number arrived at once. The pool blocks a
// request that finds no free session, and the admitter is what turns that wait
// into a bounded queue and a 429 (009-D3).
func WrapPool(m *tgo.Model, name string, n int) (*PoolEngine, error) {
	p, err := m.NewPool(n)
	if err != nil {
		return nil, err
	}
	return &PoolEngine{m: m, name: name, pool: p}, nil
}

// PoolEngine is [WrapPool]'s implementation: [Wrap]'s engine with a session
// pool in place of a session per request.
type PoolEngine struct {
	modelEngine
	pool *tgo.Pool
}

// Sessions is how many conversations the pool holds, which is also how many
// requests can generate at once.
func (e *PoolEngine) Sessions() int { return e.pool.Size() }

// Close releases every pooled session's device memory.
//
// It must be called before the [github.com/latere-ai/tgo.Model] is closed,
// which is the order accel requires, and after the last request has finished.
func (e *PoolEngine) Close() error { return e.pool.Close() }

// NewSession takes a pooled session for one request rather than allocating one.
//
// The session is chosen at the first generation and not here, because routing
// compares the request's token ids against every pooled session's history and
// the ids exist only once the prompt is rendered (019 §3).
func (e *PoolEngine) NewSession(ctx context.Context, spec SessionSpec) (Session, error) {
	l, err := e.pool.Acquire(ctx, tgo.PoolRequest{
		Tools: spec.Tools, Thinking: spec.Thinking, Key: spec.Key, Recorder: spec.Recorder,
	})
	if err != nil {
		return nil, err
	}
	return &leasedSession{l: l}, nil
}

// leasedSession forwards to a [github.com/latere-ai/tgo.Lease].
type leasedSession struct{ l *tgo.Lease }

func (s *leasedSession) Chat(ctx context.Context, msgs []chat.Message, p tgo.Policy) (Stream, error) {
	st, err := s.l.Chat(ctx, msgs, p)
	if err != nil {
		return nil, err
	}
	return st, nil
}

func (s *leasedSession) Complete(ctx context.Context, prompt string, p tgo.Policy) (Stream, error) {
	st, err := s.l.Complete(ctx, prompt, p)
	if err != nil {
		return nil, err
	}
	return st, nil
}

// Close returns the session to the pool with its history intact, truncated to
// the positions whose key/value state was actually written (019-D5). It reports
// no error because releasing a lease cannot fail: nothing is freed.
func (s *leasedSession) Close() error {
	s.l.Release()
	return nil
}

// Reused is how many leading prompt positions the request took from the pooled
// session's cache. It is [github.com/latere-ai/tgo.Usage.CachedPromptTokens]
// for a caller holding the session rather than the stream.
func (s *leasedSession) Reused() int { return s.l.Reused() }
