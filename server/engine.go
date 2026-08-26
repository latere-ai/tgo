// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

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

	// NewSession allocates one conversation. One per in-flight request.
	NewSession(spec SessionSpec) (Session, error)
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
}

// Session is one conversation, and is used by exactly one request.
type Session interface {
	// Chat renders messages through the model's template and generates.
	Chat(ctx context.Context, msgs []chat.Message, p tgo.Policy) (Stream, error)

	// Complete generates from raw text with no template, which is what
	// /v1/completions serves.
	Complete(ctx context.Context, prompt string, p tgo.Policy) (Stream, error)

	// Close releases the session's device memory, and with it the KV
	// reservation the admission semaphore counted.
	Close() error
}

// Stream yields a completion as it is produced. It is [tgo.Stream]'s surface,
// minus Text, which is Event().Text.
type Stream interface {
	Next() bool
	Event() tgo.Event
	Usage() tgo.Usage
	Err() error
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

func (e *modelEngine) NewSession(spec SessionSpec) (Session, error) {
	opts := []tgo.SessionOption{tgo.WithThinking(spec.Thinking)}
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
type modelSession struct{ s *tgo.Session }

func (s *modelSession) Chat(ctx context.Context, msgs []chat.Message, p tgo.Policy) (Stream, error) {
	// The nil check is not ceremony: a typed nil in an interface is not nil,
	// and a caller that checked the stream rather than the error would read a
	// method on it.
	st, err := s.s.Chat(ctx, msgs, p)
	if err != nil {
		return nil, err
	}
	return st, nil
}

func (s *modelSession) Complete(ctx context.Context, prompt string, p tgo.Policy) (Stream, error) {
	st, err := s.s.Complete(ctx, prompt, p)
	if err != nil {
		return nil, err
	}
	return st, nil
}

func (s *modelSession) Close() error { return s.s.Close() }
