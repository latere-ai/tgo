// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"context"
	"time"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/chat"
)

// The batched engine.
//
// specs/022-batched-serving.md. [WrapPool] gives every in-flight request its
// own [tgo.Session], and a session owns a forward pass -- so B concurrent
// requests read the weights B times per token produced, and throughput is what
// one sequence gets. A [tgo.Runner] puts every in-flight request in one step.
//
// Nothing else in this package moves. The engine is chosen where the model is
// opened; [Server] holds an [Engine] and the four dialect routes, the loss
// header, the 429 and the 499 rule are what they were (022 §10).

// WrapRunner adapts a loaded model to [Engine] with a batched runner behind it,
// so every in-flight request shares one forward pass.
//
// It needs a model opened with [github.com/latere-ai/tgo.WithPrefixCache] at
// [github.com/latere-ai/tgo.CacheProcess]: sequences that step together have
// different lengths, so a contiguous per-session cache would pad every one of
// them to the longest, and [github.com/latere-ai/tgo.Model.NewBatch] refuses a
// model without a shared block pool. The batched path and the process scope are
// one configuration (022-D1).
//
// o.Slots is the batch width and is also the concurrency: pass it to
// [WithConcurrency] so the admission semaphore and the batch are the same
// number arrived at once. [New] refuses a concurrency above it, which is the
// same refusal it already makes over a pooled engine.
func WrapRunner(m *tgo.Model, name string, o tgo.RunnerOptions) (*RunnerEngine, error) {
	r, err := m.NewRunner(o)
	if err != nil {
		return nil, err
	}
	return &RunnerEngine{m: m, name: name, r: r}, nil
}

// RunnerEngine is [WrapRunner]'s implementation: [Wrap]'s engine with a batched
// runner in place of a session per request.
type RunnerEngine struct {
	modelEngine
	r *tgo.Runner
}

// Sessions is how many requests may generate at once, which for a batched
// engine is the batch width.
//
// It is named for the interface [New] checks against rather than for what it
// is, so that the refusal 019 §8.6 added -- a concurrency above what the engine
// can run at once -- applies to this engine with no change to [New].
func (e *RunnerEngine) Sessions() int { return e.r.Slots() }

// AdmissionWait and AdmissionDepth are what the engine's own queue promises,
// so a 429 raised inside it carries the same Retry-After the admitter in front
// of it would have (021-D6).
func (e *RunnerEngine) AdmissionWait() time.Duration { return e.r.Queue().Wait() }

// AdmissionDepth is how many requests may wait for a slot at once.
func (e *RunnerEngine) AdmissionDepth() int { return e.r.Queue().MaxDepth() }

// Runner is the batch behind the engine, for a caller reporting what the queue
// measured (021 §7).
func (e *RunnerEngine) Runner() *tgo.Runner { return e.r }

// Close stops the driver and releases the batch. It must be called before the
// model is closed, which is the order accel requires.
func (e *RunnerEngine) Close() error { return e.r.Close() }

// NewSession takes nothing. A slot is taken at the first generation and not
// here, because admission is a memory promise over the prompt and the prompt
// exists only once it is rendered (008 §3).
//
// So this never blocks, and the wait a request does is inside
// [github.com/latere-ai/tgo.Queue] where it is counted and bounded.
func (e *RunnerEngine) NewSession(_ context.Context, spec SessionSpec) (Session, error) {
	return &runnerSession{r: e.r, req: tgo.RunRequest{
		Tools:    spec.Tools,
		Thinking: spec.Thinking,
		// The request's cache_salt. Under a shared block pool this is the
		// isolation domain the block hashes are seeded with rather than a
		// routing key: which slot a request lands in does not change what it
		// reuses (022-D2).
		Key:      spec.Key,
		Recorder: spec.Recorder,
	}}, nil
}

// runnerSession is one request over the batch.
type runnerSession struct {
	r   *tgo.Runner
	req tgo.RunRequest
	st  *tgo.SlotStream
}

func (s *runnerSession) Chat(ctx context.Context, msgs []chat.Message, p tgo.Policy) (
	Stream, error) {

	// The nil check is not ceremony: a typed nil in an interface is not nil,
	// and a caller that checked the stream rather than the error would read a
	// method on it.
	st, err := s.r.Chat(ctx, s.req, msgs, p)
	if err != nil {
		return nil, err
	}
	s.st = st
	return st, nil
}

func (s *runnerSession) Complete(ctx context.Context, prompt string, p tgo.Policy) (
	Stream, error) {

	st, err := s.r.Complete(ctx, s.req, prompt, p)
	if err != nil {
		return nil, err
	}
	s.st = st
	return st, nil
}

// Reused is how many leading prompt positions came from the shared block pool
// rather than from a forward pass, the same quantity the other two engines
// report.
func (s *runnerSession) Reused() int {
	if s.st == nil {
		return 0
	}
	return s.st.Usage().CachedPromptTokens
}

// Close releases the slot at the next step boundary, for a handler that
// returned before the completion ended.
//
// A dispatch in flight cannot be cancelled -- accel has no cancel on a
// submitted queue and 007-D9 records that submitting against one mid-flight
// gets a failed fence rather than a race -- so the step the slot is inside
// finishes and the blocks go back to the shared pool after it (022-D5).
func (s *runnerSession) Close() error {
	if s.st == nil {
		return nil
	}
	return s.st.Close()
}
