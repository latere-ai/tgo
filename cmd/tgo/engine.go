// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"time"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/sample"
	"github.com/latere-ai/tgo/weights"
)

// engine is what the command line needs in order to generate: one call that
// turns a prompt into a stream of text and reports what it cost.
//
// It is an interface owned by this package rather than the imported type, and
// that is a deliberate seam. [liveEngine] is the one implementation and it is
// specs/007-engine.md's Model, Session and Stream; the fake in the tests is the
// other, and it is what lets the warm-up, the reset, the streaming, the
// aggregation and both reports be exercised without a checkpoint and without
// minutes of load time.
//
// The interface is narrower than 007 §1 on purpose. `tgo run` and `tgo bench`
// are one conversation each, so sessions, typed events and tool specs are
// surface the command line does not use, and naming them here would be a
// prediction rather than a requirement.
type engine interface {
	// Generate runs one request to completion, calling req.Emit with each text
	// delta as it is produced.
	Generate(ctx context.Context, req genRequest) (genResult, error)

	// Info reports what opening the model resolved every automatic choice to.
	Info() engineInfo

	// Close releases the device memory the weights hold.
	Close() error
}

// engineInfo is what the engine resolved when it opened the model.
//
// specs/001-weights.md §5 requires the precision choice to be printed, and the
// choice that matters is the one the loader made, not the one this process
// predicted: [describe] compares the f16 footprint against a device limit read
// by a device it opened and closed, and weights.Load compares it against the
// device the model holds. The two agree on every machine tested and nothing
// makes them agree, so the resolved answer is asked for and printed, and
// [resolvedInto] states a disagreement rather than picking a winner quietly.
type engineInfo struct {
	Precision            string
	WeightBytes          int64
	CacheBytesPerSession int64
	Context              int
}

// genRequest is one generation.
type genRequest struct {
	// Prompt is the user's text. Raw says whether it goes to the model as
	// typed or through the model's chat template: a chat model saw one
	// specific string format during tuning, and getting it wrong degrades
	// quality in a way no test catches (specs/003-chat-template.md).
	Prompt string
	Raw    bool

	// Policy is the sampling policy, already checked by [checkPolicy].
	Policy sample.Policy

	// Seed is the sampler's stream. It is separate from Policy because
	// specs/006-sampling.md §4 keeps the stream out of the policy: copying a
	// policy copies the numbers and not the position in the draw sequence.
	Seed uint64

	// MaxTokens stops generation after this many produced tokens.
	MaxTokens int

	// Recorder receives the per-step breakdown and the time to first token. A
	// nil Recorder disables the instrument, which is 017-D3's default:
	// `tgo run` passes nil so that the path a user runs is not the path a
	// measurement perturbs.
	//
	// [liveEngine] records only the time to first token into it. The four-way
	// breakdown is recorded by the engine's own decode loop into a
	// bench.Recorder that specs/007-engine.md §1 exports no way to set and no
	// way to read, so a process outside that package cannot obtain 017-D1's
	// breakdown at all. [measure] reports its absence rather than a table of
	// zeros; see this package's reported discrepancies.
	Recorder *bench.Recorder

	// Emit receives each text delta. Returning an error stops generation.
	Emit func(delta string) error
}

// genResult is what one generation cost.
type genResult struct {
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
	TTFT             time.Duration `json:"ttft_ns"`
	Stop             string        `json:"stop"`
}

// engineOptions is how the model is loaded.
type engineOptions struct {
	Precision weights.Precision
	Context   int

	// Device is the accelerator the model runs on. It is the same choice
	// [openDevice] uses to describe the machine, so that the limits `tgo info`
	// prints and the device the weights land on are one device.
	Device tgo.Device

	// PrefixCache turns on reuse of the key/value state a conversation has
	// already paid for (specs/016-prefix-cache.md), scoped to one session.
	//
	// Off unless asked for, because turning it on changes what an answer costs
	// and, in the last decimal places, what it says: the reused prefix was
	// computed under a different prefill shape and floating point is not
	// associative, so a warm answer equals a cold one in distribution rather
	// than bit for bit (016-D6). Only `tgo serve` reads it, because only a
	// pooled session sees a second turn (019 §1).
	PrefixCache bool

	// Recorder instruments the engine's decode loop, which is where the
	// host/submit/device/readback breakdown comes from.
	//
	// It belongs on the engine rather than on a request because the session
	// owns the loop: specs/017-benchmarks.md 017-D1 makes that breakdown the
	// deliverable, and a throughput number without it cannot say whether a
	// regression is tgo's or accel's. Nil disables the instrument (017-D3).
	Recorder *bench.Recorder
}

// livePrecision maps the loader's precision onto the engine's.
//
// Two enumerations name the same three choices: weights.Precision is what the
// loader takes and tgo.Precision is what specs/007-engine.md §1 exports, so a
// caller who parsed a flag into one has to hand the other to Open. The engine
// keeps its own copy of this table internally; this is the third, and it exists
// because the flag is parsed against the loader's vocabulary, which is the one
// specs/001-weights.md §5 names.
func livePrecision(p weights.Precision) tgo.Precision {
	switch p {
	case weights.F16:
		return tgo.F16
	case weights.Int8:
		return tgo.Int8
	default:
		return tgo.AutoPrecision
	}
}

// liveEngine is the adapter onto specs/007-engine.md's public surface.
//
// One model and one session, because both `tgo run` and `tgo bench` are one
// conversation: run sends a single request, and bench sends a warm-up and a
// measured window that must not accumulate. The session is reset at the top of
// every Generate for exactly that reason -- a Session carries its position in
// its own cache, so a second request without a reset would prefill on top of
// the first and the prompt length in the record would be the conversation's
// rather than the prompt's.
type liveEngine struct {
	m *tgo.Model
	s *tgo.Session
}

// Generate runs one request and streams its text.
func (e *liveEngine) Generate(ctx context.Context, req genRequest) (genResult, error) {
	e.s.Reset()
	p := tgo.Policy{
		Temperature:       req.Policy.Temperature,
		TopK:              req.Policy.TopK,
		TopP:              req.Policy.TopP,
		RepetitionPenalty: req.Policy.RepetitionPenalty,
		PresencePenalty:   req.Policy.PresencePenalty,
		FrequencyPenalty:  req.Policy.FrequencyPenalty,
		PenaltyWindow:     req.Policy.PenaltyWindow,
		LogitBias:         req.Policy.LogitBias,
		Seed:              req.Seed,
		MaxTokens:         req.MaxTokens,
	}

	start := time.Now()
	st, err := e.stream(ctx, req, p)
	if err != nil {
		return genResult{}, err
	}

	var ttft time.Duration
	for st.Next() {
		if ttft == 0 {
			// Measured around the first token the caller could see, which is
			// what a user waits through: the prefill, the first decode step
			// and the detokenizer's hold-back together.
			ttft = time.Since(start)
			req.Recorder.TTFT(ttft)
		}
		if text := st.Text(); text != "" {
			if err := req.Emit(text); err != nil {
				return genResult{}, err
			}
		}
	}
	if err := st.Err(); err != nil {
		return genResult{}, err
	}
	u := st.Usage()
	return genResult{
		PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens,
		TTFT: ttft, Stop: stopReason(u, req.MaxTokens),
	}, nil
}

// stream starts the request, through the model's chat template unless the
// caller asked for the prompt as typed.
func (e *liveEngine) stream(ctx context.Context, req genRequest, p tgo.Policy) (*tgo.Stream, error) {
	if req.Raw {
		return e.s.Complete(ctx, req.Prompt, p)
	}
	return e.s.Chat(ctx, []chat.Message{{
		Role:   chat.User,
		Blocks: []chat.Block{{Type: chat.BlockText, Text: req.Prompt}},
	}}, p)
}

// Info reports what tgo.Open resolved.
func (e *liveEngine) Info() engineInfo {
	i := e.m.Info()
	return engineInfo{
		Precision: i.Precision.String(), WeightBytes: i.WeightBytes,
		CacheBytesPerSession: i.CacheBytesPerSession, Context: i.Context,
	}
}

// Close releases the session and the weights.
func (e *liveEngine) Close() error {
	err := e.s.Close()
	if cerr := e.m.Close(); err == nil {
		err = cerr
	}
	return err
}

// stopReason infers why generation ended.
//
// Inferred, because specs/007-engine.md §1's Stream reports no stop reason: a
// caller can see the tokens and the budget and nothing else, so a completion
// that ended on a stop string and one that ended on the end-of-sequence token
// are the same observation. See this package's reported discrepancies.
func stopReason(u tgo.Usage, maxTokens int) string {
	if maxTokens > 0 && u.CompletionTokens >= maxTokens {
		return "the token budget"
	}
	return "the model (end of sequence or a stop string; the engine reports no reason)"
}

// openEngine loads a model and prepares it to generate.
//
// It is a variable so that the tests can replace it with a fake, which is what
// lets every other line of `run` and `bench` be exercised without a checkpoint
// and without minutes of load time.
var openEngine = func(dir string, o engineOptions) (engine, error) {
	m, err := tgo.Open(dir,
		tgo.WithPrecision(livePrecision(o.Precision)),
		tgo.WithContext(o.Context),
		tgo.WithDevice(o.Device))
	if err != nil {
		return nil, err
	}
	var sopts []tgo.SessionOption
	if o.Recorder != nil {
		sopts = append(sopts, tgo.WithRecorder(o.Recorder))
	}
	s, err := m.NewSession(sopts...)
	if err != nil {
		m.Close()
		return nil, err
	}
	return &liveEngine{m: m, s: s}, nil
}
