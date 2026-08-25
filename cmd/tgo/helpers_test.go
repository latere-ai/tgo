// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.design/x/accel"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/model"
)

// syntheticConfig is specs/004-model-graph.md §8's two-layer model, whose
// dimensions collide nowhere: d=80, L=2, H=8, H_kv=4, d_h=48, f=176, V=112.
// A report that confused two extents does not survive it.
//
// H_kv is 4 and not 2 on purpose. §8 fixes only the layer count, and a fixture
// where L and H_kv are both 2 is the identity for every confusion between them:
// a renderer that printed the key/value head count where the layer count
// belongs would read correctly on it. The seven extents here are pairwise
// distinct, which is what makes that class of mistake visible.
func syntheticConfig() map[string]any {
	return map[string]any{
		"architectures":           []string{model.Qwen3Architecture},
		"hidden_size":             80,
		"num_hidden_layers":       2,
		"num_attention_heads":     8,
		"num_key_value_heads":     4,
		"head_dim":                48,
		"intermediate_size":       176,
		"vocab_size":              112,
		"rms_norm_eps":            1e-06,
		"rope_theta":              1000000,
		"tie_word_embeddings":     true,
		"max_position_embeddings": 4096,
	}
}

// syntheticBuilder is the model every arithmetic test below runs against. It
// reads no file: specs/000-decisions.md decision 8 keeps the real checkpoint
// out of the default run, and arithmetic that could only be checked against a
// 1.4 GiB download is arithmetic nobody checks.
func syntheticBuilder(t *testing.T) model.Builder {
	t.Helper()
	raw, err := json.Marshal(syntheticConfig())
	if err != nil {
		t.Fatalf("marshal the synthetic config: %v", err)
	}
	b, err := model.New(raw)
	if err != nil {
		t.Fatalf("model.New: %v", err)
	}
	return b
}

// syntheticDir writes a model directory holding only the config, which is all
// model.Open reads.
func syntheticDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	raw, err := json.Marshal(syntheticConfig())
	if err != nil {
		t.Fatalf("marshal the synthetic config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	return dir
}

// useCPUDevice points openDevice at the CPU backend for the duration of a test,
// and records what each command asked for.
//
// The CPU backend rather than a stub: hardware facts are read from the device
// and a stub would let the reading drift from what accel reports, which is the
// field of a benchmark report that must not be typed by hand (017-D4).
func useCPUDevice(t *testing.T) *[]tgo.Device {
	t.Helper()
	var asked []tgo.Device
	prev := openDevice
	openDevice = func(want tgo.Device) (*accel.Device, error) {
		asked = append(asked, want)
		return accel.OpenCPU(accel.CPUOptions{})
	}
	t.Cleanup(func() { openDevice = prev })
	return &asked
}

// fakeEngine stands in for specs/007-engine.md, which is unbuilt.
//
// It produces text and fills a bench.Recorder the way an engine would, so that
// everything the command line does around generation is exercised: the
// streaming, the warm-up and reset, the aggregation, and both reports.
type fakeEngine struct {
	promptTokens int
	ttft         time.Duration
	fail         error

	// noBreakdown makes the fake behave the way the live engine does today:
	// it records a time to first token and nothing else, because
	// specs/007-engine.md §1 exports no way to reach the four-term breakdown.
	// It is the arm a real run takes, so it has to be an arm a test takes.
	noBreakdown bool

	// info is what the fake resolved when it opened. The zero value leaves the
	// prediction in place, which is the agreeing case.
	info engineInfo

	closed   bool
	requests []genRequest
}

// Info reports what the fake resolved.
func (f *fakeEngine) Info() engineInfo { return f.info }

// clockFloor is how long a fake generation takes on the wall clock.
//
// The fake reports plausible per-step durations to the Recorder and returns
// immediately, so the wall clock *around* it is however long nothing takes.
// That is zero on Windows, whose timer granularity is about 15ms, and a
// tokens-per-second computed from a zero interval is zero -- which failed five
// tests there while passing everywhere else. The durations the fake reports are
// fiction either way; what has to be real is that measurable time passes, so
// the code that divides by it is exercised rather than short-circuited.
//
// 20ms rather than 1ms: comfortably above a 15.6ms tick, so the interval is a
// couple of ticks rather than a rounding of one.
const clockFloor = 20 * time.Millisecond

// Generate emits one word per token and records one prefill step and one decode
// step each, with durations that vary per step so that p50, p90 and p99 differ
// and a report that collapsed them would be visible.
func (f *fakeEngine) Generate(ctx context.Context, req genRequest) (genResult, error) {
	f.requests = append(f.requests, req)
	// See clockFloor: the caller measures the wall clock around this call.
	defer time.Sleep(clockFloor)
	if f.fail != nil {
		return genResult{}, f.fail
	}
	if err := ctx.Err(); err != nil {
		return genResult{}, err
	}
	req.Recorder.TTFT(f.ttft)
	if f.noBreakdown {
		for i := range req.MaxTokens {
			if err := req.Emit(fmt.Sprintf("t%d ", i)); err != nil {
				return genResult{}, err
			}
		}
		return genResult{
			PromptTokens: f.promptTokens, CompletionTokens: req.MaxTokens,
			TTFT: f.ttft, Stop: "max-tokens",
		}, nil
	}
	req.Recorder.Step(bench.Step{
		Phase: bench.Prefill, Tokens: f.promptTokens, Batch: 1,
		Host: 2 * time.Millisecond, Submit: 3 * time.Millisecond,
		Device: 20 * time.Millisecond, Readback: 5 * time.Millisecond,
	})
	for i := range req.MaxTokens {
		if err := req.Emit(fmt.Sprintf("t%d ", i)); err != nil {
			return genResult{}, err
		}
		n := time.Duration(i + 1)
		req.Recorder.Step(bench.Step{
			Phase: bench.Decode, Tokens: 1, Batch: 1,
			Host: n * time.Microsecond, Submit: 2 * n * time.Microsecond,
			Device: 10 * n * time.Microsecond, Readback: 3 * n * time.Microsecond,
		})
	}
	return genResult{
		PromptTokens: f.promptTokens, CompletionTokens: req.MaxTokens,
		TTFT: f.ttft, Stop: "max-tokens",
	}, nil
}

func (f *fakeEngine) Close() error { f.closed = true; return nil }

// useFakeEngine installs a fake engine for the duration of a test and returns
// it, so that the test can assert what the command asked of it.
func useFakeEngine(t *testing.T, e *fakeEngine) *fakeEngine {
	t.Helper()
	prev := openEngine
	openEngine = func(dir string, o engineOptions) (engine, error) { return e, nil }
	t.Cleanup(func() { openEngine = prev })
	return e
}

// useFailingEngine installs an engine that refuses to open.
func useFailingEngine(t *testing.T, err error) {
	t.Helper()
	prev := openEngine
	openEngine = func(dir string, o engineOptions) (engine, error) { return nil, err }
	t.Cleanup(func() { openEngine = prev })
}

// errFake is what a fake engine fails with.
var errFake = errors.New("the fake engine was told to fail")
