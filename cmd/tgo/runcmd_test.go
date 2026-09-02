// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/sample"
	"github.com/latere-ai/tgo/weights"
)

func TestParseRun(t *testing.T) {
	o, err := parseRun([]string{"--prompt", "hello", "--max-tokens", "16", "--temp", "0.7",
		"--top-p", "0.9", "--top-k", "40", "--seed", "42", "--precision", "int8", "dir"})
	if err != nil {
		t.Fatalf("parseRun: %v", err)
	}
	if o.Prompt != "hello" || o.MaxTokens != 16 || o.Seed != 42 || o.Dir != "dir" {
		t.Errorf("parsed %+v", o)
	}
	if want := (sample.Policy{Temperature: 0.7, TopP: 0.9, TopK: 40, RepetitionPenalty: 1}); !reflect.DeepEqual(o.Policy, want) {
		t.Errorf("policy = %+v, want %+v", o.Policy, want)
	}
	if o.Engine.Precision.String() != "int8" || o.Engine.Context != defaultContext {
		t.Errorf("engine options = %+v", o.Engine)
	}
	// The default policy is greedy, which is the zero Policy plus the identity
	// repetition penalty: specs/006-sampling.md makes one and zero both mean no
	// penalty, and 1 is the spelling a reader recognises.
	d, err := parseRun([]string{"dir"})
	if err != nil {
		t.Fatalf("parseRun: %v", err)
	}
	if d.Policy.Temperature != 0 || d.Prompt != defaultPrompt {
		t.Errorf("defaults = %+v", d)
	}
}

func TestParseRunRefusals(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"zero max tokens", []string{"--max-tokens", "0", "d"}, "--max-tokens is 0"},
		{"more tokens than cache", []string{"--max-tokens", "8000", "d"}, "would overrun the cache"},
		{"top-p above one", []string{"--top-p", "2", "d"}, "outside [0, 1]"},
		{"top-k above the kernel", []string{"--top-k", "500", "d"}, "clamps silently"},
		{"negative temperature", []string{"--temp", "-0.5", "d"}, "negative or NaN"},
		{"a bad precision", []string{"--precision", "q8", "d"}, "is not f16, int8, int4 or auto"},
		{"zero context", []string{"--context", "0", "d"}, "--context is 0"},
		{"two directories", []string{"a", "b"}, "one model directory"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseRun(tc.args)
			if err == nil {
				t.Fatalf("parseRun(%v) was accepted", tc.args)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want one containing %q", err, tc.want)
			}
			if !errors.Is(err, errUsage) {
				t.Error("the refusal does not wrap errUsage")
			}
		})
	}
}

// TestCmdRunStreamsToStdout pins the split: tokens go to stdout and everything
// else to stderr, so that `tgo run ... > answer.txt` holds the answer.
func TestCmdRunStreamsToStdout(t *testing.T) {
	useCPUDevice(t)
	e := useFakeEngine(t, &fakeEngine{promptTokens: 6, ttft: 12 * time.Millisecond})
	var stdout, stderr strings.Builder
	if err := cmdRun([]string{"--prompt", "hi", "--max-tokens", "3", syntheticDir(t)}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdRun: %v", err)
	}
	if got := stdout.String(); got != "t0 t1 t2 \n" {
		t.Errorf("stdout = %q, want the three deltas and a closing newline", got)
	}
	// specs/001-weights.md §5: the precision choice is printed, never silent,
	// and it goes to stderr so that it is not part of the answer.
	for _, want := range []string{"f16", "which fits", "prompt tokens", "tokens/second", "greedy"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("stderr does not mention %q:\n%s", want, stderr.String())
		}
	}
	if len(e.requests) != 1 {
		t.Fatalf("the engine saw %d requests", len(e.requests))
	}
	if e.requests[0].Recorder.Enabled() {
		t.Error("`tgo run` enabled the instrument; 017-D3 keeps it off by default")
	}
	if e.requests[0].Raw {
		t.Error("`tgo run` sent the prompt raw; a chat model expects its template")
	}
	if !e.closed {
		t.Error("cmdRun did not close the engine")
	}
}

func TestCmdRunPassesRawThrough(t *testing.T) {
	useCPUDevice(t)
	e := useFakeEngine(t, &fakeEngine{promptTokens: 1})
	var stdout, stderr strings.Builder
	if err := cmdRun([]string{"--raw", "--max-tokens", "1", syntheticDir(t)}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdRun: %v", err)
	}
	if !e.requests[0].Raw {
		t.Error("--raw did not reach the engine")
	}
}

func TestCmdRunReportsAnEngineFailure(t *testing.T) {
	useCPUDevice(t)
	useFailingEngine(t, errFake)
	var stdout, stderr strings.Builder
	if err := cmdRun([]string{syntheticDir(t)}, &stdout, &stderr); !errors.Is(err, errFake) {
		t.Fatalf("cmdRun = %v, want the engine's error", err)
	}
}

func TestRenderUsage(t *testing.T) {
	var sb strings.Builder
	renderUsage(&sb, genResult{PromptTokens: 10, CompletionTokens: 20, TTFT: 30 * time.Millisecond, Stop: "eos"}, time.Second)
	out := sb.String()
	for _, want := range []string{"10 prompt tokens", "20 generated", "stopped on eos", "30.00ms", "20.00 tokens/second"} {
		if !strings.Contains(out, want) {
			t.Errorf("usage line %q is missing %q", out, want)
		}
	}
	// A run that produced nothing reports no rate rather than a division.
	sb.Reset()
	renderUsage(&sb, genResult{PromptTokens: 3, Stop: "stop-string"}, time.Second)
	if strings.Contains(sb.String(), "tokens/second") {
		t.Errorf("a run with no tokens reported a rate: %q", sb.String())
	}
}

// TestLivePrecisionMapsEveryChoice pins the third copy of the same three
// names. weights.Precision is what the flag parses into and tgo.Precision is
// what specs/007-engine.md's Open takes, and a mapping that fell through to
// auto for int8 would silently load a model at twice the size the user asked
// for and print no refusal.
func TestLivePrecisionMapsEveryChoice(t *testing.T) {
	for _, tc := range []struct {
		in   weights.Precision
		want tgo.Precision
	}{
		{weights.F16, tgo.F16},
		{weights.Int8, tgo.Int8},
		{weights.Auto, tgo.AutoPrecision},
		{weights.Inherit, tgo.AutoPrecision},
	} {
		if got := livePrecision(tc.in); got != tc.want {
			t.Errorf("livePrecision(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// TestStopReasonNamesTheBudgetAndOtherwiseSaysItCannotTell is the honest
// reading of a gap: specs/007-engine.md §1's Stream reports no stop reason, so
// the only inference available is the token budget. A completion that ended on
// the end-of-sequence token and one that ended on a stop string are the same
// observation, and the sentence says so rather than picking one.
func TestStopReasonNamesTheBudgetAndOtherwiseSaysItCannotTell(t *testing.T) {
	if got := stopReason(tgo.Usage{CompletionTokens: 8}, 8); !strings.Contains(got, "budget") {
		t.Errorf("a completion that used its whole budget stopped on %q", got)
	}
	got := stopReason(tgo.Usage{CompletionTokens: 3}, 8)
	if strings.Contains(got, "budget") {
		t.Errorf("a completion that stopped early was blamed on the budget: %q", got)
	}
	if !strings.Contains(got, "no reason") {
		t.Errorf("stop reason %q claims to know which of the two ended the stream", got)
	}
	// A run with no budget cannot have exhausted one, whatever it produced.
	if s := stopReason(tgo.Usage{CompletionTokens: 100}, 0); strings.Contains(s, "budget") {
		t.Errorf("an unbounded run stopped on %q", s)
	}
}

// TestCmdRunPrintsTheResolvedPrecisionNotThePredictedOne is
// specs/001-weights.md §5 at the one place the two can differ: this process
// predicts a precision from a device limit and the loader chooses one on the
// device it opened. The header names what ran, and says that the prediction was
// not it.
func TestCmdRunPrintsTheResolvedPrecisionNotThePredictedOne(t *testing.T) {
	useCPUDevice(t)
	useFakeEngine(t, &fakeEngine{
		promptTokens: 2,
		info:         engineInfo{Precision: "int8", WeightBytes: 4096, CacheBytesPerSession: 8192, Context: 64},
	})
	var stdout, stderr strings.Builder
	if err := cmdRun([]string{"--max-tokens", "1", syntheticDir(t)}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdRun: %v", err)
	}
	out := stderr.String()
	if !strings.Contains(out, "resolved int8") {
		t.Errorf("the header does not say the loader chose int8:\n%s", out)
	}
	if !strings.Contains(out, weights.HumanBytes(4096+8192)) {
		t.Errorf("the header does not carry the resolved footprint:\n%s", out)
	}
}

// TestCmdRunPrintsThePolicyBesideTheThroughput is 017-D4 applied to the one
// number `tgo run` prints.
//
// [renderUsage] reports tokens per second, and a tokens-per-second figure
// without the hardware, the model, the precision and the sampling policy is
// decoration: a run at temperature 0.9 and a greedy one are different products,
// and a reader who cannot tell which produced the number cannot compare it to
// anything. `tgo bench` carries the four in its conditions table; this is the
// same rule where the command line prints a rate on its own.
func TestCmdRunPrintsThePolicyBesideTheThroughput(t *testing.T) {
	useCPUDevice(t)
	useFakeEngine(t, &fakeEngine{promptTokens: 2, ttft: time.Millisecond})
	var stdout, stderr strings.Builder
	if err := cmdRun([]string{"--temp", "0.9", "--top-k", "40", "--top-p", "0.95",
		"--seed", "7", "--max-tokens", "2", syntheticDir(t)}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdRun: %v", err)
	}
	out := stderr.String()
	if !strings.Contains(out, "tokens/second") {
		t.Fatalf("the run printed no throughput, so this test proves nothing:\n%s", out)
	}
	for what, want := range map[string]string{
		"the model":       model.Qwen3Architecture,
		"the precision":   "f16",
		"the temperature": "temperature 0.9",
		"top-k":           "top-k 40",
		"top-p":           "top-p 0.95",
		"the seed":        "seed 7",
		"the budget":      "max 2 tokens",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("a tokens-per-second figure was printed without %s (%q):\n%s", what, want, out)
		}
	}
	// The hardware, read from the device rather than typed here: whatever
	// backend the test opened has to be named beside the number.
	rep, err := openAndDescribe(syntheticDir(t), describeOptions{Context: defaultContext})
	if err != nil {
		t.Fatalf("openAndDescribe: %v", err)
	}
	if !strings.Contains(out, rep.Hardware.Backend) {
		t.Errorf("the throughput was printed without the backend %q:\n%s", rep.Hardware.Backend, out)
	}
}
