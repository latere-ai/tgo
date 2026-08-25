// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseBench(t *testing.T) {
	o, err := parseBench([]string{"--tokens", "16", "--prompt-tokens", "32", "--json", "out.json", "dir"})
	if err != nil {
		t.Fatalf("parseBench: %v", err)
	}
	if o.Tokens != 16 || o.PromptTokens != 32 || o.JSON != "out.json" || o.Dir != "dir" {
		t.Errorf("parsed %+v", o)
	}
	if o.Batch != 1 || o.Warmup != 8 || o.Context != defaultContext {
		t.Errorf("defaults are wrong: %+v", o)
	}
}

// TestParseBenchRefusals walks the refusals. The batch one is the case that
// matters: a run that silently measured batch 1 while the user asked for 8 is the
// dishonest table 017-D4 exists to prevent.
func TestParseBenchRefusals(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"a batch tgo cannot run", []string{"--batch", "8", "d"}, "008-scheduler.md is drafted and unbuilt"},
		{"zero tokens", []string{"--tokens", "0", "d"}, "--tokens is 0"},
		{"a negative warm-up", []string{"--warmup", "-1", "d"}, "--warmup is -1"},
		{"a prompt longer than the cache", []string{"--prompt-tokens", "4000", "--tokens", "200", "d"},
			"more than the --context"},
		{"an unknown precision", []string{"--precision", "q4", "d"}, "is not f16, int8 or auto"},
		{"a negative temperature", []string{"--temp", "-1", "d"}, "temperature -1 is negative"},
		{"no directory", nil, "no model directory"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseBench(tc.args)
			if err == nil {
				t.Fatalf("parseBench(%v) was accepted", tc.args)
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

func TestSyntheticPrompt(t *testing.T) {
	if got := syntheticPrompt(0); got != "" {
		t.Errorf("syntheticPrompt(0) = %q", got)
	}
	if got := syntheticPrompt(3); got != "token token token" {
		t.Errorf("syntheticPrompt(3) = %q", got)
	}
	if !strings.Contains(syntheticRecipe(3), "repeated 3 times") {
		t.Errorf("the recipe does not say how to rebuild the prompt: %q", syntheticRecipe(3))
	}
}

// TestMeasureWarmsUpThenMeasures is specs/017-benchmarks.md §4 rule 3: the
// first steps carry the plan compilation and the page faults, and they belong
// to the cold number rather than to the warm one.
func TestMeasureWarmsUpThenMeasures(t *testing.T) {
	e := useFakeEngine(t, &fakeEngine{promptTokens: 40, ttft: 25 * time.Millisecond})
	o, err := parseBench([]string{"--tokens", "20", "--prompt-tokens", "8", "--warmup", "5", "d"})
	if err != nil {
		t.Fatalf("parseBench: %v", err)
	}
	rep := describeSynthetic(t)
	rec, err := measure(context.Background(), o, rep, io.Discard)
	if err != nil {
		t.Fatalf("measure: %v", err)
	}
	if len(e.requests) != 2 {
		t.Fatalf("the engine saw %d requests, want a warm-up and a measured window", len(e.requests))
	}
	if e.requests[0].MaxTokens != 5 || e.requests[1].MaxTokens != 20 {
		t.Errorf("requests asked for %d then %d tokens, want 5 then 20",
			e.requests[0].MaxTokens, e.requests[1].MaxTokens)
	}
	// The warm-up steps were reset away: 20 decode steps, not 25.
	if got := rec.Batches[0].Report.Decode.Steps; got != 20 {
		t.Errorf("the report holds %d decode steps, want the 20 measured ones", got)
	}
	if got := rec.Batches[0].Report.Dropped; got != 0 {
		t.Errorf("%d observations were dropped; the recorder is sized above the window", got)
	}
	// And the warm-up's time to first token was reset away with its steps. A
	// cold sample left in the warm column is 017 §4 rule 3 broken in the one
	// place a reader would never see it: the percentiles would still look like
	// percentiles.
	if got := rec.Batches[0].Report.TTFT.N; got != 1 {
		t.Errorf("the warm report holds %d times to first token, want only the measured window's", got)
	}
	// The throughput is the wall clock's, and it is divided rather than left
	// at the zero a struct literal would carry.
	if rec.Batches[0].TokensPerSecond <= 0 {
		t.Errorf("tokens/second = %v over %v of wall clock",
			rec.Batches[0].TokensPerSecond, rec.Batches[0].Wall)
	}
	// Cold is the open plus the first token, and it is not the warm p50.
	cold := rec.Batches[0].Cold
	if cold.FirstToken < 25*time.Millisecond {
		t.Errorf("cold first token = %v, want at least the engine's TTFT", cold.FirstToken)
	}
	if cold.FirstToken <= cold.Open {
		t.Error("the cold first token does not include the open")
	}
	if rec.Conditions.Prompt.MeasuredTokens != 40 {
		t.Errorf("the record says %d prompt tokens, want the 40 the engine measured",
			rec.Conditions.Prompt.MeasuredTokens)
	}
	if rec.Conditions.WarmupSteps != 5 {
		t.Errorf("warm-up steps = %d, want 5", rec.Conditions.WarmupSteps)
	}
	if !rec.Conditions.Sampling.Greedy {
		t.Error("the default policy is greedy and the record does not say so")
	}
	if !e.closed {
		t.Error("measure did not close the engine")
	}
}

// TestMeasureWithNoWarmupSaysTheFirstTokenIsCold pins the other arm: with no
// warm-up the measured request is the cold one, and reporting its first token
// as warm would be the number that flatters.
func TestMeasureWithNoWarmupSaysTheFirstTokenIsCold(t *testing.T) {
	useFakeEngine(t, &fakeEngine{promptTokens: 4, ttft: 11 * time.Millisecond})
	o, err := parseBench([]string{"--tokens", "4", "--prompt-tokens", "4", "--warmup", "0", "d"})
	if err != nil {
		t.Fatalf("parseBench: %v", err)
	}
	rec, err := measure(context.Background(), o, describeSynthetic(t), io.Discard)
	if err != nil {
		t.Fatalf("measure: %v", err)
	}
	if rec.Batches[0].Cold.FirstToken < 11*time.Millisecond {
		t.Errorf("cold first token = %v, want the measured request's", rec.Batches[0].Cold.FirstToken)
	}
	if rec.Conditions.WarmupSteps != 0 {
		t.Error("the record does not say that nothing was warmed up")
	}
}

func TestMeasureReportsAnEngineFailure(t *testing.T) {
	useFakeEngine(t, &fakeEngine{fail: errFake})
	o, err := parseBench([]string{"--tokens", "2", "--prompt-tokens", "2", "d"})
	if err != nil {
		t.Fatalf("parseBench: %v", err)
	}
	_, err = measure(context.Background(), o, describeSynthetic(t), io.Discard)
	if !errors.Is(err, errFake) {
		t.Fatalf("measure error = %v, want the engine's", err)
	}
	if !strings.Contains(err.Error(), "warming up") {
		t.Errorf("error = %v, want it to name the phase that failed", err)
	}
}

// TestCmdBenchWritesBothReports is the deliverable: a Markdown table for a
// person and a JSON record for a regression check (017 §5).
func TestCmdBenchWritesBothReports(t *testing.T) {
	useCPUDevice(t)
	useFakeEngine(t, &fakeEngine{promptTokens: 8, ttft: time.Millisecond})
	dir := syntheticDir(t)
	out := filepath.Join(t.TempDir(), "bench.json")

	var stdout, stderr strings.Builder
	err := cmdBench([]string{"--tokens", "8", "--prompt-tokens", "8", "--warmup", "2",
		"--json", out, dir}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("cmdBench: %v", err)
	}
	if !strings.Contains(stdout.String(), "## Decode") {
		t.Errorf("stdout is not the Markdown table:\n%s", stdout.String())
	}
	if !strings.Contains(stderr.String(), out) {
		t.Errorf("stderr does not name the record it wrote:\n%s", stderr.String())
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read the record: %v", err)
	}
	var rec benchRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		t.Fatalf("the record does not decode: %v", err)
	}
	if rec.Schema != recordSchema || len(rec.Batches) != 1 || rec.Batches[0].Report.Decode.Steps != 8 {
		t.Errorf("the record is not the run: %+v", rec.Batches)
	}
	if rec.Conditions.Hardware.Backend == "" || rec.Conditions.Environment.Go == "" {
		t.Error("the record has no hardware or build stamp (017-D4)")
	}
}

// TestCmdBenchSaysWhenItWroteNoRecord: 017-D6 makes the JSON the artefact that
// gates a regression, so a run that produced none has to say the flag that
// would have.
func TestCmdBenchSaysWhenItWroteNoRecord(t *testing.T) {
	useCPUDevice(t)
	useFakeEngine(t, &fakeEngine{promptTokens: 2})
	var stdout, stderr strings.Builder
	if err := cmdBench([]string{"--tokens", "2", "--prompt-tokens", "2", "--warmup", "0",
		syntheticDir(t)}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdBench: %v", err)
	}
	if !strings.Contains(stderr.String(), "--json out.json") {
		t.Errorf("stderr does not name the flag that writes the record:\n%s", stderr.String())
	}
}

func TestCmdBenchRefusesAnUnwritableRecordPath(t *testing.T) {
	useCPUDevice(t)
	useFakeEngine(t, &fakeEngine{promptTokens: 2})
	var stdout, stderr strings.Builder
	err := cmdBench([]string{"--tokens", "2", "--prompt-tokens", "2", "--warmup", "0",
		"--json", filepath.Join(t.TempDir(), "no-such-dir", "bench.json"), syntheticDir(t)}, &stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "writing the benchmark record") {
		t.Fatalf("cmdBench = %v, want a refusal naming the write", err)
	}
}

func TestCmdBenchRefusesAnUnreadableModelDirectory(t *testing.T) {
	useCPUDevice(t)
	var stdout, stderr strings.Builder
	err := cmdBench([]string{filepath.Join(t.TempDir(), "not-a-model")}, &stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "read config") {
		t.Fatalf("cmdBench on a directory with no config = %v, want a refusal", err)
	}
}

// describeSynthetic is the model description the measurement tests run against.
func describeSynthetic(t *testing.T) modelReport {
	t.Helper()
	rep, err := describe("models/synthetic", syntheticBuilder(t), describeOptions{Context: defaultContext},
		hardware{Backend: "cpu", MaxPoolBytes: 1 << 30}, stampEnvironment())
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	return rep
}

// TestCmdBenchOnAnEngineWithNoBreakdown is what `tgo bench` does against the
// engine that exists today, end to end.
//
// specs/007-engine.md §1 exports no way to set or read its bench.Recorder, so
// the four terms 017-D1 calls the deliverable do not reach this process. The
// command still writes both reports, and both of them say the breakdown is
// missing and why, rather than printing a table of zeros that reads as a
// measurement.
func TestCmdBenchOnAnEngineWithNoBreakdown(t *testing.T) {
	useCPUDevice(t)
	useFakeEngine(t, &fakeEngine{promptTokens: 4, ttft: 5 * time.Millisecond, noBreakdown: true})
	out := filepath.Join(t.TempDir(), "bench.json")

	var stdout, stderr strings.Builder
	if err := cmdBench([]string{"--tokens", "4", "--prompt-tokens", "4", "--warmup", "1",
		"--json", out, syntheticDir(t)}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdBench: %v", err)
	}
	if strings.Contains(stdout.String(), "## Decode") {
		t.Errorf("a run with no breakdown printed the decode breakdown:\n%s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "## Where the time went") {
		t.Errorf("the table does not say where the breakdown went:\n%s", stdout.String())
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read the record: %v", err)
	}
	var rec benchRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		t.Fatalf("the record does not decode: %v", err)
	}
	if rec.Breakdown.Available {
		t.Error("the record claims a breakdown it does not carry")
	}
	if !strings.Contains(rec.Breakdown.Note, "007-engine.md") {
		t.Errorf("the record does not name the gap: %q", rec.Breakdown.Note)
	}
	// The throughput and the time to first token are still measured, and the
	// conditions still qualify them (017-D4).
	if rec.Batches[0].TokensPerSecond <= 0 || rec.Batches[0].Report.TTFT.N != 1 {
		t.Errorf("the measurements that do exist are missing: %+v", rec.Batches[0])
	}
	if rec.Conditions.Precision.Chosen == "" || rec.Conditions.Hardware.Backend == "" {
		t.Error("a record with no breakdown also dropped its conditions")
	}
}

// TestMeasureRecordsTheResolvedPrecision is 017-D4 at the seam a prediction can
// break: every number in the record is qualified by the precision it was
// produced at, and that is the loader's answer rather than this process's.
func TestMeasureRecordsTheResolvedPrecision(t *testing.T) {
	useFakeEngine(t, &fakeEngine{
		promptTokens: 4,
		info:         engineInfo{Precision: "int8", WeightBytes: 2048, CacheBytesPerSession: 1024, Context: 16},
	})
	o, err := parseBench([]string{"--tokens", "2", "--prompt-tokens", "2", "--warmup", "0", "d"})
	if err != nil {
		t.Fatalf("parseBench: %v", err)
	}
	rec, err := measure(context.Background(), o, describeSynthetic(t), io.Discard)
	if err != nil {
		t.Fatalf("measure: %v", err)
	}
	if rec.Conditions.Precision.Chosen != "int8" {
		t.Errorf("the record says %s and the loader used int8", rec.Conditions.Precision.Chosen)
	}
	if rec.Conditions.Memory.ResidentBytes != 2048+1024 {
		t.Errorf("resident = %d, want the resolved weights plus the resolved cache",
			rec.Conditions.Memory.ResidentBytes)
	}
	if rec.Batches[0].Resident != rec.Conditions.Memory.ResidentBytes {
		t.Error("the point's resident footprint disagrees with the conditions'")
	}
}

// TestCmdBenchReportsAMeasurementFailure: a run that could not measure writes
// no record, rather than a record of a run that did not happen.
func TestCmdBenchReportsAMeasurementFailure(t *testing.T) {
	useCPUDevice(t)
	useFakeEngine(t, &fakeEngine{fail: errFake})
	out := filepath.Join(t.TempDir(), "bench.json")
	var stdout, stderr strings.Builder
	err := cmdBench([]string{"--tokens", "1", "--prompt-tokens", "1", "--warmup", "0",
		"--json", out, syntheticDir(t)}, &stdout, &stderr)
	if !errors.Is(err, errFake) {
		t.Fatalf("cmdBench = %v, want the engine's error", err)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Error("a failed measurement still wrote a record")
	}
	if stdout.Len() != 0 {
		t.Errorf("a failed measurement wrote a table: %q", stdout.String())
	}
}
