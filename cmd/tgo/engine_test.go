// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"os"
	"strings"
	"testing"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/weights"
)

// TestDescribeAgreesWithTheLoadedModel is the pin under `tgo info`.
//
// `tgo info` reports the precision and the memory from config.json and the
// declared weight map alone, without loading a byte, because the alternative is
// uploading 1.4 GiB to the device in order to print a table. That makes two
// implementations of the same arithmetic: [describe] here and
// specs/004-model-graph.md's own inside a loaded [tgo.Model]. Nothing makes
// them agree, so this compares them on a real checkpoint.
//
// It reads the model TGO_MODEL names and is skipped otherwise
// (specs/000-decisions.md decision 8): the smallest Qwen3 is over a gigabyte,
// and a test suite that downloads one is a suite nobody runs.
func TestDescribeAgreesWithTheLoadedModel(t *testing.T) {
	dir := os.Getenv("TGO_MODEL")
	if dir == "" {
		t.Skip("TGO_MODEL is not set; this test loads a real checkpoint")
	}
	const context = 1024

	predicted, err := openAndDescribe(dir, describeOptions{Policy: weights.F16, Context: context})
	if err != nil {
		t.Fatalf("openAndDescribe: %v", err)
	}

	m, err := tgo.Open(dir, tgo.WithPrecision(tgo.F16), tgo.WithContext(context))
	if err != nil {
		t.Fatalf("tgo.Open: %v", err)
	}
	defer func() { _ = m.Close() }()
	got := m.Info()

	if got.Architecture != predicted.Model.Architecture {
		t.Errorf("architecture: the model says %q and info printed %q",
			got.Architecture, predicted.Model.Architecture)
	}
	for _, tc := range []struct {
		name       string
		got, print int
	}{
		{"layers", got.Layers, predicted.Model.Layers},
		{"hidden", got.HiddenSize, predicted.Model.HiddenSize},
		{"heads", got.Heads, predicted.Model.Heads},
		{"kv heads", got.KVHeads, predicted.Model.KVHeads},
		{"head_dim", got.HeadDim, predicted.Model.HeadDim},
		{"mlp", got.IntermediateSize, predicted.Model.IntermediateSize},
		{"vocabulary", got.VocabSize, predicted.Model.VocabSize},
		{"trained context", got.TrainedContext, predicted.Model.TrainedContext},
	} {
		if tc.got != tc.print {
			t.Errorf("%s: the model says %d and info printed %d", tc.name, tc.got, tc.print)
		}
	}
	// The footprint `tgo info` prints without loading, against what the loaded
	// weights occupy. Equal, not close: both count the same declared planes at
	// the same width, and a difference means one of them counts a tensor the
	// other does not -- which for a tied checkpoint is the largest tensor in
	// the model (004-D7).
	if got.WeightBytes != predicted.Memory.WeightBytes {
		t.Errorf("weights: the model holds %s and info printed %s",
			humanBytes(got.WeightBytes), humanBytes(predicted.Memory.WeightBytes))
	}
	// specs/005-kv-cache.md §3's M_kv, computed twice.
	if got.CacheBytesPerSession != predicted.Memory.KVBytes {
		t.Errorf("kv cache at %d positions: the model reserves %s and info printed %s; "+
			"one of the two implementations of 005 §3 is wrong, or the cache width moved",
			context, humanBytes(got.CacheBytesPerSession), humanBytes(predicted.Memory.KVBytes))
	}
	if got.Precision.String() != predicted.Precision.Chosen {
		t.Errorf("precision: the model resolved %s and info printed %s",
			got.Precision, predicted.Precision.Chosen)
	}
}

// TestLiveEngineOpensTheRealCheckpoint drives [liveEngine] against a model
// this repository did not write, as far as this hardware can drive it.
//
// It opens the model through [openEngine] -- the one line of this package that
// calls specs/007-engine.md's Open -- reads what the engine resolved, folds it
// into the description, and closes it. That covers the load, the resolved
// precision and the memory the engine actually reserved.
//
// It generates nothing, and that is a finding rather than a shortcut. Neither
// backend on this machine can run Qwen3-0.6B's forward pass in usable time:
//
//   - Metal, the automatic choice on Apple Silicon, refuses to compile the
//     prefill graph at all -- "accel: node 589: kernel Pack carries no MSL
//     artifact, so it cannot run on Metal".
//   - The CPU backend calls itself "pure Go, developer" and means it. A single
//     32-row prefill plus one decode step, at --context 32 and --max-tokens 1,
//     was still running after thirty-five minutes and was abandoned rather than
//     finished.
//
// So `tgo run` and `tgo bench` are exercised end to end against a fake engine
// (see bench_test.go and runcmd_test.go), and against a real checkpoint only up
// to the point this machine can reach. Writing a generation test that has never
// passed and calling it proof of the wiring would be the decoration 017-D4
// refuses, one layer down. See this package's reported discrepancies.
func TestLiveEngineOpensTheRealCheckpoint(t *testing.T) {
	dir := os.Getenv("TGO_MODEL")
	if dir == "" {
		t.Skip("TGO_MODEL is not set; this test loads a real checkpoint")
	}
	const context = 256

	predicted, err := openAndDescribe(dir, describeOptions{
		Policy: weights.F16, Context: context, Device: tgo.CPU})
	if err != nil {
		t.Fatalf("openAndDescribe: %v", err)
	}

	e, err := openEngine(dir, engineOptions{
		Precision: weights.F16, Context: context, Device: tgo.CPU})
	if err != nil {
		t.Fatalf("openEngine: %v", err)
	}
	defer func() {
		if err := e.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()

	got := e.Info()
	if got.Precision != "f16" {
		t.Errorf("the engine resolved %q, and --precision f16 was asked for", got.Precision)
	}
	if got.Context != context {
		t.Errorf("the engine reserved %d positions, want %d", got.Context, context)
	}
	// The footprint `tgo info` prints without loading a byte, against what the
	// loaded weights occupy. Equal, not close: both count the same declared
	// planes at the same widths, and a difference means one of them prices a
	// tensor the other does not -- which for a tied checkpoint is the largest
	// tensor in the model (004-D7), and for a norm gain is the f32 store
	// specs/004-model-graph.md §3 declares.
	if got.WeightBytes != predicted.Memory.WeightBytes {
		t.Errorf("weights: the engine holds %s and info printed %s",
			humanBytes(got.WeightBytes), humanBytes(predicted.Memory.WeightBytes))
	}
	// specs/005-kv-cache.md §3's M_kv, computed twice.
	if got.CacheBytesPerSession != predicted.Memory.KVBytes {
		t.Errorf("kv cache at %d positions: the engine reserves %s and info printed %s; "+
			"one of the two implementations of 005 §3 is wrong, or the cache width moved",
			context, humanBytes(got.CacheBytesPerSession), humanBytes(predicted.Memory.KVBytes))
	}
	// And the fold that puts the resolved numbers in front of a reader agrees
	// with the prediction, so no disagreement is announced that did not happen.
	folded := resolvedInto(predicted, got)
	if strings.Contains(folded.Precision.Why, "resolved") {
		t.Errorf("an agreeing load was reported as a disagreement: %q", folded.Precision.Why)
	}
	if folded.Memory.ResidentBytes != got.WeightBytes+got.CacheBytesPerSession {
		t.Errorf("resident = %d, want the engine's weights plus its cache", folded.Memory.ResidentBytes)
	}
	t.Logf("resolved %s, weights %s, cache %s at %d positions, resident %s",
		got.Precision, humanBytes(got.WeightBytes), humanBytes(got.CacheBytesPerSession),
		context, humanBytes(folded.Memory.ResidentBytes))
}

// TestOpenEngineRefusesADirectoryThatIsNotAModel walks the live path far enough
// to prove it is wired without a checkpoint: openEngine is the only line of
// this package that calls specs/007-engine.md's Open, and a refusal is the one
// outcome reachable with no model on disk.
func TestOpenEngineRefusesADirectoryThatIsNotAModel(t *testing.T) {
	if _, err := openEngine(t.TempDir(), engineOptions{Context: defaultContext}); err == nil {
		t.Fatal("a directory with no config.json opened as a model")
	}
}
