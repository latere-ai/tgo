// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/sample"
)

// benchOptions is `tgo bench`'s command line, parsed.
type benchOptions struct {
	Dir          string
	Tokens       int
	PromptTokens int
	Batch        int
	Warmup       int
	JSON         string
	Context      int
	Seed         uint64
	Policy       sample.Policy
	Engine       engineOptions
}

// benchFlagSet declares what `tgo bench` accepts. See [runFlagSet] for why
// declaring is separate from parsing.
func benchFlagSet() (*flag.FlagSet, *benchFlags) {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	return fs, &benchFlags{
		tokens:       fs.Int("tokens", 128, "decode steps to measure"),
		promptTokens: fs.Int("prompt-tokens", 128, "synthetic prompt length in tokens"),
		batch:        fs.Int("batch", 1, "sequences in flight"),
		warmup:       fs.Int("warmup", 8, "steps to run and discard before measuring"),
		jsonPath:     fs.String("json", "", "write the machine-readable record here"),
		precision:    fs.String("precision", "auto", "f16, int8, int4 or auto"),
		context:      fs.Int("context", defaultContext, "KV cache capacity in positions"),
		temp:         fs.Float64("temp", 0, "sampling temperature; 0 is greedy"),
		seed:         fs.Uint64("seed", 0, "the sampler seed"),
		device:       fs.String("device", "auto", "auto, cpu or metal"),
	}
}

// benchFlags holds `tgo bench`'s flag values.
type benchFlags struct {
	tokens, promptTokens, batch, warmup, context *int
	jsonPath, precision, device                  *string
	temp                                         *float64
	seed                                         *uint64
}

// parseBench parses and checks `tgo bench`'s arguments.
func parseBench(args []string) (benchOptions, error) {
	fs, f := benchFlagSet()
	dir, err := modelDir(fs, args)
	if err != nil {
		return benchOptions{}, err
	}
	policy, err := parsePrecision(*f.precision)
	if err != nil {
		return benchOptions{}, err
	}
	dev, err := parseDevice(*f.device)
	if err != nil {
		return benchOptions{}, err
	}
	for name, n := range map[string]int{"tokens": *f.tokens, "prompt-tokens": *f.promptTokens, "context": *f.context} {
		if err := positive(name, n); err != nil {
			return benchOptions{}, err
		}
	}
	if err := nonNegative("warmup", *f.warmup); err != nil {
		return benchOptions{}, err
	}
	// The batch flag exists so that asking for a batch is answered rather than
	// ignored. tgo runs one sequence at a time until specs/008-scheduler.md is
	// built, and a run that silently measured batch 1 while the record said 8
	// would be the dishonest table 017-D4 is about.
	if *f.batch != 1 {
		return benchOptions{}, fmt.Errorf("%w: --batch %d, and tgo runs one sequence at a time: "+
			"specs/008-scheduler.md is drafted and unbuilt, so there is no batched path to measure. "+
			"The report states the axis and its one point rather than pretending to a curve (017-D5)",
			errUsage, *f.batch)
	}
	if *f.promptTokens+*f.tokens > *f.context {
		return benchOptions{}, fmt.Errorf("%w: --prompt-tokens %d plus --tokens %d is more than the "+
			"--context %d the cache holds", errUsage, *f.promptTokens, *f.tokens, *f.context)
	}
	p := sample.Policy{Temperature: float32(*f.temp)}
	if err := checkPolicy(p); err != nil {
		return benchOptions{}, err
	}
	return benchOptions{
		Dir: dir, Tokens: *f.tokens, PromptTokens: *f.promptTokens, Batch: *f.batch,
		Warmup: *f.warmup, JSON: *f.jsonPath, Context: *f.context, Seed: *f.seed, Policy: p,
		Engine: engineOptions{Precision: policy, Context: *f.context, Device: dev},
	}, nil
}

// syntheticWord is the token a synthetic prompt repeats.
//
// One short common word, so that a byte-level BPE encodes each repetition to
// one token and the requested length is close to the measured one. The record
// carries both, because "close" is not "equal" and the measured number is the
// one the prefill actually did.
const syntheticWord = "token"

// syntheticPrompt builds a prompt of about n tokens.
//
// specs/017-benchmarks.md §4 rule 4 says fixed-length synthetic prompts flatter
// batching and real traces do not. This is a fixed-length synthetic prompt, and
// the record says so in as many words rather than leaving a reader to assume a
// trace.
func syntheticPrompt(n int) string {
	if n < 1 {
		return ""
	}
	return strings.TrimSpace(strings.Repeat(syntheticWord+" ", n))
}

// syntheticRecipe describes the prompt in the record, so that it can be rebuilt
// without this binary.
func syntheticRecipe(n int) string {
	return fmt.Sprintf("the word %q repeated %d times, space separated, sent without the chat template",
		syntheticWord, n)
}

// cmdBench measures a model and writes both reports.
func cmdBench(args []string, stdout, stderr io.Writer) error {
	o, err := parseBench(args)
	if err != nil {
		return err
	}
	rep, err := openAndDescribe(o.Dir, describeOptions{
		Policy: o.Engine.Precision, Context: o.Context, Device: o.Engine.Device})
	if err != nil {
		return err
	}
	record, err := measure(context.Background(), o, rep, stderr)
	if err != nil {
		return err
	}
	renderMarkdown(stdout, record)
	if o.JSON == "" {
		fmt.Fprint(stderr, "\nno JSON record was written; --json out.json writes the record a "+
			"regression check reads (017-D6)\n")
		return nil
	}
	b, err := encodeRecord(record)
	if err != nil {
		return err
	}
	if err := os.WriteFile(o.JSON, b, 0o644); err != nil {
		return fmt.Errorf("writing the benchmark record: %w", err)
	}
	fmt.Fprintf(stderr, "\nwrote %s (%s, schema %s)\n", o.JSON, humanBytes(int64(len(b))), recordSchema)
	return nil
}

// measure runs the warm-up and the measured window and assembles the record.
//
// The shape is specs/017-benchmarks.md §4 rule 3: one recorder, warm up into
// it, Reset, then measure. The first steps carry the plan compilation and the
// page faults, and averaging them into the warm numbers would report a model
// that nobody experiences after the first request.
func measure(ctx context.Context, o benchOptions, rep modelReport, log io.Writer) (benchRecord, error) {
	prompt := syntheticPrompt(o.PromptTokens)
	promptFacts := promptFacts{
		Kind: "synthetic", Recipe: syntheticRecipe(o.PromptTokens),
		RequestedTokens: o.PromptTokens, Text: prompt,
	}

	// The recorder is created before the engine because the session is what
	// instruments the loop: specs/017-benchmarks.md 017-D1 makes the
	// host/submit/device/readback breakdown the deliverable, and a recorder
	// handed over after the session exists would collect nothing. Sized above
	// the window, so a recorder filling during the warm-up does not report
	// drops for the part that mattered (bench.NewRecorder).
	r := bench.NewRecorder(o.Tokens + o.Warmup + 8)
	eo := o.Engine
	eo.Recorder = r

	openStart := time.Now()
	e, err := openEngine(o.Dir, eo)
	if err != nil {
		return benchRecord{}, err
	}
	defer e.Close()
	cold := coldFacts{Open: time.Since(openStart)}

	// 017-D4 qualifies every number by the precision it was produced at, so the
	// conditions carry what the loader resolved rather than what this process
	// predicted before the model was open.
	rep = resolvedInto(rep, e.Info())

	discard := func(string) error { return nil }

	if o.Warmup > 0 {
		fmt.Fprintf(log, "warming up: %d tokens\n", o.Warmup)
		res, err := e.Generate(ctx, genRequest{
			Prompt: prompt, Raw: true, Policy: o.Policy, Seed: o.Seed,
			MaxTokens: o.Warmup, Recorder: r, Emit: discard,
		})
		if err != nil {
			return benchRecord{}, fmt.Errorf("warming up: %w", err)
		}
		cold.FirstToken = cold.Open + res.TTFT
		r.Reset()
	}

	fmt.Fprintf(log, "measuring: %d prompt tokens, %d decode steps, batch %d\n",
		o.PromptTokens, o.Tokens, o.Batch)
	start := time.Now()
	res, err := e.Generate(ctx, genRequest{
		Prompt: prompt, Raw: true, Policy: o.Policy, Seed: o.Seed,
		MaxTokens: o.Tokens, Recorder: r, Emit: discard,
	})
	if err != nil {
		return benchRecord{}, fmt.Errorf("measuring: %w", err)
	}
	wall := time.Since(start)
	if o.Warmup == 0 {
		// With no warm-up the first request is the cold one, and saying so is
		// the honest reading: 017 §3 keeps cold and warm apart, and a cold
		// number reported as warm is the one that flatters.
		cold.FirstToken = cold.Open + res.TTFT
	}
	promptFacts.MeasuredTokens = res.PromptTokens

	return newRecord(
		conditionsOf(rep, samplingOf(o.Policy, o.Seed, o.Tokens), promptFacts, o.Warmup),
		batchAxis{Points: []int{o.Batch}, Note: singleBatchNote},
		[]batchPoint{{
			Batch: o.Batch, Cold: cold, Resident: rep.Memory.ResidentBytes,
			Tokens: res.CompletionTokens, Wall: wall, Report: r.Report(),
		}},
	), nil
}
