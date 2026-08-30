// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"time"

	"github.com/latere-ai/tgo/sample"
)

// defaultPrompt is what `tgo run` sends when the user gives no --prompt. It is
// a question rather than a greeting so that a run with no arguments produces
// enough tokens to see the stream move.
const defaultPrompt = "In one paragraph, what is a transformer?"

// runOptions is `tgo run`'s command line, parsed.
type runOptions struct {
	Dir       string
	Prompt    string
	Raw       bool
	MaxTokens int
	Context   int
	Seed      uint64
	Policy    sample.Policy
	Engine    engineOptions
}

// runFlagSet declares what `tgo run` accepts.
//
// Declaring is separate from parsing so that [TestUsageDocumentsEveryFlag] can
// enumerate the set and hold it against the usage text. A flag that exists and
// is not in the usage is a flag nobody finds, and the drift is silent: every
// other test in this package types the flag it means to exercise.
func runFlagSet() (*flag.FlagSet, *runFlags) {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	return fs, &runFlags{
		prompt:    fs.String("prompt", defaultPrompt, "the prompt text"),
		raw:       fs.Bool("raw", false, "send the prompt as typed, without the model's chat template"),
		maxTokens: fs.Int("max-tokens", 128, "stop after N generated tokens"),
		temp:      fs.Float64("temp", 0, "sampling temperature; 0 is greedy"),
		topK:      fs.Int("top-k", 0, "keep the K largest candidates; 0 disables the stage"),
		topP:      fs.Float64("top-p", 0, "nucleus mass in (0, 1]; 0 disables the stage"),
		repeat:    fs.Float64("repeat-penalty", 1, "divisive repetition penalty; 1 is none"),
		seed:      fs.Uint64("seed", 0, "the sampler seed"),
		precision: fs.String("precision", "auto", "f16, int8, int4 or auto"),
		context:   fs.Int("context", defaultContext, "KV cache capacity in positions"),
		device:    fs.String("device", "auto", "auto, cpu or metal"),
	}
}

// runFlags holds `tgo run`'s flag values.
type runFlags struct {
	prompt             *string
	raw                *bool
	maxTokens          *int
	temp, topP, repeat *float64
	topK, context      *int
	seed               *uint64
	precision, device  *string
}

// parseRun parses and checks `tgo run`'s arguments.
//
// Parsing returns a struct and checking is part of parsing, so that every
// refusal in the flag set is reachable from a test with no device, no model and
// no engine.
func parseRun(args []string) (runOptions, error) {
	fs, f := runFlagSet()
	dir, err := modelDir(fs, args)
	if err != nil {
		return runOptions{}, err
	}
	policy, err := parsePrecision(*f.precision)
	if err != nil {
		return runOptions{}, err
	}
	dev, err := parseDevice(*f.device)
	if err != nil {
		return runOptions{}, err
	}
	if err := positive("max-tokens", *f.maxTokens); err != nil {
		return runOptions{}, err
	}
	if err := positive("context", *f.context); err != nil {
		return runOptions{}, err
	}
	if *f.maxTokens > *f.context {
		return runOptions{}, fmt.Errorf("%w: --max-tokens %d is more than the --context %d the cache holds; "+
			"the run would overrun the cache", errUsage, *f.maxTokens, *f.context)
	}
	p := sample.Policy{
		Temperature:       float32(*f.temp),
		TopK:              *f.topK,
		TopP:              float32(*f.topP),
		RepetitionPenalty: float32(*f.repeat),
	}
	if err := checkPolicy(p); err != nil {
		return runOptions{}, err
	}
	return runOptions{
		Dir: dir, Prompt: *f.prompt, Raw: *f.raw, MaxTokens: *f.maxTokens,
		Context: *f.context, Seed: *f.seed, Policy: p,
		Engine: engineOptions{Precision: policy, Context: *f.context, Device: dev},
	}, nil
}

// cmdRun generates from a prompt and streams the text to stdout.
//
// Tokens go to stdout and everything else goes to stderr, so that
// `tgo run ... > answer.txt` holds the answer and nothing else. The precision
// choice is on stderr rather than suppressed, because specs/001-weights.md §5
// requires it to be printed.
func cmdRun(args []string, stdout, stderr io.Writer) error {
	o, err := parseRun(args)
	if err != nil {
		return err
	}
	rep, err := openAndDescribe(o.Dir, describeOptions{
		Policy: o.Engine.Precision, Context: o.Context, Device: o.Engine.Device})
	if err != nil {
		return err
	}
	e, err := openEngine(o.Dir, o.Engine)
	if err != nil {
		return err
	}
	defer func() { _ = e.Close() }()

	// After the engine, not before: the precision specs/001-weights.md §5
	// requires to be printed is the one the loader resolved, and until the
	// model is open this process has only its own prediction of it.
	rep = resolvedInto(rep, e.Info())
	// The sampling policy is on this line with the model, the precision and
	// the backend because [renderUsage] below prints a tokens-per-second
	// figure. 017-D4: a throughput number without the hardware, the model, the
	// precision and the policy is decoration, and two runs compared at
	// different policies are compared at nothing. `tgo bench` carries the four
	// in its conditions table; this is the same rule on the one number
	// `tgo run` prints.
	_, _ = fmt.Fprintf(stderr, "model %s, %s, %s at %d positions of context (%s)\n",
		rep.Model.Architecture, rep.Precision.Why, humanBytes(rep.Memory.ResidentBytes),
		rep.Memory.Context, rep.Hardware.Backend)
	_, _ = fmt.Fprintf(stderr, "sampling %s\n", describePolicy(samplingOf(o.Policy, o.Seed, o.MaxTokens)))

	// Ctrl-C stops the stream and prints what was produced, rather than
	// killing the process mid-token: a user who interrupts a long generation
	// still wants the text so far.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	start := time.Now()
	res, err := e.Generate(ctx, genRequest{
		Prompt: o.Prompt, Raw: o.Raw, Policy: o.Policy, Seed: o.Seed,
		MaxTokens: o.MaxTokens,
		Emit: func(delta string) error {
			_, err := io.WriteString(stdout, delta)
			return err
		},
	})
	elapsed := time.Since(start)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprint(stdout, "\n")
	renderUsage(stderr, res, elapsed)
	return nil
}

// renderUsage reports what one generation cost, with the conditions left to the
// header line cmdRun already printed.
func renderUsage(w io.Writer, res genResult, elapsed time.Duration) {
	_, _ = fmt.Fprintf(w, "\n%d prompt tokens, %d generated, stopped on %s\n",
		res.PromptTokens, res.CompletionTokens, res.Stop)
	_, _ = fmt.Fprintf(w, "time to first token %s, %s total", humanDuration(res.TTFT), humanDuration(elapsed))
	if res.CompletionTokens > 0 && elapsed > 0 {
		_, _ = fmt.Fprintf(w, ", %.2f tokens/second", float64(res.CompletionTokens)/elapsed.Seconds())
	}
	_, _ = fmt.Fprint(w, "\n")
}
