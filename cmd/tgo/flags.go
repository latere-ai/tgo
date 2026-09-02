// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"strings"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/sample"
	"github.com/latere-ai/tgo/weights"
)

// modelDir parses fs over args and returns the one positional argument the
// commands that take a model take: the model directory.
func modelDir(fs *flag.FlagSet, args []string) (string, error) {
	return onePositional(fs, args, "model directory")
}

// onePositional parses fs over args and returns the one positional argument the
// command takes, named by what so that every refusal below says which thing was
// missing, duplicated or blank.
//
// The flag package writes its own diagnostics and calls os.Exit on an unknown
// flag unless it is told otherwise, so every flag set here is created with
// [flag.ContinueOnError] and a discarded output: a command line is refused by
// returning an error, which is the form a test can assert.
func onePositional(fs *flag.FlagSet, args []string, what string) (string, error) {
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		return "", fmt.Errorf("%w: %w", errUsage, err)
	}
	switch rest := fs.Args(); len(rest) {
	case 1:
		if strings.TrimSpace(rest[0]) == "" {
			return "", fmt.Errorf("%w: the %s is empty", errUsage, what)
		}
		return rest[0], nil
	case 0:
		return "", fmt.Errorf("%w: no %s", errUsage, what)
	default:
		// The hint names the cause of the common case rather than the shape
		// of the failure. [flag.FlagSet.Parse] stops at the first argument
		// that is not a flag, so `tgo info <dir> --context 512` reaches here
		// with three positional arguments and a user reading only "one model
		// directory" has no way to see that their flag was never parsed.
		return "", fmt.Errorf("%w: one %s, and %d were given (%s); "+
			"flags go before it, since parsing stops at the first argument that is not a flag",
			errUsage, what, len(rest), strings.Join(rest, " "))
	}
}

// parsePrecision turns the --precision flag into the loader's policy.
//
// The three names are the three specs/001-weights.md §5 defines. An unknown
// value is refused with the list rather than defaulted to auto, because a user
// who typed "fp16" asked for something specific and silently choosing for them
// is the failure §5 exists to prevent.
func parsePrecision(s string) (weights.Precision, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "auto", "":
		return weights.Auto, nil
	case "f16":
		return weights.F16, nil
	case "int8":
		return weights.Int8, nil
	case "int4":
		return weights.Int4, nil
	default:
		return weights.Auto, fmt.Errorf("%w: precision %q is not f16, int8, int4 or auto",
			errUsage, s)
	}
}

// parseDevice turns the --device flag into the engine's choice.
//
// The flag exists because the automatic choice can be wrong in a way the user
// can fix and the command line otherwise gives them no way to say so: accel
// opens the best backend present, and a backend that is present can still
// refuse to lower a kernel the graph needs. Without this flag such a machine
// has no way to run the model at all, though a working backend is installed on
// it.
func parseDevice(s string) (tgo.Device, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "auto", "":
		return tgo.AutoDevice, nil
	case "cpu":
		return tgo.CPU, nil
	case "metal", "gpu":
		return tgo.Metal, nil
	default:
		return tgo.AutoDevice, fmt.Errorf("%w: device %q is not auto, cpu or metal", errUsage, s)
	}
}

// checkPolicy refuses a sampling policy before it reaches the sampler.
//
// sample.Sampler.Next panics on a policy the device could not reproduce
// (specs/006-sampling.md §4), which is right for a library whose caller is a
// program and wrong for a command line whose caller is a person: a user who
// types --temp -1 must read a sentence, not a stack trace. The rules below are
// sample's own, restated at the boundary where a human types the number.
//
// It does not check LogitBias: the command line has no flag that sets one, so
// there is no value to refuse.
func checkPolicy(p sample.Policy) error {
	switch {
	case p.Temperature < 0 || math.IsNaN(float64(p.Temperature)):
		return fmt.Errorf("%w: temperature %v is negative or NaN; 0 is greedy", errUsage, p.Temperature)
	case p.TopK < 0:
		return fmt.Errorf("%w: top-k %d is negative; 0 disables the stage", errUsage, p.TopK)
	case p.TopK > sample.TopMaxRounds:
		return fmt.Errorf("%w: top-k %d is above the %d accel's kernel can do; it clamps silently, so this is refused",
			errUsage, p.TopK, sample.TopMaxRounds)
	case p.TopP < 0 || p.TopP > 1 || math.IsNaN(float64(p.TopP)):
		return fmt.Errorf("%w: top-p %v is outside [0, 1]; 0 disables the stage", errUsage, p.TopP)
	case p.RepetitionPenalty < 0 || math.IsNaN(float64(p.RepetitionPenalty)):
		return fmt.Errorf("%w: repetition penalty %v is negative or NaN; 1 is no penalty", errUsage, p.RepetitionPenalty)
	case math.IsNaN(float64(p.PresencePenalty)) || math.IsNaN(float64(p.FrequencyPenalty)):
		return fmt.Errorf("%w: a presence or frequency penalty of NaN is not a penalty", errUsage)
	case p.PenaltyWindow < 0:
		return fmt.Errorf("%w: penalty window %d is negative; 0 reads the whole context", errUsage, p.PenaltyWindow)
	}
	return nil
}

// positive refuses a count that must be at least one, naming the flag.
func positive(flagName string, n int) error {
	if n < 1 {
		return fmt.Errorf("%w: --%s is %d, and it counts something; it must be at least 1", errUsage, flagName, n)
	}
	return nil
}

// nonNegative refuses a count that may be zero but not less.
func nonNegative(flagName string, n int) error {
	if n < 0 {
		return fmt.Errorf("%w: --%s is %d, and it counts something; it cannot be negative", errUsage, flagName, n)
	}
	return nil
}
