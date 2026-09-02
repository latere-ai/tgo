// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"errors"
	"flag"
	"io"
	"math"
	"strings"
	"testing"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/sample"
	"github.com/latere-ai/tgo/weights"
)

func TestModelDir(t *testing.T) {
	for _, tc := range []struct {
		name    string
		args    []string
		want    string
		wantErr string
	}{
		{name: "one directory", args: []string{"models/qwen3"}, want: "models/qwen3"},
		{name: "flags before it", args: []string{"-n", "3", "models/qwen3"}, want: "models/qwen3"},
		{name: "none", args: nil, wantErr: "no model directory"},
		{name: "two", args: []string{"a", "b"}, wantErr: "one model directory, and 2 were given"},
		// The commonest way to get it wrong, and the one the shape of the
		// error does not explain: flag parsing stops at the first argument
		// that is not a flag, so a flag written after the directory is never
		// applied and the refusal has to say why.
		{name: "a flag after the directory", args: []string{"models/qwen3", "-n", "3"},
			wantErr: "flags go before it"},
		{name: "empty", args: []string{"  "}, wantErr: "the model directory is empty"},
		{name: "unknown flag", args: []string{"-nope", "a"}, wantErr: "flag provided but not defined"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs := flag.NewFlagSet("t", flag.ContinueOnError)
			fs.Int("n", 0, "")
			got, err := modelDir(fs, tc.args)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error = %v, want one containing %q", err, tc.wantErr)
				}
				if !errors.Is(err, errUsage) {
					t.Errorf("error does not wrap errUsage, so main would not print the usage")
				}
				return
			}
			if err != nil {
				t.Fatalf("modelDir: %v", err)
			}
			if got != tc.want {
				t.Errorf("modelDir = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestModelDirDoesNotExitOnABadFlag pins the reason every flag set is built
// with ContinueOnError and a discarded output: the default is to print to
// stderr and call os.Exit, which would make every refusal below untestable.
func TestModelDirDoesNotExitOnABadFlag(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	var sb strings.Builder
	fs.SetOutput(&sb)
	if _, err := modelDir(fs, []string{"-nope"}); err == nil {
		t.Fatal("a bad flag was accepted")
	}
	if sb.Len() != 0 {
		t.Errorf("the flag set wrote %q; the command reports refusals through its error", sb.String())
	}
	if fs.Output() != io.Discard {
		t.Error("modelDir did not silence the flag set's own output")
	}
}

func TestParsePrecision(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want weights.Precision
	}{
		{"auto", weights.Auto}, {"", weights.Auto}, {"AUTO", weights.Auto},
		{"f16", weights.F16}, {" int8 ", weights.Int8},
	} {
		got, err := parsePrecision(tc.in)
		if err != nil {
			t.Errorf("parsePrecision(%q): %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parsePrecision(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
	// fp16 is the spelling of every other framework, and defaulting it to auto
	// would silently run a precision the user did not ask for, which is what
	// specs/001-weights.md §5 exists to prevent.
	_, err := parsePrecision("fp16")
	if err == nil || !strings.Contains(err.Error(), "is not f16, int8, int4 or auto") {
		t.Fatalf("parsePrecision(fp16) error = %v, want one naming the four", err)
	}
	if !errors.Is(err, errUsage) {
		t.Error("the refusal does not wrap errUsage")
	}
}

// TestCheckPolicyRefusesWhatTheSamplerPanicsOn walks every refusal listed on
// sample.Policy. The sampler panics on each of them, which is right for a
// library and wrong for a command line: a user who types --temp -1 must read a
// sentence.
func TestCheckPolicyRefusesWhatTheSamplerPanicsOn(t *testing.T) {
	nan := float32(math.NaN())
	for _, tc := range []struct {
		name string
		p    sample.Policy
		want string
	}{
		{"negative temperature", sample.Policy{Temperature: -1}, "temperature -1 is negative"},
		{"NaN temperature", sample.Policy{Temperature: nan}, "is negative or NaN"},
		{"negative top-k", sample.Policy{TopK: -1}, "top-k -1 is negative"},
		{"top-k above the kernel", sample.Policy{TopK: sample.TopMaxRounds + 1}, "clamps silently"},
		{"top-p above one", sample.Policy{TopP: 1.5}, "top-p 1.5 is outside"},
		{"top-p below zero", sample.Policy{TopP: -0.5}, "outside [0, 1]"},
		{"NaN top-p", sample.Policy{TopP: nan}, "outside [0, 1]"},
		{"negative repetition penalty", sample.Policy{RepetitionPenalty: -2}, "repetition penalty -2 is negative"},
		{"NaN repetition penalty", sample.Policy{RepetitionPenalty: nan}, "negative or NaN"},
		{"NaN presence penalty", sample.Policy{PresencePenalty: nan}, "NaN is not a penalty"},
		{"NaN frequency penalty", sample.Policy{FrequencyPenalty: nan}, "NaN is not a penalty"},
		{"negative window", sample.Policy{PenaltyWindow: -3}, "penalty window -3 is negative"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkPolicy(tc.p)
			if err == nil {
				t.Fatalf("checkPolicy(%+v) accepted a policy the sampler panics on", tc.p)
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

// TestCheckPolicyAcceptsWhatTheSamplerAccepts runs the accepted policies
// through the sampler itself, so that the boundary this package draws is
// sample's boundary and not a second, stricter one.
func TestCheckPolicyAcceptsWhatTheSamplerAccepts(t *testing.T) {
	logits := []float32{0.5, -1, 2, 0.25}
	for _, p := range []sample.Policy{
		{},
		{Temperature: 0.7},
		{TopK: 1},
		{TopK: sample.TopMaxRounds},
		{TopP: 1},
		{TopP: 0.9, Temperature: 1},
		{RepetitionPenalty: 1.1, PenaltyWindow: 8},
		{PresencePenalty: 0.5, FrequencyPenalty: -0.5},
	} {
		if err := checkPolicy(p); err != nil {
			t.Fatalf("checkPolicy(%+v) = %v, want nil", p, err)
		}
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("checkPolicy accepted %+v and the sampler panicked: %v", p, r)
				}
			}()
			sample.New(1).Next(logits, []int{0, 1}, p)
		}()
	}
}

func TestCountFlags(t *testing.T) {
	if err := positive("tokens", 0); err == nil || !strings.Contains(err.Error(), "--tokens is 0") {
		t.Fatalf("positive(0) = %v, want a refusal naming the flag", err)
	}
	if err := positive("tokens", 1); err != nil {
		t.Fatalf("positive(1) = %v, want nil", err)
	}
	if err := nonNegative("warmup", -1); err == nil || !strings.Contains(err.Error(), "cannot be negative") {
		t.Fatalf("nonNegative(-1) = %v, want a refusal", err)
	}
	if err := nonNegative("warmup", 0); err != nil {
		t.Fatalf("nonNegative(0) = %v, want nil: a run with no warm-up is a cold measurement, not an error", err)
	}
}

// TestParseDevice pins the flag that exists because the automatic choice can be
// wrong in a way only the user can fix: accel opens the best backend present,
// and a backend that is present can still refuse to lower a kernel the graph
// needs. Without --device such a machine cannot run the model at all.
func TestParseDevice(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want tgo.Device
	}{
		{"", tgo.AutoDevice}, {"auto", tgo.AutoDevice}, {"AUTO", tgo.AutoDevice},
		{"cpu", tgo.CPU}, {" cpu ", tgo.CPU},
		{"metal", tgo.Metal}, {"gpu", tgo.Metal},
	} {
		got, err := parseDevice(tc.in)
		if err != nil {
			t.Errorf("parseDevice(%q): %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseDevice(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
	// An unknown device is refused with the list rather than defaulted to auto:
	// a user who typed "cuda" asked for something this build does not have, and
	// silently running elsewhere would put the wrong backend in every report
	// (017-D4).
	_, err := parseDevice("cuda")
	if err == nil {
		t.Fatal("--device cuda was accepted")
	}
	if !errors.Is(err, errUsage) || !strings.Contains(err.Error(), "not auto, cpu or metal") {
		t.Errorf("error = %v, want a usage refusal naming the three", err)
	}
}
