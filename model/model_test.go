// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/latere-ai/tgo/chat"
)

// registerSeq numbers the architectures this file registers.
var registerSeq atomic.Int64

// TestOpen reads a directory the way a caller does, and gets the Qwen3 builder
// its architectures[0] names.
func TestOpen(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, configName), raw(t, good()), 0o600); err != nil {
		t.Fatal(err)
	}
	b, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got := b.Config().Architecture; got != Qwen3Architecture {
		t.Errorf("Architecture = %q, want %q", got, Qwen3Architecture)
	}
	if b.Template().TemplateChecksum() != chat.Qwen3TemplateChecksum {
		t.Error("Template() is not the Qwen3 renderer")
	}
	if len(b.Weights()) == 0 {
		t.Error("Weights() is empty")
	}
}

func TestOpenNoConfig(t *testing.T) {
	_, err := Open(t.TempDir())
	if err == nil {
		t.Fatal("Open accepted a directory with no config.json")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("error %q does not unwrap to os.ErrNotExist", err)
	}
}

// TestOpenRefusalNamesDirectory checks that a refused config still says which
// directory it came from: a caller that scans a cache of checkpoints gets a
// message that names one of them.
func TestOpenRefusalNamesDirectory(t *testing.T) {
	dir := t.TempDir()
	cfg := good()
	cfg["head_dim"] = 33
	if err := os.WriteFile(filepath.Join(dir, configName), raw(t, cfg), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Open(dir)
	if err == nil {
		t.Fatal("Open accepted an odd head_dim")
	}
	if !strings.Contains(err.Error(), dir) || !strings.Contains(err.Error(), "head_dim") {
		t.Errorf("error %q names neither the directory nor the field", err)
	}
}

// TestNewUnknownArchitecture is the refusal 004-D2 exists for. The message must
// carry the known set, or the caller's next step is to read this package.
func TestNewUnknownArchitecture(t *testing.T) {
	cfg := good()
	cfg["architectures"] = []string{"LlamaForCausalLM"}
	_, err := New(raw(t, cfg))
	if err == nil {
		t.Fatal("New accepted an unregistered architecture")
	}
	if !strings.Contains(err.Error(), "LlamaForCausalLM") {
		t.Errorf("error %q does not name the architecture asked for", err)
	}
	if !strings.Contains(err.Error(), Qwen3Architecture) {
		t.Errorf("error %q does not list the known architectures", err)
	}
}

// TestUnknownEmptyRegistry covers the message a binary that registers nothing
// would produce. It cannot be reached through New, because this package's own
// init registers Qwen3.
func TestUnknownEmptyRegistry(t *testing.T) {
	err := unknown("Whatever", nil)
	if !strings.Contains(err.Error(), "no architecture is registered") {
		t.Errorf("error %q does not say the registry is empty", err)
	}
}

func TestNewMalformed(t *testing.T) {
	cases := []struct {
		name   string
		config string
		names  string
	}{
		{"not json", "{", "parse config"},
		{"no architectures", `{"hidden_size": 64}`, "architectures[0]"},
		{"empty architectures", `{"architectures": []}`, "architectures[0]"},
		{"empty architecture name", `{"architectures": [""]}`, "architectures[0]"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(json.RawMessage(tc.config))
			if err == nil {
				t.Fatal("New accepted it")
			}
			if !strings.Contains(err.Error(), tc.names) {
				t.Errorf("error %q does not name %q", err, tc.names)
			}
		})
	}
}

// TestNewPropagatesConfigRefusal proves the registry does not swallow the
// config's own refusal: the constructor parses, and its error is the caller's.
func TestNewPropagatesConfigRefusal(t *testing.T) {
	cfg := good()
	cfg["num_key_value_heads"] = 3
	if _, err := New(raw(t, cfg)); err == nil ||
		!strings.Contains(err.Error(), "num_key_value_heads") {
		t.Fatalf("New error = %v, want one naming num_key_value_heads", err)
	}
}

func TestArchitecturesIsSortedAndCarriesQwen3(t *testing.T) {
	// Register names that straddle Qwen3ForCausalLM alphabetically before
	// asking. Without them the registry holds one entry, a one-element result
	// is sorted whatever the map iteration order was, and a sort dropped from
	// Architectures would pass this test every time. Registering in an order
	// that is not the sorted one is what makes the assertion carry weight; map
	// iteration is randomised, so a missing sort fails within a few runs and a
	// wrong sort fails on the first.
	n := registerSeq.Add(1)
	stub := func(json.RawMessage) (Builder, error) { return nil, nil }
	planted := []string{
		fmt.Sprintf("ZzzTestOnly%dForCausalLM", n),
		fmt.Sprintf("AaaTestOnly%dForCausalLM", n),
		fmt.Sprintf("MmmTestOnly%dForCausalLM", n),
	}
	for _, a := range planted {
		Register(a, stub)
	}

	got := Architectures()
	if len(got) < len(planted)+1 {
		t.Fatalf("Architectures() = %v, want at least the planted names and %q",
			got, Qwen3Architecture)
	}
	for i := 1; i < len(got); i++ {
		if got[i-1] >= got[i] {
			t.Fatalf("Architectures() is not sorted: %v", got)
		}
	}
	want := append(append([]string(nil), planted...), Qwen3Architecture)
	for _, a := range want {
		found := false
		for _, g := range got {
			found = found || g == a
		}
		if !found {
			t.Errorf("Architectures() = %v, missing %q", got, a)
		}
	}
}

// TestRegisterPanics covers each mistake that is a build-time error rather than
// a runtime condition. A second registration of one architecture is the one
// that matters: keeping either constructor silently picks the weight map of a
// model the caller did not ask for.
func TestRegisterPanics(t *testing.T) {
	stub := func(json.RawMessage) (Builder, error) { return nil, nil }
	cases := []struct {
		name string
		arch string
		new  func(json.RawMessage) (Builder, error)
		says string
	}{
		{"empty architecture", "", stub, "empty architecture"},
		{"nil constructor", "TestNilCtor", nil, "nil constructor"},
		{"duplicate", Qwen3Architecture, stub, "twice"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("Register did not panic")
				}
				if msg, ok := r.(string); !ok || !strings.Contains(msg, tc.says) {
					t.Errorf("panic %v does not say %q", r, tc.says)
				}
			}()
			Register(tc.arch, tc.new)
		})
	}
}

// TestRegisterAndOpenNewArchitecture is §6's promise that adding a model is one
// file and one init, exercised end to end through Open.
func TestRegisterAndOpenNewArchitecture(t *testing.T) {
	// A fresh name per run, so that -count=2 does not panic on a duplicate
	// registration: the registry is process-global and Register refuses a
	// second claim on one architecture, which is the behaviour under test
	// elsewhere in this file.
	arch := fmt.Sprintf("TestOnly%dForCausalLM", registerSeq.Add(1))
	Register(arch, func(r json.RawMessage) (Builder, error) {
		c, err := ParseConfig(r)
		if err != nil {
			return nil, err
		}
		return &qwen3{cfg: c}, nil
	})
	cfg := good()
	cfg["architectures"] = []string{arch}
	b, err := New(raw(t, cfg))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := b.Config().Architecture; got != arch {
		t.Errorf("Architecture = %q, want %q", got, arch)
	}
}
